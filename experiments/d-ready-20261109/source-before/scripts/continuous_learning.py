#!/usr/bin/env python3
"""Local-only continuous self-play; no LLM/API, network client, or token use."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

JST = timezone(timedelta(hours=9))
STOP_NOW = False


def backend():
    from types import SimpleNamespace
    import continuous_opening_backend as module
    module.control = SimpleNamespace(**{k:globals()[k] for k in [
        'atomic_json','atomic_copy','sha','now','runtime_check','save_state','event',
        'run_job','validate_weights','evaluation_is_clean','promotion_ok']})
    return module


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp-' + uuid.uuid4().hex)
    shutil.copyfile(source, temp)
    with temp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(temp, target)


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    cfg['_file'] = str(path)
    for key in ['project', 'root', 'runtime', 'deck', 'initial_checkpoint', 'runtime_manifest']:
        cfg[key] = str(Path(cfg[key]).resolve())
    # Resolving a venv executable's final symlink would invoke the system Python
    # without its virtualenv packages. Preserve the venv invocation path.
    cfg['python'] = str(Path(cfg['python']).absolute())
    if cfg['steps_per_cycle'] < 128 or cfg['steps_per_cycle'] % 128:
        raise ValueError('steps_per_cycle must be a positive multiple of 128')
    for key in ['eval_episodes', 'reference_episodes', 'confirmation_episodes']:
        if cfg[key] < 2 * cfg['eval_envs'] or cfg[key] % (2 * cfg['eval_envs']):
            raise ValueError(f'{key} must be divisible by eval_envs * 2')
    return cfg


def event(cfg, kind, **fields):
    entry = dict(time_jst=now(), event=kind, **fields)
    root = Path(cfg['root'])
    with (root / 'history.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def save_state(cfg, state, **updates):
    state.update(updates, updated_jst=now())
    atomic_json(Path(cfg['root']) / 'state.json', state)
    root = Path(cfg['root'])
    text = (
        '# エルフェンノーツ＋キラーチューン 常時学習\n\n'
        f'- 更新: {state["updated_jst"]}\n'
        f'- 状態: {state.get("status", "unknown")}\n'
        f'- フェーズ: {state.get("phase", "-")}\n'
        f'- 完了区間: {state.get("completed_cycles", 0)}\n'
        f'- この常時学習で保存済みの追加判断数: {state.get("added_steps", 0):,}\n'
        f'- 最新の重み: {state.get("latest_checkpoint", "-")}\n'
        f'- 評価で採用した重み: {state.get("best_checkpoint", "-")}\n'
        f'- 最後の判定: {state.get("last_decision", "未評価")}\n'
        f'- エラー: {state.get("last_error") or "なし"}\n\n'
        '最新の重みと採用した重みは別です。学習継続だけでは改善を保証しません。\n'
        '操作: %LOCALAPPDATA%/YugiohAI/continuous-elfnotes.ps1 の Stop / Status / Start。\n'
    )
    temp = root / 'STATUS.md.tmp'
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, root / 'STATUS.md')


def runtime_check(cfg):
    root = Path(cfg['root'])
    expected = read_json(root / 'snapshot/runtime.json')
    for path_key, hash_key in [('native_binary','native_sha256'), ('vocabulary','vocabulary_sha256'), ('deck','deck_sha256')]:
        if sha(expected[path_key]) != expected[hash_key]:
            raise RuntimeError(f'Pinned runtime changed: {path_key}; stopped instead of mixing versions')
    hashes = read_json(root / 'snapshot/hashes.json')
    for name, expected_hash in hashes.items():
        if sha(root / 'snapshot' / name) != expected_hash:
            raise RuntimeError(f'Snapshot changed: {name}')


def initialize(cfg):
    if cfg.get('backend') == 'verified_opening':
        return backend().initialize(cfg)
    root = Path(cfg['root'])
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'state.json').is_file():
        runtime_check(cfg)
        return read_json(root / 'state.json')
    manifest = read_json(cfg['runtime_manifest'])
    prior = read_json(Path(cfg['initial_checkpoint']).parent / 'runtime.json')
    for key in ['deck_sha256', 'vocabulary_sha256']:
        if manifest[key] != prior[key]:
            raise RuntimeError(f'Initial checkpoint runtime mismatch: {key}')
    if prior['native_sha256'] not in manifest.get('compatible_checkpoint_native_sha256', [manifest['native_sha256']]):
        raise RuntimeError('Initial checkpoint native version is incompatible')
    snapshot = root / 'snapshot'
    snapshot.mkdir(exist_ok=True)
    files = {
        'train.py': Path(cfg['runtime']) / 'agent/scripts/cleanba.py',
        'evaluator.py': Path(cfg['project']) / 'scripts/evaluate.py',
        'runtime.json': Path(cfg['runtime_manifest']),
    }
    for name, source in files.items():
        atomic_copy(source, snapshot / name)
    atomic_json(snapshot / 'hashes.json', {name: sha(snapshot / name) for name in files})
    (root / 'cycles').mkdir(exist_ok=True)
    models = root / 'models'
    models.mkdir(exist_ok=True)
    for name in ['initial.flax_model', 'latest.flax_model', 'best.flax_model']:
        atomic_copy(cfg['initial_checkpoint'], models / name)
    atomic_copy(snapshot / 'runtime.json', models / 'runtime.json')
    state = dict(schema_version=1, created_jst=now(), status='ready', phase='idle', completed_cycles=0,
                 cycle=0, added_steps=0, best_added_steps=0, failures=0, pending=None,
                 latest_checkpoint=str(models / 'latest.flax_model'), best_checkpoint=str(models / 'best.flax_model'),
                 initial_checkpoint=cfg['initial_checkpoint'], initial_sha256=sha(cfg['initial_checkpoint']),
                 initial_training_steps=cfg['initial_training_steps'], pid=None, last_error=None,
                 resume_kind='weights warm start; optimizer/RNG/duel state are not restored',
                 network_or_llm_calls=False)
    save_state(cfg, state)
    runtime_check(cfg)
    return state


def job_env(cfg):
    runtime = Path(cfg['runtime']) / 'agent'
    env = dict(os.environ, PYTHONPATH=f'{Path(cfg["root"])/"snapshot"}:{Path(cfg["project"])/"scripts"}:{runtime}:{runtime / "ygoenv"}',
               PYTHONDONTWRITEBYTECODE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MALLOC_ARENA_MAX='2',
               WANDB_MODE='disabled', WANDB_DISABLED='true', HF_HUB_OFFLINE='1')
    env['YGO_CONTINUOUS_CONFIG'] = cfg['_file']
    return env


def stop_requested(cfg):
    return STOP_NOW or (Path(cfg['root']) / 'STOP').exists()


def immediate_stop(cfg):
    path = Path(cfg['root']) / 'STOP'
    return STOP_NOW or (path.is_file() and path.read_text().strip() == 'now')


def kill_job(process):
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_job(cfg, state, command, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scope = f'ygo-learn-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    wrapped = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + scope,
               '-p', f'MemoryMax={cfg["job_memory_gib"]}G', '-p', 'MemorySwapMax=0',
               '-p', f'CPUQuota={cfg["cpu_percent"]}%', 'nice', '-n', '10', *map(str, command)]
    atomic_json(directory / (name + '-command.json'), dict(command=wrapped, started_jst=now(), network_required=False))
    timeout = cfg['job_timeout_seconds']
    started = time.monotonic()
    with (directory / (name + '.log')).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(wrapped, cwd=Path(cfg['runtime']) / 'agent/scripts', env=job_env(cfg),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_state(cfg, state, child_pid=process.pid, active_log=str(directory / (name + '.log')))
        try:
            while process.poll() is None:
                if immediate_stop(cfg):
                    raise InterruptedError('Immediate stop requested; committed checkpoint retained')
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f'{name} exceeded {timeout}s')
                time.sleep(2)
                if time.monotonic() - started > 10:
                    with (directory / (name + '.log')).open('rb') as tail:
                        tail.seek(max(0, tail.seek(0, 2) - 8192))
                        log_tail=tail.read()
                        progress = re.findall(rb'(\d+) actor_update=', log_tail)
                    if b'Traceback (most recent call last):' in log_tail:
                        raise RuntimeError(f'{name} raised an exception; terminating its remaining actor threads; see {directory / (name + ".log")}')
                    save_state(cfg, state, job_elapsed_seconds=round(time.monotonic() - started),
                               training_progress_steps=int(progress[-1]) if progress and name == 'train' else None)
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}; see {directory / (name + ".log")}')
        finally:
            kill_job(process)
            state['child_pid'] = None
    return directory / (name + '.log')


def validate_weights(source, initial, trained):
    import flax.serialization
    import numpy as np
    def flatten(value, prefix=''):
        if isinstance(value, dict):
            return {key: leaf for name, child in value.items() for key, leaf in flatten(child, prefix + '/' + name).items()}
        return {prefix: np.asarray(value)}
    reference, before, after = [flatten(flax.serialization.msgpack_restore(Path(p).read_bytes())) for p in [source, initial, trained]]
    if reference.keys() != before.keys() or before.keys() != after.keys():
        raise RuntimeError('Checkpoint structure mismatch')
    changed = 0
    for key in before:
        if before[key].shape != after[key].shape or not np.isfinite(after[key]).all():
            raise RuntimeError('Invalid or non-finite checkpoint')
        if not np.array_equal(reference[key], before[key]):
            raise RuntimeError('Training did not start at the specified checkpoint')
        changed += int(np.count_nonzero(before[key] != after[key]))
    if not changed:
        raise RuntimeError('Training did not update any weights')
    return dict(all_finite=True, changed_values=changed, initial_matches_source=True, sha256=sha(trained))


def train_cycle(cfg, state, pending):
    if cfg.get('backend') == 'verified_opening':
        return backend().train_cycle(cfg,state,pending)
    directory = Path(pending['directory'])
    checkpoints = directory / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=False)
    atomic_copy(Path(cfg['root']) / 'snapshot/runtime.json', checkpoints / 'runtime.json')
    steps = pending['steps']
    updates = steps // 128
    command = [cfg['python'], '-u', str(Path(cfg['root']) / 'snapshot/train.py'),
               '--actor-device-ids','0','--learner-device-ids','0', '--deck',cfg['deck'],
               '--max-options','128','--timeout','30', '--local-num-envs','4','--local-env-threads','2',
               '--num-actor-threads','1','--num-steps','32','--num-minibatches','4',
               '--learning-rate',str(cfg['learning_rate']),'--vloss-clip','1.0',
               '--save-interval',str(min(32, updates)), '--local-eval-episodes','4',
               '--eval-interval',str(updates),'--total-timesteps',str(steps),
               '--seed',str(pending['training_seed']), '--log-frequency','8',
               '--ckpt-dir',str(checkpoints),'--tb-dir',str(directory / 'tensorboard'),
               '--checkpoint',pending['source']]
    log_path = run_job(cfg, state, command, directory, 'train')
    log = log_path.read_text(errors='replace')
    if 'NATIVE_LUA_ERRORS 0' not in log or f'actor_update={updates},' not in log:
        raise RuntimeError('Training completion / Lua error check failed')
    if "actor_devices=['cuda:0']" not in log or "learner_devices=['cuda:0']" not in log:
        raise RuntimeError('Training did not use the GPU')
    files = list(checkpoints.glob(f'*_{steps}steps.flax_model'))
    if len(files) != 1:
        raise RuntimeError('Expected one complete final checkpoint')
    result = validate_weights(pending['source'], checkpoints / 'initial.flax_model', files[0])
    result.update(checkpoint=str(files[0]), steps=steps, completed_jst=now())
    atomic_json(directory / 'training-result.json', result)
    return files[0]


def evaluation_is_clean(summary, requested):
    result = summary['results']
    return (result['total_episodes'] == requested and result.get('verified_episodes') == requested
            and summary['terminations']['abnormal_total'] == 0
            and not result.get('requires_rerun_or_exclusion', False))


def promotion_ok(summary, requested):
    return evaluation_is_clean(summary, requested) and summary['results']['win_rate_ci95'][0] > 0.5


def evaluate_job(args, cfg):
    import gc
    import ctypes
    root = Path(cfg['root'])
    runtime_check(cfg)
    runtime = Path(cfg['runtime']) / 'agent'
    sys.path[:0] = [str(runtime), str(runtime / 'ygoenv')]
    spec = importlib.util.spec_from_file_location('continuous_pinned_evaluator', root / 'snapshot/evaluator.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = module.EvalContext(runtime / 'scripts', Path(cfg['deck']), max_options=128,
                             code_list_file=cfg.get('vocabulary','code_list.txt'),
                             num_envs=cfg['eval_envs'], env_threads=2,
                             max_steps=cfg['max_eval_steps'], timeout=60)
    if cfg.get('backend') == 'verified_opening':
        backend().install_evaluation(ctx,cfg)
    from ygoenv.ygopro.ygopro_ygoenv import get_lua_error_count
    original = ctx._run_single_wave_seat
    def tracked(*positional, **keywords):
        result = original(*positional, **keywords)
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        print('PROGRESS', keywords.get('wave_idx'), keywords.get('candidate_seat'), flush=True)
        if get_lua_error_count():
            raise RuntimeError('Lua errors during evaluation')
        return result
    ctx._run_single_wave_seat = tracked
    candidate,opponent=args.candidate,args.opponent
    if cfg.get('backend') == 'verified_opening':
        candidate=str(Path(candidate)/'base.flax_model');opponent=str(Path(opponent)/'base.flax_model')
    summary = ctx.evaluate_matchup(candidate, opponent, Path(cfg['project']),
                                  num_episodes=args.episodes, base_seed=args.seed,
                                  max_steps_per_wave=cfg['max_eval_steps'])
    if get_lua_error_count():
        raise RuntimeError('Lua errors during evaluation')
    atomic_json(args.result, summary)
    print('RESULT', json.dumps(summary['results']), flush=True)


def evaluate_candidate(cfg, state, pending, candidate):
    directory = Path(pending['directory'])
    regression=None
    if cfg.get('backend') == 'verified_opening':
        save_state(cfg,state,phase='opening_regression')
        regression=backend().regression_gate(cfg,state,pending,candidate)
        if not regression['passed']:
            decision=dict(clean=False,promoted=False,opening_regression=regression,reason='opening_regression')
            atomic_json(directory/'decision.json',decision)
            return decision
    result_path = directory / ('evaluation-' + uuid.uuid4().hex[:8] + '.json')
    command = [cfg['python'], '-u', str(Path(__file__).resolve()), 'job-eval', '--config', cfg['_file'],
               '--candidate',str(candidate),'--opponent',state['best_checkpoint'],
               '--episodes',str(cfg['eval_episodes']),'--seed',str(pending['eval_seed']), '--result',str(result_path)]
    save_state(cfg, state, phase='evaluating')
    run_job(cfg, state, command, directory, 'evaluation')
    summary = read_json(result_path)
    decision = dict(scout=summary['results'], clean=evaluation_is_clean(summary, cfg['eval_episodes']),
                    promoted=False, evidence=str(result_path))
    if regression is not None:decision['opening_regression']=regression
    reference_path = directory / ('reference-' + uuid.uuid4().hex[:8] + '.json')
    command[command.index('--opponent') + 1] = str(Path(cfg['root']) / 'models/initial.flax_model')
    command[command.index('--episodes') + 1] = str(cfg['reference_episodes'])
    command[command.index('--result') + 1] = str(reference_path)
    save_state(cfg, state, phase='reference_evaluation')
    run_job(cfg, state, command, directory, 'reference')
    reference = read_json(reference_path)
    decision.update(reference=reference['results'], reference_evidence=str(reference_path))
    reference_clean = evaluation_is_clean(reference, cfg['reference_episodes'])
    decision['clean'] = decision['clean'] and reference_clean
    command[command.index('--opponent') + 1] = state['best_checkpoint']
    # Separate fresh deals confirm promising candidates. This is an engineering gate,
    # not a time-uniform statistical guarantee over infinitely many repeated tests.
    if promotion_ok(summary, cfg['eval_episodes']) and reference_clean and reference['results']['win_rate'] >= .5:
        confirmation = directory / ('confirmation-' + uuid.uuid4().hex[:8] + '.json')
        command[command.index('--episodes') + 1] = str(cfg['confirmation_episodes'])
        command[command.index('--seed') + 1] = str(pending['eval_seed'] + 1)
        command[command.index('--result') + 1] = str(confirmation)
        save_state(cfg, state, phase='confirming')
        run_job(cfg, state, command, directory, 'confirmation')
        checked = read_json(confirmation)
        decision.update(confirmation=checked['results'], confirmation_evidence=str(confirmation),
                        promoted=promotion_ok(checked, cfg['confirmation_episodes']))
    atomic_json(directory / 'decision.json', decision)
    return decision


def prune_checkpoints(cfg, state):
    if cfg.get('backend') == 'verified_opening':return backend().prune(cfg,state)
    root = Path(cfg['root']).resolve()
    keep = cfg['keep_cycle_checkpoints']
    directories = sorted((root / 'cycles').iterdir(), key=lambda p: p.name)
    for directory in directories[:-keep]:
        # Never touch prior experiments, best/latest copies, or active candidates.
        if state.get('pending') and directory == Path(state['pending']['directory']):
            continue
        if directory.resolve().parent != root / 'cycles' or directory.is_symlink():
            raise RuntimeError('Unsafe retention path')
        source = directory / 'source.flax_model'
        if source.is_file() and source.resolve().parent == directory:
            source.unlink()
        for path in (directory / 'checkpoints').glob('*.flax_model'):
            if path.resolve().parent == directory / 'checkpoints':
                path.unlink()
        for path in (directory / 'tensorboard').rglob('*tfevents*'):
            if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
                path.unlink()


def interrupted_candidate(cfg, state):
    if cfg.get('backend') == 'verified_opening':return backend().recover(cfg,state)
    pending = state.get('pending')
    if not pending or pending['phase'] != 'training':
        return
    directory = Path(pending['directory'])
    files = list((directory / 'checkpoints').glob('*steps.flax_model'))
    files.sort(key=lambda p: int(re.search(r'_(\d+)steps', p.name)[1]), reverse=True)
    for path in files:
        try:
            result = validate_weights(pending['source'], directory / 'checkpoints/initial.flax_model', path)
        except Exception:
            continue
        steps = int(re.search(r'_(\d+)steps', path.name)[1])
        atomic_copy(path, state['latest_checkpoint'])
        state['added_steps'] = pending['starting_added_steps'] + steps
        event(cfg, 'recovered_intermediate', checkpoint=str(path), steps=steps, **result)
        break
    state['pending'] = None
    save_state(cfg, state)


def run_loop(cfg, once=False):
    global STOP_NOW
    root = Path(cfg['root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An existing learner already holds the lock')
        state = initialize(cfg)
        if (root / 'STOP').exists():
            save_state(cfg, state, status='stopped', pid=None, phase='idle')
            return
        signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__('STOP_NOW', True))
        signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('STOP_NOW', True))
        save_state(cfg, state, status='running', pid=os.getpid(), last_error=None)
        event(cfg, 'worker_started', pid=os.getpid())
        try:
            interrupted_candidate(cfg, state)
            while not stop_requested(cfg):
                try:
                    runtime_check(cfg)
                    if shutil.disk_usage(root).free < cfg['minimum_free_gib'] * 1024**3:
                        raise RuntimeError('Insufficient free disk; preserving checkpoints')
                    pending = state.get('pending')
                    if not pending:
                        cycle = state['cycle'] + 1
                        steps = cfg['bootstrap_steps'] if state['completed_cycles'] == 0 else cfg['steps_per_cycle']
                        directory = root / 'cycles' / f'{cycle:06d}-{uuid.uuid4().hex[:8]}'
                        directory.mkdir()
                        source = directory / 'source.flax_model'
                        if cfg.get('backend')=='verified_opening':
                            source=directory/'source-model';source.mkdir()
                            for item in Path(state['latest_checkpoint']).iterdir():
                                if item.is_file():atomic_copy(item,source/item.name)
                        else:atomic_copy(state['latest_checkpoint'], source)
                        pending = dict(cycle=cycle, directory=str(directory), source=str(source), steps=steps,
                                       starting_added_steps=state['added_steps'], phase='training',
                                       training_seed=cfg['training_seed_base'] + cycle,
                                       eval_seed=cfg['eval_seed_base'] + cycle * 997)
                        save_state(cfg, state, cycle=cycle, pending=pending)
                    if pending['phase'] == 'training':
                        save_state(cfg, state, phase='training')
                        event(cfg, 'training_started', cycle=pending['cycle'], steps=pending['steps'])
                        candidate = train_cycle(cfg, state, pending)
                        if cfg.get('backend')=='verified_opening':state['latest_checkpoint']=str(candidate)
                        else:atomic_copy(candidate, state['latest_checkpoint'])
                        pending.update(phase='evaluating', candidate=str(candidate))
                        save_state(cfg, state, added_steps=pending['starting_added_steps'] + pending['steps'], pending=pending)
                    if stop_requested(cfg):
                        break
                    decision = evaluate_candidate(cfg, state, pending, Path(pending['candidate']))
                    if decision['promoted']:
                        if cfg.get('backend')=='verified_opening':
                            state['best_checkpoint']=pending['candidate']
                            atomic_copy(decision['opening_regression']['evidence'],Path(cfg['root'])/'opening-best.json')
                        else:atomic_copy(pending['candidate'], state['best_checkpoint'])
                        state['best_added_steps'] = state['added_steps']
                    if decision.get('opening_regression',{}).get('passed') is False:
                        state['latest_checkpoint']=state['best_checkpoint']
                    label = ('opening_regression_rejected' if decision.get('opening_regression',{}).get('passed') is False
                             else 'promoted' if decision['promoted'] else 'not_promoted' if decision['clean']
                             else 'invalid_evaluation_not_promoted')
                    save_state(cfg, state, pending=None, completed_cycles=state['completed_cycles'] + 1,
                               phase='idle', failures=0, last_error=None, last_decision=label,
                               last_evaluation=decision, last_completed_directory=pending['directory'])
                    event(cfg, 'cycle_completed', cycle=pending['cycle'], added_steps=state['added_steps'], decision=label)
                    prune_checkpoints(cfg, state)
                    if once:
                        break
                    until = time.monotonic() + cfg['rest_seconds']
                    while time.monotonic() < until and not stop_requested(cfg):
                        time.sleep(1)
                except InterruptedError:
                    break
                except Exception as exc:
                    failures = state.get('failures', 0) + 1
                    save_state(cfg, state, failures=failures, last_error=f'{type(exc).__name__}: {exc}', phase='error_backoff')
                    event(cfg, 'job_failed', error=state['last_error'], failures=failures)
                    interrupted_candidate(cfg, state)
                    if failures >= cfg['max_consecutive_failures']:
                        save_state(cfg, state, status='error', phase='paused', pid=None)
                        return
                    until = time.monotonic() + min(300, 30 * 2**(failures - 1))
                    while time.monotonic() < until and not stop_requested(cfg):
                        time.sleep(1)
        finally:
            if state.get('status') != 'error':
                interrupted_candidate(cfg, state)
                save_state(cfg, state, status='stopped', phase='idle', pid=None, child_pid=None)
            event(cfg, 'worker_stopped', status=state['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','status','stop','resume','doctor','job-eval','job-openings'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--now', action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--opponent')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['root'])
    if args.action == 'job-openings':
        runtime_check(cfg)
        backend().openings_job(args,cfg)
    elif args.action == 'job-eval':
        evaluate_job(args, cfg)
    elif args.action == 'run':
        run_loop(cfg, args.once)
    elif args.action == 'doctor':
        state = initialize(cfg)
        print(json.dumps(dict(ok=True, root=str(root), state=state, memory_limit_gib=cfg['job_memory_gib']), ensure_ascii=False, indent=2))
    elif args.action == 'resume':
        root.mkdir(parents=True, exist_ok=True)
        (root / 'STOP').unlink(missing_ok=True)
        print('Resume enabled; start the Windows scheduled task or run the worker.')
    elif args.action == 'stop':
        root.mkdir(parents=True, exist_ok=True)
        (root / 'STOP').write_text('now' if args.now else 'after_phase', encoding='utf-8')
        print('Stop requested. Complete checkpoints and best model are retained.')
    else:
        if not (root / 'state.json').exists():
            print('Not initialized')
            return
        state = read_json(root / 'state.json')
        with (root / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state['worker_running'] = False
            except BlockingIOError:
                state['worker_running'] = True
        print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
