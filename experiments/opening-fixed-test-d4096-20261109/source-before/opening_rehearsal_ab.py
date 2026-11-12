#!/usr/bin/env python3
"""Finite A/B: current PPO vs PPO + train-split opening rehearsal, in isolated roots.

- Never writes to the production root, its STOP, state or adopted model; refuses
  any arm root inside it and verifies its fingerprint before and after.
- Never starts or resumes the resident loop and never adopts a candidate.
- Each arm gets its own freshly generated snapshot (trainer, policy code, data).
- No LLM/API/network client.

Usage:
  python -B scripts/opening_rehearsal_ab.py run --plan PLAN.json [--arms a,b] [--seeds 1,2]
  python -B scripts/opening_rehearsal_ab.py summarize --plan PLAN.json
  python -B scripts/opening_rehearsal_ab.py diagnose --plan PLAN.json --arms ARM --seeds S1,S2

A run directory is reused only when its recorded conditions (run-conditions.json) and the
job commands/files it actually used match the current plan exactly; otherwise the runner
refuses. Every arm the runner touched is left with STOP and a final state, also on failure.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuous_learning as C  # noqa: E402
import opening_rehearsal as R  # noqa: E402

PROJECT = HERE.parent
PRODUCTION_CONFIG = PROJECT / 'config/continuous-elfnotes.json'
EVALUATION_VERSION = 'opening-eval-v1: validation 66 free-run openings; every baseline-passing case_id must stay passing'
CONDITION_FORMAT = 2  # 2: effective learning rate and the saved arm config are part of the conditions
RESULT_FUNCTIONS = ('job_teacher_forced', 'load_split_for_diagnostics', 'family_counts', 'weight_change')
ARM_SETTINGS = ('rehearsal', 'learning_rate')  # the only per-arm overrides; anything else is refused
SHARED_SNAPSHOT_FILES = ['evaluator.py', 'opening-manifest.json', 'opening-policy.json', 'runtime.json', 'meta-run-manifest.json']


# ---------------------------------------------------------------- safety

def production_root():
    return Path(C.read_json(PRODUCTION_CONFIG)['root']).resolve()


def assert_isolated(path):
    path, production = Path(path).resolve(), production_root()
    if path == production or path.is_relative_to(production) or production.is_relative_to(path):
        raise RuntimeError(f'Refusing to use the production root: {path}')
    return path


def production_fingerprint():
    root = production_root()
    names = ['STOP', 'state.json', 'opening-baseline.json', 'snapshot/hashes.json',
             'models/initial/base.flax_model', 'models/initial/chain-head.flax_model', 'models/initial/metadata.json']
    value = {name: C.sha(root / name) for name in names}
    value['STOP_content'] = (root / 'STOP').read_text().strip()
    value['production_root'] = str(root)
    value['config/continuous-elfnotes.json'] = C.sha(PRODUCTION_CONFIG)
    policy_config = Path(C.read_json(PRODUCTION_CONFIG)['verified_policy_config'])
    value['verified_policy_config'] = str(policy_config)
    value['verified_policy_config_sha256'] = C.sha(policy_config)
    value['config/latest-verified-opening-policy.json'] = C.sha(PROJECT / 'config/latest-verified-opening-policy.json')
    state = C.read_json(root / 'state.json')
    value['state_status'], value['state_pid'] = state['status'], state['pid']
    if value['STOP_content'] != 'now' or state['status'] != 'stopped' or state['pid'] is not None:
        raise RuntimeError('Production is not in the stopped state; refusing to run experiments')
    return value


def resident_learners():
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines()
            if 'continuous_learning.py' in line and ' run' in line and 'ps -eo' not in line]


# ---------------------------------------------------------------- resource accounting

class GpuPeak:
    def __init__(self):
        self.samples, self.stop = [], threading.Event()

    @staticmethod
    def sample():
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10)
            return [float(x) for x in r.stdout.strip().splitlines()[0].split(',')]
        except Exception:
            return None

    def __enter__(self):
        self.before = self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        while not self.stop.wait(2):
            value = self.sample()
            if value:
                self.samples.append(value)

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=15)

    def summary(self):
        return dict(gpu_memory_used_before_mib=self.before[0] if self.before else None,
                    gpu_memory_used_peak_mib=max((s[0] for s in self.samples), default=None),
                    gpu_utilization_peak_percent=max((s[1] for s in self.samples), default=None),
                    gpu_samples=len(self.samples))


def timed(original):
    def run_job(cfg, state, command, directory, name):
        assert_isolated(cfg['root'])
        started = time.monotonic()
        with GpuPeak() as gpu:
            log = original(cfg, state, ['/usr/bin/time', '-v', *map(str, command)], directory, name)
        text = Path(log).read_text(errors='replace')
        rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', text)
        C.atomic_json(Path(directory) / f'{name}-resources.json', dict(
            name=name, wall_seconds=round(time.monotonic() - started, 1),
            max_rss_mib=round(int(rss[-1]) / 1024, 1) if rss else None, **gpu.summary()))
        return log
    return run_job


def backend():
    module = C.backend()
    module.control.run_job = timed(C.run_job)
    return module


# ---------------------------------------------------------------- per-arm setup

def validate_arm(name, arm):
    unknown = sorted(set(arm) - set(ARM_SETTINGS))
    if unknown:
        raise ValueError(f'Arm {name}: unsupported settings {unknown}; allowed: {list(ARM_SETTINGS)}')
    if arm.get('rehearsal'):
        R.validate_settings(arm['rehearsal'])
    if 'learning_rate' in arm:
        rate = arm['learning_rate']
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < float(rate) < 1:
            raise ValueError(f'Arm {name}: learning_rate must be a number in (0, 1)')


def build_arm_config(plan, name, arm):
    """The effective config of an arm: base config + ONLY the allowed per-arm settings."""
    validate_arm(name, arm)
    base = C.read_json(plan.get('base_config', PRODUCTION_CONFIG))
    root = assert_isolated(Path(plan['experiment_root']) / name)
    cfg = dict(base, root=str(root), experiment_arm=name, experiment_plan=str(plan['_file']),
               resident_operation_allowed=False)
    cfg.pop('rehearsal', None)
    if arm.get('rehearsal'):
        cfg['rehearsal'] = dict(arm['rehearsal'])
    if 'learning_rate' in arm:
        cfg['learning_rate'] = float(arm['learning_rate'])
    return cfg


def arm_config(plan, name, arm):
    cfg = build_arm_config(plan, name, arm)
    path = Path(plan['experiment_root']) / 'configs' / f'{name}.json'
    if path.exists():
        if C.read_json(path) != cfg:
            raise RuntimeError(f'Arm config changed after the arm was created: {path}')
    else:
        C.atomic_json(path, cfg)
    return C.load_config(path)


def check_snapshot(cfg, arm):
    root, production = Path(cfg['root']), production_root()
    ours, theirs = C.read_json(root / 'snapshot/hashes.json'), C.read_json(production / 'snapshot/hashes.json')
    shared = [n for n in theirs if n.startswith('policy-code/')] + SHARED_SNAPSHOT_FILES

    def content(base, name):
        # pin_helper_metadata() writes the root's own snapshot path into one helper.
        return (base / 'snapshot' / name).read_bytes().replace(str(base / 'snapshot').encode(), b'<SNAPSHOT>')
    different = [n for n in shared if ours.get(n) != theirs.get(n) and content(root, n) != content(production, n)]
    path_only = [n for n in shared if ours.get(n) != theirs.get(n) and n not in different]
    if different:
        raise RuntimeError(f'Arm snapshot differs from the production snapshot: {different}')
    identical_trainer = ours['train.py'] == theirs['train.py']
    if bool(arm.get('rehearsal')) == identical_trainer:
        raise RuntimeError('Trainer identity does not match the arm definition')
    for name in ['base.flax_model', 'chain-head.flax_model']:
        if C.sha(root / 'models/initial' / name) != C.sha(production / 'models/initial' / name):
            raise RuntimeError('Arm initial model differs from the adopted model')
    extra = sorted(n for n in ours if n not in theirs)
    if any(n.endswith(('validation.npz', 'holdout.npz')) for n in ours):
        raise RuntimeError('A non-train split was pinned into the snapshot')
    return dict(shared_files_identical=len(shared), files_differing_only_by_root_path=path_only,
                trainer_identical_to_production=identical_trainer,
                trainer_sha256=ours['train.py'], backend_sha256=ours['continuous_opening_backend.py'],
                rehearsal_files=extra, initial_model_identical_to_adopted=True)


def setup_arm(B, plan, name, arm, cfg=None):
    cfg = cfg or arm_config(plan, name, arm)
    root = Path(cfg['root'])
    if (root / 'arm-setup.json').exists() and (root / 'STOP').exists():
        (root / 'STOP').unlink()  # this runner's own completion marker for an experiment arm; never production
    state = B.initialize(cfg)
    snapshot = check_snapshot(cfg, arm)
    baseline_path = root / 'opening-baseline.json'
    if not baseline_path.exists():
        B.control.run_job(cfg, state, [cfg['python'], '-B', '-u', str(HERE / 'continuous_learning.py'), 'job-openings',
                                       '--config', cfg['_file'], '--candidate', str(root / 'models/initial'),
                                       '--result', str(baseline_path)], root / 'baseline', 'opening-baseline')
    baseline = C.read_json(baseline_path)
    reference = C.read_json(production_root() / 'opening-baseline.json')
    same = baseline['passed_case_ids'] == reference['passed_case_ids'] and baseline['total'] == reference['total']
    if not same or baseline['lua_errors'] != 0:
        raise RuntimeError('Pre-training opening baseline does not reproduce the adopted 65/66 cases')
    teacher_initial = root / 'baseline/teacher-forced-initial.json'
    if not teacher_initial.exists():
        teacher_forced(B, cfg, state, root / 'models/initial', teacher_initial, root / 'baseline', 'teacher-forced-initial')
    return cfg, state, dict(snapshot=snapshot, baseline_passed=baseline['passed'], baseline_total=baseline['total'],
                            baseline_reproduced=same, baseline_evidence=str(baseline_path))


def teacher_forced(B, cfg, state, checkpoint, result, directory, name):
    B.control.run_job(cfg, state, [cfg['python'], '-B', '-u', str(Path(__file__).resolve()), 'job-teacher-forced',
                                   '--config', cfg['_file'], '--candidate', str(checkpoint), '--result', str(result)],
                      directory, name)
    return C.read_json(result)


# ---------------------------------------------------------------- run conditions

def result_code_fingerprint():
    """Code that determines a run's recorded numbers (the pinned snapshot is fingerprinted separately)."""
    source = ''.join(inspect.getsource(globals()[name]) for name in RESULT_FUNCTIONS)
    return {'scripts/continuous_learning.py': C.sha(HERE / 'continuous_learning.py'),
            'scripts/continuous_opening_backend.py': C.sha(HERE / 'continuous_opening_backend.py'),
            'scripts/opening_rehearsal.py': C.sha(HERE / 'opening_rehearsal.py'),
            'runner_result_functions': hashlib.sha256(source.encode()).hexdigest()}


def requested_conditions(plan, name, arm, cfg, seed):
    root = Path(cfg['root'])
    return dict(format=CONDITION_FORMAT, arm=name, steps=int(plan['steps']), training_seed=int(seed),
                battle_episodes=int(plan['battle_episodes']), battle_seed=int(plan['battle_seed']),
                battle_opponent='models/initial', learning_rate=float(cfg['learning_rate']),
                arm_config_sha256=C.sha(cfg['_file']),
                initial_base_sha256=C.sha(root / 'models/initial/base.flax_model'),
                initial_head_sha256=C.sha(root / 'models/initial/chain-head.flax_model'),
                rehearsal=R.validate_settings(arm['rehearsal']) if arm.get('rehearsal') else None,
                snapshot_hashes_sha256=C.sha(root / 'snapshot/hashes.json'),
                evaluation_version=EVALUATION_VERSION, code=result_code_fingerprint())


def condition_differences(stored, wanted):
    return sorted(k for k in set(stored) | set(wanted) if stored.get(k) != wanted.get(k))


def recorded_flag(path, flag):
    command = [str(x) for x in C.read_json(path)['command']]
    return command[command.index(flag) + 1]


def evidence_problems(directory, root, wanted):
    """Compare what the recorded job commands and files actually used with the requested conditions."""
    problems = []
    if (directory / 'train-command.json').exists():
        flag = lambda f: recorded_flag(directory / 'train-command.json', f)
        if int(flag('--total-timesteps')) != wanted['steps']:
            problems.append('training steps in train-command.json')
        if int(flag('--seed')) != wanted['training_seed']:
            problems.append('training seed in train-command.json')
        if float(flag('--learning-rate')) != wanted['learning_rate']:
            problems.append('learning rate in train-command.json')
        if Path(flag('--checkpoint')).resolve().parent != directory.resolve():
            problems.append('training checkpoint outside the run directory')
        for name, key in [('base.flax_model', 'initial_base_sha256'), ('chain-head.flax_model', 'initial_head_sha256')]:
            if C.sha(directory / 'source-model' / name) != wanted[key]:
                problems.append(f'starting {name}')
    if (directory / 'battle-command.json').exists():
        flag = lambda f: recorded_flag(directory / 'battle-command.json', f)
        if int(flag('--episodes')) != wanted['battle_episodes']:
            problems.append('battle episodes in battle-command.json')
        if int(flag('--seed')) != wanted['battle_seed']:
            problems.append('battle seed in battle-command.json')
        if Path(flag('--opponent')).resolve() != (root / 'models/initial').resolve():
            problems.append('battle opponent')
        if Path(flag('--candidate')).resolve() != (directory / 'candidate').resolve():
            problems.append('battle candidate')
    if (directory / 'run-result.json').exists():
        result = C.read_json(directory / 'run-result.json')
        if (result['steps'], result['seed']) != (wanted['steps'], wanted['training_seed']):
            problems.append('steps/seed in run-result.json')
        for name, key in [('base.flax_model', 'base_sha256'), ('chain-head.flax_model', 'head_sha256')]:
            if C.sha(directory / 'candidate' / name) != result[key]:
                problems.append(f'candidate {name} hash')
        found = (result.get('battle') or {}).get('results')
        if found and found.get('total_episodes') != wanted['battle_episodes']:
            problems.append('battle episodes in run-result.json')
    return problems


def verified_reuse(directory, root, wanted):
    """Refuse, never guess: a run directory is reusable only under identical recorded conditions."""
    path = directory / 'run-conditions.json'
    if not path.exists():
        raise RuntimeError(f'{directory}: no recorded run conditions (earlier runner version); refusing to reuse it. '
                           'Use a new experiment_root.')
    different = condition_differences(C.read_json(path), wanted)
    if different:
        raise RuntimeError(f'{directory}: recorded run conditions differ from the current plan {different}; '
                           'refusing to reuse it. Use a new experiment_root.')
    problems = evidence_problems(directory, root, wanted)
    if problems:
        raise RuntimeError(f'{directory}: recorded evidence does not match the requested conditions {problems}')


# ---------------------------------------------------------------- one training run

def run_one(B, plan, name, arm, cfg, state, seed):
    root = Path(cfg['root'])
    directory = root / 'cycles' / f"{plan['steps']}-seed{seed}"
    done = directory / 'run-result.json'
    wanted = requested_conditions(plan, name, arm, cfg, seed)
    trained = (directory / 'training-result.json').exists() and (directory / 'candidate/chain-head.flax_model').exists()
    if done.exists() or trained:
        verified_reuse(directory, root, wanted)
        if done.exists():
            return C.read_json(done)
    elif directory.exists():
        directory.rename(directory.with_name(directory.name + '-incomplete-' + uuid.uuid4().hex[:6]))
    if trained:  # resume after a later stage failed; the validated candidate is reused, not retrained
        pending = C.read_json(directory / 'pending.json')
        candidate = directory / 'candidate'
    else:
        directory.mkdir(parents=True)
        C.atomic_json(directory / 'run-conditions.json', wanted)
        source = directory / 'source-model'
        source.mkdir()
        for item in (root / 'models/initial').iterdir():
            if item.is_file():
                C.atomic_copy(item, source / item.name)
        pending = dict(cycle=0, directory=str(directory), source=str(source), steps=plan['steps'], starting_added_steps=0,
                       phase='training', training_seed=seed, eval_seed=plan['battle_seed'])
        C.atomic_json(directory / 'pending.json', pending)
        candidate = B.train_cycle(cfg, state, pending)
    training = C.read_json(directory / 'training-result.json')
    log = (directory / 'train.log').read_text(errors='replace')
    updates = plan['steps'] // 128
    rehearsal = None
    if arm.get('rehearsal'):
        rows = [json.loads(line) for line in (directory / 'rehearsal-metrics.jsonl').read_text().splitlines()]
        if len(rows) != updates or f'OPENING_REHEARSAL_UPDATES {updates}' not in log:
            raise RuntimeError('Rehearsal did not run on every PPO update')
        keys = R.METRIC_NAMES
        rehearsal = dict(updates=len(rows), minibatch_updates=sum(len(r['minibatches']) for r in rows),
                         sequences_sampled=sum(len(m) for r in rows for shard in r['sampled_sequences'] for m in shard),
                         loss_decisions=int(sum(m['loss_steps'] for r in rows for m in r['minibatches'])),
                         first=rows[0]['mean'], last=rows[-1]['mean'],
                         mean={k: sum(r['mean'][k] for r in rows) / len(rows) for k in keys},
                         learner_seconds_mean=sum(r['learner_seconds'] for r in rows) / len(rows))
    elif 'OPENING_REHEARSAL' in log:
        raise RuntimeError('Control arm trainer reported rehearsal activity')
    baseline = C.read_json(root / 'opening-baseline.json')
    if (directory / 'opening-regression.json').exists():
        current = C.read_json(directory / 'opening-regression.json')
        regression = dict(passed=B.regression_ok(baseline, current), baseline_passed=baseline['passed'],
                          candidate_passed=current['passed'], total=current['total'],
                          evidence=str(directory / 'opening-regression.json'),
                          lost_case_ids=sorted(set(baseline['passed_case_ids']) - set(current['passed_case_ids'])))
    else:
        regression = B.regression_gate(cfg, state, pending, candidate)
    openings = C.read_json(regression['evidence'])
    if (directory / 'teacher-forced.json').exists():
        teacher = C.read_json(directory / 'teacher-forced.json')
    else:
        teacher = teacher_forced(B, cfg, state, candidate, directory / 'teacher-forced.json', directory, 'teacher-forced')
    battle_path = directory / 'battle.json'
    try:
        if not battle_path.exists():
            B.control.run_job(cfg, state, [cfg['python'], '-B', '-u', str(HERE / 'continuous_learning.py'), 'job-eval',
                                           '--config', cfg['_file'], '--candidate', str(candidate),
                                           '--opponent', str(root / 'models/initial'), '--episodes', str(plan['battle_episodes']),
                                           '--seed', str(plan['battle_seed']), '--result', str(battle_path)], directory, 'battle')
        found = C.read_json(battle_path)
        battle = dict(status='complete', results=found['results'], terminations=found['terminations'])
    except RuntimeError as exc:
        # A crash is an abnormal evaluation, recorded with its cause; it is never dropped or retried silently.
        text = (directory / 'battle.log').read_text(errors='replace').splitlines()
        cause = next((line for line in reversed(text) if re.match(r'^[A-Za-z_.]+(Error|Exception)\b', line)), None)
        battle = dict(status='evaluation_error', error=str(exc), exception=cause, log=str(directory / 'battle.log'),
                      results=None, terminations=None, verified_episodes=0)
    resources = {p.stem.replace('-resources', ''): C.read_json(p) for p in directory.glob('*-resources.json')}
    result = dict(
        arm=name, seed=seed, steps=plan['steps'], ppo_updates=updates, rehearsal_settings=arm.get('rehearsal'),
        conditions=wanted,
        candidate=str(candidate), base_sha256=training['base_sha256'], head_sha256=training['head_sha256'],
        changed_values=training['changed_values'], all_finite=training['all_finite'],
        regression_passed=regression['passed'], adoption_eligible_by_openings=regression['passed'], adopted=False,
        opening_passed=openings['passed'], opening_total=openings['total'], opening_lua_errors=openings['lua_errors'],
        lost_case_ids=regression['lost_case_ids'],
        new_case_ids=sorted(set(openings['passed_case_ids']) - set(baseline['passed_case_ids'])),
        cases={c['case_id']: c['passed'] for c in openings['cases']},
        families=family_counts(openings['cases']),
        teacher_forced=teacher, rehearsal=rehearsal, learning_rate=wanted['learning_rate'],
        weight_change=weight_change(directory / 'source-model', candidate),
        battle=battle,
        resources=resources, native_lua_errors_in_training=0 if 'NATIVE_LUA_ERRORS 0' in log else None)
    C.atomic_json(done, result)
    return result


def weight_change(source, candidate):
    """Size of the update for each network of the pair (candidate minus starting weights)."""
    import numpy as np
    import flax.serialization as ser

    def leaves(value):
        if isinstance(value, dict):
            return [x for k in sorted(value) for x in leaves(value[k])]
        return [np.asarray(value, np.float64).ravel()]
    out = {}
    for name in ['base.flax_model', 'chain-head.flax_model']:
        before = np.concatenate(leaves(ser.msgpack_restore((Path(source) / name).read_bytes())['params']))
        after = np.concatenate(leaves(ser.msgpack_restore((Path(candidate) / name).read_bytes())['params']))
        delta = after - before
        out[name.split('.')[0]] = dict(parameters=int(delta.size), changed=int(np.count_nonzero(delta)),
                                       l2=float(np.linalg.norm(delta)), relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(before)),
                                       max_abs=float(np.abs(delta).max()), mean_abs=float(np.abs(delta).mean()))
    return out


def family_counts(cases):
    out = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        out[c['family']][0] += int(c['passed'])
        out[c['family']][1] += 1
    return {k: dict(passed=v[0], total=v[1]) for k, v in sorted(out.items())}


# ---------------------------------------------------------------- teacher-forced diagnostics (no training)

def load_split_for_diagnostics(manifest_path, data_dir, split):
    """Diagnostic CE only. Never passed to any optimizer; holdout is refused."""
    if split == 'holdout':
        raise ValueError('holdout is not used in this experiment')
    if split == R.SPLIT:
        return R.load_train_split(manifest_path, data_dir)
    import numpy as np
    manifest = C.read_json(manifest_path)
    record = manifest['decks'][R.DECK]
    path = Path(data_dir) / f'{split}.npz'
    if C.sha(path) != record[split]['sha256']:
        raise ValueError(f'{split}.npz does not match the manifest checksum')
    witnesses = C.read_json(Path(data_dir) / f'{split}-witnesses.json')
    with np.load(path) as z:
        prepared = R.prepare({k: z[k] for k in z.files})
    if not np.array_equal(prepared['lengths'], [w['decisions'] for w in witnesses]):
        raise ValueError(f'{split} witnesses do not match')
    prepared['families'] = [w['family'] for w in witnesses]
    return prepared


def job_teacher_forced(args):
    import numpy as np
    cfg = C.load_config(args.config)
    C.runtime_check(cfg)
    B = C.backend()
    common = B.policy_common(cfg)
    import jax
    agent = B.CompositeAgent(common.AGENT, common.HEAD)
    checkpoint = Path(args.candidate)
    base, head = common.load(checkpoint / 'base.flax_model'), common.load(checkpoint / 'chain-head.flax_model')
    variables = {'params': {'base': base['params'], 'head': head['params']}, 'batch_stats': base.get('batch_stats', {})}

    @jax.jit
    def per_step(v, steps):
        logits = R.unroll_logits(agent, v, steps)
        _, metrics = R.acceptable_action_ce(logits, steps)
        hit = jax.numpy.take_along_axis(steps['targets'] > 0, logits.argmax(-1)[..., None], axis=-1)[..., 0]
        return metrics['per_step'], hit
    manifest = Path(cfg['root']) / 'snapshot/opening-manifest.json'
    data_dir = R.dataset_directory(cfg['opening_manifest'])
    out = dict(checkpoint=str(checkpoint), base_sha256=C.sha(checkpoint / 'base.flax_model'),
               head_sha256=C.sha(checkpoint / 'chain-head.flax_model'), used_for_training=False, holdout_used=False,
               adapter_steps_excluded=True)
    for split in ['train', 'validation']:
        prepared = load_split_for_diagnostics(manifest, data_dir, split)
        n, width, length = len(prepared['lengths']), 16, int(prepared['lengths'].max())
        ce_sum, hit_sum, steps_sum = collections.Counter(), collections.Counter(), collections.Counter()
        for start in range(0, n, width):
            idx = list(range(start, min(start + width, n)))
            real = len(idx)
            idx += [idx[-1]] * (width - real)
            steps = R.batch(prepared, idx, length)
            ce, hit = map(np.asarray, per_step(variables, steps))
            mask = steps['loss_mask']
            for k in range(real):
                family = prepared['families'][idx[k]]
                for key in ['all', family]:
                    ce_sum[key] += float((ce[:, k] * mask[:, k]).sum())
                    hit_sum[key] += float((hit[:, k] * mask[:, k]).sum())
                    steps_sum[key] += float(mask[:, k].sum())
        out[split] = dict(sequences=n, loss_steps=int(steps_sum['all']), ce=ce_sum['all'] / steps_sum['all'],
                          accuracy=hit_sum['all'] / steps_sum['all'],
                          families={k: dict(ce=ce_sum[k] / steps_sum[k], accuracy=hit_sum[k] / steps_sum[k],
                                            loss_steps=int(steps_sum[k])) for k in sorted(steps_sum) if k != 'all'})
        print('TEACHER_FORCED', split, json.dumps({k: out[split][k] for k in ['ce', 'accuracy', 'loss_steps']}), flush=True)
    C.atomic_json(args.result, out)


# ---------------------------------------------------------------- driver

def load_plan(path):
    plan = C.read_json(path)
    plan['_file'] = str(Path(path).resolve())
    assert_isolated(plan['experiment_root'])
    if plan['steps'] < 128 or plan['steps'] % 128:
        raise ValueError('steps must be a positive multiple of 128')
    if plan['battle_episodes'] % 8:
        raise ValueError('battle_episodes must be divisible by eval_envs * 2')
    for name, arm in plan['arms'].items():
        validate_arm(name, arm)
    limits = plan.get('limits')
    if limits:
        runs = len(plan['arms']) * len(plan['seeds'])
        if runs * plan['steps'] > limits['max_environment_decisions']:
            raise ValueError('Plan exceeds its environment-decision limit')
        if runs * plan['battle_episodes'] > limits['max_battle_episodes']:
            raise ValueError('Plan exceeds its battle limit')
        if runs > limits['max_candidates']:
            raise ValueError('Plan exceeds its candidate limit')
    return plan


def leftover_jobs(root):
    marker = str(Path(root).resolve()) + '/'
    out = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
    own = {os.getpid(), os.getppid()}
    return [int(line.split(None, 1)[0]) for line in out.splitlines()
            if marker in line and int(line.split(None, 1)[0]) not in own]


LEFTOVER_WAIT_SECONDS = 10


def terminate_leftover_jobs(root, wait=None):
    """run_job() kills its job on every exit path; confirm that nothing referencing this arm survives."""
    wait = LEFTOVER_WAIT_SECONDS if wait is None else wait
    found, deadline = leftover_jobs(root), time.monotonic() + wait
    while found and time.monotonic() < deadline:
        time.sleep(0.5)
        found = leftover_jobs(root)
    terminated = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not found:
            break
        for pid in found:
            try:
                os.kill(pid, sig)
                terminated.append(pid)
            except ProcessLookupError:
                pass
        time.sleep(2)
        found = leftover_jobs(root)
    if found:
        raise RuntimeError(f'Job processes for {root} are still running: {found}')
    return sorted(set(terminated))


def close_arm(cfg, state, outcome, error=None):
    """Leave an experiment arm stopped on every exit path: no live job, STOP present, final state saved."""
    root = assert_isolated(cfg['root'])
    if not root.exists():
        return None
    try:
        terminated = terminate_leftover_jobs(root)
    finally:
        (root / 'STOP').write_text('now')
    if state is None and (root / 'state.json').exists():
        state = C.read_json(root / 'state.json')
    if state is not None:
        C.save_state(cfg, state, status='stopped', phase='idle', pid=None, child_pid=None,
                     last_decision=('finite_ab_experiment_complete_not_adopted' if outcome == 'complete'
                                    else 'finite_ab_experiment_failed_not_adopted'),
                     last_error=None if error is None else f'{type(error).__name__}: {error}',
                     leftover_jobs_terminated=terminated)
    return terminated


def raise_on_termination_signal():
    """SIGTERM/SIGHUP become exceptions so that every finally-block (job kill, STOP) runs."""
    def handler(signum, frame):
        raise SystemExit(f'terminated by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


def run(args):
    plan = load_plan(args.plan)
    root = Path(plan['experiment_root'])
    before = production_fingerprint()
    if resident_learners():
        raise RuntimeError(f'A resident learner is running: {resident_learners()}')
    if not (root / 'production-fingerprint-before.json').exists():
        C.atomic_json(root / 'production-fingerprint-before.json', before)
    arms = [a for a in plan['arms'] if not args.arms or a in args.arms.split(',')]
    seeds = [s for s in plan['seeds'] if not args.seeds or str(s) in args.seeds.split(',')]
    raise_on_termination_signal()
    touched, outcome, error = {}, 'failed', None
    try:
        B = backend()
        for name in arms:
            cfg = arm_config(plan, name, plan['arms'][name])
            touched[name] = [cfg, None]  # registered before setup_arm() lifts this arm's STOP
            cfg, state, info = setup_arm(B, plan, name, plan['arms'][name], cfg)
            touched[name][1] = state
            C.atomic_json(Path(cfg['root']) / 'arm-setup.json', info)
            print('ARM_READY', name, json.dumps(info), flush=True)
        for seed in seeds:
            for name in arms:  # interleave arms per seed
                cfg, state = touched[name]
                started = time.monotonic()
                result = run_one(B, plan, name, plan['arms'][name], cfg, state, seed)
                print('RUN_DONE', name, seed, f'{time.monotonic() - started:.0f}s',
                      json.dumps(dict(openings=f"{result['opening_passed']}/{result['opening_total']}",
                                      regression_passed=result['regression_passed'], lost=result['lost_case_ids'],
                                      new=result['new_case_ids'],
                                      battle=(result['battle']['results'] or {}).get('win_rate', result['battle'].get('status')))), flush=True)
        outcome = 'complete'
    except BaseException as exc:
        error = exc
        raise
    finally:
        failures = []
        for name, (cfg, state) in touched.items():
            try:
                close_arm(cfg, state, outcome, error)
            except Exception as exc:  # keep closing the other arms
                failures.append(f'{name}: {exc}')
        after = production_fingerprint()
        if after != before:
            raise RuntimeError('Production fingerprint changed during the experiment')
        if failures:
            raise RuntimeError(f'Could not close experiment arms cleanly: {failures}')
    C.atomic_json(root / 'production-fingerprint-after.json', after)
    summarize(args)


def summarize(args):
    """Summarize ONLY runs whose recorded conditions and evidence match the current plan; refuse otherwise."""
    plan = load_plan(args.plan)
    root = Path(plan['experiment_root'])
    runs, mismatches, missing = [], [], []
    for name, arm in plan['arms'].items():
        config = root / 'configs' / f'{name}.json'
        cfg = C.load_config(config) if config.exists() else None
        if cfg is not None and C.read_json(config) != build_arm_config(plan, name, arm):
            mismatches.append(f'{config}: saved arm config differs from the current plan')
            cfg = None
        cycles = root / name / 'cycles'
        seen = set()
        for path in sorted(cycles.glob('*/run-result.json')) if cycles.exists() else []:
            directory, result = path.parent, C.read_json(path)
            if cfg is None:
                mismatches.append(f'{directory}: arm config is missing')
            elif result['seed'] not in plan['seeds']:
                mismatches.append(f"{directory}: seed {result['seed']} is not in the plan")
            else:
                try:
                    verified_reuse(directory, Path(cfg['root']), requested_conditions(plan, name, arm, cfg, result['seed']))
                    runs.append(result)
                    seen.add(result['seed'])
                except RuntimeError as exc:
                    mismatches.append(str(exc))
        missing.extend(f'{name}/{seed}' for seed in plan['seeds'] if seed not in seen)
    if mismatches:
        raise RuntimeError('Refusing to summarize runs that do not match the current plan:\n' + '\n'.join(mismatches))
    arms = {}
    for name in plan['arms']:
        rs = [r for r in runs if r['arm'] == name]
        if not rs:
            continue
        b = [r['battle']['results'] for r in rs if r['battle'].get('results')]
        arms[name] = dict(
            runs=len(rs), seeds=[r['seed'] for r in rs], regression_passed_runs=sum(r['regression_passed'] for r in rs),
            opening_passed=[r['opening_passed'] for r in rs], lost_case_ids={str(r['seed']): r['lost_case_ids'] for r in rs},
            new_case_ids={str(r['seed']): r['new_case_ids'] for r in rs},
            lost_case_union=sorted(set().union(*[r['lost_case_ids'] for r in rs])),
            train_ce_after=[r['teacher_forced']['train']['ce'] for r in rs],
            validation_ce_after=[r['teacher_forced']['validation']['ce'] for r in rs],
            battle_verified_wins=sum(x['wins'] for x in b), battle_verified_losses=sum(x['losses'] for x in b),
            battle_draws=sum(x['draws'] for x in b), battle_verified_episodes=sum(x['verified_episodes'] for x in b),
            battle_abnormal=sum(r['battle']['terminations']['abnormal_total'] for r in rs if r['battle'].get('terminations')),
            battle_evaluation_errors={str(r['seed']): r['battle'].get('exception') for r in rs if r['battle'].get('status') == 'evaluation_error'},
            battle_truncated=sum(r['battle']['terminations']['truncated'] for r in rs if r['battle'].get('terminations')),
            battle_engine_timeout=sum(r['battle']['terminations']['engine_timeout'] for r in rs if r['battle'].get('terminations')),
            battle_timeout_suspected=sum(r['battle']['terminations']['timeout_suspected'] for r in rs if r['battle'].get('terminations')),
            learning_rate=sorted({r.get('learning_rate') for r in rs}),
            train_wall_seconds=[r['resources'].get('train', {}).get('wall_seconds') for r in rs],
            train_max_rss_mib=[r['resources'].get('train', {}).get('max_rss_mib') for r in rs],
            train_gpu_peak_mib=[r['resources'].get('train', {}).get('gpu_memory_used_peak_mib') for r in rs])
    summary = dict(plan=plan, runs=runs, arms=arms, generated_jst=C.now(), adopted_any=False,
                   production_started=False, network_or_llm_calls=False, conditions_verified=True,
                   missing_runs=missing, complete=not missing)
    C.atomic_json(root / 'summary.json', summary)
    print(json.dumps(arms, indent=2, ensure_ascii=False))


class arm_active:
    """Temporarily lift this runner's own STOP marker in an experiment arm (never production)."""

    def __init__(self, cfg):
        self.root = assert_isolated(cfg['root'])

    def __enter__(self):
        if not (self.root / 'arm-setup.json').exists():
            raise RuntimeError('Not an experiment arm created by this runner')
        (self.root / 'STOP').unlink(missing_ok=True)

    def __exit__(self, *exc):
        try:
            terminate_leftover_jobs(self.root)
        finally:
            (self.root / 'STOP').write_text('now')


def diagnose(args):
    """Failure reasons for every candidate + perturbation checks for selected control runs."""
    plan = load_plan(args.plan)
    root = Path(plan['experiment_root'])
    if production_fingerprint() != C.read_json(root / 'production-fingerprint-before.json'):
        raise RuntimeError('Production fingerprint changed')
    raise_on_termination_signal()
    B = backend()
    for name in plan['arms']:
        cfg = C.load_config(root / 'configs' / f'{name}.json')
        state = C.read_json(Path(cfg['root']) / 'state.json')
        with arm_active(cfg):
            for run_dir in sorted((Path(cfg['root']) / 'cycles').glob(f"{plan['steps']}-seed*")):
                if '-incomplete-' in run_dir.name or (run_dir / 'opening-failures.json').exists():
                    continue
                B.control.run_job(cfg, state, [cfg['python'], '-B', '-u', str(Path(__file__).resolve()), 'job-opening-failures',
                                               '--config', cfg['_file'], '--candidate', str(run_dir / 'candidate'),
                                               '--evidence', str(run_dir / 'opening-regression.json'),
                                               '--result', str(run_dir / 'opening-failures.json')], run_dir, 'opening-failures')
            if args.arms == name:
                for seed in args.seeds.split(','):
                    args_one = argparse.Namespace(plan=args.plan, arms=name, seeds=seed)
                    perturbation(args_one)
    if production_fingerprint() != C.read_json(root / 'production-fingerprint-before.json'):
        raise RuntimeError('Production fingerprint changed')


def job_opening_failures(args):
    """Diagnostic re-run of failed validation openings to record WHY they failed.

    Uses the unchanged pinned witness code and the same policy stack as job-openings;
    the pass/fail verdicts are only compared, never replaced."""
    cfg = C.load_config(args.config)
    C.runtime_check(cfg)
    B = C.backend()
    common = B.policy_common(cfg)
    import all_opening_cases as cases
    evidence = C.read_json(args.evidence)
    failed = [c['case_id'] for c in evidence['cases'] if not c['passed']]
    manifest = C.read_json(Path(cfg['root']) / 'snapshot/opening-manifest.json')
    by_id = {}
    for case in manifest['decks']['ElfnotesKewlTune']['cases']['validation']:
        by_id.setdefault(case['case_id'], case)
    policy = B.create_policy(cfg, Path(args.candidate))
    rows = []
    for case_id in failed:
        result, _ = common.v.run(cases.normalize(by_id[case_id]), policy=policy)
        rows.append(dict(case_id=case_id, passed=bool(result['passed']), failure=result.get('failure'),
                         board=result.get('board'), chosen_variant=result.get('chosen_variant'),
                         interruption_verified=result.get('interruption_verified'),
                         crystal_first_preference_met=result.get('crystal_first_preference_met'),
                         decisions=len(result['trace'])))
        print('FAILURE', json.dumps(rows[-1]), flush=True)
    C.atomic_json(args.result, dict(candidate=args.candidate, evidence=args.evidence, cases=rows,
                                    verdicts_reproduced=all(not r['passed'] for r in rows),
                                    lua_errors=common.v.native.get_lua_error_count(), diagnostic_only=True))


def perturbation(args):
    """Called from diagnose() inside arm_active(). Diagnostic only (no training, no adoption): is the loss tied to the PPO update
    direction or to any change of that size? Evaluates theta0 + alpha*delta and
    theta0 + |delta|*random_sign on the unchanged 66 validation openings."""
    import numpy as np
    import flax.serialization as ser
    plan = load_plan(args.plan)
    root = Path(plan['experiment_root'])
    cfg = C.load_config(root / 'configs' / f'{args.arms}.json')
    state = C.read_json(Path(cfg['root']) / 'state.json')
    B = backend()
    initial = Path(cfg['root']) / 'models/initial'
    reference = Path(cfg['root']) / 'cycles' / f"{plan['steps']}-seed{args.seeds}" / 'candidate'
    out = root / 'diagnostics' / f'{args.arms}-seed{args.seeds}'
    baseline = C.read_json(Path(cfg['root']) / 'opening-baseline.json')
    rng = np.random.default_rng(20261109)
    rows = []
    for name, alpha, randomize in [('interpolate-0.5', 0.5, False), ('random-sign-same-magnitude', 1.0, True)]:
        target = out / name
        result = target / 'openings.json'
        if not result.exists():
            target.mkdir(parents=True, exist_ok=True)
            for file in ['base.flax_model', 'chain-head.flax_model']:
                before = ser.msgpack_restore((initial / file).read_bytes())
                after = ser.msgpack_restore((reference / file).read_bytes())

                def mix(a, b):
                    if isinstance(a, dict):
                        return {k: mix(a[k], b[k]) for k in a}
                    a, b = np.asarray(a), np.asarray(b)
                    delta = (b - a) * alpha
                    if randomize:
                        delta = np.abs(delta) * rng.choice(np.array([-1, 1], a.dtype), size=a.shape)
                    return (a + delta).astype(a.dtype)
                (target / file).write_bytes(ser.to_bytes(mix(before, after)))
            B.control.run_job(cfg, state, [cfg['python'], '-B', '-u', str(HERE / 'continuous_learning.py'), 'job-openings',
                                           '--config', cfg['_file'], '--candidate', str(target), '--result', str(result)],
                              target, 'openings')
        found = C.read_json(result)
        rows.append(dict(variant=name, passed=found['passed'], total=found['total'], lua_errors=found['lua_errors'],
                         lost_case_ids=sorted(set(baseline['passed_case_ids']) - set(found['passed_case_ids'])),
                         new_case_ids=sorted(set(found['passed_case_ids']) - set(baseline['passed_case_ids'])),
                         base_sha256=C.sha(target / 'base.flax_model'), head_sha256=C.sha(target / 'chain-head.flax_model')))
        print('PERTURBATION', json.dumps(rows[-1]), flush=True)
    C.atomic_json(out / 'summary.json', dict(reference_candidate=str(reference), initial=str(initial), variants=rows,
                                              diagnostic_only=True, adopted=False, training_performed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['run', 'summarize', 'job-teacher-forced', 'job-opening-failures', 'diagnose'])
    parser.add_argument('--plan')
    parser.add_argument('--arms')
    parser.add_argument('--seeds')
    parser.add_argument('--config')
    parser.add_argument('--candidate')
    parser.add_argument('--result')
    parser.add_argument('--evidence')
    args = parser.parse_args()
    if args.action == 'job-teacher-forced':
        job_teacher_forced(args)
    elif args.action == 'run':
        run(args)
    elif args.action == 'job-opening-failures':
        job_opening_failures(args)
    elif args.action == 'diagnose':
        diagnose(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
