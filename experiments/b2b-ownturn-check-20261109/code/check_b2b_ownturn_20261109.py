"""Isolated own-turn B2B witnesses. No model, training, or production writes.

Two finite scripted lines, each recorded and replayed once (hard cap 4 duels).
This checks the pinned local engine, not the Master Duel client.
"""
import argparse
import collections
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-ownturn-check-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
CASE_ID = 'cue-reco-jj-zaren-g00-p0'
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C

B2B, TRACK, REMIX, CUE, RECO, MIX, ASH, VEILER, CLIP, SYNC = (
    65961304, 42781164, 88170262, 16387555, 89392810,
    16509007, 14558127, 97268402, 43904702, 78058681)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def protected(cfg):
    root = Path(cfg['root'])
    paths = [CONFIG, P / 'scripts/continuous_learning.py', P / 'assets/deck/ElfnotesKewlTune.ydk', root / 'STOP', root / 'state.json']
    paths += [x for x in (root / 'snapshot').rglob('*') if x.is_file()]
    state = json.loads((root / 'state.json').read_text())
    for key in ['initial_checkpoint', 'best_checkpoint', 'latest_checkpoint']:
        where = Path(state[key])
        paths += [x for x in where.rglob('*') if x.is_file()] if where.is_dir() else [where]
    return {str(x): C.sha(x) for x in sorted(set(paths))}


def prepare():
    cfg = C.load_config(CONFIG)
    C.runtime_check(cfg)
    root = Path(cfg['root'])
    assert (root / 'STOP').read_text().strip() == 'now'
    state = json.loads((root / 'state.json').read_text())
    assert state['status'] == 'stopped' and not state.get('pid') and not state.get('child_pid')
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_opening_cases as cases
    import all_opening_witness as w
    spec = importlib.util.spec_from_file_location('existing_probe_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    manifest = json.loads((root / 'snapshot/opening-manifest.json').read_text())['decks']['ElfnotesKewlTune']['cases']
    raw = next(c for c in manifest['train'] if c['case_id'] == CASE_ID)
    hand_key = tuple(sorted(raw['hand']))
    assert not any(tuple(sorted(c['hand'])) == hand_key for split in ['validation', 'holdout'] for c in manifest[split])
    return cfg, cases.normalize(raw), w, analysis


def route_case(raw, variant):
    case = copy.deepcopy(raw)
    # Both routes start from the same train hand and seed. No altered card state.
    index = next(i for i, a in enumerate(case['plan']) if a['act'] == 'SpSummon' and a['code'] == REMIX)
    case['plan'] = case['plan'][:index] + [dict(act='SpSummon', code=REMIX), dict(act='SpSummon', code=B2B), dict(act='Set', code=SYNC)]
    case['materials'][B2B] = [[(VEILER, 'h'), (TRACK, 'm'), (REMIX, 'm')]]
    if variant == 'no-grave-recovery':
        # JJ releases Cue rather than Ash, keeping Ash as the level-3 material.
        # Mix searches Clip from deck rather than recovering the single Reco.
        jj_at = next(i for i, a in enumerate(case['plan']) if a['act'] == 'None' and a['code'] == ASH and a.get('after_location') == 16)
        case['plan'][jj_at]['code'] = CUE
        mix_at = next(i for i, a in enumerate(case['plan']) if a['act'] == 'Activate' and a['code'] == MIX)
        case['plan'][mix_at + 1] = dict(act='None', code=CLIP, after_location=2)
        case['materials'][REMIX] = [[(MIX, 'm'), (ASH, 'm')]]
    return case


def play(case, w, analysis, recorded=None):
    import numpy as np
    v = w.v
    teacher = w.teacher(case) if recorded is None else None
    env, obs, info = v.create(case)
    audit = w.VisibilityAudit()
    trace, mismatch, failure = [], None, None
    try:
        for step in range(220):
            s = v.native.benchmark_snapshot()
            compact = v.compact(s)
            if recorded is None and s['turn_player'] == 1 and s['chain_size'] == 0:
                assert not teacher.steps, teacher.steps
                break
            if recorded is not None and step == len(recorded['trace']):
                break
            if s['to_play'] == 0:
                audit.check(obs, info, s)
            if recorded is None:
                idx = teacher(s) if s['to_play'] == 0 else v.passive(s)
            else:
                row = recorded['trace'][step]
                idx = row['selected']['index']
                if idx >= len(s['actions']):
                    mismatch = dict(step=step, keys=['index_out_of_range'])
                    break
                keys = analysis.compare_step(row, compact, s['actions'][idx])
                if keys:
                    mismatch = dict(step=step, keys=keys)
                    break
            trace.append(dict(step=step, snapshot=compact, selected=s['actions'][idx]))
            obs, reward, term, trunc, info = env.step(np.array([idx], dtype=np.int32))
            assert not term[0] and not trunc[0], 'unexpected duel end'
        else:
            failure = 'step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(v.native.benchmark_snapshot())
    lua = v.native.get_lua_error_count()
    env.close()
    del env
    gc.collect()
    assert v.native.get_env_resource_stats()['live_duels'] == 0
    board = [x for x in end['cards'] if x['controller'] == 0 and x['location'] == 4 and x['position'] & 5]
    summoned = [row['step'] for row in trace if row['snapshot']['turn_player'] == 0 and row['selected']['act'] == 'SpSummon' and row['selected']['code'] == B2B]
    ok = not failure and not mismatch and lua == 0 and any(x['code'] == B2B for x in board) and bool(summoned)
    if recorded is not None:
        ok = ok and len(trace) == len(recorded['trace']) and analysis.normalized(end) == analysis.normalized(recorded['end'])
    return dict(ok=ok, failure=failure, mismatch=mismatch, lua_errors=lua, trace=trace, end=end,
                b2b_summon_steps=summoned, visibility_checks=audit.checks)


def main(variant, replay):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = C.load_config(CONFIG)
    before = protected(cfg)
    write(OUT / 'protected-before.json', before) if not (OUT / 'protected-before.json').exists() else None
    cfg, raw, w, analysis = prepare()
    case = route_case(raw, variant)
    ledger = analysis.Ledger(OUT / 'engine-runs.jsonl', limit=4)
    target = OUT / 'records' / f'{variant}.json'
    if not replay and target.exists():
        raise RuntimeError('Refusing to overwrite a completed run')
    recorded = json.loads(target.read_text()) if replay else None
    run = ledger.start(('replay' if replay else 'record') + ':' + variant, CASE_ID, C.now())
    try:
        result = play(case, w, analysis, recorded)
        result.update(variant=variant, engine_run=run, case_id=CASE_ID, split='train', hand=case['hand'], seed=case['seed'],
                      case=case, model_used=False, training=False, master_duel_client_used=False,
                      source_sha256=C.sha(Path(__file__)), deck_sha256=C.sha(w.v.DECK))
        if replay:
            result['replays_run'] = recorded['engine_run']
        write(OUT / 'replays' / f'{variant}.json' if replay else target, result)
        ledger.finish(run, 'ok' if result['ok'] else 'finding', C.now(), failure=result['failure'], mismatch=result['mismatch'])
        print(json.dumps({k: result[k] for k in ['variant', 'engine_run', 'ok', 'failure', 'mismatch', 'lua_errors', 'b2b_summon_steps']}, ensure_ascii=False), flush=True)
    finally:
        after = protected(cfg)
        changed = [name for name in set(before) | set(after) if before.get(name) != after.get(name)]
        write(OUT / f'protected-after-{variant}-{"replay" if replay else "record"}.json', dict(changed=changed, hashes=after))
        assert not changed, changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['standard', 'no-grave-recovery'])
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--job', action='store_true')
    args = parser.parse_args()
    if args.job:
        main(args.variant, args.replay)
    else:
        cfg = C.load_config(CONFIG)
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / 'code/check_b2b_ownturn_20261109.py'
        target.parent.mkdir(exist_ok=True)
        source = Path(__file__).read_bytes()
        if target.exists():
            assert target.read_bytes() == source, 'Refusing to change the diagnostic after first execution'
        else:
            target.write_bytes(source)
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(target), args.variant, '--job']
        if args.replay:
            cmd.append('--replay')
        log = OUT / f'{args.variant}-{"replay" if args.replay else "record"}.log'
        with log.open('w') as handle:
            done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT, timeout=90)
        print(log.read_text()[-5500:])
        sys.exit(done.returncode)
