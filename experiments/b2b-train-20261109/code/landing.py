"""Evaluation positions for the B2B training trial: the own-turn B2B landing on the pre-registered TRAIN hands g03 / g05.

Route (as verified for g00 in experiments/b2b-zaren-ownturn-check-20261109): each case's own teacher plan, cut before
Jukejoint (3) releases Ash, then S summon Zaren (Trackmaker + Ash) and B2B (Zaren + Cue), set Kewl Tune Synchro.
The opponent passes. One record + one replay per hand. No model, no training; production files only read.
Engine runs share this root's 12-run B2B evaluation ledger.

  landing.py launch <check|record|replay> <g03|g05>
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-train-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
PLAN = json.loads((OUT / 'plan.json').read_text())
POSITIONS = {p['name']: p for p in PLAN['evaluation']['positions']}
LIMIT = 12
B2B, TRACK, REMIX, CUE, ASH, VEILER, SYNC, ZAREN, JUKE = 65961304, 42781164, 88170262, 16387555, 14558127, 97268402, 78058681, 4891376, 14442329
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_json(path, json.loads(json.dumps(value, default=str)))


def via_zaren(case):
    case = copy.deepcopy(case)
    index = next(i for i, a in enumerate(case['plan']) if a['act'] == 'SpSummon' and a['code'] == REMIX)
    case['plan'] = case['plan'][:index] + [dict(act='SpSummon', code=REMIX), dict(act='SpSummon', code=B2B), dict(act='Set', code=SYNC)]
    cost_at = next(i for i, a in enumerate(case['plan']) if a['act'] == 'None' and a['code'] == ASH and a.get('after_location') == 16)
    assert case['plan'][cost_at - 1] == dict(act='Activate', code=JUKE), case['plan'][cost_at - 1]
    case['plan'] = case['plan'][:cost_at - 1] + [dict(act='SpSummon', code=ZAREN), dict(act='SpSummon', code=B2B), dict(act='Set', code=SYNC)]
    case['materials'][ZAREN] = [[(ASH, 'm'), (TRACK, 'm')]]
    case['materials'][B2B] = [[(CUE, 'm'), (ZAREN, 'm')]]
    return case


def setup(name):
    cfg = C.load_config(CONFIG)
    root = Path(cfg['root'])
    C.runtime_check(cfg)
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_openings_common as common  # installs the teacher preferences, as the gate and the earlier records did
    import all_opening_cases as cases
    manifest = json.loads((root / 'snapshot/opening-manifest.json').read_text())['decks']['ElfnotesKewlTune']['cases']
    raw = next(c for c in manifest['train'] if c['case_id'] == POSITIONS[name]['case_id'])
    assert raw['seed'] == POSITIONS[name]['seed'] and raw['hand'] == POSITIONS[name]['hand']
    key = tuple(sorted(raw['hand']))
    assert not [s for s in ['validation', 'holdout'] if any(tuple(sorted(c['hand'])) == key for c in manifest[s])]
    return cfg, common, via_zaren(cases.normalize(raw)), raw


def play(common, case, recorded=None):
    import numpy as np
    w, v, native = common.v, common.v.v, common.v.native
    teacher = w.teacher(case) if recorded is None else None
    env, obs, info = v.create(case)
    audit = w.VisibilityAudit()
    trace, mismatch, failure = [], None, None
    try:
        for step in range(260):
            s = native.benchmark_snapshot()
            compact = v.compact(s)
            if recorded is None and s['turn_player'] == 1 and s['chain_size'] == 0:
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
                    mismatch = dict(step=step, keys=['index_out_of_range']); break
                keys = A.compare_step(row, compact, s['actions'][idx])
                if keys:
                    mismatch = dict(step=step, keys=keys); break
            trace.append(dict(step=step, snapshot=compact, selected=s['actions'][idx]))
            obs, reward, term, trunc, info = env.step(np.array([idx], dtype=np.int32))
            if term[0] or trunc[0]:
                failure = 'unexpected duel end'; break
        else:
            failure = 'step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(native.benchmark_snapshot()) if failure != 'unexpected duel end' else trace[-1]['snapshot']
    lua = native.get_lua_error_count()
    remaining = list(getattr(teacher, 'steps', [])) if teacher else None
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    field = [x for x in end['cards'] if x['controller'] == 0 and x['location'] == 4 and x['position'] & 5]
    spells = [x['code'] for x in end['cards'] if x['controller'] == 0 and x['location'] == 8]
    landed = (any(x['code'] == B2B for x in field) and JUKE in spells and SYNC in spells and end['turn_player'] == 1 and end['chain_size'] == 0)
    return dict(trace=trace, end=end, failure=failure, mismatch=mismatch, lua_errors=lua, remaining_plan=remaining,
                landed=bool(landed), b2b_summon_steps=[r['step'] for r in trace if r['selected']['act'] == 'SpSummon' and r['selected']['code'] == B2B],
                visibility_checks=audit.checks)


def main(command, name):
    cfg, common, case, raw = setup(name)
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    path = OUT / 'landings' / f'{name}.json'
    if command == 'check':
        print('CHECK', json.dumps(dict(position=name, case_id=raw['case_id'], seed=raw['seed'], plan_tail=case['plan'][-4:], runs=len(ledger.started())), default=str))
        return
    recorded = json.loads(path.read_text()) if command == 'replay' else None
    if command == 'record' and path.exists():
        raise RuntimeError('refusing to overwrite a recorded landing')
    run = ledger.start(f'landing {command} {name}', raw['case_id'], C.now())
    try:
        got = play(common, case, recorded)
        ok = got['failure'] is None and got['mismatch'] is None and got['lua_errors'] == 0 and got['landed']
        if command == 'record':
            ok = ok and not got['remaining_plan']
            write(path, dict(position=name, case_id=raw['case_id'], hand=raw['hand'], seed=raw['seed'], split='train', engine_run=run, ok=ok,
                             case_plan_tail=case['plan'][-6:], landing_sha_source=C.sha(Path(__file__)), **got))
        else:
            ok = ok and len(got['trace']) == len(recorded['trace']) and A.normalized(got['end']) == A.normalized(recorded['end'])
            write(OUT / 'landings' / f'{name}-replay.json', dict(position=name, engine_run=run, replayed_engine_run=recorded['engine_run'], ok=ok,
                                                                  reproduced_steps=len(got['trace']), of=len(recorded['trace']), failure=got['failure'],
                                                                  mismatch=got['mismatch'], end_matches=A.normalized(got['end']) == A.normalized(recorded['end'])))
        ledger.finish(run, 'ok' if ok else 'finding', C.now(), failure=got['failure'], mismatch=got['mismatch'])
        field = sorted(x['code'] for x in got['end']['cards'] if x['controller'] == 0 and x['location'] == 4)
        print('LANDING', json.dumps(dict(run=run, command=command, position=name, ok=ok, landed=got['landed'], failure=got['failure'], mismatch=got['mismatch'],
                                         steps=len(got['trace']), b2b_steps=got['b2b_summon_steps'], remaining=got['remaining_plan'], field=field), default=str), flush=True)
    except BaseException as exc:
        ledger.finish(run, 'error', C.now(), error=f'{type(exc).__name__}: {exc}'); raise


def launch(command, name):
    cfg = C.load_config(CONFIG)
    log = OUT / 'logs' / f'landing-{command}-{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(Path(__file__).resolve()),
           'job', command, name]
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT)
    keep = ('CHECK', 'LANDING', 'Traceback', 'AssertionError', 'RuntimeError', 'KeyError', 'TypeError', 'ValueError', 'IndexError')
    print(''.join(l for l in log.read_text().splitlines(True) if l.startswith(keep)), end='')
    print(f'exit={done.returncode} log={log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'job'])
    parser.add_argument('command', choices=['check', 'record', 'replay'])
    parser.add_argument('name', choices=sorted(POSITIONS))
    args = parser.parse_args()
    (launch if args.mode == 'launch' else main)(args.command, args.name)
