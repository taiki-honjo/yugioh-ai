"""Kewl Tune B2B finite function test: six conditions on one teacher-built position (shadow evaluation).

Position: own turn = the recorded P1 teacher route (TRAIN case cue-reco-jj-zaren-g00-p0), replayed step by step with
checks. Opponent turn: the existing O1 opponent (Rukina (1) to the centre, Rukina (2) search), then it ends its turn.
Our decisions follow b2b_policy.py (R0-R3 recovery/summon conditions, A0/A1 attack conditions); every scripted choice
is taken only if it is a legal action, otherwise it is logged and answered passively (no card-processing changes, no
search). No model, no training, no battles; production files are only read. Engine runs are capped at 12 in this
root's own ledger (6 conditions + 6 replays).

  b2b.py launch <check|run|replay> <R0|R1|R2|R3|A0|A1>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-probe-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
RECORD = P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json'
CASE_ID = 'cue-reco-jj-zaren-g00-p0'
LIMIT = 12
STEP_CAP = 400
KEY_CARDS = [65961304, 88170262, 4891376, 16387555, 42781164, 14558127, 89392810, 97268402, 13597785]
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import b2b_policy as BP  # noqa: E402
from b2b_policy import CONDITIONS  # noqa: E402
import continuous_learning as C  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_json(path, json.loads(json.dumps(value, default=str)))


def setup():
    cfg = C.load_config(CONFIG)
    root = Path(cfg['root'])
    C.runtime_check(cfg)
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_openings_common as common
    import all_opening_cases as cases
    for name in ['all_openings_common', 'all_opening_witness', 'rukina_recovery_witness']:
        assert Path(sys.modules[name].__file__).resolve().is_relative_to((root / 'snapshot/policy-code').resolve()), name
    manifest = json.loads((root / 'snapshot/opening-manifest.json').read_text())['decks']['ElfnotesKewlTune']['cases']
    raw = next(c for c in manifest['train'] if c['case_id'] == CASE_ID)
    key = tuple(sorted(raw['hand']))
    assert not [s for s in ['validation', 'holdout'] if any(tuple(sorted(c['hand'])) == key for c in manifest[s])]
    hashes = json.loads((root / 'snapshot/hashes.json').read_text())
    runtime = json.loads((root / 'snapshot/runtime.json').read_text())
    scripts = Path(cfg['runtime']) / 'agent/scripts/script'
    versions = dict(snapshot_hashes_sha256=C.sha(root / 'snapshot/hashes.json'),
                    policy_code={k: v for k, v in hashes.items() if k.startswith('policy-code/') and k.endswith('.py')},
                    native_sha256=C.sha(runtime['native_binary']), deck_sha256=C.sha(runtime['deck']),
                    card_scripts={code: C.sha(scripts / f'c{code}.lua') for code in KEY_CARDS}, card_script_dir=str(scripts.resolve()),
                    position_record=str(RECORD), position_record_sha256=C.sha(RECORD),
                    b2b_sha256=C.sha(Path(__file__)), b2b_policy_sha256=C.sha(OUT / 'code/b2b_policy.py'),
                    capability_analysis_sha256=C.sha(P / 'experiments/capability-probe-20261109/code/analysis.py'))
    return cfg, common, cases.normalize(raw), raw, versions


def o1_opponent(v, s, state):
    """Unchanged O1 opponent of all_opening_witness.run()."""
    if s['turn_player'] == 1 and s['phase'] == 4:
        target = state['steps'][0] if state['steps'] else None
        found = next((a for a in s['actions'] if target and all(a[k] == value for k, value in target.items())), None)
        if found:
            state['steps'].pop(0)
            if found['act'] == 'Activate':
                state['search_started'] = True
            return found['index']
        if s['message'] == 'select_place':
            return next(a['index'] for a in s['actions'] if a['place'] == 3)
        if s['message'] == 'select_position':
            return 0
        if s['message'] in ['select_card', 'select_unselect_card']:
            return next((a['index'] for a in s['actions'] if a['code'] == v.SPIRIT), 0)
    return v.passive(s)


def play(common, case, position, condition=None, record=None):
    import numpy as np
    v, native = common.v.v, common.v.native
    env, obs, info = v.create(case)
    rows, failure, mismatch = [], None, None
    st = BP.new_state(condition) if condition else None
    prefix = record['trace'] if record else [r for r in position['trace'] if r['snapshot']['turn_player'] == 0]
    live = dict(steps=[v.a('SpSummon', v.RUK), v.a('Activate', v.RUK)], search_started=False)
    try:
        for row in prefix:  # recorded steps, checked one by one
            raw = native.benchmark_snapshot()
            s = v.compact(raw)
            idx = row['selected']['index']
            if idx >= len(raw['actions']):
                mismatch = dict(step=row['step'], keys=['action_index_out_of_range']); break
            keys = A.compare_step(row, s, raw['actions'][idx])
            if keys:
                mismatch = dict(step=row['step'], keys=keys); break
            rows.append(dict(step=row['step'], snapshot=s, selected=raw['actions'][idx], source=row.get('source', 'record') if record else 'record'))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                failure = 'terminated_or_truncated'; break
        own_end = v.compact(native.benchmark_snapshot())
        if record is None and mismatch is None and failure is None:
            step = rows[-1]['step'] + 1
            for _ in range(STEP_CAP):
                raw = native.benchmark_snapshot()
                if BP.finished(raw, st):
                    break
                if raw['to_play'] == 0:
                    idx, source = BP.decide(raw, st, step, v.passive)
                else:
                    idx, source = o1_opponent(v, raw, live), 'opponent'
                rows.append(dict(step=step, snapshot=v.compact(raw), selected=raw['actions'][idx], source=source))
                obs, _, term, trunc, info = env.step(np.array([idx], np.int32)); step += 1
                if term[0] or trunc[0]:
                    failure = 'terminated_or_truncated'; break
            else:
                failure = 'step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(native.benchmark_snapshot())
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, own_end=own_end, end=end, state=st, failure=failure, mismatch=mismatch, lua_errors=lua)


def main(command, condition):
    cfg, common, case, raw, versions = setup()
    position = json.loads(RECORD.read_text())
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    meta = dict(condition=condition, case_id=CASE_ID, split='train', seed=raw['seed'], versions=versions,
                position='own turn replayed from the P1 teacher record; teacher-built, not the model reaching it',
                scripted=True, model_used=False)
    path = OUT / 'conditions' / f'{condition}.json'
    if command == 'check':
        own_turn = sum(1 for r in position['trace'] if r['snapshot']['turn_player'] == 0)
        print('CHECK', json.dumps(dict(case=CASE_ID, runs=len(ledger.started()), limit=LIMIT, own_turn_steps=own_turn,
                                       b2b_script=versions['card_scripts'][65961304])), flush=True)
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    run = ledger.start(f'{command} condition={condition}', CASE_ID, C.now())
    try:
        got = play(common, case, position, condition if command == 'run' else None, record)
        if command == 'run':
            out = dict(meta, engine_run=run, failure=got['failure'], mismatch=got['mismatch'], lua_errors=got['lua_errors'],
                       script_state=got['state'], own_end_matches_record=A.normalized(got['own_end']) == A.normalized(position['own_end']),
                       end=got['end'], trace=got['rows'])
            write(path, out)
            st = got['state']
            print('RUN', json.dumps(dict(run=run, condition=condition, failure=got['failure'], mismatch=got['mismatch'],
                                         remix=st['remix'], b2b=st['b2b'], attacks=st['attacks'], battle=st['battle_entered'],
                                         notable=[e for e in st['log'] if 'NOT' in e['label'] or 'UNEXPECTED' in e['label']][:6],
                                         labels=[e['label'] for e in st['log'] if e['label'] not in ('pass', 'yesno-pass')][:40])), flush=True)
        else:
            out = dict(meta, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                       lua_errors=got['lua_errors'], reproduced_steps=len(got['rows']), of=len(record['trace']),
                       end_matches=A.normalized(got['end']) == A.normalized(record['end']))
            write(OUT / 'replays' / f'{condition}.json', out)
            print('REPLAY', json.dumps({k: out[k] for k in ['engine_run', 'mismatch', 'failure', 'reproduced_steps', 'of', 'end_matches']}), flush=True)
        ledger.finish(run, 'ok' if not got['failure'] and not got['mismatch'] else 'completed-with-findings', C.now(),
                      failure=got['failure'], mismatch=got['mismatch'])
    except BaseException as exc:
        ledger.finish(run, 'error', C.now(), error=f'{type(exc).__name__}: {exc}'); raise


def launch(command, condition):
    cfg = C.load_config(CONFIG)
    log = OUT / 'logs' / f'{command}-{condition}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(Path(__file__).resolve()),
           'job', command, condition]
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT)
    keep = ('CHECK', 'RUN', 'REPLAY', 'Traceback', 'AssertionError', 'RuntimeError', 'KeyError', 'TypeError', 'ValueError', 'IndexError')
    print(''.join(l for l in log.read_text().splitlines(True) if l.startswith(keep)), end='')
    print(f'exit={done.returncode} log={log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'job'])
    parser.add_argument('command', choices=['check', 'run', 'replay'])
    parser.add_argument('condition', choices=CONDITIONS)
    args = parser.parse_args()
    (launch if args.mode == 'launch' else main)(args.command, args.condition)
