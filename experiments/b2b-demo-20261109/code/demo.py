"""Candidate B2B demonstrations (step 1 of the B2B-use plan): record + replay, observations saved (shadow, isolated root).

Start: the own-turn B2B landing (experiments/b2b-zaren-ownturn-check-20261109 record, TRAIN case cue-reco-jj-zaren-g00-p0,
seed 151080), replayed step by step with checks. Opponent: O1 (Rukina special summon + search), passes otherwise.
Our side from turn 2: demo_policy.py (DM-A recovery/deploy, DM-B attack). The model's input observations at each of our
decisions (cards_, global_, actions_, h_actions_, public-chain features) and the chosen index are saved; a replay
re-creates them and must match exactly. No model, no training, no battles; production files only read.
Engine runs are capped at 6 in this root's ledger.

  demo.py launch <check|run|replay> <DM-A|DM-B>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-demo-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
RECORD = P / 'experiments/b2b-zaren-ownturn-check-20261109/records/via-zaren.json'
CASE_ID = 'cue-reco-jj-zaren-g00-p0'
LIMIT = 6
STEP_CAP = 400
KEY_CARDS = [65961304, 4891376, 16387555, 42781164, 97268402, 13597785, 78058681]
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import demo_policy as BP  # noqa: E402
from demo_policy import CONDITIONS  # noqa: E402
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
                    demo_sha256=C.sha(Path(__file__)), demo_policy_sha256=C.sha(OUT / 'code/demo_policy.py'),
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


def capture(common, obs, info, idx, step):
    import numpy as np
    return dict(step=step, label=int(idx), chain=np.asarray(common.features(info))[0].copy(),
                **{k: np.asarray(obs[k])[0].copy() for k in ['cards_', 'global_', 'actions_', 'h_actions_']})


def play(common, case, position, condition=None, record=None):
    import numpy as np
    v, native = common.v.v, common.v.native
    env, obs, info = v.create(case)
    rows, failure, mismatch, game_over, observed = [], None, None, None, []
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
            if raw['to_play'] == 0:
                observed.append(capture(common, obs, info, idx, row['step']))
            rows.append(dict(step=row['step'], snapshot=s, selected=raw['actions'][idx], source=row.get('source', 'landing-record') if record else 'landing-record'))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                # never read the engine state after the duel has ended (that crashed run 2); use the step's own result
                game_over = terminal(row['step'], s, reward, term, trunc, info)
                break
        if record is None and mismatch is None and failure is None and game_over is None:
            step = rows[-1]['step'] + 1
            for _ in range(STEP_CAP):
                raw = native.benchmark_snapshot()
                if BP.finished(raw):
                    break
                if raw['to_play'] == 0:
                    idx, source = BP.decide(raw, st, step, v.passive)
                    observed.append(capture(common, obs, info, idx, step))
                else:
                    idx, source = o1_opponent(v, raw, live), 'opponent'
                rows.append(dict(step=step, snapshot=v.compact(raw), selected=raw['actions'][idx], source=source))
                obs, reward, term, trunc, info = env.step(np.array([idx], np.int32)); step += 1
                if term[0] or trunc[0]:
                    game_over = terminal(step - 1, rows[-1]['snapshot'], reward, term, trunc, info)
                    break
            else:
                failure = 'step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = rows[-1]['snapshot'] if game_over else v.compact(native.benchmark_snapshot())  # last snapshot before the final step when the duel ended
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, end=end, state=st, failure=failure, mismatch=mismatch, lua_errors=lua, game_over=game_over, observed=observed)


def terminal(step, last_snapshot, reward, term, trunc, info):
    import numpy as np
    pick = lambda k: (np.asarray(info[k]).tolist() if k in info else None)
    return dict(step=step, terminated=bool(term[0]), truncated=bool(trunc[0]), reward=np.asarray(reward).tolist(),
                win_reason=pick('win_reason'), to_play=pick('to_play'), lp_before_final_step=(last_snapshot['lp0'], last_snapshot['lp1']),
                note='duel ended on this step; the engine state is not read afterwards')


def save_observations(path, observed):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.stack([o[k] for o in observed]) for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain']},
                        step=np.array([o['step'] for o in observed]), label=np.array([o['label'] for o in observed]))


def same_observations(path, observed):
    import numpy as np
    saved = np.load(path)
    return all(np.array_equal(saved[k], np.stack([o[k] for o in observed])) for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain']) \
        and np.array_equal(saved['step'], [o['step'] for o in observed]) and np.array_equal(saved['label'], [o['label'] for o in observed])


def main(command, condition):
    cfg, common, case, raw, versions = setup()
    position = json.loads(RECORD.read_text())
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    meta = dict(condition=condition, case_id=CASE_ID, split='train', seed=raw['seed'], versions=versions,
                position='own turn replayed from the own-turn B2B record; teacher-built', scripted=True, model_used=False,
                use='candidate demonstration for review; not yet training data')
    path = OUT / 'conditions' / f'{condition}.json'
    obs_path = OUT / 'observations' / f'{condition}.npz'
    if command == 'check':
        own_turn = sum(1 for r in position['trace'] if r['snapshot']['turn_player'] == 0)
        print('CHECK', json.dumps(dict(case=CASE_ID, runs=len(ledger.started()), limit=LIMIT, own_turn_steps=own_turn,
                                       record_sha256=versions['position_record_sha256'][:16])), flush=True)
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    run = ledger.start(f'{command} condition={condition}', CASE_ID, C.now())
    try:
        got = play(common, case, position, condition if command == 'run' else None, record)
        if command == 'run':
            save_observations(obs_path, got['observed'])
            out = dict(meta, engine_run=run, failure=got['failure'], mismatch=got['mismatch'], lua_errors=got['lua_errors'],
                       game_over=got['game_over'], script_state=got['state'], end=got['end'], trace=got['rows'],
                       observations=dict(path=str(obs_path), sha256=C.sha(obs_path), decisions=len(got['observed'])))
            write(path, out)
            st = got['state']
            print('RUN', json.dumps(dict(run=run, condition=condition, failure=got['failure'], mismatch=got['mismatch'], game_over=got['game_over'],
                                         b2b=st['b2b'], track_search=st['track_search'], attacks=st['attacks'], decisions=len(got['observed']),
                                         notable=[e for e in st['log'] if 'NOT' in e['label'] or 'UNEXPECTED' in e['label']][:6],
                                         labels=[e['label'] for e in st['log'] if e['label'] not in ('pass', 'yesno-pass')][:40],
                                         end_lp=(got['end']['lp0'], got['end']['lp1']))), flush=True)
        else:
            out = dict(meta, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                       lua_errors=got['lua_errors'], game_over=got['game_over'], reproduced_steps=len(got['rows']), of=len(record['trace']),
                       end_matches=A.normalized(got['end']) == A.normalized(record['end']),
                       observations_match=same_observations(obs_path, got['observed']))
            write(OUT / 'replays' / f'{condition}.json', out)
            print('REPLAY', json.dumps({k: out[k] for k in ['engine_run', 'mismatch', 'failure', 'game_over', 'reproduced_steps', 'of', 'end_matches', 'observations_match']}), flush=True)
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
