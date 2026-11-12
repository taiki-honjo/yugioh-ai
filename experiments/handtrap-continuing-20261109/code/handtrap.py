"""Maxx C / Droll & Lock Bird continuing-effect conditions (shadow evaluation, isolated experiment root).

Same opening hands, seed and deploying script in all four conditions:
- player 0 (deploying side) follows the fixed teacher route of the TRAIN case elf-rukina-regina-g00-p0 during turn 1
  (the P2 route of experiments/capability-probe-20261109);
- player 1 holds the standard fixed hand (Rukina, Ash Blossom, Effect Veiler, Droll & Lock Bird, Maxx C) and only
  activates Maxx C and/or Droll according to the condition, at the first window where each is legal; it passes on
  everything else.
If the deploying script cannot continue (its next planned action is not offered), the block is recorded and the
deploying side ends its turn without new actions (a property of this script, reported as such).

No model, no training, no battles; production files are only read. Engine runs are capped at 8 in this root's own
ledger (4 conditions + 4 replays); the capability-probe ledger is not used or changed.

  handtrap.py launch <check|run|replay> <none|g|droll|both>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/handtrap-continuing-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
CASE_ID = 'elf-rukina-regina-g00-p0'
CONDITIONS = {'none': (), 'g': ('g',), 'droll': ('droll',), 'both': ('g', 'droll')}
LIMIT = 8
KEY_CARDS = [23434538, 94145021, 13597785, 42302563, 56651978, 59581480]
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import handtrap_analysis as H  # noqa: E402
import continuous_learning as C  # noqa: E402
A = H.A


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_json(path, json.loads(json.dumps(value, default=str)))


def setup():
    cfg = C.load_config(CONFIG)
    root = Path(cfg['root'])
    C.runtime_check(cfg)
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_openings_common as common  # same import as the gate job (installs teacher preferences)
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
                    handtrap_sha256=C.sha(Path(__file__)), handtrap_analysis_sha256=C.sha(OUT / 'code/handtrap_analysis.py'),
                    capability_analysis_sha256=C.sha(P / 'experiments/capability-probe-20261109/code/analysis.py'))
    return cfg, common, cases.normalize(raw), raw, versions


def handtrap_side(v, raw, condition, used, step):
    acts = raw['actions']
    for key, code in [('g', H.G), ('droll', H.DROLL)]:
        if key in condition and used[key] is None:
            found = next((a for a in acts if a['act'] == 'Activate' and a['code'] == code), None)
            if found:
                used[key] = step
                return found['index'], 'handtrap-' + key
    return v.passive(raw), 'handtrap-pass'


def deploying_side(v, teacher, raw, state, step):
    if state['blocked'] is None:
        try:
            return teacher(raw), 'script'
        except Exception as exc:  # the script cannot continue: record what it wanted and what was legal
            state['blocked'] = dict(step=step, message=raw['message'], phase=raw['phase'], error=f'{type(exc).__name__}: {exc}'[:1500],
                                    next_plan_steps=[dict(x) for x in list(getattr(teacher, 'steps', []))[:6]],
                                    legal=[A.meaning(a) for a in raw['actions']])
    try:
        return v.passive(raw), 'fallback-pass'
    except Exception:
        return 0, 'fallback-first-legal'


def play(common, case, condition=None, record=None):
    """condition: run that condition live. record: replay its index sequence, checking every step."""
    import numpy as np
    w, v, native = common.v, common.v.v, common.v.native
    env, obs, info = v.create(case)
    teacher = w.teacher(case) if record is None else None
    rows, used, state, failure, mismatch = [], dict(g=None, droll=None), dict(blocked=None), None, None
    start = v.compact(native.benchmark_snapshot())
    recorded = record['trace'] if record else None
    try:
        for step in range(420 if record is None else len(recorded)):
            raw = native.benchmark_snapshot()
            s = v.compact(raw)
            if record is None:
                if raw['turn_player'] == 1 and raw['chain_size'] == 0:
                    break  # end of the deploying side's turn
                if raw['to_play'] == 0:
                    idx, source = deploying_side(v, teacher, raw, state, step)
                else:
                    idx, source = handtrap_side(v, raw, CONDITIONS[condition], used, step)
            else:
                row = recorded[step]
                idx, source = row['selected']['index'], 'replay'
                if idx >= len(raw['actions']):
                    mismatch = dict(step=step, keys=['action_index_out_of_range']); break
                keys = A.compare_step(row, s, raw['actions'][idx])
                if keys:
                    mismatch = dict(step=step, keys=keys); break
            rows.append(dict(step=step, snapshot=s, selected=raw['actions'][idx], source=source))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                failure = 'terminated_or_truncated'; break
        else:
            if record is None:
                failure = 'decision_limit'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(native.benchmark_snapshot())
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, start=start, end=end, used=used, blocked=state['blocked'], failure=failure, mismatch=mismatch, lua_errors=lua)


def main(command, condition):
    cfg, common, case, raw, versions = setup()
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    meta = dict(condition=condition, case_id=CASE_ID, split='train', seed=raw['seed'], deploying_hand=raw['hand'],
                handtrap_side_hand='standard fixed player-1 hand (rukina_recovery_witness.OPP_HAND)', versions=versions,
                handtrap_policy='activate Maxx C / Droll at the first window where each is legal (per condition); pass otherwise',
                deploying_script='teacher route of the train case (player 0, turn 1); on a block: end the turn without new actions')
    path = OUT / 'conditions' / f'{condition}.json'
    if command == 'check':
        print('CHECK', json.dumps(dict(case=CASE_ID, seed=raw['seed'], runs=len(ledger.started()), limit=LIMIT,
                                       g_script=versions['card_scripts'][H.G], droll_script=versions['card_scripts'][H.DROLL])), flush=True)
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    run = ledger.start(f'{command} condition={condition}', CASE_ID, C.now())
    try:
        got = play(common, case, condition if command == 'run' else None, record)
        if command == 'run':
            out = dict(meta, engine_run=run, failure=got['failure'], lua_errors=got['lua_errors'], used=got['used'],
                       script_blocked=got['blocked'], start=got['start'], end=got['end'], trace=got['rows'],
                       summary=H.summarize(got['rows'], got['end'], got['start'], got['blocked']))
            write(path, out)
            print('RUN', json.dumps(dict(run=run, condition=condition, failure=got['failure'], used=got['used'],
                                         blocked=bool(got['blocked']), summary={k: out['summary'][k] for k in
                                         ['p0_adds_from_hidden', 'p0_adds_after_droll', 'p0_monster_entries', 'p1_gained_total',
                                          'p1_gained_after_droll', 'deploying_decisions']})), flush=True)
        else:
            out = dict(meta, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                       lua_errors=got['lua_errors'], reproduced_steps=len(got['rows']), of=len(record['trace']),
                       start_matches=A.normalized(got['start']) == A.normalized(record['start']),
                       end_matches=A.normalized(got['end']) == A.normalized(record['end']))
            write(OUT / 'replays' / f'{condition}.json', out)
            print('REPLAY', json.dumps({k: out[k] for k in ['engine_run', 'mismatch', 'failure', 'reproduced_steps', 'of', 'start_matches', 'end_matches']}), flush=True)
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
    keep = ('CHECK', 'RUN', 'REPLAY', 'Traceback', 'AssertionError', 'RuntimeError', 'KeyError', 'TypeError', 'ValueError')
    print(''.join(l for l in log.read_text().splitlines(True) if l.startswith(keep)), end='')
    print(f'exit={done.returncode} log={log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'job'])
    parser.add_argument('command', choices=['check', 'run', 'replay'])
    parser.add_argument('condition', choices=list(CONDITIONS))
    args = parser.parse_args()
    (launch if args.mode == 'launch' else main)(args.command, args.condition)
