"""Zaren (2) chain-condition test: four conditions on one teacher-built position (shadow evaluation).

Position: own turn = the recorded P1 teacher route (TRAIN case cue-reco-jj-zaren-g00-p0; board Zaren + Remix,
Jukejoint, set Kewl Tune Synchro; hand Effect Veiler x3), replayed step by step with checks. Opponent turn: the
existing O1 opponent (Rukina (1) to the centre, Rukina (2) search). Our side passes everything (no Remix preparation)
except, per condition:
  chain1       nothing: Rukina's search stays the only link (Zaren must not be offered)
  veiler       Effect Veiler on Rukina as chain link 2; Zaren not used (control)
  zaren-link1  Veiler as link 2, then Zaren (2) choosing description 3: negate/destroy the effect link 2 chained to
               (Rukina's search, link 1)
  zaren-link2  Veiler as link 2, then Zaren (2) choosing description 2: negate/destroy that effect (our Veiler, link 2)
The last condition only checks the evaluator; it is not a line to learn. If a step cannot be built legally the
condition is recorded as not constructed. No model, no training, no battles; production files are only read.
Engine runs are capped at 8 in this root's own ledger.

  zaren.py launch <check|run|replay> <chain1|veiler|zaren-link1|zaren-link2>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/zaren-chain-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
RECORD = P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json'
CASE_ID = 'cue-reco-jj-zaren-g00-p0'
CONDITIONS = ['chain1', 'veiler', 'zaren-link1', 'zaren-link2']
WANTED_OPTION = {'zaren-link1': 3, 'zaren-link2': 2}
LIMIT = 8
OPP_STEP_CAP = 250
KEY_CARDS = [4891376, 97268402, 13597785, 88170262]
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import zaren_analysis as Z  # noqa: E402
import continuous_learning as C  # noqa: E402
A = Z.A


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
                    zaren_sha256=C.sha(Path(__file__)), zaren_analysis_sha256=C.sha(OUT / 'code/zaren_analysis.py'),
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


def our_side(v, raw, condition, st, step):
    acts, msg, chain = raw['actions'], raw['message'], raw['public_chain']
    activate = lambda code, effect=None: next((a for a in acts if a['act'] == 'Activate' and a['code'] == code
                                               and (effect is None or A.effect_index(a) == effect)), None)
    if msg == 'select_option' and st['zaren'] is not None and st['option'] is None:
        offered = sorted(A.effect_index(a) for a in acts if a['code'] == Z.ZAREN)
        found = activate(Z.ZAREN, WANTED_OPTION[condition])
        st['option'] = dict(step=step, offered=offered, wanted=WANTED_OPTION[condition], constructed=found is not None)
        return (found or acts[0])['index'], 'zaren-option' if found else 'zaren-option-NOT-OFFERED'
    if msg in ['select_card', 'select_unselect_card'] and st['veiler'] is not None and st['veiler_target'] is None:
        found = next((a for a in acts if a['code'] == Z.RUK), None)
        st['veiler_target'] = dict(step=step, constructed=found is not None)
        return (found or acts[0])['index'], 'veiler-target' if found else 'veiler-target-NOT-OFFERED'
    if msg == 'select_chain':
        if condition != 'chain1' and st['veiler'] is None and len(chain) == 1 and chain[0]['player'] == 1 and chain[0]['code'] == Z.RUK:
            found = activate(Z.VEILER)
            if found:
                st['veiler'] = step
                # the env resolves one-candidate prompts itself; keep the candidates to show the target is unambiguous
                st['veiler_candidates'] = sorted(c['code'] for c in raw['cards'] if c['controller'] == 1 and c['location'] == 4 and c['position'] & 5)
                return found['index'], 'veiler'
            st['veiler_missing'] = step
        if condition in WANTED_OPTION and st['veiler'] is not None and st['zaren'] is None and len(chain) == 2 \
                and chain[1]['player'] == 0 and chain[1]['code'] == Z.VEILER:
            found = activate(Z.ZAREN, 1)
            if found:
                st['zaren'] = step
                return found['index'], 'zaren'
            st['zaren_missing'] = step
    return v.passive(raw), 'pass'


def play(common, case, position, condition=None, record=None):
    import numpy as np
    w, v, native = common.v, common.v.v, common.v.native
    env, obs, info = v.create(case)
    rows, failure, mismatch = [], None, None
    st = dict(veiler=None, veiler_target=None, veiler_candidates=None, zaren=None, option=None, veiler_missing=None, zaren_missing=None)
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
            for _ in range(OPP_STEP_CAP):
                raw = native.benchmark_snapshot()
                if live['search_started'] and raw['chain_size'] == 0:
                    break
                if raw['turn_count'] > 2:
                    failure = 'opponent_turn_passed'; break
                if raw['to_play'] == 0:
                    idx, source = our_side(v, raw, condition, st, step)
                else:
                    idx, source = o1_opponent(v, raw, live), 'o1-opponent'
                rows.append(dict(step=step, snapshot=v.compact(raw), selected=raw['actions'][idx], source=source))
                obs, _, term, trunc, info = env.step(np.array([idx], np.int32)); step += 1
                if term[0] or trunc[0]:
                    failure = 'terminated_or_truncated'; break
            else:
                failure = 'opponent_step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(native.benchmark_snapshot())
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, own_end=own_end, end=end, state=st, failure=failure, mismatch=mismatch, lua_errors=lua)


def constructed(condition, st):
    need = {'chain1': [], 'veiler': ['veiler', 'veiler_target'], 'zaren-link1': ['veiler', 'veiler_target', 'zaren', 'option'],
            'zaren-link2': ['veiler', 'veiler_target', 'zaren', 'option']}[condition]
    target_ok = st['veiler'] is None or (st['veiler_target']['constructed'] if st['veiler_target'] is not None
                                         else st.get('veiler_candidates') == [Z.RUK])  # no prompt: only Rukina could be targeted
    need = [k for k in need if k != 'veiler_target']
    return all(st[k] is not None for k in need) and target_ok and (st['option'] is None or st['option']['constructed'])


def main(command, condition):
    cfg, common, case, raw, versions = setup()
    position = json.loads(RECORD.read_text())
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    meta = dict(condition=condition, case_id=CASE_ID, split='train', seed=raw['seed'], versions=versions,
                position='own turn replayed from the P1 teacher record; teacher-built, not the model reaching it',
                evaluator_check_only=condition == 'zaren-link2')
    path = OUT / 'conditions' / f'{condition}.json'
    if command == 'check':
        own_turn = sum(1 for r in position['trace'] if r['snapshot']['turn_player'] == 0)
        print('CHECK', json.dumps(dict(case=CASE_ID, runs=len(ledger.started()), limit=LIMIT, own_turn_steps=own_turn,
                                       zaren_script=versions['card_scripts'][Z.ZAREN])), flush=True)
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    run = ledger.start(f'{command} condition={condition}', CASE_ID, C.now())
    try:
        got = play(common, case, position, condition if command == 'run' else None, record)
        if command == 'run':
            ok = got['failure'] is None and got['mismatch'] is None and constructed(condition, got['state'])
            out = dict(meta, engine_run=run, failure=got['failure'], mismatch=got['mismatch'], lua_errors=got['lua_errors'],
                       script_state=got['state'], constructed=ok, own_end_matches_record=A.normalized(got['own_end']) == A.normalized(position['own_end']),
                       end=got['end'], trace=got['rows'],
                       analysis=dict(zaren_legality=Z.zaren_legality(got['rows']), zaren_option=Z.zaren_option(got['rows']),
                                     outcome=Z.outcome(got['rows'], got['end'])))
            write(path, out)
            o = out['analysis']['outcome']
            print('RUN', json.dumps(dict(run=run, condition=condition, constructed=ok, failure=got['failure'], mismatch=got['mismatch'],
                                         state=got['state'], search=o['search'], rukina_after=o['rukina_locations_after'],
                                         chain=[c['code'] for c in o['longest_chain']], legality=out['analysis']['zaren_legality'])), flush=True)
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
