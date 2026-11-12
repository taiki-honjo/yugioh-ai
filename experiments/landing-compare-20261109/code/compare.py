"""Two landings x two opponents, recorded at three time points (shadow evaluation, isolated root).

Landings (own turn 1, replayed step by step from their records with checks; the model's RNN history and placement
memory are teacher-forced over these steps):
  P1   Zaren + Remix, Jukejoint, set Kewl Tune Synchro, Veiler x3 (experiments/capability-probe-20261109 P1 record)
  B2B  Kewl Tune B2B, Jukejoint, set Synchro, Veiler x3 (experiments/b2b-zaren-ownturn-check-20261109 record)
Both start from the same TRAIN hand and seed (cue-reco-jj-zaren-g00-p0, 151080), so the opponent's hand/deck are equal.
Opponents (turn 2, see compare_policy.py): search (O1) or synchro (the elf teacher route, mirrored to player 1);
on a block the same adopted model plays the rest of the opponent's turn (common fallback); in turn 3 it passes.
Our side from turn 2 on: the adopted model (base 86b4358d / head 6ca3221f, production adapters). This is NOT a test of
whether the model can build either landing. Time points: T1 end of our turn 1, T2 end of the opponent's turn 2,
T3 end of our turn 3. No training, no normal battles, production files only read; 8 engine runs in this root's ledger.

  compare.py launch <check|run|replay> <P1-search|P1-synchro|B2B-search|B2B-synchro>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/landing-compare-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
ADOPTED = P / 'experiments/adoption-d16384-20261109/model'
LANDINGS = {'P1': P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json',
            'B2B': P / 'experiments/b2b-zaren-ownturn-check-20261109/records/via-zaren.json'}
CASE_ID, SYNCHRO_CASE = 'cue-reco-jj-zaren-g00-p0', 'elf-rukina-regina-g00-p0'
CONDITIONS = [f'{l}-{o}' for l in LANDINGS for o in ['search', 'synchro']]
LIMIT, LIVE_CAP = 8, 900
KEY_CARDS = [65961304, 4891376, 88170262, 14558127, 97268402, 13597785, 42302563, 50954680, 84815190]
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import compare_policy as CP  # noqa: E402
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
    raw = {c['case_id']: c for c in manifest['train'] if c['case_id'] in (CASE_ID, SYNCHRO_CASE)}
    for case in raw.values():
        key = tuple(sorted(case['hand']))
        assert not [s for s in ['validation', 'holdout'] if any(tuple(sorted(c['hand'])) == key for c in manifest[s])], case['case_id']
    hashes = json.loads((root / 'snapshot/hashes.json').read_text())
    runtime = json.loads((root / 'snapshot/runtime.json').read_text())
    scripts = Path(cfg['runtime']) / 'agent/scripts/script'
    versions = dict(snapshot_hashes_sha256=C.sha(root / 'snapshot/hashes.json'),
                    policy_code={k: v for k, v in hashes.items() if k.startswith('policy-code/') and k.endswith('.py')},
                    native_sha256=C.sha(runtime['native_binary']), deck_sha256=C.sha(runtime['deck']),
                    card_scripts={code: C.sha(scripts / f'c{code}.lua') for code in KEY_CARDS}, card_script_dir=str(scripts.resolve()),
                    landing_records={k: dict(path=str(v), sha256=C.sha(v)) for k, v in LANDINGS.items()},
                    model=dict(path=str(ADOPTED), base_sha256=C.sha(ADOPTED / 'base.flax_model'), head_sha256=C.sha(ADOPTED / 'chain-head.flax_model')),
                    compare_sha256=C.sha(Path(__file__)), compare_policy_sha256=C.sha(OUT / 'code/compare_policy.py'),
                    capability_analysis_sha256=C.sha(P / 'experiments/capability-probe-20261109/code/analysis.py'))
    return cfg, common, cases, raw, versions


def forced_observe(common, policy, obs, info, idx):
    """Advance the model's RNN on the executed route (state update does not depend on the chosen action) and let the
    placement helper observe the EXECUTED action (same as experiments/capability-probe-20261109/code/probe.py)."""
    import numpy as np
    placement = policy.base
    core = placement.base
    core.state, _ = core.infer(core.base, core.head, core.state, obs, common.features(info))
    actions = np.asarray(obs['actions_'])[0]
    cards = np.asarray(obs['cards_'])[0]
    row = actions[idx]
    code = placement.id_codes.get(int(row[1]) * 256 + int(row[2]), 0)
    spec = int(row[0])
    hand = bool(spec and spec <= len(cards) and int(cards[spec - 1, 2]) == 2 and int(cards[spec - 1, 4]) == 0)
    placement.memory.observe(int(row[3]), int(row[4]), code, hand, int(np.asarray(info['public_chain_length'])[0]))


def play(cfg, common, cases, raw, condition=None, record=None):
    import numpy as np
    w, v, native = common.v, common.v.v, common.v.native
    case = cases.normalize(raw[CASE_ID])
    env, obs, info = v.create(case)
    audit, audit_failures = w.VisibilityAudit(), []
    rows, failure, mismatch, times = [], None, None, {}
    landing, opponent_kind = condition.split('-') if condition else (None, None)
    if record is None:
        backend = C.backend()
        ours, theirs = backend.create_policy(cfg, ADOPTED), backend.create_policy(cfg, ADOPTED)
        ours.reset(); theirs.reset()
        prefix = [r for r in json.loads(LANDINGS[landing].read_text())['trace'] if r['snapshot']['turn_player'] == 0]
        script = CP.SearchScript() if opponent_kind == 'search' else CP.TeacherScript(w.teacher(cases.normalize(raw[SYNCHRO_CASE])))
        opp = CP.Opponent(script, lambda o, i: theirs.act(o, i), v.passive)
    else:
        prefix, opp = record['trace'], None

    def check_audit(raw_snapshot, step):
        try:
            audit.check(obs, info, raw_snapshot)
        except AssertionError as exc:  # recorded as a visibility finding; the run continues
            audit_failures.append(dict(step=step, error=str(exc)[:300]))

    try:
        for row in prefix:
            s_raw = native.benchmark_snapshot()
            s = v.compact(s_raw)
            idx = row['selected']['index']
            if idx >= len(s_raw['actions']):
                mismatch = dict(step=row['step'], keys=['action_index_out_of_range']); break
            keys = A.compare_step(row, s, s_raw['actions'][idx])
            if keys:
                mismatch = dict(step=row['step'], keys=keys); break
            if record is None:
                if s_raw['to_play'] == 0:
                    check_audit(s_raw, row['step'])
                    forced_observe(common, ours, obs, info, idx)
                else:
                    forced_observe(common, theirs, obs, info, idx)
            if s_raw['turn_count'] >= 3 and 'T2' not in times:
                times['T2'] = dict(step=row['step'], snapshot=s)
            rows.append(dict(step=row['step'], snapshot=s, selected=s_raw['actions'][idx], source=row.get('source', 'landing-record')))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                failure = 'terminated_or_truncated'; break
        if record is None:
            times['T1'] = dict(step=rows[-1]['step'] + 1 if rows else 0, snapshot=v.compact(native.benchmark_snapshot()))
        if record is None and mismatch is None and failure is None:
            step = rows[-1]['step'] + 1
            for _ in range(LIVE_CAP):
                s_raw = native.benchmark_snapshot()
                if s_raw['turn_count'] >= 3 and 'T2' not in times:
                    times['T2'] = dict(step=step, snapshot=v.compact(s_raw))
                if CP.finished(s_raw):
                    break
                if s_raw['to_play'] == 0:
                    check_audit(s_raw, step)
                    idx, source = ours.act(obs, info), 'model'
                else:
                    idx, source = opp.decide(s_raw, obs, info, step)
                    if source != 'fallback-model' and s_raw['turn_count'] <= 2:
                        forced_observe(common, theirs, obs, info, idx)
                rows.append(dict(step=step, snapshot=v.compact(s_raw), selected=s_raw['actions'][idx], source=source))
                obs, _, term, trunc, info = env.step(np.array([idx], np.int32)); step += 1
                if term[0] or trunc[0]:
                    failure = 'terminated_or_truncated'; break
            else:
                failure = 'live_step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = v.compact(native.benchmark_snapshot())
    times.setdefault('T3', dict(step=rows[-1]['step'] + 1 if rows else 0, snapshot=end))
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, times=times, end=end, failure=failure, mismatch=mismatch, lua_errors=lua,
                opponent=dict(mode=opp.mode, blocked=opp.blocked) if opp else None,
                visibility=dict(checks=audit.checks, failures=audit_failures))


def main(command, condition):
    cfg, common, cases, raw, versions = setup()
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    landing, opponent_kind = condition.split('-')
    meta = dict(condition=condition, landing=landing, opponent=opponent_kind, case_id=CASE_ID, synchro_script_case=SYNCHRO_CASE,
                split='train', seed=raw[CASE_ID]['seed'], versions=versions,
                our_side='own turn 1 replayed from the landing record (teacher-forced model history); adopted model from turn 2',
                opponent_side='script in turn 2; common fallback = adopted model on a block; passive in turn 3',
                not_measured='whether the model can build either landing by itself')
    path = OUT / 'conditions' / f'{condition}.json'
    if command == 'check':
        print('CHECK', json.dumps(dict(condition=condition, runs=len(ledger.started()), limit=LIMIT,
                                       landing_rows=sum(1 for r in json.loads(LANDINGS[landing].read_text())['trace'] if r['snapshot']['turn_player'] == 0),
                                       model=versions['model'])), flush=True)
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    run = ledger.start(f'{command} condition={condition}', CASE_ID, C.now())
    try:
        got = play(cfg, common, cases, raw, condition if command == 'run' else None, record)
        if command == 'run':
            import jax
            out = dict(meta, engine_run=run, device=str(jax.devices()[0]), failure=got['failure'], mismatch=got['mismatch'],
                       lua_errors=got['lua_errors'], opponent_script=got['opponent'], visibility=got['visibility'],
                       time_points=got['times'], end=got['end'], trace=got['rows'])
            write(path, out)
            srcs = {}
            for r in got['rows']:
                srcs[r['source']] = srcs.get(r['source'], 0) + 1
            T = lambda k: got['times'].get(k, {}).get('snapshot', {})
            field = lambda s, c: sorted(x['code'] for x in s.get('cards', []) if x['controller'] == c and x['location'] == 4)
            times = {k: dict(our=field(T(k), 0), opp=field(T(k), 1), lp=(T(k).get('lp0'), T(k).get('lp1')), turn=T(k).get('turn_count'))
                     for k in ['T1', 'T2', 'T3']}
            print('RUN', json.dumps(dict(run=run, condition=condition, failure=got['failure'], mismatch=got['mismatch'], sources=srcs,
                                         opponent=got['opponent'], audit_failures=len(got['visibility']['failures']), T=times),
                                    default=str), flush=True)
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
