"""Before/after comparison of the B2B training trial on the pre-registered g03 / g05 B2B landings (shadow, isolated root).

Own turn 1: replayed from this root's landing record, step by step with checks; both models' RNN histories and placement
memories are teacher-forced over it. Turn 2: opponent = O1 (Rukina special summon + search); if O1 cannot continue,
the common fallback (adopted model) plays the rest of the opponent's turn; after O1 the opponent ends its turn. In turn 3
the opponent passes. Our side from turn 2: 'before' = adopted pair, 'after' = the single trained candidate.
Stop when turn 4 begins or when the duel ends; after the duel has ended the engine state is never read.
Time points: T1 end of turn 1, T2 end of turn 2, T3 end of turn 3. Engine runs share this root's 12-run ledger.

  compare_b2b.py launch <check|run|replay> <g03-before|g05-before|g03-after|g05-after>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-train-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
ADOPTED = P / 'experiments/adoption-d16384-20261109/model'
PLAN = json.loads((OUT / 'plan.json').read_text())
CASES = {p['name']: p['case_id'] for p in PLAN['evaluation']['positions']}
CONDITIONS = [f'{l}-{m}' for l in CASES for m in ['before', 'after']]
LIMIT, LIVE_CAP = 12, 900
sys.path.insert(0, str(P / 'experiments/landing-compare-20261109/code'))  # compare_policy.py (unchanged, read only)
sys.path.insert(0, str(P / 'scripts'))
import compare_policy as CP  # noqa: E402
import continuous_learning as C  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_json(path, json.loads(json.dumps(value, default=str)))


def model_dir(which):
    if which == 'before':
        return ADOPTED
    record = json.loads((OUT / 'candidate.json').read_text())  # written once by the training step
    path = Path(record['candidate'])
    assert C.sha(path / 'base.flax_model') == record['base_sha256'] and C.sha(path / 'chain-head.flax_model') == record['head_sha256']
    return path


def setup(landing):
    cfg = C.load_config(CONFIG)
    root = Path(cfg['root'])
    C.runtime_check(cfg)
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_openings_common as common
    import all_opening_cases as cases
    manifest = json.loads((root / 'snapshot/opening-manifest.json').read_text())['decks']['ElfnotesKewlTune']['cases']
    raw = next(c for c in manifest['train'] if c['case_id'] == CASES[landing])
    return cfg, common, cases.normalize(raw), raw


def forced_observe(common, policy, obs, info, idx):
    """Same as experiments/landing-compare-20261109/code/compare.py: advance the RNN on the executed route and let the
    placement helper observe the executed action."""
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


def terminal(step, last_snapshot, reward, term, trunc, info):
    import numpy as np
    pick = lambda k: (np.asarray(info[k]).tolist() if k in info else None)
    return dict(step=step, terminated=bool(term[0]), truncated=bool(trunc[0]), reward=np.asarray(reward).tolist(),
                win_reason=pick('win_reason'), lp_before_final_step=(last_snapshot['lp0'], last_snapshot['lp1']))


def play(cfg, common, case, landing_record, which=None, record=None):
    import numpy as np
    w, v, native = common.v, common.v.v, common.v.native
    env, obs, info = v.create(case)
    audit, audit_failures = w.VisibilityAudit(), []
    rows, failure, mismatch, times, ended = [], None, None, {}, None
    if record is None:
        backend = C.backend()
        ours, theirs = backend.create_policy(cfg, model_dir(which)), backend.create_policy(cfg, ADOPTED)
        ours.reset(); theirs.reset()
        prefix = landing_record['trace']
        opp = CP.Opponent(CP.SearchScript(), lambda o, i: theirs.act(o, i), v.passive)
    else:
        prefix, opp = record['trace'], None

    def check_audit(raw, step):
        try:
            audit.check(obs, info, raw)
        except AssertionError as exc:
            audit_failures.append(dict(step=step, error=str(exc)[:300]))

    def advance(idx, step, s):
        nonlocal obs, info, ended
        obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
        if term[0] or trunc[0]:
            ended = terminal(step, s, reward, term, trunc, info)
        return ended is None

    try:
        for row in prefix:
            raw = native.benchmark_snapshot()
            s = v.compact(raw)
            idx = row['selected']['index']
            if idx >= len(raw['actions']):
                mismatch = dict(step=row['step'], keys=['action_index_out_of_range']); break
            keys = A.compare_step(row, s, raw['actions'][idx])
            if keys:
                mismatch = dict(step=row['step'], keys=keys); break
            if record is None:
                if raw['to_play'] == 0:
                    check_audit(raw, row['step'])
                    forced_observe(common, ours, obs, info, idx)
                else:
                    forced_observe(common, theirs, obs, info, idx)
            for key, turn in [('T2', 3), ('T3', 4)]:
                if raw['turn_count'] >= turn and key not in times:
                    times[key] = dict(step=row['step'], snapshot=s)
            rows.append(dict(step=row['step'], snapshot=s, selected=raw['actions'][idx], source=row.get('source', 'landing-record')))
            if not advance(idx, row['step'], s):
                break
        if record is None and mismatch is None and ended is None:
            times['T1'] = dict(step=rows[-1]['step'] + 1, snapshot=v.compact(native.benchmark_snapshot()))
            step = rows[-1]['step'] + 1
            for _ in range(LIVE_CAP):
                raw = native.benchmark_snapshot()
                s = v.compact(raw)
                for key, turn in [('T2', 3), ('T3', 4)]:
                    if raw['turn_count'] >= turn and key not in times:
                        times[key] = dict(step=step, snapshot=s)
                if CP.finished(raw):
                    break
                if raw['to_play'] == 0:
                    check_audit(raw, step)
                    idx, source = ours.act(obs, info), 'model'
                else:
                    idx, source = opp.decide(raw, obs, info, step)
                    if source != 'fallback-model' and raw['turn_count'] <= 2:
                        forced_observe(common, theirs, obs, info, idx)
                rows.append(dict(step=step, snapshot=s, selected=raw['actions'][idx], source=source))
                ok = advance(idx, step, s)
                step += 1
                if not ok:
                    break
            else:
                failure = 'live_step_cap'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    end = rows[-1]['snapshot'] if ended else v.compact(native.benchmark_snapshot())
    if ended:
        times.setdefault('T2' if end['turn_count'] < 3 else 'T3', dict(step=rows[-1]['step'], snapshot=end, duel_ended=True))
    times.setdefault('T3', dict(step=rows[-1]['step'] + 1 if rows else 0, snapshot=end))
    lua = native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, times=times, end=end, ended=ended, failure=failure, mismatch=mismatch, lua_errors=lua,
                opponent=dict(mode=opp.mode, blocked=opp.blocked) if opp else None, visibility=dict(checks=audit.checks, failures=audit_failures))


def main(command, condition):
    landing, which = condition.split('-')
    cfg, common, case, raw = setup(landing)
    landing_path = OUT / 'landings' / f'{landing}.json'
    landing_record = json.loads(landing_path.read_text())
    assert landing_record['ok'] and json.loads((OUT / 'landings' / f'{landing}-replay.json').read_text())['ok']
    ledger = A.Ledger(OUT / 'engine-runs.jsonl', limit=LIMIT)
    path = OUT / 'conditions' / f'{condition}.json'
    if command == 'check':
        mdir = model_dir(which) if which == 'before' or (OUT / 'candidate.json').exists() else None
        print('CHECK', json.dumps(dict(condition=condition, runs=len(ledger.started()), model=str(mdir), landing_steps=len(landing_record['trace']))))
        return
    record = json.loads(path.read_text()) if command == 'replay' else None
    if command == 'run' and path.exists():
        raise RuntimeError('refusing to overwrite a recorded comparison')
    mdir = model_dir(which)
    run = ledger.start(f'compare {command} {condition}', raw['case_id'], C.now())
    try:
        got = play(cfg, common, case, landing_record, which if command == 'run' else None, record)
        if command == 'run':
            import jax
            out = dict(condition=condition, landing=landing, model=which, case_id=raw['case_id'], seed=raw['seed'], split='train',
                       engine_run=run, device=str(jax.devices()[0]),
                       our_model=dict(path=str(mdir), base_sha256=C.sha(mdir / 'base.flax_model'), head_sha256=C.sha(mdir / 'chain-head.flax_model')),
                       fallback_model=dict(path=str(ADOPTED), base_sha256=C.sha(ADOPTED / 'base.flax_model')),
                       landing_record=dict(path=str(landing_path), sha256=C.sha(landing_path)),
                       code=dict(compare_b2b_sha256=C.sha(Path(__file__)), compare_policy_sha256=C.sha(P / 'experiments/landing-compare-20261109/code/compare_policy.py')),
                       failure=got['failure'], mismatch=got['mismatch'], lua_errors=got['lua_errors'], duel_ended=got['ended'],
                       opponent_script=got['opponent'], visibility=got['visibility'], time_points=got['times'], end=got['end'], trace=got['rows'])
            write(path, out)
            T = lambda k: got['times'].get(k, {}).get('snapshot', {})
            field = lambda s, c: sorted(x['code'] for x in s.get('cards', []) if x['controller'] == c and x['location'] == 4)
            print('RUN', json.dumps(dict(run=run, condition=condition, failure=got['failure'], mismatch=got['mismatch'], ended=got['ended'],
                                         opponent=got['opponent'], audit_failures=len(got['visibility']['failures']),
                                         T={k: dict(our=field(T(k), 0), opp=field(T(k), 1), lp=(T(k).get('lp0'), T(k).get('lp1'))) for k in ['T1', 'T2', 'T3']}),
                                    default=str), flush=True)
        else:
            out = dict(condition=condition, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                       lua_errors=got['lua_errors'], duel_ended=got['ended'], reproduced_steps=len(got['rows']), of=len(record['trace']),
                       end_matches=A.normalized(got['end']) == A.normalized(record['end']))
            write(OUT / 'replays' / f'{condition}.json', out)
            print('REPLAY', json.dumps({k: out[k] for k in ['engine_run', 'mismatch', 'failure', 'reproduced_steps', 'of', 'end_matches', 'duel_ended']}, default=str), flush=True)
        ledger.finish(run, 'ok' if not got['failure'] and not got['mismatch'] else 'completed-with-findings', C.now(),
                      failure=got['failure'], mismatch=got['mismatch'])
    except BaseException as exc:
        ledger.finish(run, 'error', C.now(), error=f'{type(exc).__name__}: {exc}'); raise


def launch(command, condition):
    cfg = C.load_config(CONFIG)
    log = OUT / 'logs' / f'compare-{command}-{condition}.log'
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
