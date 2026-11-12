"""Artmage first finite stage: 4 known-hand regressions + 2 new-opening demos (record + replay). Cap: 8 engine runs.

  artmage_runs.py register                      write registration.json (offline; refuses if it exists)
  artmage_runs.py launch <regression|record|replay> [C1|C2]
  artmage_runs.py job ...                       (internal; run inside the resource-limited scope)

One engine run = one duel from the opening hand. Every run is written to engine-runs.jsonl before it starts;
the ledger refuses a 9th. Failed runs count. Jobs refuse to start if any registered code or pinned file changed.
CPU only (JAX_PLATFORMS=cpu); no training, no battles, no server access, common registry not read.
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
from datetime import datetime, timedelta, timezone

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
CODE = W / 'code'
REG = W / 'registration.json'
LEDGER = W / 'engine-runs.jsonl'
LIMIT = 8
PYTHON = '/opt/ygo-ai/.venv/bin/python'
RUNTIME = P / 'evaluation/opponent-article-basics-20261108/runtime'
OLD_MANIFEST = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json'
ORIGINAL_REGRESSION = P / 'evaluation/all-deck-position-20261109/verified/regression/Artmage/after'
ANALYSIS = P / 'experiments/capability-probe-20261109/code/analysis.py'
REGRESSION_CASES = ['Artmage-g08-p0', 'Artmage-g09-p0', 'Artmage-g10-p0', 'Artmage-g11-p0']
REGISTERED_CODE = ['artmage_pins.py', 'artmage_entry.py', 'artmage_teacher.py', 'artmage_runs.py', 'split_check.py']
LIMITS = dict(job_memory_gib=5, memory_swap_max=0, cpu_percent=300, job_timeout_seconds=1800, nice=10,
              env=dict(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       MALLOC_ARENA_MAX='2', PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled', HF_HUB_OFFLINE='1'))
RECORD_STEP_CAP = 700
JST = timezone(timedelta(hours=9))
sys.path.insert(0, str(CODE))
import artmage_teacher as T  # noqa: E402


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + '\n')
    tmp.replace(path)


def analysis():
    spec = importlib.util.spec_from_file_location('capability_analysis_for_artmage', ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- registration
def plan_json(cond):
    c = T.CONDITIONS[cond]
    return json.loads(json.dumps(dict(c, plan=c['plan'])))


def register():
    if REG.exists():
        raise SystemExit(f'{REG} exists; registration is fixed before runs and not rewritten')
    split = json.loads((W / 'split-check.json').read_text())
    assert split['passed'], 'split check must pass before registration'
    old = json.loads(OLD_MANIFEST.read_text())['decks']['Artmage']['cases']['holdout']
    regression = []
    for cid in REGRESSION_CASES:
        case = next(c for c in old if c['case_id'] == cid)
        original = ORIGINAL_REGRESSION / f'{cid}.json'
        regression.append(dict(case_id=cid, seed=case['seed'], hand=case['hand'], original_record=str(original), original_sha256=sha(original)))
    import artmage_pins
    pins = artmage_pins.verify_pins()
    reg = dict(
        status='registered_before_engine_runs', registered_jst=now(),
        engine_run_cap=LIMIT, ledger=str(LEDGER), counting='one engine run = one duel from the opening hand; failed runs count; ledger never rolled back',
        order=['regression x4 (g08,g09,g10,g11)', 'record C1', 'replay C1', 'record C2', 'replay C2'],
        allocation=dict(regression=4, C1=dict(record=1, replay=1), C2=dict(record=1, replay=1)),
        regression=dict(cases=regression, policy='work-area copy of the registered Artmage base+chain-head with the common position adapter (isolated entry)',
                        loop='same stop rules and step cap (320) as evaluation/all-deck-position-20261109/code/check_other_positions.py (after mode)',
                        compare=['passed', 'reason', 'goal', 'per-step normalized snapshot and selected-action meaning', 'final snapshot', 'position events']),
        demos={c: dict(hand=T.CONDITIONS[c]['hand'], seed=T.CONDITIONS[c]['seed'], reference=T.CONDITIONS[c]['reference'],
                       label=T.CONDITIONS[c]['label'], plan=plan_json(c)['plan']) for c in T.CONDITIONS},
        demo_rules=[
            'Hands, seeds and plans fixed here before any run; no extra runs to search for other hands or plan fixes.',
            'Player 1 holds the same hand and always passes (article_witness.passive order).',
            'Player 0 = scripted Teacher wrapped by the common position adapter (PositionTeacher).',
            'Stop at the first decision of turn 3 (start of the next own turn), at game over (handled from env.step returns, no snapshot afterwards), or at the step cap.',
            'If a planned step is not legal, stop that run and save the plan head and the legal options at that prompt.',
            'Replay = the next run of the same condition: same fixture and seed, recorded action indices, compare every step (normalized snapshot, action meaning) and the saved model-input observations. If the record failed, the replay reproduces the recorded prefix.',
            'Observations saved at every player-0 decision: cards_, global_, actions_, h_actions_, public-chain features, chosen index.',
        ],
        split_check=dict(path=str(W / 'split-check.json'), sha256=sha(W / 'split-check.json')),
        code={name: sha(CODE / name) for name in REGISTERED_CODE},
        pins={k: dict(path=str(artmage_pins.PINS[k][0]), sha256=pins[k]) for k in pins},
        ledger_class=dict(path=str(ANALYSIS), sha256=sha(ANALYSIS)),
        old_manifest=dict(path=str(OLD_MANIFEST), sha256=sha(OLD_MANIFEST)),
        limits=LIMITS, python=PYTHON, stop_file='STOP stays "now"; these finite CPU jobs are not connected to it',
        not_included=['training', 'battles', 'adoption', 'server replay posts', 'common registry changes', 'Elfnotes production changes'],
    )
    write(REG, reg)
    print('REGISTERED', sha(REG))


def check_registration():
    reg = json.loads(REG.read_text())
    for name, digest in reg['code'].items():
        if sha(CODE / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    for key in ['ledger_class', 'old_manifest', 'split_check']:
        if sha(reg[key]['path']) != reg[key]['sha256']:
            raise RuntimeError(f'registered file changed: {key}')
    import artmage_pins
    if {k: x['sha256'] for k, x in reg['pins'].items()} != artmage_pins.verify_pins():
        raise RuntimeError('pinned files differ from registration')
    for c in T.CONDITIONS:
        assert reg['demos'][c]['plan'] == plan_json(c)['plan'] and reg['demos'][c]['hand'] == T.CONDITIONS[c]['hand']
    return reg


# ---------------------------------------------------------------------------- engine helpers
def observation(obs, info):
    length = int(info['public_chain_length'][0])
    owner = int(info['public_chain_player'][0])
    chain = [length, 0 if not length else 1 if owner == 0 else 2, int(info['public_chain_card_id'][0]), int(info['public_chain_effect'][0])]
    import numpy as np
    out = {k: np.asarray(obs[k])[0].copy() for k in ['cards_', 'global_', 'actions_', 'h_actions_']}
    out['chain'] = np.array(chain, np.int32)
    return out


def save_observations(path, rows):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack([r['obs'][k] for r in rows]) for k in rows[0]['obs']} if rows else {}
    arrays.update(labels=np.array([r['label'] for r in rows], np.int32), steps=np.array([r['step'] for r in rows], np.int32),
                  turns=np.array([r['turn'] for r in rows], np.int32), num_options=np.array([r['n'] for r in rows], np.int32))
    np.savez_compressed(path, **arrays)


def same_observations(path, rows):
    import numpy as np
    saved = np.load(path)
    if len(rows) != len(saved['labels']):
        return False
    for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain']:
        if not np.array_equal(saved[k], np.stack([r['obs'][k] for r in rows])):
            return False
    return bool(np.array_equal(saved['labels'], np.array([r['label'] for r in rows], np.int32)))


def board(s):
    own = lambda loc: sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == loc)
    face_up = lambda loc: sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == loc and c['position'] & 5)
    return dict(turn=s['turn_count'], phase=s['phase'], monsters=own(4), monsters_face_up=face_up(4),
                spell_trap=own(8), spell_trap_face_up=face_up(8), hand=own(2), graveyard=own(16), banished=own(32),
                extra_face_up=sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == 64 and c['position'] & 5),
                deck_count=sum(1 for c in s['cards'] if c['controller'] == 0 and c['location'] == 1),
                lp=s.get('lp'), opponent_hand_count=sum(1 for c in s['cards'] if c['controller'] == 1 and c['location'] == 2))


# ---------------------------------------------------------------------------- regression
def run_regression(E, A, ledger):
    v, np = E.v, E.np
    old = json.loads(OLD_MANIFEST.read_text())['decks']['Artmage']['cases']['holdout']
    results = []
    for cid in REGRESSION_CASES:
        case = next(c for c in old if c['case_id'] == cid)
        run = ledger.start('regression', cid, now())
        env = None
        try:
            policy = E.create_policy()
            env, obs, info = E.env_for(case)
            policy.reset()
            trace, reason = [], 'decision_limit'
            for step in range(320):
                s = v.native.benchmark_snapshot()
                goal = v.check_goal('Artmage', s)
                if s['turn_player'] == 1 and s['chain_size'] == 0 and all(goal.values()):
                    reason = 'goal_reached'; break
                if s['turn_player'] == 1 and s['phase'] > 4 and s['chain_size'] == 0:
                    reason = 'opponent_main_phase_passed'; break
                if s['turn_count'] > 2:
                    reason = 'two_turns_passed'; break
                assert int(info['raw_num_options'][0]) <= 128
                v.check_observation(obs, info, s)
                idx = policy.act(obs, info) if s['to_play'] == 0 else v.passive(s)
                assert idx in [x['index'] for x in s['actions']]
                trace.append(dict(step=step, snapshot=v.compact(s), selected=s['actions'][idx]))
                obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
                if term[0] or trunc[0]:
                    reason = 'episode_finished'; break
            final = v.compact(v.native.benchmark_snapshot()) if reason != 'episode_finished' else None
            goal = v.check_goal('Artmage', final) if final else None
            row = dict(deck='Artmage', mode='after', case_id=cid, passed=bool(final) and all(goal.values()) and final['turn_player'] == 1,
                       goal=goal, reason=reason, final=final, trace=trace, events=copy.deepcopy(policy.positions.events),
                       engine_run=run, policy=policy.policy_metadata)
            original = json.loads((ORIGINAL_REGRESSION / f'{cid}.json').read_text())
            per_step = [A.compare_step(o, n['snapshot'], n['selected']) for o, n in zip(original['trace'], trace)]
            first_diff = next((i for i, d in enumerate(per_step) if d), None)
            cmp = dict(case_id=cid, passed=row['passed'], original_passed=original['passed'], reason=row['reason'], original_reason=original['reason'],
                       goal_equal=row['goal'] == original['goal'], steps=len(trace), original_steps=len(original['trace']),
                       first_differing_step=first_diff, differing_keys=per_step[first_diff] if first_diff is not None else [],
                       final_equal=A.normalized(final) == A.normalized(original['final']),
                       events_equal=A.normalized(row['events']) == A.normalized(original['events']))
            cmp['identical'] = (cmp['passed'] == cmp['original_passed'] and cmp['reason'] == cmp['original_reason'] and cmp['goal_equal']
                                and cmp['steps'] == cmp['original_steps'] and first_diff is None and cmp['final_equal'] and cmp['events_equal'])
            row['comparison'] = cmp
            write(W / 'regression' / f'{cid}.json', row)
            results.append(cmp)
            print('REGRESSION', json.dumps(cmp, ensure_ascii=False), flush=True)
            ledger.finish(run, 'ok', now(), passed=row['passed'], identical=cmp['identical'])
        except BaseException as exc:
            ledger.finish(run, 'error', now(), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if env is not None:
                E.close(env)
                del env
            gc.collect()
            assert v.native.get_env_resource_stats()['live_duels'] == 0
    write(W / 'regression' / 'summary.json', dict(results=results, lua_errors=v.native.get_lua_error_count(),
                                                  all_identical=all(r['identical'] for r in results), passed=sum(r['passed'] for r in results)))


# ---------------------------------------------------------------------------- demos
def play(E, cond, replay_record=None, obs_path=None):
    """One duel of condition cond. Teacher-driven (record) or index-driven (replay)."""
    v, np = E.v, E.np
    A = analysis()
    c = T.CONDITIONS[cond]
    case = dict(case_id=cond, hand=c['hand'], seed=c['seed'])
    teacher = None if replay_record else E.create_teacher(T.Teacher(c['plan']))
    env, obs, info = E.env_for(case)
    rows, observed, opponent_turn_offers = [], [], []
    failure = mismatch = game_over = blocking = None
    checkpoints, reason = {}, 'step_cap'
    previous_turn_player = 0
    try:
        for step in range(RECORD_STEP_CAP):
            s = v.native.benchmark_snapshot()
            if s['turn_player'] != previous_turn_player and 'own_turn_end' not in checkpoints:
                checkpoints['own_turn_end'] = board(s)
            previous_turn_player = s['turn_player']
            if s['turn_count'] >= 3:
                checkpoints['next_own_turn_start'] = board(s)
                reason = 'reached_turn3'; break
            if replay_record and step >= len(replay_record['trace']):
                reason = 'end_of_record'; break
            assert int(info['raw_num_options'][0]) <= 128, 'Action set truncated'
            assert len(s['actions']) == int(info['num_options'][0])
            v.check_observation(obs, info, s)
            snap = v.compact(s)
            if replay_record:
                rec = replay_record['trace'][step]
                idx = rec['selected']['index']
                diff = A.compare_step(rec, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')) if idx < len(s['actions']) else {})
                if diff:
                    mismatch = dict(step=step, keys=diff); break
                why = rec.get('why')
            elif s['to_play'] == 0:
                try:
                    idx = teacher(s)
                except T.Unresolved as exc:
                    failure = str(exc)
                    blocking = dict(step=step, message=s['message'], turn=s['turn_count'], phase=s['phase'], plan_head=teacher.base.head(),
                                    legal=[A.meaning(dict(x, name=v.names.get(x['code'], ''))) for x in s['actions']], board=board(s))
                    break
                why = dict(teacher.base.log[-1]['why'], adapter_changed=idx != teacher.base.log[-1]['index'])
            else:
                idx = v.passive(s)
                why = dict(kind='opponent-passive')
            if s['to_play'] == 0:
                observed.append(dict(obs=observation(obs, info), label=idx, step=step, turn=s['turn_count'], n=len(s['actions'])))
                if s['turn_player'] == 1:
                    opponent_turn_offers.append(dict(step=step, phase=s['phase'], message=s['message'], chain=s['public_chain'],
                                                     offered=[A.meaning(dict(x, name=v.names.get(x['code'], ''))) for x in s['actions']],
                                                     chosen=idx))
            rows.append(dict(step=step, snapshot=snap, selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), why=why))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                game_over = dict(step=step, reward=float(np.asarray(reward).reshape(-1)[0]), terminated=bool(term[0]), truncated=bool(trunc[0]),
                                 win_reason=int(np.asarray(info.get('win_reason', [-1])).reshape(-1)[0]) if 'win_reason' in info else None)
                reason = 'game_over'; break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua_errors = v.native.get_lua_error_count()
    E.close(env)
    del env
    gc.collect()
    assert v.native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, observed=observed, opponent_turn_offers=opponent_turn_offers, failure=failure, mismatch=mismatch,
                game_over=game_over, blocking=blocking, checkpoints=checkpoints, reason=reason, final=final, lua_errors=lua_errors,
                teacher=None if teacher is None else dict(remaining_required=teacher.base.remaining_required(), remaining=teacher.base.steps,
                                                          pending=teacher.base.pending, skipped_optional=teacher.base.skipped_optional,
                                                          marks=teacher.base.marks, log=teacher.base.log,
                                                          unmatched_chain_heads=teacher.base.unmatched_chain_heads,
                                                          position_events=teacher.positions.events))


def run_demo(E, ledger, command, cond):
    d = W / 'demos' / cond
    record_path, obs_path = d / 'record.json', d / 'observations.npz'
    record = json.loads(record_path.read_text()) if command == 'replay' else None
    if command == 'record' and record_path.exists():
        raise RuntimeError('record exists; a condition is recorded once')
    if command == 'replay' and (d / 'replay.json').exists():
        raise RuntimeError('replay exists')
    run = ledger.start(f'{command} {cond}', cond, now())
    try:
        got = play(E, cond, replay_record=record)
        A = analysis()
        if command == 'record':
            if got['observed']:
                save_observations(obs_path, got['observed'])
            t = got['teacher']
            out = dict(condition=cond, engine_run=run, **{k: T.CONDITIONS[cond][k] for k in ['label', 'reference', 'reference_title', 'article_headings', 'hand', 'seed']},
                       plan=plan_json(cond)['plan'], failure=got['failure'], blocking=got['blocking'], reason=got['reason'], game_over=got['game_over'],
                       lua_errors=got['lua_errors'], script_completed=got['failure'] is None and not t['remaining_required'],
                       remaining_required=t['remaining_required'], remaining=t['remaining'], pending=t['pending'],
                       skipped_optional=t['skipped_optional'], marks=t['marks'],
                       unmatched_chain_heads=t['unmatched_chain_heads'], position_events=t['position_events'],
                       checkpoints=got['checkpoints'], opponent_turn_offers=got['opponent_turn_offers'], final=got['final'], trace=got['rows'],
                       observations=dict(path=str(obs_path), sha256=sha(obs_path) if got['observed'] else None, decisions=len(got['observed']),
                                         decisions_with_choice=sum(1 for r in got['observed'] if r['n'] > 1),
                                         own_turn=sum(1 for r in got['observed'] if r['turn'] == 1),
                                         opponent_turn=sum(1 for r in got['observed'] if r['turn'] == 2)))
            write(record_path, out)
            print('RECORD', json.dumps(dict(run=run, condition=cond, script_completed=out['script_completed'], failure=out['failure'], reason=out['reason'],
                                            remaining_required=out['remaining_required'][:3], decisions=out['observations']['decisions'],
                                            own_turn_end=got['checkpoints'].get('own_turn_end'), lua_errors=out['lua_errors']), ensure_ascii=False), flush=True)
            status = 'ok' if out['script_completed'] else 'completed-with-findings'
        else:
            out = dict(condition=cond, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                       game_over=got['game_over'], reproduced_steps=len(got['rows']), of=len(record['trace']), reason=got['reason'],
                       final_equal=None if record['final'] is None else A.normalized(got['final']) == A.normalized(record['final']),
                       checkpoints_equal=A.normalized(got['checkpoints']) == A.normalized(record['checkpoints']),
                       observations_match=same_observations(obs_path, got['observed']) if obs_path.exists() else not got['observed'],
                       lua_errors=got['lua_errors'])
            write(d / 'replay.json', out)
            print('REPLAY', json.dumps(out, ensure_ascii=False), flush=True)
            status = 'ok' if not got['mismatch'] and not got['failure'] else 'completed-with-findings'
        ledger.finish(run, status, now(), failure=got['failure'], mismatch=got['mismatch'])
    except BaseException as exc:
        ledger.finish(run, 'error', now(), error=f'{type(exc).__name__}: {exc}')
        raise


def job(command, cond):
    reg = check_registration()
    A = analysis()
    ledger = A.Ledger(LEDGER, limit=reg['engine_run_cap'])
    remaining = reg['engine_run_cap'] - len(ledger.started())
    need = len(REGRESSION_CASES) if command == 'regression' else 1
    if remaining < need:
        raise RuntimeError(f'not enough engine runs left: {remaining} < {need}')
    import artmage_entry as E
    print('PINS', json.dumps({k: v[:12] for k, v in E.verify_pins().items()}), flush=True)
    if command == 'regression':
        run_regression(E, A, ledger)
    else:
        run_demo(E, ledger, command, cond)
    print('LEDGER', len(ledger.started()), '/', reg['engine_run_cap'], flush=True)


def launch(command, cond):
    check_registration()
    log = W / 'logs' / f'{command}{"-" + cond if cond else ""}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={LIMITS['job_memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={LIMITS['cpu_percent']}%", 'nice', '-n', str(LIMITS['nice']), 'timeout', str(LIMITS['job_timeout_seconds']),
           PYTHON, '-B', '-u', str(Path(__file__).resolve()), 'job', command] + ([cond] if cond else [])
    env = dict(os.environ, **LIMITS['env'])
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=RUNTIME / 'agent/scripts', env=env, stdout=handle, stderr=subprocess.STDOUT)
    keep = ('PINS', 'REGRESSION', 'RECORD', 'REPLAY', 'LEDGER', 'Traceback', 'AssertionError', 'RuntimeError', 'KeyError', 'TypeError', 'ValueError', 'IndexError')
    print(''.join(l for l in log.read_text().splitlines(True) if l.startswith(keep)), end='')
    print(f'exit={done.returncode} log={log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'launch', 'job'])
    parser.add_argument('command', nargs='?', choices=['regression', 'record', 'replay'])
    parser.add_argument('condition', nargs='?', choices=list(T.CONDITIONS))
    args = parser.parse_args()
    if args.mode == 'register':
        register()
    else:
        if args.command != 'regression' and not args.condition:
            parser.error('record/replay need a condition')
        (launch if args.mode == 'launch' else job)(args.command, args.condition)
