"""Evaluation of the RL candidate (CPU). Engine cap: 141 new episodes in eval-runs.jsonl (128 battles + 13 retention);
failures count, nothing is retried.

  rlt_eval.py retain NAME ...   retention: hb_runs.rt1 (Artmage-g06..g11) / hb_runs.free_run (F0-V..F6-V), frozen copy
                                88e69412 with its output root rebound to this study; fixture duels (no normal draws)
  rlt_eval.py battles           128 natural duels candidate vs start: 64 registered seeds x candidate seat 0 / 1,
                                normal dealing and draws, both sides argmax + the common position adapter, no learning.

Battles: each side is its own isolated-entry policy (explicit base/head hashes), with its own RNN and adapter, reset
every game. Per game: result, end classification (scripts/evaluate.py rules + the spec's pair exclusion), decisions
and inference time per side, every decision (actor, message, executed action meaning, public chain) and the public
board at each turn start (LP, hand/deck/extra counts, face-up field / graveyard / face-up banished codes; face-down
field cards without identity). The native snapshot is read for these logs only; the models see the env observation.
"""
import collections
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

T_PROCESS = time.time()
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
import artmage_teacher  # noqa: E402,F401  (cached from THIS root before hb_runs inserts the hand-branch root)
import batch_plans  # noqa: E402,F401
import hb_plans  # noqa: E402,F401
import hb_runs as HR  # noqa: E402
_hb_code = str(Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110/code'))
while _hb_code in sys.path:
    sys.path.remove(_hb_code)
for _m in (HR, artmage_teacher, batch_plans, hb_plans):
    assert Path(_m.__file__).resolve().parent == CODE, _m.__file__
import rlt_train as RT  # noqa: E402
from rlt_stats import classify_game, evaluate_module, pair_summary  # noqa: E402,F401

FROZEN = {'hb_runs.py': '88e694125079d2a9687490f1120186b536939b8750f731266efae56b167a0c14'}
LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 141
RETAIN = [f'Artmage-g{i:02d}-p0' for i in range(6, 12)] + [f'F{i}-V' for i in range(7)]
BATTLE_WALL = 6600  # seconds since process start: no new game after this (job killed at 7,000 s)
BATTLES = ROOT / 'eval/battles'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate():
    result = json.loads((ROOT / 'train/result.json').read_text())
    if result.get('status') != 'complete':
        raise RuntimeError('no complete candidate')
    pair = (ROOT / 'train/candidate/base.flax_model', ROOT / 'train/candidate/chain-head.flax_model')
    want = (result['saved']['candidate_base_sha256'], result['saved']['candidate_head_sha256'])
    if (sha(pair[0]), sha(pair[1])) != want:
        raise RuntimeError('candidate files differ from the training result')
    return pair, want


def bind():
    for n, digest in FROZEN.items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'frozen module changed: {n}')
    pair, want = candidate()
    HR.ROOT = ROOT
    assert HR.models('candidate') == (pair, want)
    return pair, want


# ------------------------------------------------------------------ public logs
def public_board(s):
    cards = s['cards']
    count = lambda c, loc: sum(1 for x in cards if x['controller'] == c and x['location'] == loc)
    out = dict(turn=s['turn_count'], turn_player=s['turn_player'], phase=s['phase'], lp=[s['lp0'], s['lp1']])
    for c in (0, 1):
        field = []
        for x in cards:
            if x['controller'] == c and x['location'] in (4, 8):
                up = bool(x['position'] & 5)
                field.append(dict(location=x['location'], sequence=x['sequence'], face_up=up, code=x['code'] if up else None,
                                  position=x['position']))
        out[f'p{c}'] = dict(hand=count(c, 2), deck=count(c, 1), extra=count(c, 64), field=sorted(field, key=lambda r: (r['location'], r['sequence'])),
                            graveyard=sorted(x['code'] for x in cards if x['controller'] == c and x['location'] == 16),
                            banished_face_up=sorted(x['code'] for x in cards if x['controller'] == c and x['location'] == 32 and x['position'] & 5),
                            banished_face_down=sum(1 for x in cards if x['controller'] == c and x['location'] == 32 and not x['position'] & 5),
                            extra_face_up=sorted(x['code'] for x in cards if x['controller'] == c and x['location'] == 64 and x['position'] & 5))
    return out


def check(v, obs, info, s, audit, notes, step, previous):
    """hb_runs.check_observation rules, except that action-identity / public-chain mismatches and more than 128 raw
    options are recorded instead of stopping a full game (tokens or long games were never seen in the fixture runs).
    Opponent hand/deck identities: audited by origin; a clear exposure (visible before any action or more than 15 at
    once) stops the evaluation."""
    actions = np.asarray(obs['actions_'])[0]
    if int(info['raw_num_options'][0]) > 128:
        notes['raw_options_over_128'] += 1
    for action in s['actions']:
        if action['index'] >= 128:
            continue
        row = actions[action['index']]
        if v.ID_CODES.get(int(row[1]) * 256 + int(row[2]), 0) != action['code']:
            notes['action_identity_mismatch'] += 1
            examples = notes.setdefault('mismatch_examples', [])
            if len(examples) < 5:
                examples.append(dict(step=step, code=action['code'], act=action['act']))
    if int(info['public_chain_length'][0]) != len(s['public_chain']):
        notes['public_chain_length_mismatch'] += 1
    cards = np.asarray(obs['cards_'])[0]
    action_ids = {int(x[1]) * 256 + int(x[2]) for x in actions[actions[:, 3] != 0]}
    hidden = (cards[:, 4] == 1) & np.isin(cards[:, 2], [1, 2])
    idx = np.where(hidden & ((cards[:, 0] > 0) | (cards[:, 1] > 0)))[0]
    if len(idx):
        shown = []
        for i in idx:
            cid = int(cards[i, 0]) * 256 + int(cards[i, 1])
            shown.append(dict(row=int(i), location='deck' if int(cards[i, 2]) == 1 else 'hand', card_id=cid, code=v.ID_CODES.get(cid, 0),
                              origin='selection_prompt' if cid in action_ids else 'unverified'))
        reason = 'visible before any action' if step == 0 else f'{len(shown)} visible at once' if len(shown) > 15 else None
        audit.append(dict(step=step, turn=s['turn_count'], phase=s['phase'], to_play=s['to_play'], message=s['message'],
                          public_chain=s['public_chain'], previous_action=previous, cards=shown, stopped=reason))
        if reason:
            raise RuntimeError('clear exposure: ' + reason)


def action_meaning(a):
    return {k: a.get(k) for k in ('act', 'code', 'effect', 'spec', 'place', 'position', 'phase', 'finish')}


# ------------------------------------------------------------------ one battle
def battle(E, policies, seed, candidate_seat, pair):
    v = E.v
    env, obs, info = RT.make_env(E, seed)  # same construction as training
    roles = {candidate_seat: 'candidate', 1 - candidate_seat: 'start'}
    for p in policies.values():
        p.reset()
    trace, boards, audit = [], [], []
    infer_s, decisions, max_wall = collections.Counter(), collections.Counter(), 0.0
    last_turn, result, error, step = None, None, None, 0
    notes = collections.Counter()
    try:
        while True:
            s = v.native.benchmark_snapshot()
            if s['turn_count'] != last_turn:
                boards.append(dict(step=step, **public_board(s)))
                last_turn = s['turn_count']
            check(v, obs, info, s, audit, notes, step, trace[-1]['selected'] if trace else None)
            tp = int(info['to_play'][0])
            if tp != s['to_play']:
                raise RuntimeError('to_play differs between info and snapshot')
            role = roles[tp]
            t = time.time()
            idx = int(policies[role].act(obs, info))
            infer_s[role] += time.time() - t
            decisions[role] += 1
            if not 0 <= idx < int(info['num_options'][0]):
                raise RuntimeError('illegal action index')
            trace.append(dict(step=step, seat=tp, role=role, turn=s['turn_count'], phase=s['phase'], message=s['message'],
                              selected=action_meaning(s['actions'][idx]), public_chain=s['public_chain']))
            t = time.time()
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            max_wall = max(max_wall, time.time() - t)
            step += 1
            if term[0] or trunc[0]:
                tr = info.get('termination_reason')
                wr = info.get('win_reason')
                result = classify_game(float(np.asarray(reward).reshape(-1)[0]), term[0], trunc[0], None if tr is None else int(tr[0]),
                                       None if wr is None else int(wr[0]), tp, candidate_seat, max_wall)
                result['last_actor'] = role
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        E.close(env)
        del env
        gc.collect()
    positions = {role: dict(decisions=len(p.positions.events), changed=sum(1 for x in p.positions.events if x.get('changed'))) for role, p in policies.items()}
    return dict(pair=pair, seed=seed, candidate_seat=candidate_seat, roles={str(k): v_ for k, v_ in roles.items()}, deck=E.DECK_KEY,
                result=result, error=error, env_steps=step, notes=dict(notes), decisions=dict(decisions), inference_seconds=dict(infer_s), max_step_wall_seconds=max_wall,
                last_turn=last_turn, boards=boards, trace=trace, position_adapter=positions, disclosed=audit,
                lua_errors=v.native.get_lua_error_count())


def run_battles(E, A):
    pair, want = bind()
    seeds = json.loads((ROOT / 'data/seeds.json').read_text())['evaluation']
    policies = dict(candidate=E.create_policy(pair[0], pair[1], positions=True, expected=want),
                    start=E.create_policy(RT.START[0], RT.START[1], positions=True, expected=RT.START_SHA))
    meta = {r: p.policy_metadata for r, p in policies.items()}
    BATTLES.mkdir(parents=True, exist_ok=True)
    (BATTLES / 'policies.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1) + '\n')
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    done = {(json.loads(l)['pair'], json.loads(l)['candidate_seat']) for l in (BATTLES / 'games.jsonl').read_text().splitlines()} if (BATTLES / 'games.jsonl').exists() else set()
    started = {r['case_id'] for r in ledger.started()}
    for i, seed in enumerate(seeds):
        for cs in (0, 1):
            name = f'battle-{i:02d}-cand{cs}'
            if name in started or (i, cs) in done:
                continue
            if time.time() - T_PROCESS >= BATTLE_WALL:
                print('WALL', 'battle wall-time limit reached; no new game', flush=True)
                return
            run = ledger.start('battle', name, HR.now())
            g = None
            try:
                g = battle(E, policies, seed, cs, i)
                g['engine_run'] = run
                (BATTLES / 'games').mkdir(exist_ok=True)
                RT.write_json(BATTLES / 'games' / f'{name}.json', g)
                row = {k: g[k] for k in ['pair', 'seed', 'candidate_seat', 'result', 'error', 'env_steps', 'notes', 'decisions', 'inference_seconds',
                                         'max_step_wall_seconds', 'last_turn', 'position_adapter', 'lua_errors']}
                row.update(engine_run=run, visible=len(g['disclosed']), visible_unverified=sum(c['origin'] == 'unverified' for d in g['disclosed'] for c in d['cards']),
                           file_sha256=sha(BATTLES / 'games' / f'{name}.json'))
                RT.append(BATTLES / 'games.jsonl', row)
                status = 'ok' if not g['error'] else 'error'
                ledger.finish(run, status, HR.now(), **({'error': g['error']} if g['error'] else {}))
            except BaseException as exc:
                ledger.finish(run, 'error', HR.now(), error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                assert E.v.native.get_env_resource_stats()['live_duels'] == 0
            r = g['result'] or {}
            print('BATTLE', json.dumps(dict(name=name, seed=seed, outcome=r.get('outcome'), reason=r.get('reason_label'), steps=g['env_steps'],
                                            turn=g['last_turn'], error=g['error'], visible=len(g['disclosed']))), f'{time.time() - T_PROCESS:.0f}s', flush=True)
            if g['error'] and 'clear exposure' in g['error']:
                raise RuntimeError('clear exposure in a battle: evaluation not established')
    print('LEDGER', len(ledger.started()), '/', LIMIT, flush=True)


def run_retain(E, A, names):
    bind()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    for name in names:
        if name not in RETAIN:
            raise RuntimeError(f'{name} is not a registered retention case')
        if (ROOT / 'eval/retain/candidate' / f'{name}.json').exists():
            raise RuntimeError(f'{name} retained already')
    for name in names:
        run = ledger.start('retain', name, HR.now())
        try:
            if name.startswith('Artmage-g'):
                HR.rt1(E, A, 'candidate', name)
                status = 'ok'
            else:
                out = HR.free_run(E, A, 'candidate', name, out_dir='retain')
                status = 'ok' if not out['failure'] else 'completed-with-findings'
            ledger.finish(run, status, HR.now())
        except BaseException as exc:
            ledger.finish(run, 'error', HR.now(), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            gc.collect()
            assert E.v.native.get_env_resource_stats()['live_duels'] == 0
    print('LEDGER', len(ledger.started()), '/', LIMIT, flush=True)


def job(command, names):
    RT.registered()
    import artmage_entry as E
    assert Path(E.__file__).resolve().parent == CODE
    E.verify_pins()
    A = HR.analysis()
    evaluate_module()
    if command == 'retain':
        run_retain(E, A, names)
    elif command == 'battles':
        run_battles(E, A)
    else:
        raise SystemExit('retain NAME... | battles')


if __name__ == '__main__':
    try:
        job(sys.argv[1], sys.argv[2:])
    except BaseException:
        traceback.print_exc()
        raise
