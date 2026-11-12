"""Functional comparison of the two 学都 branches (CPU, engine). Cap: 16 episodes in eval-runs.jsonl.

  flc_runs.py record NAME...   one condition: replay own turn-1 choices up to the branch BY MEANING (the model reads
                               every observation once), execute the registered Graphrea set target, then the model plays
                               freely to the end of own turn 3; player 1 follows the registered script
  flc_runs.py replay NAME...   same fixture/seed, recorded action indices; snapshots, action meanings, model inputs
                               and end state compared
NAME = BV1-<1cost|2cost>-<VARNISH|ASSAULT>-<OM|OI>.
Fixture: player 0 holds the registered BV1 hand, player 1 the registered opponent hand; deck order from the same
seed as before (player 0's deck order equals the earlier BV1 runs). No normal draws in fixture duels.
"""
import collections
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import traceback
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
sys.path.insert(0, str(ROOT / 'code'))
import hb_plans as HP  # noqa: E402

LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 16
STEP_CAP = 600
ANALYSIS = P / 'experiments/capability-probe-20261109/code/analysis.py'
MODEL = (ROOT / 'models/candidate/base.flax_model', ROOT / 'models/candidate/chain-head.flax_model')
MODEL_SHA = ('b9c7751f5c1f7179faa4911e1a52e8603e332fd9700a0f301dd4e759f8208f66', '77191d5d9a9aaa65d63902c93e47c0e3274009e7a91ee1e8e8abb48fcd73c6a2')
MEDIUS, INFINITE, ASH, USAGI, WARASHI, BEAST = HP.MEDIUS, HP.INFINITE, HP.ASH, HP.USAGI, HP.WARASHI, HP.BEAST
MAIDEN, DIAKTOROS, FINEMELT, SHI, GRAPH = HP.MAIDEN, HP.DIAKTOROS, HP.FINEMELT, 31822037, HP.GRAPH
VARNISH, ASSAULT, AWAKENING, IMPASTO = HP.VARNISH, HP.ASSAULT, HP.AWAKENING, 44654994
OPPONENT_HAND = [MEDIUS, INFINITE, ASH, USAGI, WARASHI]
HANDS = {'1cost': 'BV1-B-1cost', '2cost': 'BV1-B-2cost'}
BRANCH = {'VARNISH': VARNISH, 'ASSAULT': ASSAULT}
TARGET_PRIORITY = [MAIDEN, DIAKTOROS, FINEMELT, SHI]
JST = timezone(timedelta(hours=9))


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
    spec = importlib.util.spec_from_file_location('capability_analysis_for_flc', ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(name):
    _, cost, branch, opp = name.split('-')
    return dict(hand_name=HANDS[cost], cost=cost, branch=branch, opp=opp)


def conditions():
    return [f'BV1-{c}-{b}-{o}' for c in ['1cost', '2cost'] for b in ['VARNISH', 'ASSAULT'] for o in ['OM', 'OI']]


def fixture(deck, hand0, hand1, seed):
    """article_witness.fixture with a separate hand per player (same RNG use: player 0 first, then player 1)."""
    rows, rng = [], random.Random(seed)
    for player, hand in [(0, hand0), (1, hand1)]:
        remaining = list(deck['main'])
        for i, code in enumerate(hand):
            remaining.remove(code)
            rows.append([code, player, player, 2, i, 8])
        rng.shuffle(remaining)
        rows.extend([code, player, player, 1, i, 8] for i, code in enumerate(remaining))
        rows.extend([code, player, player, 64, i, 8] for i, code in enumerate(deck['extra']))
    assert collections.Counter((x[1], x[0]) for x in rows) == collections.Counter((p, c) for p in [0, 1] for c in deck['main'] + deck['extra'])
    return rows


def meaning_key(action):
    return tuple(action.get(k) for k in ['act', 'code', 'effect', 'spec', 'place', 'position', 'phase', 'finish'])


def is_branch_prompt(s):
    offered = {a['code'] for a in s['actions']}
    return (s['to_play'] == 0 and s['turn_count'] == 1 and s['message'] == 'select_card' and {VARNISH, ASSAULT} <= offered
            and not offered & {AWAKENING, IMPASTO} and any(c['code'] == GRAPH for c in s['public_chain']))


def prefix_of(hand_name):
    """Own turn-1 decisions of the candidate's earlier free run on this hand, up to (excluding) the Graphrea set prompt."""
    d = json.loads((HB / 'eval/free/candidate' / f'{hand_name}.json').read_text())
    out = []
    for r in d['trace']:
        s = r['snapshot']
        if s['to_play'] != 0:
            continue
        if is_branch_prompt(s):
            return out, dict(source=str(HB / 'eval/free/candidate' / f'{hand_name}.json'), source_sha256=sha(HB / 'eval/free/candidate' / f'{hand_name}.json'),
                             branch_step_in_source=r['step'], source_choice=r['selected']['code'])
        out.append(dict(message=s['message'], meaning=meaning_key(r['selected']), name=r['selected'].get('name')))
    raise RuntimeError('branch prompt not found in the source record')


class Opponent:
    """Player 1 script. OM: in its own main phase, normal summon Medius, activate Medius① and special summon 獄神獣
    from the deck; otherwise pass. OI: in its own main phase with an empty field, activate 無限泡影 from the hand on the
    first legal target by TARGET_PRIORITY (same name: lowest zone); otherwise pass. Hand traps are never used."""

    def __init__(self, mode, v):
        self.mode, self.v = mode, v
        self.events = []
        self.state = dict(summoned=False, activated=False, picked=False, option=False, target=False)

    def log(self, s, what, **kw):
        self.events.append(dict(step=s.get('_step'), turn=s['turn_count'], phase=s['phase'], message=s['message'], what=what, **kw))

    def passive(self, s):
        try:
            return self.v.passive(s), False
        except RuntimeError:
            return s['actions'][0]['index'], True

    def __call__(self, s):
        acts, msg, st = s['actions'], s['message'], self.state
        own_turn = s['turn_player'] == 1 and s['turn_count'] == 2
        if own_turn and self.mode == 'OM':
            if msg == 'select_idlecmd' and s['phase'] == 4 and not st['summoned']:
                idx = next((a['index'] for a in acts if a['act'] == 'Summon' and a['code'] == MEDIUS), None)
                self.log(s, 'Medius normal summon ' + ('offered' if idx is not None else 'NOT offered'), legal=[(a['act'], a['code']) for a in acts])
                if idx is not None:
                    st['summoned'] = True
                    return idx, 'script'
            if st['summoned'] and not st['activated'] and msg in ('select_effectyn', 'select_chain'):
                idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == MEDIUS), None)
                self.log(s, 'Medius① ' + ('offered' if idx is not None else 'NOT offered'), legal=[(a['act'], a['code'], a.get('effect')) for a in acts])
                if idx is not None:
                    st['activated'] = True
                    return idx, 'script'
            if st['activated'] and not st['picked'] and msg == 'select_card':
                idx = next((a['index'] for a in acts if a['code'] == BEAST), None)
                self.log(s, '獄神獣 ' + ('offered' if idx is not None else 'NOT offered'), legal=[a['code'] for a in acts])
                if idx is not None:
                    st['picked'] = True
                    return idx, 'script'
            if st['picked'] and not st['option'] and msg == 'select_option':
                idx = next((a['index'] for a in acts if a.get('effect') == 1152), None)
                self.log(s, 'special summon option ' + ('offered' if idx is not None else 'NOT offered'), legal=[a.get('effect') for a in acts])
                if idx is not None:
                    st['option'] = True
                    return idx, 'script'
            if msg in ('select_place', 'select_position') and st['summoned'] and not (st['option'] and msg == 'select_place' and False):
                if msg == 'select_position':
                    return next((a['index'] for a in acts if a['position'] == 1), acts[0]['index']), 'script-default'
                return acts[0]['index'], 'script-default'
        if own_turn and self.mode == 'OI':
            if msg == 'select_idlecmd' and s['phase'] == 4 and not st['activated']:
                idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == INFINITE), None)
                field = [c for c in s['cards'] if c['controller'] == 1 and c['location'] in (4, 8)]
                self.log(s, '無限泡影 from hand ' + ('offered' if idx is not None else 'NOT offered'), own_field=len(field),
                         legal=[(a['act'], a['code']) for a in acts])
                if idx is not None:
                    st['activated'] = True
                    return idx, 'script'
            if st['activated'] and not st['target'] and msg in ('select_card', 'select_unselect_card'):
                options = [a for a in acts if not a['finish']]
                pick = None
                for code in TARGET_PRIORITY:
                    same = sorted([a for a in options if a['code'] == code], key=lambda a: str(a.get('spec')))
                    if same:
                        pick = same[0]
                        break
                if pick is None and options:
                    pick = sorted(options, key=lambda a: str(a.get('spec')))[0]
                self.log(s, '無限泡影 target', legal=[(a['code'], a.get('spec')) for a in options], chosen=(pick or {}).get('code'))
                if pick is not None:
                    st['target'] = True
                    return pick['index'], 'script'
            if msg in ('select_place',) and st['activated'] and not st['target']:
                return acts[0]['index'], 'script-default'
        idx, fb = self.passive(s)
        return idx, 'fallback' if fb else 'pass'


def board(s):
    own = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc)
    up = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc and x['position'] & 5)
    return dict(turn=s['turn_count'], phase=s['phase'], lp0=s['lp0'], lp1=s['lp1'], monsters=own(0, 4), monsters_face_up=up(0, 4),
                spell_trap=own(0, 8), spell_trap_face_up=up(0, 8), hand=own(0, 2), graveyard=own(0, 16), banished=own(0, 32),
                extra_face_up=sorted(x['code'] for x in s['cards'] if x['controller'] == 0 and x['location'] == 64 and x['position'] & 5),
                opponent_monsters=own(1, 4), opponent_spell_trap=own(1, 8), opponent_graveyard=own(1, 16), opponent_banished=own(1, 32),
                opponent_hand_count=len(own(1, 2)))


def check_observation(v, obs, info, s, audit, step, previous):
    import numpy as np
    actions = np.asarray(obs['actions_'])[0]
    for action in s['actions']:
        row = actions[action['index']]
        assert v.ID_CODES.get(int(row[1]) * 256 + int(row[2]), 0) == action['code'], ('Action identity mismatch', s['message'], action)
    assert int(info['public_chain_length'][0]) == len(s['public_chain'])
    cards = np.asarray(obs['cards_'])[0]
    action_ids = {int(x[1]) * 256 + int(x[2]) for x in actions[actions[:, 3] != 0]}
    hidden = (cards[:, 4] == 1) & np.isin(cards[:, 2], [1, 2])
    idx = np.where(hidden & ((cards[:, 0] > 0) | (cards[:, 1] > 0)))[0]
    if len(idx):
        shown = [dict(row=int(i), location='deck' if int(cards[i, 2]) == 1 else 'hand', card_id=int(cards[i, 0]) * 256 + int(cards[i, 1]),
                      code=v.ID_CODES.get(int(cards[i, 0]) * 256 + int(cards[i, 1]), 0),
                      origin='selection_prompt' if int(cards[i, 0]) * 256 + int(cards[i, 1]) in action_ids else 'unverified') for i in idx]
        reason = 'visible before any action' if step == 0 else f'{len(shown)} visible at once' if len(shown) > 15 else None
        audit.append(dict(step=step, turn=s['turn_count'], to_play=s['to_play'], message=s['message'], previous_action=previous, cards=shown, stopped=reason))
        if reason:
            raise RuntimeError('clear exposure: ' + reason)


def observation(obs, info):
    import numpy as np
    length, owner = int(info['public_chain_length'][0]), int(info['public_chain_player'][0])
    out = {k: np.asarray(obs[k])[0].copy() for k in ['cards_', 'global_', 'actions_', 'h_actions_']}
    out['chain'] = np.array([length, 0 if not length else 1 if owner == 0 else 2, int(info['public_chain_card_id'][0]), int(info['public_chain_effect'][0])], np.int32)
    return out


def play(E, A, name, record=None):
    v, np = E.v, E.np
    c = parse(name)
    spec = HP.HANDS[c['hand_name']]
    deck = v.deck_cards('Artmage')
    rows_fx = fixture(deck, spec['hand'], OPPONENT_HAND, spec['seed'])
    case = dict(case_id=name, hand=spec['hand'], seed=spec['seed'])
    env, obs, info = E.env_for(case, rows=rows_fx)
    policy = None if record else E.create_policy(MODEL[0], MODEL[1], positions=True, expected=MODEL_SHA)
    if policy:
        policy.reset()
    opponent = Opponent(c['opp'], v)
    prefix, source = prefix_of(c['hand_name'])
    pos, branched, branch_info = 0, False, None
    rows, observed, audit, checkpoints, prefix_log = [], [], [], {}, []
    reason, game_over, failure, mismatch = 'step_cap', None, None, None
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            s['_step'] = step
            for t, label in [(2, 'own_turn1_end'), (3, 'opp_turn2_end'), (4, 'own_turn3_end')]:
                if s['turn_count'] >= t and label not in checkpoints:
                    checkpoints[label] = board(s)
            if s['turn_count'] >= 4:
                reason = 'own_turn3_end'; break
            if record and step >= len(record['trace']):
                reason = 'end_of_record'; break
            assert int(info['raw_num_options'][0]) <= 128
            check_observation(v, obs, info, s, audit, step, rows[-1]['selected'] if rows else None)
            snap = v.compact({k: x for k, x in s.items() if k != '_step'})
            if record:
                rec = record['trace'][step]
                idx = rec['selected']['index']
                diff = A.compare_step(rec, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')) if idx < len(s['actions']) else {})
                if diff:
                    mismatch = dict(step=step, keys=diff); break
                who = rec['who']
            elif s['to_play'] == 0:
                model_idx = int(policy.act(obs, info))
                if not branched and pos < len(prefix):
                    want = prefix[pos]
                    if s['message'] != want['message']:
                        raise RuntimeError(f"prefix message differs at own decision {pos}: {s['message']} != {want['message']}")
                    idx = next((a['index'] for a in s['actions'] if meaning_key(a) == tuple(want['meaning'])), None)
                    if idx is None:
                        raise RuntimeError(f'prefix action not legal at own decision {pos}: {want}')
                    prefix_log.append(dict(step=step, model_agrees=model_idx == idx))
                    pos += 1
                    who = 'prefix-replay'
                elif not branched:
                    if not is_branch_prompt(s):
                        raise RuntimeError(f"expected the Graphrea set prompt at step {step}, got {s['message']}")
                    idx = next(a['index'] for a in s['actions'] if a['code'] == BRANCH[c['branch']])
                    branch_info = dict(step=step, executed=v.names.get(BRANCH[c['branch']]), model_choice=v.names.get(s['actions'][model_idx]['code']),
                                       offered=[v.names.get(a['code'], a['code']) for a in s['actions']])
                    branched = True
                    who = 'branch-control'
                else:
                    idx, who = model_idx, 'model'
            else:
                idx, who = opponent(s)
                who = 'opponent-' + who
            if s['to_play'] == 0:
                observed.append(dict(obs=observation(obs, info), label=idx, step=step, turn=s['turn_count']))
            rows.append(dict(step=step, snapshot=snap, selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), who=who))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                r = float(np.asarray(reward).reshape(-1)[0])
                game_over = dict(step=step, reward=r, acting_player=s['to_play'], winner=(s['to_play'] if r > 0 else 1 - s['to_play']) if r else None, turn=s['turn_count'])
                reason = 'game_over'; break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua = v.native.get_lua_error_count()
    E.close(env)
    return dict(name=name, **c, hand=spec['hand'], seed=spec['seed'], opponent_hand=OPPONENT_HAND, model_sha256=list(MODEL_SHA),
                prefix_source=source, prefix_length=len(prefix), prefix_replayed=pos, prefix_log=prefix_log, branch=branch_info,
                reason=reason, failure=failure, mismatch=mismatch, game_over=game_over, checkpoints=checkpoints, final=final,
                opponent_events=opponent.events, position_events=copy.deepcopy(policy.positions.events) if policy else None,
                lua_errors=lua, disclosed=audit, trace=rows, draws='none (fixture duel: draw count 0)'), observed


def save_observations(path, rows):
    import numpy as np
    arrays = {k: np.stack([r['obs'][k] for r in rows]) for k in rows[0]['obs']}
    arrays.update(labels=np.array([r['label'] for r in rows], np.int32), steps=np.array([r['step'] for r in rows], np.int32),
                  turns=np.array([r['turn'] for r in rows], np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
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


def job(command, names):
    reg = json.loads((ROOT / 'registration.json').read_text())
    for n, digest in reg['code'].items():
        if sha(ROOT / 'code' / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    A = analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    E.verify_pins()
    if len(ledger.started()) + len(names) > LIMIT:
        raise RuntimeError('engine episode cap would be exceeded')
    for name in names:
        d = ROOT / 'runs' / name
        if command == 'record' and (d / 'record.json').exists():
            raise RuntimeError(f'{name} recorded already')
        if command == 'replay' and (d / 'replay.json').exists():
            raise RuntimeError(f'{name} replayed already')
        run = ledger.start(f'{command} {name}', name, now())
        try:
            if command == 'record':
                out, observed = play(E, A, name)
                if observed:
                    save_observations(d / 'observations.npz', observed)
                out['engine_run'] = run
                out['observations'] = dict(sha256=sha(d / 'observations.npz') if observed else None, decisions=len(observed))
                write(d / 'record.json', out)
                cp = out['checkpoints']
                print('RECORD', json.dumps(dict(name=name, failure=out['failure'], reason=out['reason'], prefix=f"{out['prefix_replayed']}/{out['prefix_length']}",
                                                prefix_model_agrees=sum(x['model_agrees'] for x in out['prefix_log']), branch=out['branch'] and out['branch']['executed'],
                                                model_would=out['branch'] and out['branch']['model_choice'],
                                                t1=(cp.get('own_turn1_end') or {}).get('monsters'), t2_opp=(cp.get('opp_turn2_end') or {}).get('opponent_monsters'),
                                                lp1_t3=(cp.get('own_turn3_end') or {}).get('lp1'), game_over=out['game_over'],
                                                opp=[e['what'] for e in out['opponent_events']], visible=len(out['disclosed']), lua=out['lua_errors']),
                                           ensure_ascii=False), flush=True)
                status = 'ok' if not out['failure'] else 'completed-with-findings'
            else:
                record = json.loads((d / 'record.json').read_text())
                got, observed = play(E, A, name, record=record)
                rep = dict(name=name, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                           reproduced_steps=len(got['trace']), of=len(record['trace']), reason=got['reason'],
                           final_equal=None if record['final'] is None else A.normalized(got['final']) == A.normalized(record['final']),
                           checkpoints_equal=A.normalized(got['checkpoints']) == A.normalized(record['checkpoints']),
                           game_over_equal=got['game_over'] == record['game_over'],
                           observations_match=same_observations(d / 'observations.npz', observed) if (d / 'observations.npz').exists() else not observed,
                           lua_errors=got['lua_errors'])
                write(d / 'replay.json', rep)
                print('REPLAY', json.dumps({k: rep[k] for k in ['name', 'mismatch', 'failure', 'reproduced_steps', 'of', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']}), flush=True)
                status = 'ok' if not rep['mismatch'] and not rep['failure'] else 'completed-with-findings'
            ledger.finish(run, status, now())
        except BaseException as exc:
            ledger.finish(run, 'error', now(), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            gc.collect()
            assert E.v.native.get_env_resource_stats()['live_duels'] == 0
    print('LEDGER', len(ledger.started()), '/', LIMIT, flush=True)


if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2:])
