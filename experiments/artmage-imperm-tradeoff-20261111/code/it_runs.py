"""Imperm / Diaktoros② trade-off: scripted functional runs (CPU, engine; no model, no training).
Cap: 14 engine episodes in eval-runs.jsonl (failures count).

  it_runs.py record NAME...   it_runs.py replay NAME...
NAME:
  A-I-N, A-I-P, A-IM-N, A-IM-P   start A: the functional-comparison record BV1-1cost-VARNISH-OI replayed BY MEANING (snapshot equal at
                                 every step, the opponent script re-run and checked) up to OUR first response to 無限泡影 (whose
                                 target is the Maiden in zone 0 that already used its ② this turn). Then N = Diaktoros② (27184601,
                                 10011) / P = pass. I: the opponent does nothing more. IM: after the Imperm chain resolved the
                                 opponent normal summons Medius, activates ① and special summons 獄神獣. At our Medius① window:
                                 Diaktoros② once if legal, else pass. Stop at the first decision of turn 3.
  B-N, B-P                       start B: the record BV1-2cost-ASSAULT-OI replayed for turn 1 only. From turn 2 every optional own
                                 activation is declined in every phase until the opponent's 無限泡影 (target: the turn-1 Maiden, by the
                                 same priority rule). N / P as above. After the chain: the ORIGINAL Maiden's ② once if offered
                                 (fusion target: EX Maiden; materials Finemelt + Litera); stop at the first decision after that chain.
Own rule otherwise: decline (legal Cancel) on select_chain / select_effectyn / select_yesno; other prompts are forced: select_place first,
select_position face-up attack else first, anything else the first option (logged). Opponent: pass except the script; no attack.
No normal draws (fixture duel); effect draws occur.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
import artmage_teacher  # noqa: E402,F401
import batch_plans  # noqa: E402,F401
import hb_plans  # noqa: E402,F401
import flc_runs as F  # noqa: E402
OLD = str(F.ROOT / 'code')
while OLD in sys.path:
    sys.path.remove(OLD)
for _m in (F, hb_plans):
    assert Path(_m.__file__).resolve().parent == CODE

P = Path('/home/hatk8/projects/yugioh-ai')
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 14
STEP_CAP = 600
MEDIUS, INFINITE, BEAST, MAIDEN, DIAKTOROS, FINEMELT, LITERA = 97556336, 10045474, 23829452, 74631897, 27184601, 34541940, 97434754
CONDITIONS = ['A-I-N', 'A-I-P', 'A-IM-N', 'A-IM-P', 'B-N', 'B-P']
START = {'A': dict(old='BV1-1cost-VARNISH-OI', hand='BV1-B-1cost'), 'B': dict(old='BV1-2cost-ASSAULT-OI', hand='BV1-B-2cost')}
PASS_MESSAGES = ('select_chain', 'select_effectyn', 'select_yesno')
TARGET_PRIORITY = [MAIDEN, DIAKTOROS, FINEMELT, 31822037]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(name):
    parts = name.split('-')
    if parts[0] == 'A':
        return dict(start='A', opp=parts[1], response=parts[2])
    return dict(start='B', opp='I', response=parts[1])


def chain_top(s):
    return s['public_chain'][-1] if s['public_chain'] else None


def is_imperm_window(s):
    t = chain_top(s)
    return s['to_play'] == 0 and s['message'] == 'select_chain' and t is not None and t['player'] == 1 and t['code'] == INFINITE


def is_medius_window(s):
    t = chain_top(s)
    return (s['to_play'] == 0 and s['message'] == 'select_chain' and len(s['public_chain']) == 1 and t['player'] == 1 and t['code'] == MEDIUS
            and t['description'] == MEDIUS * 16)


class Opponent:
    """O-I (+ O-M afterwards for IM). Same Imperm rule as the comparison's O-I script (priority, lowest zone)."""

    def __init__(self, mode, v):
        self.mode, self.v = mode, v
        self.st = dict(imperm=False, target=False, imperm_resolved=False, summoned=False, medius=False, picked=False, option=False)
        self.events = []

    def log(self, s, what, **kw):
        self.events.append(dict(step=s['_step'], turn=s['turn_count'], phase=s['phase'], message=s['message'], what=what, **kw))

    def __call__(self, s):
        acts, msg, st = s['actions'], s['message'], self.st
        mine = s['turn_player'] == 1 and s['turn_count'] == 2
        if st['imperm'] and not st['imperm_resolved'] and s['chain_size'] == 0:
            st['imperm_resolved'] = True
        if mine and msg == 'select_idlecmd' and s['phase'] == 4 and not st['imperm']:
            idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == INFINITE), None)
            self.log(s, '無限泡影 from hand ' + ('offered' if idx is not None else 'NOT offered'))
            if idx is not None:
                st['imperm'] = True
                return idx, 'script'
        if st['imperm'] and not st['target'] and msg in ('select_card', 'select_unselect_card'):
            options = [a for a in acts if not a['finish']]
            pick = None
            for code in TARGET_PRIORITY:
                same = sorted([a for a in options if a['code'] == code], key=lambda a: str(a.get('spec')))
                if same:
                    pick = same[0]; break
            if pick is None and options:
                pick = sorted(options, key=lambda a: str(a.get('spec')))[0]
            self.log(s, '無限泡影 target', legal=[(a['code'], a.get('spec')) for a in options], chosen=(pick or {}).get('code'), spec=(pick or {}).get('spec'))
            if pick is not None:
                st['target'] = True
                return pick['index'], 'script'
        if st['imperm'] and not st['target'] and msg == 'select_place':
            return acts[0]['index'], 'script-default'
        if self.mode == 'IM' and mine and st['imperm_resolved']:
            if msg == 'select_idlecmd' and s['phase'] == 4 and not st['summoned']:
                idx = next((a['index'] for a in acts if a['act'] == 'Summon' and a['code'] == MEDIUS), None)
                self.log(s, 'Medius normal summon ' + ('offered' if idx is not None else 'NOT offered'))
                if idx is not None:
                    st['summoned'] = True
                    return idx, 'script'
            if st['summoned'] and not st['medius'] and msg in ('select_effectyn', 'select_chain'):
                idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == MEDIUS), None)
                self.log(s, 'Medius① ' + ('offered' if idx is not None else 'NOT offered'))
                if idx is not None:
                    st['medius'] = True
                    return idx, 'script'
            if st['medius'] and not st['picked'] and msg == 'select_card':
                idx = next((a['index'] for a in acts if a['code'] == BEAST), None)
                self.log(s, '獄神獣 ' + ('offered' if idx is not None else 'NOT offered'))
                if idx is not None:
                    st['picked'] = True
                    return idx, 'script'
            if st['picked'] and not st['option'] and msg == 'select_option':
                idx = next((a['index'] for a in acts if a.get('effect') == 1152), None)
                self.log(s, 'special summon option ' + ('offered' if idx is not None else 'NOT offered'))
                if idx is not None:
                    st['option'] = True
                    return idx, 'script'
            if st['summoned'] and msg in ('select_place', 'select_position'):
                if msg == 'select_position':
                    return next((a['index'] for a in acts if a['position'] == 1), acts[0]['index']), 'script-default'
                return acts[0]['index'], 'script-default'
        try:
            return self.v.passive(s), 'pass'
        except RuntimeError:
            self.log(s, 'fallback: first option')
            return acts[0]['index'], 'fallback'


class Own:
    """Scripted own side after the replayed part."""

    def __init__(self, c):
        self.c = c
        self.st = dict(responded=False, medius_done=False, maiden_zone=None, fusion=False, fusion_target=False, materials=[], fusion_chain=False)
        self.log = []

    def note(self, s, rule, idx, **kw):
        a = next(x for x in s['actions'] if x['index'] == idx)
        self.log.append(dict(step=s['_step'], turn=s['turn_count'], phase=s['phase'], message=s['message'], rule=rule,
                             executed=dict(act=a['act'], code=a['code'], effect=a.get('effect'), spec=a.get('spec')), **kw))
        return idx

    def __call__(self, s):
        acts, msg, st = s['actions'], s['message'], self.st
        cancel = next((a['index'] for a in acts if a['act'] == 'Cancel'), None) if msg in PASS_MESSAGES else None
        legal = [(a['act'], a['code'], a.get('effect'), a.get('spec')) for a in acts]
        if not st['responded'] and is_imperm_window(s):
            st['responded'] = True
            if self.c['response'] == 'N':
                idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == DIAKTOROS and a.get('effect') == 10011), None)
                if idx is None:
                    return None, 'not established: Diaktoros② not legal at the Imperm window'
                return self.note(s, 'Imperm window: Diaktoros②', idx, legal=legal), 'rule'
            return self.note(s, 'Imperm window: pass', cancel, legal=legal), 'rule'
        if self.c['opp'] == 'IM' and not st['medius_done'] and is_medius_window(s):
            st['medius_done'] = True
            idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == DIAKTOROS and a.get('effect') == 10011), None)
            if idx is not None:
                return self.note(s, 'Medius① window: Diaktoros② (legal)', idx, legal=legal), 'rule'
            return self.note(s, 'Medius① window: pass (Diaktoros② not legal)', cancel, legal=legal), 'rule'
        if self.c['start'] == 'B' and st['responded'] and s['chain_size'] == 0 and not st['fusion'] and st['maiden_zone'] is not None and msg in PASS_MESSAGES:
            spec = f"m{st['maiden_zone'] + 1}"
            idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == MAIDEN and a.get('effect') == 10011 and a.get('spec') == spec), None)
            if idx is not None:
                st['fusion'] = True
                return self.note(s, 'original Maiden② (fusion)', idx, legal=legal), 'rule'
        if st['fusion'] and not st['fusion_target'] and msg == 'select_card' and any(a['code'] == MAIDEN and str(a.get('spec', '')).startswith('x') for a in acts):
            idx = next(a['index'] for a in acts if a['code'] == MAIDEN and str(a.get('spec', '')).startswith('x'))
            st['fusion_target'] = True
            return self.note(s, 'fusion target: EX Maiden', idx, legal=legal), 'rule'
        if st['fusion_target'] and msg in ('select_unselect_card', 'select_card') and len(st['materials']) < 2:
            want = [FINEMELT, LITERA][len(st['materials'])]
            idx = next((a['index'] for a in acts if a['code'] == want and not a['finish']), None)
            if idx is not None:
                st['materials'].append(want)
                return self.note(s, f'fusion material {len(st["materials"])}', idx, legal=legal), 'rule'
            return None, f'not established: material {want} not offered'
        if cancel is not None:
            return cancel, 'decline'
        if msg == 'select_place':
            return acts[0]['index'], 'forced-first'
        if msg == 'select_position':
            return next((a['index'] for a in acts if a['position'] == 1), acts[0]['index']), 'forced-position'
        if msg == 'select_unselect_card':
            fin = next((a['index'] for a in acts if a['finish']), None)
            if fin is not None:
                return fin, 'forced-finish'
        return self.note(s, 'forced: first option', acts[0]['index'], legal=legal), 'forced-first'


def play(E, A, name, record=None):
    v, np = E.v, E.np
    c = parse(name)
    base = START[c['start']]
    spec = F.HP.HANDS[base['hand']]
    old = json.loads((FLC / 'runs' / base['old'] / 'record.json').read_text())
    rows_fx = F.fixture(v.deck_cards('Artmage'), spec['hand'], F.OPPONENT_HAND, spec['seed'])
    env, obs, info = E.env_for(dict(case_id=name, hand=spec['hand'], seed=spec['seed']), rows=rows_fx)
    opponent, own = Opponent(c['opp'], v), Own(c)
    rows, observed, audit, checkpoints = [], [], [], {}
    reason, game_over, failure, mismatch, replaying = 'step_cap', None, None, None, True
    fusion_chain_seen = False
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            s['_step'] = step
            if c['start'] == 'A' and s['turn_count'] >= 3:
                checkpoints.setdefault('first_turn3_decision', dict(F.board(s), step=step)); reason = 'first turn-3 decision'; break
            if c['start'] == 'B' and own.st['fusion'] and s['chain_size'] >= 1:
                fusion_chain_seen = True
            if c['start'] == 'B' and own.st['fusion'] and fusion_chain_seen and s['chain_size'] == 0:
                checkpoints.setdefault('after_fusion_chain', dict(F.board(s), step=step)); reason = 'fusion chain resolved'; break
            if c['start'] == 'B' and s['turn_count'] >= 3:
                checkpoints.setdefault('first_turn3_decision', dict(F.board(s), step=step)); reason = 'turn 3 reached (Maiden② not used)'; break
            if record and step >= len(record['trace']):
                reason = 'end_of_record'; break
            # checkpoints from public snapshots (record and replay alike)
            if is_imperm_window(s) and 'imperm_window' not in checkpoints:
                checkpoints['imperm_window'] = dict(F.board(s), step=step)
            if opponent.st['imperm'] and opponent.st['target'] and s['chain_size'] == 0 and 'after_imperm_chain' not in checkpoints:
                checkpoints['after_imperm_chain'] = dict(F.board(s), step=step)
            if is_medius_window(s) and 'medius_window' not in checkpoints:
                checkpoints['medius_window'] = dict(F.board(s), step=step)
            if opponent.st['medius'] and s['chain_size'] == 0 and 'after_medius_chain' not in checkpoints:
                checkpoints['after_medius_chain'] = dict(F.board(s), step=step, beast_on_opponent_field=BEAST in F.board(s)['opponent_monsters'],
                                                         medius_on_opponent_field=MEDIUS in F.board(s)['opponent_monsters'])
            if s['turn_count'] >= 2 and 'own_turn1_end' not in checkpoints:
                checkpoints['own_turn1_end'] = F.board(s)
            assert int(info['raw_num_options'][0]) <= 128
            F.check_observation(v, obs, info, s, audit, step, rows[-1]['selected'] if rows else None)
            snap = v.compact({k: x for k, x in s.items() if k != '_step'})
            if record:
                rec = record['trace'][step]
                idx = rec['selected']['index']
                diff = A.compare_step(rec, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')) if idx < len(s['actions']) else {})
                if diff:
                    mismatch = dict(step=step, keys=diff); break
                who = rec['who']
                if s['to_play'] == 1:
                    o_idx, _ = opponent(s)
                    if o_idx != idx:
                        mismatch = dict(step=step, keys=['opponent script differs from the record']); break
                elif not who.startswith('replay'):
                    o_idx, _ = own(s)  # keep the own rule state in step; its choice must equal the record
                    if o_idx != idx:
                        mismatch = dict(step=step, keys=['own rule differs from the record']); break
            else:
                in_replay = replaying and ((c['start'] == 'A' and not is_imperm_window(s)) or (c['start'] == 'B' and s['turn_count'] == 1))
                if in_replay:
                    o = old['trace'][step]
                    if s['to_play'] == 0:
                        idx = next((a['index'] for a in s['actions'] if F.meaning_key(a) == F.meaning_key(o['selected'])), None)
                        if idx is None:
                            raise RuntimeError(f'old action not legal at step {step}')
                        who = 'replay-old'
                    else:
                        idx, _ = opponent(s)
                        who = 'opponent-replay-old'
                    diff = A.compare_step(o, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')))
                    if diff:
                        raise RuntimeError(f'path differs from the old record at step {step}: {diff}')
                else:
                    replaying = False
                    if s['to_play'] == 0:
                        idx, who = own(s)
                        if idx is None:
                            reason = who; break
                        who = 'own-' + who
                    else:
                        idx, who = opponent(s)
                        who = 'opponent-' + who
            # B: remember the original Maiden's zone at turn-1 end (the only Maiden then)
            if c['start'] == 'B' and s['turn_count'] == 2 and own.st['maiden_zone'] is None:
                z = [x['sequence'] for x in s['cards'] if x['controller'] == 0 and x['location'] == 4 and x['code'] == MAIDEN]
                if len(z) == 1:
                    own.st['maiden_zone'] = z[0]
            if s['to_play'] == 0:
                observed.append(dict(obs=F.observation(obs, info), label=idx, step=step, turn=s['turn_count']))
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
    return dict(name=name, start=c['start'], opponent_mode=c['opp'], response=c['response'], old_record=str(FLC / 'runs' / base['old'] / 'record.json'),
                old_record_sha256=sha(FLC / 'runs' / base['old'] / 'record.json'), hand=spec['hand'], seed=spec['seed'], opponent_hand=F.OPPONENT_HAND,
                maiden_zone_B=own.st['maiden_zone'], reason=reason, failure=failure, mismatch=mismatch, game_over=game_over, checkpoints=checkpoints,
                final=final, opponent_events=opponent.events, own_rule_log=own.log, lua_errors=lua, disclosed=audit, trace=rows,
                draws='no normal draws (fixture duel: draw count 0); effect draws occur', model='none (scripted functional test)'), observed


def job(command, names):
    latest = sorted(ROOT.glob('amendment-*.json'))
    reg = json.loads((latest[-1] if latest else ROOT / 'registration.json').read_text())
    for n, digest in reg['code'].items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    for n in names:
        if n not in reg['to_run']:
            raise RuntimeError(f'{n} is not registered to run')
    A = F.analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    assert Path(E.__file__).resolve().parent == CODE
    E.verify_pins()
    if len(ledger.started()) + len(names) > LIMIT:
        raise RuntimeError('engine episode cap would be exceeded')
    for name in names:
        d = ROOT / 'runs' / name
        if command == 'record' and (d / 'record.json').exists():
            raise RuntimeError(f'{name} recorded already')
        if command == 'replay' and (d / 'replay.json').exists():
            raise RuntimeError(f'{name} replayed already')
        run = ledger.start(f'{command} {name}', name, F.now())
        try:
            if command == 'record':
                out, observed = play(E, A, name)
                if observed:
                    F.save_observations(d / 'observations.npz', observed)
                out['engine_run'] = run
                out['observations'] = dict(sha256=sha(d / 'observations.npz') if observed else None, decisions=len(observed), note='saved, not used for training')
                F.write(d / 'record.json', out)
                print('RECORD', json.dumps(dict(name=name, reason=out['reason'], failure=out['failure'], opp=[e['what'] for e in out['opponent_events']],
                                                own=[(x['step'], x['rule']) for x in out['own_rule_log']], visible=len(out['disclosed']), lua=out['lua_errors']),
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
                           observations_match=F.same_observations(d / 'observations.npz', observed) if (d / 'observations.npz').exists() else not observed,
                           lua_errors=got['lua_errors'])
                F.write(d / 'replay.json', rep)
                print('REPLAY', json.dumps({k: rep[k] for k in ['name', 'mismatch', 'failure', 'reproduced_steps', 'of', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']}), flush=True)
                status = 'ok' if not rep['mismatch'] and not rep['failure'] else 'completed-with-findings'
            ledger.finish(run, status, F.now())
        except BaseException as exc:
            ledger.finish(run, 'error', F.now(), error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            gc.collect()
            assert E.v.native.get_env_resource_stats()['live_duels'] == 0
    print('LEDGER', len(ledger.started()), '/', LIMIT, flush=True)


if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2:])
