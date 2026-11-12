"""Unseen-response probe (CPU, engine; diagnosis only). Cap: 26 engine episodes in eval-runs.jsonl (failures count).

  up_runs.py record NAME...   up_runs.py replay NAME...
NAME = <H1|H2>-<ASH|GAMMA>-<BASE|CAND>  (policy comparison) or H1-<ASH|GAMMA>-CTRL-<G|PASS> (local functional control)
  H1 = BV1-B-1cost hand, H2 = BV1-B-2cost hand, seed 1112103; opponent hand: Ash Blossom, PSY-Framegear Gamma, Ghost Belle
  (幽鬼うさぎ), Ghost Sister (屋敷わらし), Medius; PSY-Frame Driver stays in the opponent deck. Fixture duels: no normal draws.
  Own side: the compared policy decides from the start of turn 1 (nothing forced). Opponent: passes until the TARGET =
  its first chance to respond to OUR first Medius① when that Medius① is chain link 1; there it activates Ash (ASH) or
  Gamma (GAMMA); later it chooses Driver for Gamma, face-up attack position (else the first option) and the first zone,
  and passes everything else. The run ends at the first decision after the target chain has fully resolved.
  BASE = the frozen base + chain head (2-file entry); CAND = the coverage candidate (3-file entry, adapter 00977941).
  CTRL (H1 only): the base policy before the target; at OUR first response after the opponent's activation exactly one
  substitution (G activation or pass; not established if illegal -> the run stops there); after it every optional own
  activation is declined (legal Cancel), forced choices are the base policy's.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-unseen-response-probe-20261111')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
import artmage_teacher  # noqa: E402,F401
import batch_plans  # noqa: E402,F401
import hb_plans  # noqa: E402,F401
import sa_model as SM  # noqa: E402  (isolated entry from THIS root)
import flc_runs as F  # noqa: E402
OLD = str(F.ROOT / 'code')
while OLD in sys.path:
    sys.path.remove(OLD)
import artmage_entry as _entry  # noqa: E402
for _m in (F, SM, _entry, hb_plans):
    assert Path(_m.__file__).resolve().parent == CODE, _m.__file__

LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 26
STEP_CAP = 600
BASE = (ROOT / 'models/base/base.flax_model', ROOT / 'models/base/chain-head.flax_model')
CAND = ROOT / 'models/candidate'
MEDIUS, ASH, GAMMA, DRIVER, USAGI, WARASHI, G = 97556336, 14558127, 38814750, 49036338, 59438930, 73642296, 23434538
OPPONENT_HAND = [ASH, GAMMA, USAGI, WARASHI, MEDIUS]
HANDS = {'H1': F.HP.HANDS['BV1-B-1cost'], 'H2': F.HP.HANDS['BV1-B-2cost']}
COMPARISON = [f'{h}-{o}-{p}' for h in ['H1', 'H2'] for o in ['ASH', 'GAMMA'] for p in ['BASE', 'CAND']]
CONTROLS = [f'H1-{o}-CTRL-{c}' for o in ['ASH', 'GAMMA'] for c in ['G', 'PASS']]
PASS_MESSAGES = ('select_chain', 'select_effectyn', 'select_yesno')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(name):
    parts = name.split('-')
    h, o = parts[0], parts[1]
    if parts[2] == 'CTRL':
        return dict(hand=h, opp=o, policy='BASE', control=parts[3])
    return dict(hand=h, opp=o, policy=parts[2], control=None)


def own_medius1(link):
    return link['player'] == 0 and link['code'] == MEDIUS and link['description'] == MEDIUS * 16


class Opponent:
    def __init__(self, mode, v):
        self.mode, self.v = mode, v
        self.code = ASH if mode == 'ASH' else GAMMA
        self.state = dict(seen_medius=False, target=None, activated=False)
        self.events = []

    def log(self, s, what, **kw):
        self.events.append(dict(step=s['_step'], turn=s['turn_count'], phase=s['phase'], message=s['message'], what=what, **kw))

    def __call__(self, s):
        acts, st = s['actions'], self.state
        ch = s['public_chain']
        if not st['seen_medius'] and any(own_medius1(c) for c in ch):
            st['seen_medius'] = True
            if len(ch) == 1 and s['message'] == 'select_chain':
                idx = next((a['index'] for a in acts if a['act'] == 'Activate' and a['code'] == self.code), None)
                self.log(s, f'target: {self.mode} ' + ('activated' if idx is not None else 'NOT legal'), legal=[(a['act'], a['code'], a.get('effect')) for a in acts])
                st['target'] = 'established' if idx is not None else 'opponent effect not legal'
                if idx is not None:
                    st['activated'] = True
                    return idx, 'script'
            else:
                st['target'] = 'first own Medius① not at chain link 1 or no opponent chain window'
                self.log(s, st['target'], chain=[(c['player'], c['code']) for c in ch])
        if st['activated']:
            if s['message'] in ('select_card', 'select_unselect_card') and any(a['code'] == DRIVER for a in acts):
                idx = next(a['index'] for a in acts if a['code'] == DRIVER)
                self.log(s, 'Driver chosen')
                return idx, 'script'
            if s['message'] == 'select_position':
                return next((a['index'] for a in acts if a['position'] == 1), acts[0]['index']), 'script-default'
            if s['message'] == 'select_place':
                return acts[0]['index'], 'script-default'
        try:
            return self.v.passive(s), 'pass'
        except RuntimeError:
            self.log(s, 'fallback: first option', legal=[(a['act'], a['code']) for a in acts])
            return acts[0]['index'], 'fallback'


def make_policy(E, kind):
    if kind == 'BASE':
        pol = E.create_policy(BASE[0], BASE[1], positions=True, expected=SM.FROZEN_SHA)
        info = dict(type='2-file original entry', base=str(BASE[0]), base_sha256=sha(BASE[0]), head=str(BASE[1]), head_sha256=sha(BASE[1]))
        return pol, info
    pol = SM.create_adapter_policy(CAND, positions=True)
    return pol, dict(pol.policy_metadata)


def play(E, A, name, record=None):
    v, np = E.v, E.np
    c = parse(name)
    spec = HANDS[c['hand']]
    rows_fx = F.fixture(v.deck_cards('Artmage'), spec['hand'], OPPONENT_HAND, spec['seed'])
    env, obs, info = E.env_for(dict(case_id=name, hand=spec['hand'], seed=spec['seed']), rows=rows_fx)
    policy, pinfo = (None, None) if record else make_policy(E, c['policy'])
    if policy:
        policy.reset()
    opponent = Opponent(c['opp'], v)
    rows, observed, audit, checkpoints, control_log = [], [], [], {}, []
    reason, game_over, failure, mismatch = 'step_cap', None, None, None
    first_response_done = False
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            s['_step'] = step
            if opponent.state['activated'] and s['chain_size'] == 0 and 'target_chain_resolved' not in checkpoints:
                checkpoints['target_chain_resolved'] = dict(F.board(s), step=step)
                reason = 'target_chain_resolved'; break
            if opponent.state['target'] and not opponent.state['activated'] and s['chain_size'] == 0:
                reason = f"target not established: {opponent.state['target']}"; break
            if s['turn_count'] >= 2 and not opponent.state['seen_medius']:
                reason = 'target not reached in turn 1'; break
            if record and step >= len(record['trace']):
                reason = 'end_of_record'; break
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
                if s['to_play'] == 1:  # keep the opponent script state in step (its choice must equal the record)
                    opp_idx, _ = opponent(s)
                    if opp_idx != idx:
                        mismatch = dict(step=step, keys=['opponent script differs from the record']); break
            elif s['to_play'] == 0:
                model_idx = int(policy.act(obs, info))
                idx, who = model_idx, 'model'
                if c['control'] and opponent.state['activated']:
                    if not first_response_done:
                        first_response_done = True
                        if c['control'] == 'G':
                            sub = next((a['index'] for a in s['actions'] if a['act'] == 'Activate' and a['code'] == G), None)
                        else:
                            sub = next((a['index'] for a in s['actions'] if a['act'] == 'Cancel'), None) if s['message'] in PASS_MESSAGES else None
                        control_log.append(dict(step=step, rule=f"first response -> {c['control']}", message=s['message'],
                                                model_choice=dict(act=s['actions'][model_idx]['act'], code=s['actions'][model_idx]['code'], effect=s['actions'][model_idx].get('effect')),
                                                established=sub is not None))
                        if sub is None:
                            reason = f"control not established ({c['control']} not legal)"; break
                        idx, who = sub, f"control-{c['control']}"
                    else:
                        cancel = next((a['index'] for a in s['actions'] if a['act'] == 'Cancel'), None) if s['message'] in PASS_MESSAGES else None
                        control_log.append(dict(step=step, rule='decline optional' if cancel is not None else 'forced: base policy', message=s['message'],
                                                model_choice=dict(act=s['actions'][model_idx]['act'], code=s['actions'][model_idx]['code'], effect=s['actions'][model_idx].get('effect'))))
                        if cancel is not None:
                            idx, who = cancel, 'control-decline'
                        else:
                            who = 'control-forced-base'
            else:
                idx, who = opponent(s)
                who = 'opponent-' + who
            if s['to_play'] == 1 and opponent.state['activated'] and 'opponent_activation' not in checkpoints:
                checkpoints['opponent_activation'] = dict(F.board(s), step=step)  # the opponent's activation decision (record and replay alike)
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
    if policy is not None and c['policy'] == 'CAND':
        pinfo = dict(pinfo, decisions=policy.adapter_policy.calls, adapter_active=policy.adapter_policy.log)
    return dict(name=name, hand_name=c['hand'], hand=spec['hand'], seed=spec['seed'], opponent_hand=OPPONENT_HAND, opponent_mode=c['opp'], policy_kind=c['policy'],
                control=c['control'], policy=pinfo, target=opponent.state['target'], reason=reason, failure=failure, mismatch=mismatch, game_over=game_over,
                checkpoints=checkpoints, final=final, opponent_events=opponent.events, control_log=control_log, lua_errors=lua, disclosed=audit, trace=rows,
                draws='no normal draws (fixture duel: draw count 0); effect draws occur'), observed


def job(command, names):
    reg_path = sorted(ROOT.glob('amendment-*.json'))
    reg = json.loads((reg_path[-1] if reg_path else ROOT / 'registration.json').read_text())
    for n, digest in reg['code'].items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    allowed = set(COMPARISON) | set(reg.get('controls_to_run', CONTROLS))
    for n in names:
        if n not in allowed:
            raise RuntimeError(f'{n} is not registered to run')
    A = F.analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    assert Path(E.__file__).resolve().parent == CODE
    E.verify_pins()
    if (sha(BASE[0]), sha(BASE[1])) != SM.FROZEN_SHA:
        raise RuntimeError('base pair changed')
    SM.read_manifest(CAND)
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
                out['observations'] = dict(sha256=sha(d / 'observations.npz') if observed else None, decisions=len(observed))
                F.write(d / 'record.json', out)
                print('RECORD', json.dumps(dict(name=name, reason=out['reason'], failure=out['failure'], target=out['target'],
                                                opp=[e['what'] for e in out['opponent_events']], control=[(x['step'], x['rule']) for x in out['control_log']][:3],
                                                adapter=(out['policy'] or {}).get('adapter_active') and [(x['call'], x['frozen_argmax'], x['final_argmax']) for x in out['policy']['adapter_active']],
                                                end=(out['checkpoints'].get('target_chain_resolved') or {}).get('hand'), visible=len(out['disclosed']), lua=out['lua_errors']),
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
