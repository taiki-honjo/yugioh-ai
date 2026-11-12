"""Opponent-turn probe: keep vs reshape in the opponent's draw phase (D) x the first response to Medius① (R).
CPU, engine. Cap: 12 engine episodes in eval-runs.jsonl (failures count).

  otp_runs.py record NAME...   play one probe condition from the start (fixture, prefix replay, Graphrea branch, model)
  otp_runs.py replay NAME...   same fixture/seed, recorded action indices; snapshots, meanings, model inputs, end state compared

NAME = <S1|S2>-D<0|1>R<0|1>
  S1 = BV1-1cost-VARNISH-OM, S2 = BV1-2cost-ASSAULT-OM (the earlier functional comparison, same hands/seed/opponent script)
  D1 = in the opponent's turn-2 DRAW phase only, every optional own activation is declined with the legal Cancel; a prompt
       without a legal Cancel is left to the model and logged as forced. From the standby phase on, the model decides.
  R1 = at the FIRST own select_chain of turn 2 (opponent main phase 1) whose public chain is exactly one link, player 1's
       Medius① (code 97556336, description aux.Stringid(97556336,0)), activate once, in this priority:
       ファインメルト③ (34541940, effect 10011) else ディアクトロス② (27184601, effect 10011); same card: zone order.
       If neither is legal, the substitution is recorded as not established and the model's choice is executed.
       All later decisions are the model's.
The model reads every own observation exactly once (also when its choice is overridden); the executed action enters the
environment and the history. The duel loop, fixture, prefix replay, branch control and opponent script are the frozen
helpers of the comparison runner (code/flc_runs.py, sha256 8af7cbae…, imported unchanged).
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
import hb_plans  # noqa: E402,F401  (cached from THIS root before flc_runs inserts its own root into sys.path)
import flc_runs as F  # noqa: E402
OLD_CODE = str(F.ROOT / 'code')
while OLD_CODE in sys.path:
    sys.path.remove(OLD_CODE)
assert Path(F.__file__).resolve().parent == CODE and Path(hb_plans.__file__).resolve().parent == CODE

LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 12
STEP_CAP = 600
MODEL = (ROOT / 'models/candidate/base.flax_model', ROOT / 'models/candidate/chain-head.flax_model')
MODEL_SHA = F.MODEL_SHA
SITES = {'S1': 'BV1-1cost-VARNISH-OM', 'S2': 'BV1-2cost-ASSAULT-OM'}
NEW = ['S1-D1R0', 'S1-D0R1', 'S1-D1R1', 'S2-D1R0', 'S2-D1R1']
FINEMELT, DIAKTOROS = 34541940, 27184601
R1_PRIORITY = [(FINEMELT, 10011, 'ファインメルト③'), (DIAKTOROS, 10011, 'ディアクトロス②')]
MEDIUS_EFFECT1 = F.MEDIUS * 16  # aux.Stringid(97556336, 0): Medius① in the public chain description
DRAW_PHASE, MAIN1 = 1, 4
PASS_MESSAGES = ('select_chain', 'select_effectyn', 'select_yesno')


def parse(name):
    site, cond = name.split('-')
    assert site in SITES and len(cond) == 4 and cond[0] == 'D' and cond[2] == 'R' and cond[1] in '01' and cond[3] in '01', name
    return dict(site=site, old=SITES[site], d1=cond[1] == '1', r1=cond[3] == '1')


def meaning(a):
    return dict(act=a['act'], code=a['code'], effect=a.get('effect'), spec=a.get('spec'), index=a['index'])


def is_draw_phase_window(s):
    return s['to_play'] == 0 and s['turn_count'] == 2 and s['turn_player'] == 1 and s['phase'] == DRAW_PHASE


def is_medius_window(s):
    """Own select_chain in the opponent's main phase 1 of turn 2 with exactly player 1's Medius① on the chain."""
    ch = s['public_chain']
    return (s['to_play'] == 0 and s['turn_count'] == 2 and s['turn_player'] == 1 and s['phase'] == MAIN1 and s['message'] == 'select_chain'
            and len(ch) == 1 and ch[0]['player'] == 1 and ch[0]['code'] == F.MEDIUS and ch[0]['description'] == MEDIUS_EFFECT1)


def zone_key(a):
    spec = a.get('spec') or ''
    return (spec[:1], int(spec[1:]) if spec[1:].isdigit() else 99, a['index'])


def r1_pick(s):
    for code, effect, label in R1_PRIORITY:
        same = sorted([a for a in s['actions'] if a['act'] == 'Activate' and a['code'] == code and a.get('effect') == effect], key=zone_key)
        if same:
            return same[0], label
    return None, None


class Intervener:
    """Decides overrides; never advances the model (the caller runs the model once per observation)."""

    def __init__(self, d1, r1):
        self.d1, self.r1 = d1, r1
        self.r1_seen = False
        self.log = []

    def __call__(self, s, step, model_idx):
        if self.d1 and is_draw_phase_window(s):
            cancel = next((a['index'] for a in s['actions'] if a['act'] == 'Cancel'), None) if s['message'] in PASS_MESSAGES else None
            self.log.append(dict(rule='D1', step=step, message=s['message'], legal=[meaning(a) for a in s['actions']],
                                 model_choice=meaning(s['actions'][model_idx]), executed='Cancel' if cancel is not None else 'model (no legal Cancel: forced)',
                                 forced=cancel is None, model_agrees=cancel == model_idx if cancel is not None else True))
            return cancel
        if self.r1 and not self.r1_seen and is_medius_window(s):
            self.r1_seen = True
            pick, label = r1_pick(s)
            self.log.append(dict(rule='R1', step=step, message=s['message'], legal=[meaning(a) for a in s['actions']],
                                 model_choice=meaning(s['actions'][model_idx]), established=pick is not None,
                                 executed=label if pick else 'not established (neither effect legal): model choice executed',
                                 model_agrees=(pick['index'] == model_idx) if pick else None))
            return pick['index'] if pick else None
        return None


class Probe:
    """Extra checkpoints computed from snapshots only (identical in record and replay)."""

    def __init__(self):
        self.cp, self.medius_on_chain = {}, False

    def __call__(self, s):
        cp = self.cp
        if s['turn_count'] == 2 and s['phase'] > DRAW_PHASE and 'opp_draw_phase_end' not in cp:
            cp['opp_draw_phase_end'] = F.board(s)
        if is_medius_window(s) and 'medius_response_window' not in cp:
            cp['medius_response_window'] = dict(F.board(s), step=s['_step'], candidates=[meaning(a) for a in s['actions']])
        if s['turn_count'] == 2 and any(c['player'] == 1 and c['code'] == F.MEDIUS for c in s['public_chain']):
            self.medius_on_chain = True
        if self.medius_on_chain and s['chain_size'] == 0 and 'after_medius_chain' not in cp:
            b = F.board(s)
            cp['after_medius_chain'] = dict(b, step=s['_step'], beast_arrived=F.BEAST in b['opponent_monsters'], medius_on_field=F.MEDIUS in b['opponent_monsters'])


def play(E, A, name, record=None):
    """flc_runs.play with the two registered interventions and the extra checkpoints (marked PROBE)."""
    v, np = E.v, E.np
    pc = parse(name)
    c = F.parse(pc['old'])
    spec = F.HP.HANDS[c['hand_name']]
    deck = v.deck_cards('Artmage')
    rows_fx = F.fixture(deck, spec['hand'], F.OPPONENT_HAND, spec['seed'])
    case = dict(case_id=name, hand=spec['hand'], seed=spec['seed'])
    env, obs, info = E.env_for(case, rows=rows_fx)
    policy = None if record else E.create_policy(MODEL[0], MODEL[1], positions=True, expected=MODEL_SHA)
    if policy:
        policy.reset()
    opponent = F.Opponent(c['opp'], v)
    prefix, source = F.prefix_of(c['hand_name'])
    intervene, probe = Intervener(pc['d1'], pc['r1']), Probe()  # PROBE
    pos, branched, branch_info = 0, False, None
    rows, observed, audit, checkpoints, prefix_log = [], [], [], {}, []
    reason, game_over, failure, mismatch = 'step_cap', None, None, None
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            s['_step'] = step
            for t, label in [(2, 'own_turn1_end'), (3, 'opp_turn2_end'), (4, 'own_turn3_end')]:
                if s['turn_count'] >= t and label not in checkpoints:
                    checkpoints[label] = F.board(s)
            probe(s)  # PROBE
            if s['turn_count'] >= 4:
                reason = 'own_turn3_end'; break
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
            elif s['to_play'] == 0:
                model_idx = int(policy.act(obs, info))
                if not branched and pos < len(prefix):
                    want = prefix[pos]
                    if s['message'] != want['message']:
                        raise RuntimeError(f"prefix message differs at own decision {pos}: {s['message']} != {want['message']}")
                    idx = next((a['index'] for a in s['actions'] if F.meaning_key(a) == tuple(want['meaning'])), None)
                    if idx is None:
                        raise RuntimeError(f'prefix action not legal at own decision {pos}: {want}')
                    prefix_log.append(dict(step=step, model_agrees=model_idx == idx))
                    pos += 1
                    who = 'prefix-replay'
                elif not branched:
                    if not F.is_branch_prompt(s):
                        raise RuntimeError(f"expected the Graphrea set prompt at step {step}, got {s['message']}")
                    idx = next(a['index'] for a in s['actions'] if a['code'] == F.BRANCH[c['branch']])
                    branch_info = dict(step=step, executed=v.names.get(F.BRANCH[c['branch']]), model_choice=v.names.get(s['actions'][model_idx]['code']),
                                       offered=[v.names.get(a['code'], a['code']) for a in s['actions']])
                    branched = True
                    who = 'branch-control'
                else:
                    forced = intervene(s, step, model_idx)  # PROBE
                    if forced is None:
                        idx, who = model_idx, 'model'
                    else:
                        idx, who = forced, 'probe-' + intervene.log[-1]['rule']
            else:
                idx, who = opponent(s)
                who = 'opponent-' + who
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
    checkpoints.update(probe.cp)  # PROBE
    r1_log = [x for x in intervene.log if x['rule'] == 'R1']
    return dict(name=name, site=pc['site'], old_condition=pc['old'], d1=pc['d1'], r1=pc['r1'],
                hand_name=c['hand_name'], cost=c['cost'], branch_choice=c['branch'], opp=c['opp'], hand=spec['hand'], seed=spec['seed'],
                opponent_hand=F.OPPONENT_HAND, model_sha256=list(MODEL_SHA),
                prefix_source=source, prefix_length=len(prefix), prefix_replayed=pos, prefix_log=prefix_log, branch=branch_info,
                interventions=intervene.log,
                r1_status=None if not pc['r1'] or record else ('not reached' if not r1_log else 'established' if r1_log[0]['established'] else 'not established'),
                reason=reason, failure=failure, mismatch=mismatch, game_over=game_over, checkpoints=checkpoints, final=final,
                opponent_events=opponent.events, position_events=copy.deepcopy(policy.positions.events) if policy else None,
                lua_errors=lua, disclosed=audit, trace=rows,
                draws='no normal draws (fixture duel: draw count 0); effect draws occur'), observed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job(command, names):
    reg_path = sorted(ROOT.glob('amendment-*.json'))
    reg = json.loads((reg_path[-1] if reg_path else ROOT / 'registration.json').read_text())
    for n, digest in reg['code'].items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    for n in names:
        if n not in NEW:
            raise RuntimeError(f'{n} is not a registered new condition (reused conditions are not run)')
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
                out['observations'] = dict(sha256=sha(d / 'observations.npz') if observed else None, decisions=len(observed))
                F.write(d / 'record.json', out)
                cp = out['checkpoints']
                am = cp.get('after_medius_chain') or {}
                print('RECORD', json.dumps(dict(name=name, failure=out['failure'], reason=out['reason'], prefix=f"{out['prefix_replayed']}/{out['prefix_length']}",
                                                branch=out['branch'] and out['branch']['executed'], r1=out['r1_status'],
                                                interventions=[(x['rule'], x['step'], x['executed'], x['model_choice']['code'], x['model_choice']['effect']) for x in out['interventions']],
                                                beast_arrived=am.get('beast_arrived'), medius_on_field=am.get('medius_on_field'),
                                                t2_own=(cp.get('opp_turn2_end') or {}).get('monsters'), lp1_t3=(cp.get('own_turn3_end') or {}).get('lp1'),
                                                game_over=out['game_over'], visible=len(out['disclosed']), lua=out['lua_errors']), ensure_ascii=False), flush=True)
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
