"""Evaluation for the Artmage learning pilot. Cap: 18 evaluation episodes (own ledger eval-runs.jsonl).

  variant C1v|C2v          scripted feasibility of an NT1 variant (teacher, passive opponent, stop at turn 3 start)
  rt1 start|candidate ID   free model on a known Assault hand; stop rules identical to check_other_positions (after mode)
  nt1 start|candidate C    free model on C1 / C2 / C1v / C2v; observe to the end of own turn 3 (stop at the first
                           decision of turn 4), game over (from env.step returns) or 600 environment decisions
CPU only. Player 1 = common pass policy (article_witness.passive order); if it cannot decide, the first legal option
(logged as a fallback). Fixture duels: both players hold the same five cards and NO normal draws happen (the native
benchmark fixture sets draw count 0), so turn 3 is observed without a turn-3 draw.
"""
import collections
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_teacher as T  # noqa: E402

LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 18
NT1_CAP = 600
RT1_CAP = 320
OLD_MANIFEST = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json'
OLD_REGRESSION = P / 'experiments/artmage-parallel-20261110/regression'
ANALYSIS = P / 'experiments/capability-probe-20261109/code/analysis.py'
RT1_CASES = ['Artmage-g06-p0', 'Artmage-g07-p0', 'Artmage-g08-p0', 'Artmage-g09-p0', 'Artmage-g10-p0', 'Artmage-g11-p0']
SANSEN, FUNSAI = 25311006, 97045737
VARIANTS = {
    'C1v': dict(base='C1', hand=[T.ASSAULT, FUNSAI, T.USAGI, T.PSY_GAMMA, T.PSY_DRIVER], seed=1110003,
                filler_priority=[T.PSY_DRIVER, T.PSY_GAMMA, T.USAGI], cost_card='聖王の粉砕（罠）'),
    'C2v': dict(base='C2', hand=[T.FIELD, SANSEN, T.USAGI, T.PSY_GAMMA, T.PSY_DRIVER], seed=1110004,
                filler_priority=[T.PSY_DRIVER, T.PSY_GAMMA, T.USAGI], cost_card='三戦の才（魔法）'),
}
NT1_CONDITIONS = {
    'C1': dict(hand=T.CONDITIONS['C1']['hand'], seed=T.CONDITIONS['C1']['seed'], reference='C1'),
    'C2': dict(hand=T.CONDITIONS['C2']['hand'], seed=T.CONDITIONS['C2']['seed'], reference='C2'),
    'C1v': dict(hand=VARIANTS['C1v']['hand'], seed=VARIANTS['C1v']['seed'], reference='C1'),
    'C2v': dict(hand=VARIANTS['C2v']['hand'], seed=VARIANTS['C2v']['seed'], reference='C2'),
}
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
    spec = importlib.util.spec_from_file_location('capability_analysis_for_pilot', ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VariantTeacher(T.Teacher):
    """The registered C1/C2 plan; the Medius② cost rule uses the variant's own filler order."""
    def __init__(self, plan, filler_priority):
        super().__init__(plan)
        self.filler_priority = list(filler_priority)

    def _rule(self, name, s):
        if name != 'medius_cost' or s['message'] not in ['select_card', 'select_unselect_card']:
            return None
        candidates = {x['code']: x['index'] for x in s['actions'] if x['code'] and str(x.get('spec', '')).startswith('h')}
        if not candidates:
            return None
        before = collections.Counter(self.marks.get('before_draw', {}))
        for code in (T.own_hand(s) - before):
            if code in T.NON_ENGINE_MONSTERS and code in candidates:
                return candidates[code]
        return next((candidates[c] for c in self.filler_priority if c in candidates), None)


def board(s):
    own = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc)
    up = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc and x['position'] & 5)
    return dict(turn=s['turn_count'], phase=s['phase'], lp0=s['lp0'], lp1=s['lp1'], monsters=own(0, 4), monsters_face_up=up(0, 4),
                spell_trap=own(0, 8), spell_trap_face_up=up(0, 8), hand=own(0, 2), graveyard=own(0, 16), banished=own(0, 32),
                extra_face_up=sorted(x['code'] for x in s['cards'] if x['controller'] == 0 and x['location'] == 64 and x['position'] & 5),
                opponent_monsters=own(1, 4), opponent_hand_count=len(own(1, 2)))


def opponent_choice(v, s):
    try:
        return v.passive(s), False
    except RuntimeError:
        return s['actions'][0]['index'], True


def models(which):
    if which == 'start':
        pair = (ROOT / 'models/initial/base.flax_model', ROOT / 'models/initial/chain-head.flax_model')
        want = ('12500c7724b2d723983a2963a960d75172a6df51fb55ea4f3acf102eeb34132e', '0bc7534cab102cd87586e5d22bd4ad0cd3b55d52d08e284146976506ae6193fc')
    else:
        result = json.loads((ROOT / 'train/result.json').read_text())
        if result.get('status') != 'complete':
            raise RuntimeError('no complete candidate')
        pair = (ROOT / 'train/candidate/base.flax_model', ROOT / 'train/candidate/chain-head.flax_model')
        want = (result['saved']['candidate_base_sha256'], result['saved']['candidate_head_sha256'])
    return pair, want


def save_observations(path, rows):
    import numpy as np
    arrays = {k: np.stack([r['obs'][k] for r in rows]) for k in rows[0]['obs']}
    arrays.update(labels=np.array([r['label'] for r in rows], np.int32), steps=np.array([r['step'] for r in rows], np.int32),
                  turns=np.array([r['turn'] for r in rows], np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def observation(obs, info):
    import numpy as np
    length, owner = int(info['public_chain_length'][0]), int(info['public_chain_player'][0])
    out = {k: np.asarray(obs[k])[0].copy() for k in ['cards_', 'global_', 'actions_', 'h_actions_']}
    out['chain'] = np.array([length, 0 if not length else 1 if owner == 0 else 2, int(info['public_chain_card_id'][0]), int(info['public_chain_effect'][0])], np.int32)
    return out


# ---------------------------------------------------------------- runs
def run_variant(E, A, name):
    v, np = E.v, E.np
    spec = VARIANTS[name]
    plan = T.CONDITIONS[spec['base']]['plan']
    teacher = E.create_teacher(VariantTeacher(plan, spec['filler_priority']))
    case = dict(case_id=name, hand=spec['hand'], seed=spec['seed'])
    env, obs, info = E.env_for(case)
    rows, observed, failure, blocking, game_over, checkpoints, reason = [], [], None, None, None, {}, 'step_cap'
    previous = 0
    try:
        for step in range(700):
            s = v.native.benchmark_snapshot()
            if s['turn_player'] != previous and 'own_turn_end' not in checkpoints:
                checkpoints['own_turn_end'] = board(s)
            previous = s['turn_player']
            if s['turn_count'] >= 3:
                checkpoints['next_own_turn_start'] = board(s)
                reason = 'reached_turn3'
                break
            v.check_observation(obs, info, s)
            if s['to_play'] == 0:
                try:
                    idx = teacher(s)
                except T.Unresolved as exc:
                    failure = str(exc)
                    blocking = dict(step=step, message=s['message'], plan_head=teacher.base.head(),
                                    legal=[A.meaning(dict(x, name=v.names.get(x['code'], ''))) for x in s['actions']], board=board(s))
                    break
                why = teacher.base.log[-1]['why']
                observed.append(dict(obs=observation(obs, info), label=idx, step=step, turn=s['turn_count']))
            else:
                idx, _ = opponent_choice(v, s)
                why = dict(kind='opponent-passive')
            rows.append(dict(step=step, snapshot=v.compact(s), selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), why=why))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                game_over = dict(step=step, reward=float(np.asarray(reward).reshape(-1)[0]))
                reason = 'game_over'
                break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua = v.native.get_lua_error_count()
    E.close(env)
    t = teacher.base
    out = dict(condition=name, hand=spec['hand'], seed=spec['seed'], cost_card=spec['cost_card'], base_plan=spec['base'],
               script_completed=failure is None and not t.remaining_required(), failure=failure, blocking=blocking, reason=reason,
               game_over=game_over, lua_errors=lua, remaining_required=t.remaining_required(), skipped_optional=t.skipped_optional,
               marks=t.marks, checkpoints=checkpoints, final=final, trace=rows, position_events=teacher.positions.events)
    if observed:
        save_observations(ROOT / 'eval/variants' / f'{name}-observations.npz', observed)
    write(ROOT / 'eval/variants' / f'{name}.json', out)
    print('VARIANT', json.dumps(dict(condition=name, script_completed=out['script_completed'], failure=failure, reason=reason,
                                     own_turn_end=checkpoints.get('own_turn_end'), lua_errors=lua), ensure_ascii=False), flush=True)
    return out


def run_rt1(E, A, which, cid):
    v, np = E.v, E.np
    split = 'validation' if cid in ['Artmage-g06-p0', 'Artmage-g07-p0'] else 'holdout'
    case = next(c for c in json.loads(OLD_MANIFEST.read_text())['decks']['Artmage']['cases'][split] if c['case_id'] == cid)
    pair, want = models(which)
    policy = E.create_policy(pair[0], pair[1], positions=True, expected=want)
    env, obs, info = E.env_for(case)
    policy.reset()
    trace, reason = [], 'decision_limit'
    try:
        for step in range(RT1_CAP):
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
            trace.append(dict(step=step, snapshot=v.compact(s), selected=s['actions'][idx]))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                reason = 'episode_finished'; break
        final = v.compact(v.native.benchmark_snapshot()) if reason != 'episode_finished' else None
    finally:
        E.close(env)
    goal = v.check_goal('Artmage', final) if final else None
    row = dict(model=which, model_sha256=list(want), case_id=cid, split_origin=split, seed=case['seed'], hand=case['hand'],
               passed=bool(final) and all(goal.values()) and final['turn_player'] == 1, goal=goal, reason=reason,
               final=final, board=board(final) if final else None, trace=trace, events=copy.deepcopy(policy.positions.events),
               lua_errors=v.native.get_lua_error_count())
    write(ROOT / 'eval/rt1' / which / f'{cid}.json', row)
    print('RT1', json.dumps({k: row[k] for k in ['model', 'case_id', 'passed', 'reason', 'goal']}, ensure_ascii=False), flush=True)
    return row


def run_nt1(E, A, which, cond):
    v, np = E.v, E.np
    spec = NT1_CONDITIONS[cond]
    pair, want = models(which)
    policy = E.create_policy(pair[0], pair[1], positions=True, expected=want)
    case = dict(case_id=cond, hand=spec['hand'], seed=spec['seed'])
    env, obs, info = E.env_for(case)
    policy.reset()
    rows, observed, checkpoints, fallbacks = [], [], {}, 0
    reason, game_over, failure = 'step_cap', None, None
    try:
        for step in range(NT1_CAP):
            s = v.native.benchmark_snapshot()
            for t, label in [(2, 'own_turn1_end'), (3, 'opp_turn2_end'), (4, 'own_turn3_end')]:
                if s['turn_count'] >= t and label not in checkpoints:
                    checkpoints[label] = board(s)
            if s['turn_count'] >= 4:
                reason = 'own_turn3_end'
                break
            assert int(info['raw_num_options'][0]) <= 128
            v.check_observation(obs, info, s)
            if s['to_play'] == 0:
                idx = int(policy.act(obs, info))
                observed.append(dict(obs=observation(obs, info), label=idx, step=step, turn=s['turn_count']))
                who = 'model'
            else:
                idx, fb = opponent_choice(v, s)
                fallbacks += int(fb)
                who = 'opponent-fallback' if fb else 'opponent-pass'
            rows.append(dict(step=step, snapshot=v.compact(s), selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), who=who))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                r = float(np.asarray(reward).reshape(-1)[0])
                acting = s['to_play']
                game_over = dict(step=step, reward=r, acting_player=acting,
                                 winner=(acting if r > 0 else 1 - acting) if r != 0 else None, turn=s['turn_count'])
                reason = 'game_over'
                break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua = v.native.get_lua_error_count()
    E.close(env)
    out = dict(model=which, model_sha256=list(want), condition=cond, hand=spec['hand'], seed=spec['seed'], reason=reason, failure=failure,
               game_over=game_over, checkpoints=checkpoints, final=final, opponent_fallbacks=fallbacks, lua_errors=lua,
               legacy_goal_at_own_turn1_end=None, trace=rows, position_events=copy.deepcopy(policy.positions.events),
               decisions=len(rows), model_decisions=len(observed), draws='none (fixture duel: draw count 0)')
    t1 = [r for r in rows if r['snapshot']['turn_count'] == 1]
    if t1:
        out['legacy_goal_at_own_turn1_end'] = v.check_goal('Artmage', rows[len(t1)]['snapshot']) if len(rows) > len(t1) else None
    if observed:
        save_observations(ROOT / 'eval/nt1' / which / f'{cond}-observations.npz', observed)
    write(ROOT / 'eval/nt1' / which / f'{cond}.json', out)
    print('NT1', json.dumps(dict(model=which, condition=cond, reason=reason, failure=failure, game_over=game_over, fallbacks=fallbacks,
                                 own_turn1_end=checkpoints.get('own_turn1_end', {}).get('monsters'), lp_at_turn3_end=(checkpoints.get('own_turn3_end') or {}).get('lp1'),
                                 lua=lua), ensure_ascii=False), flush=True)
    return out


def job(command, args):
    reg = json.loads((ROOT / 'registration.json').read_text())
    for name, digest in reg['code'].items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    A = analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    E.verify_pins()
    items = []
    if command == 'variant':
        items = [('variant', a) for a in args]
    elif command == 'rt1':
        items = [('rt1', args[0], c) for c in args[1:]]
    elif command == 'nt1':
        items = [('nt1', args[0], c) for c in args[1:]]
    if len(ledger.started()) + len(items) > LIMIT:
        raise RuntimeError('evaluation cap would be exceeded')
    for item in items:
        run = ledger.start(' '.join(item), item[-1], now())
        try:
            if item[0] == 'variant':
                out = run_variant(E, A, item[1]); status = 'ok' if out['script_completed'] else 'completed-with-findings'
            elif item[0] == 'rt1':
                out = run_rt1(E, A, item[1], item[2]); status = 'ok'
            else:
                out = run_nt1(E, A, item[1], item[2]); status = 'ok' if not out['failure'] else 'completed-with-findings'
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
