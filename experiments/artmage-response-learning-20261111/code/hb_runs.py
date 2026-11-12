"""Engine runs for the hand-branch study. Cap: 72 episodes in eval-runs.jsonl.

  substitute NAME...     one-move diagnostic on F1-V/F3-V/F4-V (start model; first divergent choice replaced by the script's)

  record NAME...         new train demonstration (scripted teacher, passive opponent, stop at the first turn-3 decision)
  replay NAME...         same fixture/seed, recorded indices; every step, the end state and the saved observations compared
  validate NAME...       scripted feasibility of a validation hand (stored as JSON only; no observation arrays)
  free MODEL NAME...     free model on a validation hand (start | candidate); observe to the end of own turn 3
  retain NAME...         candidate on RT1 cases (Artmage-g06..g11) and on the earlier F0-V..F6-V hands
CPU only. Fixture duels: both players hold the same five cards and no normal draws happen (native draw count 0).
Player 1 always uses the common pass policy (first legal option only when it cannot decide, logged).
Opponent hand/deck identities visible in an observation are audited (origin selection_prompt / unverified); a clear
exposure (visible before any action or more than 15 at once) stops the run.
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

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
BATCH = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import batch_plans as BP  # noqa: E402
import hb_plans as HP  # noqa: E402
import artmage_teacher as T  # noqa: E402

LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 72
STEP_CAP = 700
FREE_CAP = 600
RT1_CAP = 320
OLD_MANIFEST = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json'
ANALYSIS = P / 'experiments/capability-probe-20261109/code/analysis.py'
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
    spec = importlib.util.spec_from_file_location('capability_analysis_for_batch', ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered_code():
    amendments = sorted(ROOT.glob('amendment-*.json'))
    return json.loads((amendments[-1] if amendments else ROOT / 'registration.json').read_text())['code']


def board(s):
    own = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc)
    up = lambda c, loc: sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc and x['position'] & 5)
    return dict(turn=s['turn_count'], phase=s['phase'], lp0=s['lp0'], lp1=s['lp1'], monsters=own(0, 4), monsters_face_up=up(0, 4),
                spell_trap=own(0, 8), spell_trap_face_up=up(0, 8), hand=own(0, 2), graveyard=own(0, 16), banished=own(0, 32),
                extra_face_up=sorted(x['code'] for x in s['cards'] if x['controller'] == 0 and x['location'] == 64 and x['position'] & 5),
                opponent_monsters=own(1, 4), opponent_hand_count=len(own(1, 2)))


def check_observation(v, obs, info, s, audit, step, previous):
    """Action identities and the public chain strict; opponent hand/deck identities audited (see module docstring)."""
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


def observation(obs, info):
    import numpy as np
    length, owner = int(info['public_chain_length'][0]), int(info['public_chain_player'][0])
    out = {k: np.asarray(obs[k])[0].copy() for k in ['cards_', 'global_', 'actions_', 'h_actions_']}
    out['chain'] = np.array([length, 0 if not length else 1 if owner == 0 else 2, int(info['public_chain_card_id'][0]), int(info['public_chain_effect'][0])], np.int32)
    return out


def save_observations(path, rows):
    import numpy as np
    arrays = {k: np.stack([r['obs'][k] for r in rows]) for k in rows[0]['obs']}
    arrays.update(labels=np.array([r['label'] for r in rows], np.int32), steps=np.array([r['step'] for r in rows], np.int32),
                  turns=np.array([r['turn'] for r in rows], np.int32), num_options=np.array([r['n'] for r in rows], np.int32))
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


def opponent_choice(v, s):
    try:
        return v.passive(s), False
    except RuntimeError:
        return s['actions'][0]['index'], True


def models(which):
    if which == 'start':
        pair = (ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
        want = ('f47cd2b664b6153bec85484a037ecd888244baafd1b503e01655681a6f929186', '92b452daf1a8e9b30d657963fa48db8ddc46d8dfb552e9802d878540e7a823bd')
    else:
        result = json.loads((ROOT / 'train/result.json').read_text())
        if result.get('status') != 'complete':
            raise RuntimeError('no complete candidate')
        pair = (ROOT / 'train/candidate/base.flax_model', ROOT / 'train/candidate/chain-head.flax_model')
        want = (result['saved']['candidate_base_sha256'], result['saved']['candidate_head_sha256'])
    return pair, want


def hand_spec(name):
    if name in HP.HANDS:
        x = HP.HANDS[name]
        return dict(condition=x['plan'], hand=x['hand'], seed=x['seed'])
    return BP.VALIDATION[name]


def make_teacher(E, name):
    spec = HP.HANDS[name]
    return E.create_teacher(HP.BranchTeacher(HP.PLANS[spec['plan']])), f"BranchTeacher {spec['plan']}"


# ---------------------------------------------------------------- scripted record / validate / replay
def scripted(E, A, name, record=None):
    """Teacher-driven (record=None) or index-driven replay of a recorded run. Stops at the first decision of turn 3."""
    v, np = E.v, E.np
    spec = hand_spec(name)
    case = dict(case_id=name, hand=spec['hand'], seed=spec['seed'])
    teacher, teacher_kind = (None, 'replay') if record else make_teacher(E, name)
    env, obs, info = E.env_for(case)
    rows, observed, audit, checkpoints = [], [], [], {}
    failure = blocking = mismatch = game_over = None
    reason, previous = 'step_cap', 0
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            if s['turn_player'] != previous and 'own_turn_end' not in checkpoints:
                checkpoints['own_turn_end'] = board(s)
            previous = s['turn_player']
            if s['turn_count'] >= 3:
                checkpoints['next_own_turn_start'] = board(s)
                reason = 'reached_turn3'
                break
            if record and step >= len(record['trace']):
                reason = 'end_of_record'
                break
            assert int(info['raw_num_options'][0]) <= 128, 'Action set truncated'
            check_observation(v, obs, info, s, audit, step, rows[-1]['selected'] if rows else None)
            snap = v.compact(s)
            if record:
                rec = record['trace'][step]
                idx = rec['selected']['index']
                diff = A.compare_step(rec, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')) if idx < len(s['actions']) else {})
                if diff:
                    mismatch = dict(step=step, keys=diff)
                    break
                why = rec.get('why')
            elif s['to_play'] == 0:
                try:
                    idx = teacher(s)
                except (T.Unresolved, RuntimeError) as exc:
                    failure = f'{type(exc).__name__}: {exc}'
                    head = teacher.base.head() if hasattr(teacher.base, 'head') else (teacher.base.steps[:1] if hasattr(teacher.base, 'steps') else None)
                    blocking = dict(step=step, message=s['message'], turn=s['turn_count'], phase=s['phase'], plan_head=head,
                                    legal=[A.meaning(dict(x, name=v.names.get(x['code'], ''))) for x in s['actions']], board=board(s))
                    break
                log = getattr(teacher.base, 'log', None)
                why = dict(log[-1]['why'], adapter_changed=idx != log[-1]['index']) if log else dict(kind='old-teacher')
            else:
                idx, fb = opponent_choice(v, s)
                why = dict(kind='opponent-fallback' if fb else 'opponent-passive')
            if s['to_play'] == 0:
                observed.append(dict(obs=observation(obs, info), label=idx, step=step, turn=s['turn_count'], n=len(s['actions'])))
            rows.append(dict(step=step, snapshot=snap, selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), why=why))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                game_over = dict(step=step, reward=float(np.asarray(reward).reshape(-1)[0]), acting_player=s['to_play'])
                reason = 'game_over'
                break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua = v.native.get_lua_error_count()
    E.close(env)
    out = dict(name=name, condition=spec['condition'], hand=spec['hand'], seed=spec['seed'], teacher=teacher_kind, failure=failure,
               blocking=blocking, mismatch=mismatch, reason=reason, game_over=game_over, lua_errors=lua, checkpoints=checkpoints,
               final=final, trace=rows, disclosed=audit, decisions=len(observed))
    if teacher is not None:
        base = teacher.base
        remaining = base.remaining_required() if hasattr(base, 'remaining_required') else [x for x in base.steps]
        out.update(script_completed=failure is None and not remaining, remaining_required=remaining,
                   remaining_opponent_turn=[x for x in getattr(base, 'steps', []) if isinstance(x, dict) and x.get('_opp')],
                   skipped_optional=getattr(base, 'skipped_optional', []), marks=getattr(base, 'marks', {}),
                   unmatched_chain_heads=getattr(base, 'unmatched_chain_heads', []), adaptations=getattr(base, 'adaptations', []),
                   position_events=teacher.positions.events)
    return out, observed


# ---------------------------------------------------------------- free model runs
def free_run(E, A, which, name, out_dir='free', substitute=None):
    """Free model run. substitute = dict(step, meaning): at that step the model still reads the observation (RNN
    updated), then the given legal action is executed instead of its choice; earlier steps must equal the record."""
    v, np = E.v, E.np
    spec = hand_spec(name)
    pair, want = models(which)
    policy = E.create_policy(pair[0], pair[1], positions=True, expected=want)
    case = dict(case_id=name, hand=spec['hand'], seed=spec['seed'])
    env, obs, info = E.env_for(case)
    policy.reset()
    rows, checkpoints, audit, fallbacks = [], {}, [], 0
    reason, game_over, failure = 'step_cap', None, None
    try:
        for step in range(FREE_CAP):
            s = v.native.benchmark_snapshot()
            for t, label in [(2, 'own_turn1_end'), (3, 'opp_turn2_end'), (4, 'own_turn3_end')]:
                if s['turn_count'] >= t and label not in checkpoints:
                    checkpoints[label] = board(s)
            if s['turn_count'] >= 4:
                reason = 'own_turn3_end'
                break
            assert int(info['raw_num_options'][0]) <= 128
            check_observation(v, obs, info, s, audit, step, rows[-1]['selected'] if rows else None)
            if s['to_play'] == 0:
                idx, who = int(policy.act(obs, info)), 'model'
                if substitute is not None and step < substitute['step'] and idx != substitute['recorded'][step]:
                    substitute['prefix_mismatch'] = substitute.get('prefix_mismatch') or step
                if substitute is not None and step == substitute['step']:
                    want = tuple(substitute['meaning'])
                    sub = next((x['index'] for x in s['actions'] if (x['act'], x['code'], x.get('effect')) == want), None)
                    if sub is None:
                        raise RuntimeError(f'substitute action not legal at step {step}: {want}')
                    substitute['model_choice'] = dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], ''))
                    substitute['legal'] = [(x['act'], v.names.get(x['code'], x['code']), x.get('effect')) for x in s['actions']]
                    idx, who = sub, 'substituted'
            else:
                idx, fb = opponent_choice(v, s)
                fallbacks += int(fb)
                who = 'opponent-fallback' if fb else 'opponent-pass'
            rows.append(dict(step=step, snapshot=v.compact(s), selected=dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')), who=who))
            obs, reward, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                r = float(np.asarray(reward).reshape(-1)[0])
                game_over = dict(step=step, reward=r, acting_player=s['to_play'], winner=(s['to_play'] if r > 0 else 1 - s['to_play']) if r else None,
                                 turn=s['turn_count'])
                reason = 'game_over'
                break
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = v.compact(v.native.benchmark_snapshot()) if game_over is None else None
    lua = v.native.get_lua_error_count()
    E.close(env)
    out = dict(model=which, model_sha256=list(want), name=name, condition=spec['condition'], hand=spec['hand'], seed=spec['seed'], reason=reason, substitute=substitute,
               failure=failure, game_over=game_over, checkpoints=checkpoints, final=final, opponent_fallbacks=fallbacks, lua_errors=lua,
               trace=rows, position_events=copy.deepcopy(policy.positions.events), disclosed=audit,
               draws='none (fixture duel: draw count 0)')
    write(ROOT / 'eval' / out_dir / which / f'{name}.json', out)
    print('FREE' if out_dir == 'free' else out_dir.upper(), json.dumps(dict(model=which, name=name, reason=reason, failure=failure, game_over=game_over,
                                  own_turn1_end=checkpoints.get('own_turn1_end', {}).get('monsters'),
                                  lp1_turn3_end=(checkpoints.get('own_turn3_end') or {}).get('lp1'), visible=len(audit), lua=lua)), flush=True)
    return out


def rt1(E, A, which, cid):
    v, np = E.v, E.np
    split = 'validation' if cid in ['Artmage-g06-p0', 'Artmage-g07-p0'] else 'holdout'
    case = next(c for c in json.loads(OLD_MANIFEST.read_text())['decks']['Artmage']['cases'][split] if c['case_id'] == cid)
    pair, want = models(which)
    policy = E.create_policy(pair[0], pair[1], positions=True, expected=want)
    env, obs, info = E.env_for(case)
    policy.reset()
    trace, reason, audit = [], 'decision_limit', []
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
            check_observation(v, obs, info, s, audit, step, trace[-1]['selected'] if trace else None)
            idx = policy.act(obs, info) if s['to_play'] == 0 else v.passive(s)
            trace.append(dict(step=step, snapshot=v.compact(s), selected=s['actions'][idx]))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                reason = 'episode_finished'; break
        final = v.compact(v.native.benchmark_snapshot()) if reason != 'episode_finished' else None
    finally:
        E.close(env)
    goal = v.check_goal('Artmage', final) if final else None
    row = dict(model=which, model_sha256=list(want), case_id=cid, seed=case['seed'], hand=case['hand'],
               passed=bool(final) and all(goal.values()) and final['turn_player'] == 1, goal=goal, reason=reason, final=final,
               board=board(final) if final else None, trace=trace, events=copy.deepcopy(policy.positions.events),
               lua_errors=v.native.get_lua_error_count(), disclosed=audit)
    write(ROOT / 'eval/retain' / which / f'{cid}.json', row)
    print('RT1', json.dumps({k: row[k] for k in ['model', 'case_id', 'passed', 'reason']}), flush=True)
    return row


def main_moves(trace, turn=1):
    """(step, meaning) of player-0 choices in a turn, skipping passes, placements and positions (as in the batch summary)."""
    out = []
    for r in trace:
        snap = r['snapshot']
        if snap['to_play'] != 0 or snap['turn_count'] != turn:
            continue
        a_ = r['selected']
        if a_['act'] == 'Cancel' or snap['message'] in ('select_place', 'select_position', 'select_disfield'):
            continue
        if snap['message'] in ('select_chain', 'select_effectyn', 'select_yesno') and a_['act'] != 'Activate':
            continue
        out.append((r['step'], (a_['act'], a_['code'], a_.get('effect'))))
    return out


def substitution_plan(name):
    """From the batch records (base candidate free run and the validation script): the first differing choice."""
    free = json.loads((BATCH / 'eval/free/candidate' / f'{name}.json').read_text())
    script = json.loads((BATCH / 'validation' / f'{name}.json').read_text())
    m, r = main_moves(free['trace']), main_moves(script['trace'])
    k = next(i for i, (x, y) in enumerate(zip(m, r)) if x[1] != y[1])
    recorded = {row['step']: row['selected']['index'] for row in free['trace'] if row['snapshot']['to_play'] == 0}
    return dict(name=name, move_index=k, step=m[k][0], model_meaning=list(m[k][1]), meaning=list(r[k][1]), recorded=recorded,
                source_free=str(BATCH / 'eval/free/candidate' / f'{name}.json'), source_script=str(BATCH / 'validation' / f'{name}.json'))


# ---------------------------------------------------------------- job
def job(command, args):
    for name, digest in registered_code().items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    A = analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    E.verify_pins()
    if command in ('record', 'replay', 'validate', 'substitute', 'retain'):
        items = [(command, n) for n in args]
    else:
        items = [(command, args[0], n) for n in args[1:]]
    if len(ledger.started()) + len(items) > LIMIT:
        raise RuntimeError('engine episode cap would be exceeded')
    for item in items:
        name = item[-1]
        if item[0] == 'record' and (ROOT / 'demos' / name / 'record.json').exists():
            raise RuntimeError(f'{name} already recorded')
        if item[0] == 'replay' and (ROOT / 'demos' / name / 'replay.json').exists():
            raise RuntimeError(f'{name} already replayed')
        if item[0] == 'validate' and (ROOT / 'validation' / f'{name}.json').exists():
            raise RuntimeError(f'{name} already validated')
        run = ledger.start(' '.join(item), name, now())
        status = 'ok'
        try:
            if item[0] == 'record':
                out, observed = scripted(E, A, name)
                d = ROOT / 'demos' / name
                if observed:
                    save_observations(d / 'observations.npz', observed)
                out['observations'] = dict(path=str(d / 'observations.npz'), sha256=sha(d / 'observations.npz') if observed else None,
                                           decisions=len(observed), own_turn=sum(1 for r in observed if r['turn'] == 1),
                                           opponent_turn=sum(1 for r in observed if r['turn'] == 2))
                out['engine_run'] = run
                write(d / 'record.json', out)
                status = 'ok' if out['script_completed'] else 'completed-with-findings'
                print('RECORD', json.dumps(dict(name=name, completed=out['script_completed'], failure=out['failure'], reason=out['reason'],
                                                remaining=out['remaining_required'][:2], remaining_opp=len(out['remaining_opponent_turn']),
                                                end=out['checkpoints'].get('own_turn_end', {}).get('monsters'), decisions=len(observed),
                                                visible=len(out['disclosed']), lua=out['lua_errors']), ensure_ascii=False), flush=True)
            elif item[0] == 'replay':
                d = ROOT / 'demos' / name
                record = json.loads((d / 'record.json').read_text())
                got, observed = scripted(E, A, name, record=record)
                rep = dict(name=name, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
                           reproduced_steps=len(got['trace']), of=len(record['trace']), reason=got['reason'],
                           final_equal=None if record['final'] is None else A.normalized(got['final']) == A.normalized(record['final']),
                           checkpoints_equal=A.normalized(got['checkpoints']) == A.normalized(record['checkpoints']),
                           observations_match=same_observations(d / 'observations.npz', observed) if (d / 'observations.npz').exists() else not observed,
                           lua_errors=got['lua_errors'], disclosed=got['disclosed'])
                write(d / 'replay.json', rep)
                status = 'ok' if not rep['mismatch'] and not rep['failure'] else 'completed-with-findings'
                print('REPLAY', json.dumps({k: rep[k] for k in ['name', 'mismatch', 'failure', 'reproduced_steps', 'of', 'final_equal', 'checkpoints_equal', 'observations_match']}), flush=True)
            elif item[0] == 'validate':
                out, _ = scripted(E, A, name)
                out['engine_run'] = run
                write(ROOT / 'validation' / f'{name}.json', out)
                status = 'ok' if out['script_completed'] else 'completed-with-findings'
                print('VALIDATE', json.dumps(dict(name=name, completed=out['script_completed'], failure=out['failure'], reason=out['reason'],
                                                  remaining=out['remaining_required'][:2], end=out['checkpoints'].get('own_turn_end', {}).get('monsters'),
                                                  visible=len(out['disclosed']), lua=out['lua_errors']), ensure_ascii=False), flush=True)
            elif item[0] == 'free':
                out = free_run(E, A, item[1], name)
                status = 'ok' if not out['failure'] else 'completed-with-findings'
            elif item[0] == 'substitute':
                plan = substitution_plan(name)
                out = free_run(E, A, 'start', name, out_dir='substitute', substitute=plan)
                status = 'ok' if not out['failure'] else 'completed-with-findings'
            elif item[0] == 'retain':
                if name.startswith('Artmage-g'):
                    rt1(E, A, 'candidate', name)
                else:
                    out = free_run(E, A, 'candidate', name, out_dir='retain')
                    status = 'ok' if not out['failure'] else 'completed-with-findings'
            else:
                raise ValueError(item[0])
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
