"""Engine evaluation of the response-study candidate (CPU). Cap: 28 engine episodes in eval-runs.jsonl (failures count).

  rl_runs.py a-record S1|S2 ...      A: replay the old D0R0 path (functional comparison record) BY MEANING up to the first
                                      own Medius① window, the candidate reading every observation once; from the window the
                                      candidate decides freely to the end of own turn 3 (no D1 / R1)
  rl_runs.py a-replay S1|S2 ...      same fixture/seed, recorded indices; snapshots, meanings, model inputs, end state compared
  rl_runs.py b-record NAME ...        B: flc_runs.play (frozen copy, 8af7cbae) with the candidate: prefix replay + the Graphrea
  rl_runs.py b-replay NAME ...           branch control as in the comparison; everything else is the candidate
  rl_runs.py retain NAME ...          C: hb_runs.rt1 / hb_runs.free_run (frozen copy, 88e69412) with the candidate
The frozen evaluation modules are used unchanged; only their output root (hb_runs.ROOT) and the candidate model
(flc_runs.MODEL / MODEL_SHA) are rebound to this study, and the module files are asserted to come from this root.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
import artmage_teacher  # noqa: E402,F401  (cached from THIS root before the frozen modules insert their own roots)
import batch_plans  # noqa: E402,F401
import hb_plans  # noqa: E402,F401
import flc_runs as F  # noqa: E402
import hb_runs as HR  # noqa: E402
for _old in {str(F.ROOT / 'code'), str(HR.ROOT / 'code')}:
    while _old in sys.path:
        sys.path.remove(_old)
for _m in (F, HR, artmage_teacher, batch_plans, hb_plans):
    assert Path(_m.__file__).resolve().parent == CODE, _m.__file__
FROZEN = {'flc_runs.py': '8af7cbae6ce9d6d8322b043326f6b78d3f819ef8758ceb0e11597488f33e840f',
          'hb_runs.py': '88e694125079d2a9687490f1120186b536939b8750f731266efae56b167a0c14'}

FLC = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')
LEDGER = ROOT / 'eval-runs.jsonl'
LIMIT = 28
STEP_CAP = F.STEP_CAP
A_SITES = {'S1': ('BV1-1cost-VARNISH-OM', 149), 'S2': ('BV1-2cost-ASSAULT-OM', 169)}
B_NAMES = ['BV1-1cost-VARNISH-OM', 'BV1-2cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OM', 'BV1-2cost-VARNISH-OM']
C_NAMES = [f'Artmage-g{i:02d}-p0' for i in range(6, 12)] + [f'F{i}-V' for i in range(7)]
MEDIUS = F.MEDIUS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_medius_window(s):
    """Copy of otp_runs.is_medius_window (opponent-turn probe)."""
    ch = s['public_chain']
    return (s['to_play'] == 0 and s['turn_count'] == 2 and s['turn_player'] == 1 and s['phase'] == 4 and s['message'] == 'select_chain'
            and len(ch) == 1 and ch[0]['player'] == 1 and ch[0]['code'] == MEDIUS and ch[0]['description'] == MEDIUS * 16)


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
    """Point the frozen modules at this study (outputs and candidate); nothing else is changed."""
    for n, digest in FROZEN.items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'frozen module changed: {n}')
    pair, want = candidate()
    F.MODEL, F.MODEL_SHA = pair, want
    HR.ROOT = ROOT
    assert HR.models('candidate') == (pair, want)
    return pair, want


def play_a(E, A, site, record=None):
    """Old D0R0 path replayed by meaning up to the window (the candidate reads every observation), then the candidate."""
    v, np = E.v, E.np
    old_name, window = A_SITES[site]
    old = json.loads((FLC / 'runs' / old_name / 'record.json').read_text())
    if next(r['step'] for r in old['trace'] if is_medius_window(r['snapshot'])) != window:
        raise RuntimeError('registered window differs from the old record')
    c = F.parse(old_name)
    spec = F.HP.HANDS[c['hand_name']]
    rows_fx = F.fixture(v.deck_cards('Artmage'), spec['hand'], F.OPPONENT_HAND, spec['seed'])
    env, obs, info = E.env_for(dict(case_id=f'A-{site}', hand=spec['hand'], seed=spec['seed']), rows=rows_fx)
    policy = None if record else E.create_policy(F.MODEL[0], F.MODEL[1], positions=True, expected=F.MODEL_SHA)
    if policy:
        policy.reset()
    opponent = F.Opponent(c['opp'], v)
    rows, observed, audit, checkpoints, agree = [], [], [], {}, []
    reason, game_over, failure, mismatch, window_info = 'step_cap', None, None, None, None
    try:
        for step in range(STEP_CAP):
            s = v.native.benchmark_snapshot()
            s['_step'] = step
            for t, label in [(2, 'own_turn1_end'), (3, 'opp_turn2_end'), (4, 'own_turn3_end')]:
                if s['turn_count'] >= t and label not in checkpoints:
                    checkpoints[label] = F.board(s)
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
            elif step < window:
                rec = old['trace'][step]
                model_idx = int(policy.act(obs, info)) if s['to_play'] == 0 else None
                idx = next((a['index'] for a in s['actions'] if F.meaning_key(a) == F.meaning_key(rec['selected'])), None)
                if idx is None:
                    raise RuntimeError(f'old action not legal at step {step}')
                diff = A.compare_step(rec, snap, dict(s['actions'][idx], name=v.names.get(s['actions'][idx]['code'], '')))
                if diff:
                    raise RuntimeError(f'path differs from the old record at step {step}: {diff}')
                if s['to_play'] == 0:
                    agree.append(dict(step=step, turn=s['turn_count'], candidate_agrees=model_idx == idx,
                                      candidate=F.meaning_key(s['actions'][model_idx]), replayed=F.meaning_key(s['actions'][idx])))
                    who = 'replay-old'
                else:
                    opp_idx, _ = opponent(s)
                    if opp_idx != idx:
                        raise RuntimeError(f'opponent script differs from the old record at step {step}')
                    who = 'opponent-replay-old'
            elif s['to_play'] == 0:
                model_idx = int(policy.act(obs, info))
                if step == window:
                    if not is_medius_window(s):
                        raise RuntimeError(f'step {step} is not the Medius① window')
                    old_choice = old['trace'][step]['selected']
                    window_info = dict(step=step, candidates=[dict(index=a['index'], act=a['act'], code=a['code'], effect=a.get('effect'), spec=a.get('spec')) for a in s['actions']],
                                       candidate_choice=dict(act=s['actions'][model_idx]['act'], code=s['actions'][model_idx]['code'], effect=s['actions'][model_idx].get('effect'),
                                                             spec=s['actions'][model_idx].get('spec')),
                                       start_model_choice_in_old_record=dict(act=old_choice['act'], code=old_choice['code'], effect=old_choice.get('effect')))
                idx, who = model_idx, 'candidate'
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
    return dict(name=f'A-{site}', site=site, old_condition=old_name, old_record_sha256=sha(FLC / 'runs' / old_name / 'record.json'), window_step=window,
                model_sha256=list(F.MODEL_SHA), hand=spec['hand'], seed=spec['seed'], opponent_hand=F.OPPONENT_HAND,
                window=window_info, replayed_agreement=agree, reason=reason, failure=failure, mismatch=mismatch, game_over=game_over,
                checkpoints=checkpoints, final=final, opponent_events=opponent.events, lua_errors=lua, disclosed=audit, trace=rows,
                draws='no normal draws (fixture duel: draw count 0); effect draws occur'), observed


def run_dir(kind, name):
    return ROOT / 'runs' / (f'A-{name}' if kind == 'a' else f'B-{name}')


def job(command, names):
    reg = json.loads((sorted(ROOT.glob('amendment-*.json'))[-1] if list(ROOT.glob('amendment-*.json')) else ROOT / 'registration.json').read_text())
    for n, digest in reg['code'].items():
        if sha(CODE / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    allowed = {'a-record': A_SITES, 'a-replay': A_SITES, 'b-record': B_NAMES, 'b-replay': B_NAMES, 'retain': C_NAMES}[command]
    for n in names:
        if n not in allowed:
            raise RuntimeError(f'{n} is not registered for {command}')
    A = F.analysis()
    ledger = A.Ledger(LEDGER, limit=LIMIT)
    import artmage_entry as E
    assert Path(E.__file__).resolve().parent == CODE
    E.verify_pins()
    bind()
    if len(ledger.started()) + len(names) > LIMIT:
        raise RuntimeError('engine episode cap would be exceeded')
    for name in names:
        kind = command[0]
        if command == 'retain':
            target = ROOT / 'eval/retain/candidate' / f'{name}.json'
            if target.exists():
                raise RuntimeError(f'{name} retained already')
        else:
            d = run_dir(kind, name)
            if command.endswith('record') and (d / 'record.json').exists():
                raise RuntimeError(f'{name} recorded already')
            if command.endswith('replay') and (d / 'replay.json').exists():
                raise RuntimeError(f'{name} replayed already')
        run = ledger.start(f'{command} {name}', name, F.now())
        status = 'ok'
        try:
            if command == 'retain':
                if name.startswith('Artmage-g'):
                    HR.rt1(E, A, 'candidate', name)
                else:
                    out = HR.free_run(E, A, 'candidate', name, out_dir='retain')
                    status = 'ok' if not out['failure'] else 'completed-with-findings'
            elif command.endswith('record'):
                out, observed = play_a(E, A, name) if kind == 'a' else F.play(E, A, name)
                if observed:
                    F.save_observations(d / 'observations.npz', observed)
                out['engine_run'] = run
                out['evaluation'] = 'A (old path to the window, then candidate)' if kind == 'a' else 'B (flc_runs.play with the candidate)'
                out['observations'] = dict(sha256=sha(d / 'observations.npz') if observed else None, decisions=len(observed))
                F.write(d / 'record.json', out)
                cp = out['checkpoints']
                print('RECORD', json.dumps(dict(name=d.name, failure=out['failure'], reason=out['reason'], window=out.get('window') and out['window']['candidate_choice'],
                                                prefix=out.get('prefix_replayed'), branch=(out.get('branch') or {}).get('executed'),
                                                t2_own=(cp.get('opp_turn2_end') or {}).get('monsters'), t2_opp=(cp.get('opp_turn2_end') or {}).get('opponent_monsters'),
                                                lp1_t3=(cp.get('own_turn3_end') or {}).get('lp1'), game_over=out['game_over'], opp=[e['what'] for e in out['opponent_events']],
                                                visible=len(out['disclosed']), lua=out['lua_errors']), ensure_ascii=False), flush=True)
                status = 'ok' if not out['failure'] else 'completed-with-findings'
            else:
                record = json.loads((d / 'record.json').read_text())
                got, observed = play_a(E, A, name, record=record) if kind == 'a' else F.play(E, A, name, record=record)
                rep = dict(name=d.name, engine_run=run, replayed_engine_run=record['engine_run'], mismatch=got['mismatch'], failure=got['failure'],
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
