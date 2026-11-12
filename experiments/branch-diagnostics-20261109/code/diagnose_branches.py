"""Branch diagnostics for two repeatedly failing validation openings. NO training, NO battles.

Exactly 2 cases x 7 models = 14 free runs with the production v2 snapshot code (read-only),
the unchanged witness/evaluator, the same hands and seeds. For every own decision it records
the public-information/history digest, the final base+chain-head logits (each model's RNN
state comes from its own run through the shared history), the neural choice, the executed
choice after the placement/position adapters, and the adapter memory. Teacher demonstrations
are used only to label diagnostic states; nothing is fed to the policy.
"""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal_ab as AB

CASES = ['mixed-rukina-cue-g10-p0', 'cue-reco-jj-zaren-g09-p0']
X = P / 'experiments/opening-rehearsal-lr-v2-20261109'
cfg = C.load_config(P / 'config/continuous-elfnotes.json')
C.runtime_check(cfg)
before = AB.production_fingerprint()
root = Path(cfg['root'])
MODELS = {'adopted': root / 'models/initial'}
for arm, label in [('c-lr1.5e-5', 'C'), ('d-lr1.5e-5-l5', 'D')]:
    for seed in [10000555, 10000777, 10000999]:
        MODELS[f'{label}-{seed}'] = X / arm / 'cycles' / f'256-seed{seed}' / 'candidate'
STORED = {'adopted': root / 'opening-baseline.json'}
for name in MODELS:
    if name != 'adopted':
        arm = 'c-lr1.5e-5' if name[0] == 'C' else 'd-lr1.5e-5-l5'
        STORED[name] = X / arm / 'cycles' / f"256-seed{name.split('-')[1]}" / 'opening-regression.json'

B = C.backend()
common = B.policy_common(cfg)
import all_opening_cases as cases
import center_placement
v = common.v
manifest = C.read_json(root / 'snapshot/opening-manifest.json')
by_id = {}
for case in manifest['decks']['ElfnotesKewlTune']['cases']['validation']:
    by_id.setdefault(case['case_id'], case)

def digest(obs, info):
    h = hashlib.sha256()
    for key in ['cards_', 'global_', 'actions_', 'h_actions_']:
        h.update(np.ascontiguousarray(np.asarray(obs[key])[0], np.uint8).tobytes())
    h.update(np.ascontiguousarray(common.features(info)[0], np.int64).tobytes())
    return h.hexdigest()

# Teacher demonstrations of these hands (validation split, diagnostics only).
teacher = {}
data_dir = Path(cfg['opening_manifest']).parent / 'dataset/ElfnotesKewlTune'
assert C.sha(data_dir / 'validation.npz') == manifest['decks']['ElfnotesKewlTune']['validation']['sha256']
witnesses = C.read_json(data_dir / 'validation-witnesses.json')
hand_of = {c['case_id']: tuple(sorted(c['hand'])) for c in manifest['decks']['ElfnotesKewlTune']['cases']['validation']}
wanted_hands = {hand_of[c] for c in CASES}
with np.load(data_dir / 'validation.npz') as z:
    arrays = {k: z[k] for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain', 'targets', 'labels', 'lengths']}
for i, w in enumerate(witnesses):
    if hand_of[w['case_id']] not in wanted_hands:
        continue
    for t in range(int(arrays['lengths'][i])):
        h = hashlib.sha256()
        for key in ['cards_', 'global_', 'actions_', 'h_actions_']:
            h.update(np.ascontiguousarray(arrays[key][i, t], np.uint8).tobytes())
        h.update(np.ascontiguousarray(arrays['chain'][i, t], np.int64).tobytes())
        teacher.setdefault(h.hexdigest(), []).append(dict(teacher_case=w['case_id'], step=t, label=int(arrays['labels'][i, t]),
                                                           acceptable=np.flatnonzero(arrays['targets'][i, t]).tolist()))
del arrays

class Recorder:
    """Wraps the production policy stack; observes, never changes, any decision."""
    def __init__(self, stack):
        self.stack, self.rows = stack, []
        neural = stack.base.base
        original = neural.infer
        def infer(*args):
            state, logits = original(*args)
            self.logits = np.asarray(logits)[0].astype(np.float64)
            return state, logits
        neural.infer = infer
    def reset(self):
        self.stack.reset(); self.rows = []
    def memory(self):
        m = self.stack.base.memory
        return dict(incoming=int(m.incoming), locked=bool(m.locked), remix=None if m.remix_targets is None else list(m.remix_targets))
    def act(self, obs, info):
        n = int(info['num_options'][0]); mem_before = self.memory()
        placements, positions = len(self.stack.base.memory.events), len(self.stack.positions.events)
        state_digest = digest(obs, info)
        executed = int(self.stack.act(obs, info))
        logits = self.logits[:n]; raw = int(logits.argmax())
        p = np.exp(logits - logits.max()); p /= p.sum()
        self.rows.append(dict(k=len(self.rows), digest=state_digest, n=n, logits=[round(float(x), 5) for x in logits],
                              probs=[round(float(x), 5) for x in p], neural=raw, executed=executed,
                              message=int(np.asarray(obs['actions_'])[0, raw, 3]), memory_before=mem_before, memory_after=self.memory(),
                              placement_event=copy.deepcopy(self.stack.base.memory.events[placements:]) or None,
                              position_event=copy.deepcopy(self.stack.positions.events[positions:]) or None,
                              on_teacher_path=teacher.get(state_digest)))
        return executed

def describe(a):
    keep = ['index', 'act', 'code', 'name', 'spec', 'place', 'position', 'finish', 'phase']
    return {k: a[k] for k in keep if k in a}

def own_cards(s, location, face_up=None):
    out = []
    for x in s['cards']:
        if x['controller'] == 0 and x['location'] == location and (face_up is None or bool(x['position'] & 5) == face_up):
            out.append(x['code'])
    return sorted(out)

def goal_parts(case, result):
    s = result.get('own_end')
    if not s:
        return dict(own_turn_ended=False)
    sp, w = v.spirit, v.v
    ms, active, sets, hand = set(own_cards(s, 4, True)), set(own_cards(s, 8, True)), set(own_cards(s, 8, False)), own_cards(s, 2)
    parts = dict(own_turn_ended=True, field_monsters=sorted(ms), active_spells_traps=sorted(active), set_spells_traps=sorted(sets), hand=hand)
    if case['kind'] == 'tune':
        parts.update(track=sp.TRACK in ms, remix=sp.REMIX in ms, juke_active=sp.JUKE in active, sync_set=sp.SYNC in sets,
                     crystal=w.CRYSTAL in ms, baronne=w.BARONNE in ms, parallelism=w.PAR in active, rhapsodia=w.RHAP in active)
    else:
        parts.update(endings={e['id']: dict(board=ms == set(e['board']), retained=(not e.get('retained')) or e['retained'] in hand)
                              for e in case['accepted_endings']},
                     juke_active=sp.JUKE in active, sync_set=sp.SYNC in sets)
    parts.update({k: result.get(k) for k in ['passed', 'board', 'failure', 'chosen_variant', 'jj_release_step', 'ash_used_as_synchro_material',
                                             'recovered_ash_activated', 'opponent_search_negated', 'opponent_search_resolved',
                                             'crystal_first_preference_met', 'boss_field_arrival_order']})
    parts['ash_recovered'] = result.get('ash_recovered') is not None
    return parts

runs, versions = {}, {}
for name, checkpoint in MODELS.items():
    versions[name] = dict(checkpoint=str(checkpoint), base_sha256=C.sha(checkpoint / 'base.flax_model'),
                          head_sha256=C.sha(checkpoint / 'chain-head.flax_model'))
    recorder = Recorder(B.create_policy(cfg, checkpoint))
    stored = {c['case_id']: c['passed'] for c in C.read_json(STORED[name])['cases']}
    for case_id in CASES:
        case = by_id[case_id]
        result, _ = v.run(cases.normalize(case), policy=recorder)
        own = [t for t in result['trace'] if t['snapshot']['to_play'] == 0]
        assert len(own) == len(recorder.rows), (name, case_id, len(own), len(recorder.rows))
        for row, t in zip(recorder.rows, own):
            assert row['executed'] == t['selected']['index']
            row.update(step=t['step'], message_name=t['snapshot']['message'], selected=describe(t['selected']),
                       options=[describe(a) for a in t['snapshot']['actions']],
                       public_chain=[dict(code=x['code'], player=x['player']) for x in t['snapshot'].get('public_chain', [])],
                       turn=t['snapshot'].get('turn_count'), phase=t['snapshot'].get('phase'))
        runs[(name, case_id)] = dict(model=name, case_id=case_id, passed=bool(result['passed']), stored_passed=stored[case_id],
                                     verdict_reproduced=bool(result['passed']) == stored[case_id], decisions=len(own),
                                     total_steps=len(result['trace']), goal=goal_parts(case, result), rows=recorder.rows,
                                     lua_errors=v.native.get_lua_error_count())
        print('RUN', name, case_id, 'passed', result['passed'], 'stored', stored[case_id], 'own decisions', len(own), flush=True)

def top(row, k=5):
    order = np.argsort(row['logits'])[::-1][:k]
    return [dict(index=int(i), option=row['options'][i], logit=row['logits'][i], prob=row['probs'][i]) for i in order]

comparisons = []
for case_id in CASES:
    ref = runs[('adopted', case_id)]['rows']
    for name in MODELS:
        if name == 'adopted':
            continue
        cand = runs[(name, case_id)]['rows']
        first_exec = next((k for k in range(min(len(ref), len(cand))) if ref[k]['executed'] != cand[k]['executed'] or ref[k]['digest'] != cand[k]['digest']), None)
        first_neural = next((k for k in range(min(len(ref), len(cand))) if ref[k]['neural'] != cand[k]['neural']), None)
        entry = dict(case_id=case_id, model=name, candidate_passed=runs[(name, case_id)]['passed'],
                     adopted_passed=runs[('adopted', case_id)]['passed'], first_executed_divergence=first_exec,
                     first_neural_divergence=first_neural, decisions_adopted=len(ref), decisions_candidate=len(cand))
        if first_exec is None:
            entry['identical_through_end'] = len(ref) == len(cand)
        else:
            k = first_exec
            a, c = ref[k], cand[k]
            entry['shared_history_identical_before'] = all(ref[j]['digest'] == cand[j]['digest'] and ref[j]['executed'] == cand[j]['executed'] for j in range(k))
            entry['same_state_at_divergence'] = a['digest'] == c['digest']
            if entry['same_state_at_divergence']:
                ia, ic = a['executed'], c['executed']
                entry.update(step=a['step'], turn=a['turn'], phase=a['phase'], message=a['message_name'], public_chain=a['public_chain'],
                             options=a['options'], adopted_choice=a['options'][ia], candidate_choice=c['options'][ic],
                             adopted_top=top(a), candidate_top=top(c),
                             adopted_gap_top1_top2=round(sorted(a['logits'])[-1] - sorted(a['logits'])[-2], 5) if a['n'] > 1 else None,
                             candidate_gap_top1_top2=round(sorted(c['logits'])[-1] - sorted(c['logits'])[-2], 5) if c['n'] > 1 else None,
                             adopted_logit_margin_for_its_choice=round(a['logits'][ia] - a['logits'][ic], 5),
                             candidate_logit_margin_for_its_choice=round(c['logits'][ic] - c['logits'][ia], 5),
                             adapter=dict(adopted=dict(neural=a['neural'], executed=ia, placement=a['placement_event'], position=a['position_event'],
                                                       memory=a['memory_before']),
                                          candidate=dict(neural=c['neural'], executed=ic, placement=c['placement_event'], position=c['position_event'],
                                                         memory=c['memory_before'])),
                             teacher_at_state=a['on_teacher_path'],
                             teacher_path_until=max((j for j in range(k + 1) if ref[j]['on_teacher_path']), default=None))
            entry['after_divergence'] = dict(adopted=[r['selected'] for r in ref[k:k + 25]], candidate=[r['selected'] for r in cand[k:k + 25]])
        entry['goal_adopted'] = runs[('adopted', case_id)]['goal']; entry['goal_candidate'] = runs[(name, case_id)]['goal']
        comparisons.append(entry)
        print('COMPARE', case_id, name, 'passed', entry['candidate_passed'], 'first executed divergence', first_exec,
              'first neural divergence', first_neural, flush=True)

after = AB.production_fingerprint()
snapshot = C.read_json(root / 'snapshot/hashes.json')
report = dict(cases=CASES, models=versions, free_runs=len(runs), training=False, battles=False,
              versions=dict(production_root=str(root), snapshot_hashes_sha256=C.sha(root / 'snapshot/hashes.json'),
                            center_placement_sha256=snapshot['policy-code/center_placement.py'],
                            battle_position_sha256=snapshot['policy-code/battle_position.py'],
                            all_opening_witness_sha256=snapshot['policy-code/all_opening_witness.py'],
                            evaluator_sha256=snapshot['evaluator.py'], center_placement_module=center_placement.__file__),
              case_definitions={c: dict(hand=by_id[c]['hand'], seed=by_id[c]['seed'], kind=by_id[c]['kind']) for c in CASES},
              teacher_states_indexed=len(teacher), comparisons=comparisons,
              runs={f'{m}|{c}': r for (m, c), r in runs.items()},
              production_fingerprint_unchanged=before == after, lua_errors=v.native.get_lua_error_count())
(OUT / 'result.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
print('DONE', json.dumps(dict(free_runs=len(runs), production_unchanged=before == after, lua_errors=report['lua_errors'],
                              reproduced=all(r['verdict_reproduced'] for r in runs.values()))), flush=True)
