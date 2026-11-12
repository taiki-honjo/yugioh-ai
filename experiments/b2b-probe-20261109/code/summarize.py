"""Offline summary of the six B2B conditions (no engine)."""
import collections, json, sqlite3, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); O = P / 'experiments/b2b-probe-20261109'
sys.path.insert(0, str(O / 'code')); sys.path.insert(0, str(P / 'scripts'))
import b2b_policy as BP
import continuous_learning as C
import importlib.util
spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
nm = lambda c: (db.execute('select name from texts where id=?', (int(c),)).fetchone() or ['?'])[0] if c else None
Z = lambda s, c, l: collections.Counter(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == l)
names = lambda counter: sorted(nm(c) for c in counter.elements())
docs = {c: json.loads((O / 'conditions' / f'{c}.json').read_text()) for c in BP.CONDITIONS}
reps = {c: json.loads((O / 'replays' / f'{c}.json').read_text()) for c in BP.CONDITIONS}
position = json.loads((P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json').read_text())

def rows_where(d, pred): return [r for r in d['trace'] if pred(r)]
def by_label(d, label): return [r for r in d['trace'] if r['source'] == label]
def zones(s, c=0): return {k: names(Z(s, c, l)) for k, l in [('field', 4), ('spell_trap', 8), ('hand', 2), ('graveyard', 16), ('banished', 32)]}

out = {}
for c, d in docs.items():
    st = d['script_state']; tr = d['trace']
    item = dict(engine_run=d['engine_run'], failure=d['failure'], mismatch=d['mismatch'], lua_errors=d['lua_errors'],
                own_end_matches_record=d['own_end_matches_record'],
                replay=dict(engine_run=reps[c]['engine_run'], reproduced=f"{reps[c]['reproduced_steps']}/{reps[c]['of']}", end_matches=reps[c]['end_matches']))
    # route to B2B (opponent draw phase)
    if st['remix'] is not None:
        remix_rows = [r for r in tr if r['step'] >= st['remix'] and r['snapshot']['turn_count'] == 2 and r['snapshot']['phase'] == 1]
        before = next(r for r in tr if r['step'] == st['remix'])['snapshot']
        after = next(r for r in tr if r['step'] > st['remix'] and r['snapshot']['chain_size'] == 0 and r['snapshot']['to_play'] in (0, 1)
                     and Z(r['snapshot'], 0, 4)[BP.B2B])['snapshot']
        item['route'] = dict(remix_step=st['remix'], phase='opponent draw phase', picks=[nm(r['selected']['code']) for r in by_label(d, 'remix-pick')],
                             synchro_yes=bool(by_label(d, 'remix-synchro-yes')),
                             field_before=names(Z(before, 0, 4)), field_after=names(Z(after, 0, 4)),
                             hand_gained=names(Z(after, 0, 2) - Z(before, 0, 2)), graveyard_gained=names(Z(after, 0, 16) - Z(before, 0, 16)),
                             b2b_monster_and_material_prompts='none shown (one candidate each; resolved by the env)')
    # B2B (3)
    window = next((r for r in tr if r['snapshot']['to_play'] == 0 and r['snapshot']['public_chain'][-1:] and r['snapshot']['public_chain'][-1]['player'] == 1
                   and r['snapshot']['public_chain'][-1]['code'] == BP.RUK and len(r['snapshot']['public_chain']) == 1), None)
    if window:
        item['b2b3_legal_at_search_window'] = any(a['act'] == 'Activate' and a['code'] == BP.B2B and BP.effect_index(a) == 0 for a in window['snapshot']['actions'])
    if st['b2b']:
        tgt = by_label(d, 'b2b-pick')[0]
        opt = by_label(d, 'b2b-option')
        yes = by_label(d, 'b2b-synchro-yes'); no = [e for e in st['log'] if e['label'] == 'b2b-synchro-no']
        resolved = next(r for r in tr if r['step'] > st['b2b']['step'] and r['snapshot']['chain_size'] == 0)['snapshot']
        act_snap = next(r for r in tr if r['step'] == st['b2b']['step'])['snapshot']
        synchro = dict(prompt_shown=bool(yes or no), chosen='yes' if yes else 'no' if no else None)
        if yes:
            pick_rows = by_label(d, 'b2b-synchro-pick')
            synchro.update(candidates=[nm(a['code']) for a in pick_rows[0]['snapshot']['actions']], monster=nm(pick_rows[0]['selected']['code']),
                           material_prompts=[dict(offered=[nm(a['code']) + '@' + a['spec'] for a in r['snapshot']['actions']], chosen=nm(r['selected']['code']))
                                             for r in pick_rows[1:]])
        item['b2b3'] = dict(step=st['b2b']['step'], chain_link=2, targets_offered=[nm(a['code']) for a in tgt['snapshot']['actions']],
                            target=nm(tgt['selected']['code']), options_offered=st['b2b'].get('option_offered'),
                            option=opt[0]['selected']['effect'] if opt else None, synchro=synchro,
                            our_change=dict(field=dict(gained=names(Z(resolved, 0, 4) - Z(act_snap, 0, 4)), lost=names(Z(act_snap, 0, 4) - Z(resolved, 0, 4))),
                                            hand=dict(gained=names(Z(resolved, 0, 2) - Z(act_snap, 0, 2)), lost=names(Z(act_snap, 0, 2) - Z(resolved, 0, 2))),
                                            graveyard=dict(gained=names(Z(resolved, 0, 16) - Z(act_snap, 0, 16)), lost=names(Z(act_snap, 0, 16) - Z(resolved, 0, 16)))))
        later = [r for r in tr if r['step'] > st['b2b']['step'] and r['snapshot']['to_play'] == 0 and r['snapshot']['turn_count'] == 2]
        first_after = next((r for r in later if r['snapshot']['chain_size'] == 0), None)
        item['newly_activatable_after_chain'] = dict(step=first_after['step'], activations=[f"{nm(a['code'])}({A.effect_index(a)})@{a['spec']}"
                                                     for a in first_after['snapshot']['actions'] if a['act'] == 'Activate']) if first_after else None
    o1 = A.o1_stages(tr, d['end'])
    item['opponent_search'] = dict(outcome=o1['outcome'], opponent_hand=o1['evidence'].get('opponent_hand'))
    ruk = [(r['step'], x['level']) for r in tr if r['snapshot']['turn_count'] >= 2 for x in r['snapshot']['cards']
           if x['controller'] == 1 and x['code'] == BP.RUK and x['location'] == 4]
    item['rukina_level_observed'] = sorted(set(l for _, l in ruk))
    end2 = next((r['snapshot'] for r in tr if r['snapshot']['turn_count'] >= 3), d['end'])  # start of our turn 3 (or the end)
    item['our_resources_at_turn3_start'] = zones(end2)
    if c.startswith('A'):
        bc = [r for r in tr if r['snapshot']['message'] == 'select_battlecmd' and r['snapshot']['to_play'] == 0]
        item['battle'] = dict(
            commands=[dict(step=r['step'], offered=[f"{a['act']}:{nm(a['code'])}" for a in r['snapshot']['actions'] if a['act'] in ('Attack', 'DirectAttack')],
                           chosen=f"{r['selected']['act']}:{nm(r['selected']['code'])}" if r['selected']['act'] in ('Attack', 'DirectAttack') else r['selected'].get('phase'),
                           lp_opponent=r['snapshot']['lp1'], opponent_field=names(Z(r['snapshot'], 1, 4))) for r in bc],
            zaren_attacks_executed=len(st['attacks']), lp_opponent_end=d['end']['lp1'], lp_ours_end=d['end']['lp0'],
            opponent_field_end=names(Z(d['end'], 1, 4)), opponent_graveyard_end=names(Z(d['end'], 1, 16)))
    out[c] = item
summary = dict(created_jst=C.now(), position='P1 teacher route (train cue-reco-jj-zaren-g00-p0), own turn replayed; opponent O1 then ends its turn',
               engine_runs=dict(used=12, limit=12), conditions=out,
               not_established=['B2B (3) second and third use in one turn', 'B2B (2) for B2B itself (offered, not executed)',
                                'value of the recovered/summoned cards in later turns, model use of B2B, win/loss',
                                'A0 and A1 boards differ beyond B2B (Remix vs B2B on field, Reco in hand in A1, Rukina level 6 vs 8)'])
C.atomic_json(O / 'summary.json', summary)
print(json.dumps({c: {k: v.get(k) for k in ['b2b3_legal_at_search_window', 'opponent_search', 'rukina_level_observed']} for c, v in out.items()}, ensure_ascii=False))
for c in ['R1', 'R2', 'R3']:
    print(c, json.dumps(out[c]['b2b3'], ensure_ascii=False)[:900]); print('   after:', out[c]['newly_activatable_after_chain'])
print('route', json.dumps(out['R0']['route'], ensure_ascii=False))
for c in ['A0', 'A1']: print(c, json.dumps(out[c]['battle'], ensure_ascii=False))
