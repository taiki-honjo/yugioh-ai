"""Offline summary of the four Maxx C / Droll conditions (no engine)."""
import json, sqlite3, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); O = P / 'experiments/handtrap-continuing-20261109'
sys.path.insert(0, str(O / 'code')); sys.path.insert(0, str(P / 'scripts'))
import handtrap_analysis as H
import continuous_learning as C
db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
nm = lambda c: (db.execute('select name from texts where id=?', (int(c),)).fetchone() or ['?'])[0]
names = lambda cs: [nm(c) for c in cs]
docs = {c: json.loads((O / 'conditions' / f'{c}.json').read_text()) for c in ['none', 'g', 'droll', 'both']}
replays = {c: json.loads((O / 'replays' / f'{c}.json').read_text()) for c in docs}
non_pass = lambda t: [H.A.normalized(H.A.meaning(r['selected'])) for r in t if r['snapshot']['to_play'] == 0 and r['selected']['act'] != 'Cancel']
base = docs['none']
rows = {}
for c, d in docs.items():
    s = d['summary']
    rows[c] = dict(
        engine_run=d['engine_run'], replay=dict(engine_run=replays[c]['engine_run'], reproduced=f"{replays[c]['reproduced_steps']}/{replays[c]['of']}",
                                               start_matches=replays[c]['start_matches'], end_matches=replays[c]['end_matches'], mismatch=replays[c]['mismatch']),
        g_activation_step=s['g_activation_step'], droll_activation_step=s['droll_activation_step'], droll_in_effect_from_step=s['droll_in_effect_from_step'],
        deploying=dict(monster_entries=s['p0_monster_entries'], entries_after_droll=s['p0_monster_entries_after_droll'],
                       adds_from_deck=s['p0_adds_from_hidden'], add_events=[dict(step=e['step'], cards=names(sum(([k] * v for k, v in e['cards'].items()), [])))
                                                                           for e in s['p0_add_events']],
                       adds_after_droll=s['p0_adds_after_droll'], decisions=s['deploying_decisions'], non_pass_actions=len(non_pass(d['trace'])),
                       non_pass_identical_to_none=non_pass(d['trace']) == non_pass(base['trace']),
                       end=dict(monsters=names(s['p0_end']['monsters']), spell_trap=names(s['p0_end']['spell_trap']), hand_count=s['p0_end']['hand_count']),
                       script_blocked=None if not d['script_blocked'] else dict(step=d['script_blocked']['step'], message=d['script_blocked']['message'],
                                                                                next_plan_step=d['script_blocked']['next_plan_steps'][:2])),
        handtrap_side=dict(gained_from_deck=s['p1_gained_total'], gained_after_droll=s['p1_gained_after_droll'],
                           gain_events=[dict(step=e['step'], cards=names(sum(([k] * v for k, v in e['cards'].items()), []))) for e in s['p1_gain_events']],
                           hand_start=names(s['p1_hand_start']), hand_end=names(s['p1_hand_end']), hand_end_count=len(s['p1_hand_end'])),
        failure=d['failure'], lua_errors=d['lua_errors'])
rows['droll']['deploying']['non_pass_identical_to_both'] = non_pass(docs['droll']['trace']) == non_pass(docs['both']['trace'])
# Droll's direct restriction: which deploying-side activations offered in the baseline right after Strelitzia were missing
def offered_after_strel(d):
    i = next(i for i, r in enumerate(d['trace']) if r['snapshot']['to_play'] == 0 and r['selected']['act'] == 'SpSummon' and r['selected']['code'] == 42302563)
    nxt = next(r for r in d['trace'][i + 1:] if r['snapshot']['to_play'] == 0 and any(a['act'] == 'Activate' for a in r['snapshot']['actions']))
    return nxt['step'], sorted(set((nm(a['code']), a['spec'][:1]) for a in nxt['snapshot']['actions'] if a['act'] == 'Activate'))
evidence = {c: offered_after_strel(docs[c]) for c in ['none', 'droll', 'both']}
summary = dict(created_jst=C.now(), case='elf-rukina-regina-g00-p0 (train), seed 151080', engine_runs=dict(used=8, limit=8),
               setting='player 0 deploys with the fixed teacher route in turn 1; player 1 (standard fixed hand) activates Maxx C / Droll at the first legal window per condition',
               conditions=rows,
               droll_direct_restriction=dict(
                   description='first deploying-side window offering activations right after Strelitzia was special summoned: graveyard effects offered',
                   offered={c: dict(step=v[0], activations=[f'{a}({loc})' for a, loc in v[1]]) for c, v in evidence.items()}),
               not_established=[
                   'that Droll reduced the final board: after the block the script ends its turn (fallback); Strelitzia (field effect) and a normal summon were still legal',
                   'the value of the cards Maxx C drew (next-turn follow-up not measured)', 'any change of opponent decisions caused by Maxx C (fixed script does not react)',
                   'the order Droll -> Maxx C, other activation timings, other hands/seeds/scripts', 'effects on win/loss'])
C.atomic_json(O / 'summary.json', summary)
print(json.dumps({c: dict(entries=r['deploying']['monster_entries'], adds=r['deploying']['adds_from_deck'], adds_after_droll=r['deploying']['adds_after_droll'],
                          gained=r['handtrap_side']['gained_from_deck'], gained_after_droll=r['handtrap_side']['gained_after_droll'], hand_end=r['handtrap_side']['hand_end_count'],
                          end=r['deploying']['end'], identical=r['deploying']['non_pass_identical_to_none'], blocked=r['deploying']['script_blocked'] and r['deploying']['script_blocked']['step'])
                  for c, r in rows.items()}, ensure_ascii=False, indent=1))
print(json.dumps(summary['droll_direct_restriction']['offered'], ensure_ascii=False))
