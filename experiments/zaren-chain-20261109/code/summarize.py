"""Offline summary of the four Zaren conditions (no engine)."""
import json, sqlite3, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); O = P / 'experiments/zaren-chain-20261109'
sys.path.insert(0, str(O / 'code')); sys.path.insert(0, str(P / 'scripts'))
import zaren_analysis as Z
import continuous_learning as C
db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
nm = lambda c: (db.execute('select name from texts where id=?', (int(c),)).fetchone() or ['?'])[0]
LOC = {2: 'hand', 4: 'monster zone', 16: 'graveyard'}
conds = ['chain1', 'veiler', 'zaren-link1', 'zaren-link2']
docs = {c: json.loads((O / 'conditions' / f'{c}.json').read_text()) for c in conds}
reps = {c: json.loads((O / 'replays' / f'{c}.json').read_text()) for c in conds}
def target_unambiguous(d):
    row = next((r for r in d['trace'] if r['source'] == 'veiler'), None)
    if row is None:
        return None
    return sorted(c['code'] for c in row['snapshot']['cards'] if c['controller'] == 1 and c['location'] == 4 and c['position'] & 5) == [Z.RUK]
out = {}
for c, d in docs.items():
    o = d['analysis']['outcome']; st = d['script_state']
    need_option = c in ('zaren-link1', 'zaren-link2')
    constructed = d['failure'] is None and d['mismatch'] is None and (c == 'chain1' or (st['veiler'] is not None and target_unambiguous(d))) \
        and (not need_option or (st['zaren'] is not None and st['option'] and st['option']['constructed']))
    out[c] = dict(engine_run=d['engine_run'], harness_sha256=d['versions']['zaren_sha256'], own_end_matches_record=d['own_end_matches_record'],
                  constructed=constructed, constructed_flag_in_run=d['constructed'],
                  veiler_target='Rukina (only face-up opponent effect monster; the env resolved the one-candidate prompt)' if st['veiler'] is not None else None,
                  zaren_legal_by_chain_length={k: v for k, v in d['analysis']['zaren_legality'].items()},
                  zaren_option=d['analysis']['zaren_option'],
                  chain=[f"{'opp' if x['player'] == 1 else 'own'}:{nm(x['code'])}" for x in o['longest_chain']],
                  search=o['search'], opponent_hand_change=o['search_evidence'].get('opponent_hand'),
                  rukina_after=[LOC.get(x, x) for x in o['rukina_locations_after']],
                  our_veiler_after=[LOC.get(x, x) for x in o['our_veiler_locations_after']], our_zaren_after=[LOC.get(x, x) for x in o['our_zaren_locations_after']],
                  replay=dict(engine_run=reps[c]['engine_run'], reproduced=f"{reps[c]['reproduced_steps']}/{reps[c]['of']}", end_matches=reps[c]['end_matches'],
                              mismatch=reps[c]['mismatch']),
                  lua_errors=d['lua_errors'])
summary = dict(created_jst=C.now(), position='P1 teacher route (train cue-reco-jj-zaren-g00-p0), own turn replayed; opponent O1; our side passes except the scripted Veiler/Zaren',
               engine_runs=dict(used=8, limit=8), conditions=out,
               harness_note='zaren.py changed after run 2 only in the construction check (one-candidate target prompts are resolved by the env); '
                            'runs 1-2 store the stricter in-run flag, this summary applies the corrected rule to all four',
               zaren_options='description 2 = negate/destroy the link Zaren responds to (link 2, our Veiler); description 3 = the link it chained to (link 1, Rukina)',
               not_established=['that Zaren alone stops the search in zaren-link1 (Veiler on link 2 also stops it, see veiler)',
                                'whether the destroy part of option 2 does anything when link 2 is our Veiler already in the graveyard (no observable change)',
                                'chains built by the opponent (e.g. the opponent chaining to our effect), other positions, model use of Zaren'])
C.atomic_json(O / 'summary.json', summary)
print(json.dumps({c: {k: v[k] for k in ['constructed', 'constructed_flag_in_run', 'zaren_option', 'chain', 'search', 'rukina_after', 'our_veiler_after', 'our_zaren_after', 'replay']}
                  for c, v in out.items()}, ensure_ascii=False, indent=1))
print({c: v['zaren_legal_by_chain_length'] for c, v in out.items()})
