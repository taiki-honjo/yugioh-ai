"""Offline summary of the two-landing x two-opponent comparison (no engine)."""
import collections, json, sqlite3, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); O = P / 'experiments/landing-compare-20261109'
sys.path.insert(0, str(O / 'code')); sys.path.insert(0, str(P / 'scripts'))
import compare_policy as CP
import continuous_learning as C
import importlib.util
spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
nm = lambda c: (db.execute('select name from texts where id=?', (int(c),)).fetchone() or ['?'])[0] if c else None
Z = lambda s, c, l: collections.Counter(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == l)
names = lambda counter: sorted(nm(c) for c in counter.elements())
conds = ['P1-search', 'B2B-search', 'P1-synchro', 'B2B-synchro']
docs = {c: json.loads((O / 'conditions' / f'{c}.json').read_text()) for c in conds}
reps = {c: json.loads((O / 'replays' / f'{c}.json').read_text()) for c in conds}


def state(s):
    return dict(lp=dict(ours=s['lp0'], opponent=s['lp1']),
                ours=dict(monsters=[f"{nm(x['code'])}(L{x['level']})" for x in s['cards'] if x['controller'] == 0 and x['location'] == 4],
                          spell_trap=names(Z(s, 0, 8)), hand=names(Z(s, 0, 2)), graveyard=names(Z(s, 0, 16)), banished=names(Z(s, 0, 32)),
                          extra_count=sum(Z(s, 0, 64).values())),
                opponent=dict(monsters=[f"{nm(x['code'])}(L{x['level']},{'face-down' if not x['position'] & 5 else 'face-up'})" for x in s['cards']
                                        if x['controller'] == 1 and x['location'] == 4],
                              hand_count=sum(Z(s, 1, 2).values()), graveyard=names(Z(s, 1, 16)), banished=names(Z(s, 1, 32))))


def actions(rows, who, turn):
    out = []
    for r in rows:
        s, a = r['snapshot'], r['selected']
        if s['turn_count'] != turn or (s['to_play'] != who) or a['act'] in ('Cancel',) or (a['act'] == 'None' and not a['code'] and a.get('phase') in (None, 'None')):
            continue
        if a['act'] == 'None' and s['message'] in ('select_place', 'select_position'):
            continue
        label = a['act'] if a['act'] != 'None' else f"choose({s['message']})"
        eff = A.effect_index(a)
        out.append(dict(step=r['step'], source=r['source'], action=f"{label}:{nm(a['code']) or a.get('phase')}" + (f"#{eff}" if eff is not None else ''),
                        chain=[nm(x['code']) for x in s['public_chain']]))
    return out


summary = dict(created_jst=C.now(), engine_runs=dict(used=8, limit=8), conditions={})
for c, d in docs.items():
    rows = [r for r in d['trace'] if r['source'] != 'landing-record']
    t = d['time_points']
    # our turn-2 response options (distinct card/effect seen as legal) and what was used
    offered, used = collections.Counter(), []
    for r in rows:
        s = r['snapshot']
        if s['turn_count'] == 2 and s['to_play'] == 0:
            for a in s['actions']:
                if a['act'] == 'Activate' and a['code']:
                    offered[f"{nm(a['code'])}#{A.effect_index(a)}"] += 0 or 1
            if r['selected']['act'] == 'Activate':
                used.append(dict(step=r['step'], card=nm(r['selected']['code']), chain=[nm(x['code']) for x in s['public_chain']]))
    search_window = next((r for r in rows if r['snapshot']['to_play'] == 0 and len(r['snapshot']['public_chain']) == 1
                          and r['snapshot']['public_chain'][0]['player'] == 1 and r['snapshot']['public_chain'][0]['code'] == CP.RUK), None)
    o1 = A.o1_stages(rows, d['end'])
    blocked = d['opponent_script']['blocked']
    summary['conditions'][c] = dict(
        engine_run=d['engine_run'], replay=dict(engine_run=reps[c]['engine_run'], reproduced=f"{reps[c]['reproduced_steps']}/{reps[c]['of']}", end_matches=reps[c]['end_matches']),
        failure=d['failure'], lua_errors=d['lua_errors'], visibility=dict(checks=d['visibility']['checks'], failures=len(d['visibility']['failures'])),
        T1=state(t['T1']['snapshot']), T2=state(t['T2']['snapshot']), T3=state(t['T3']['snapshot']),
        turn2=dict(our_activation_options_seen=sorted(offered), our_activations=used,
                   our_options_at_search_window=[f"{nm(a['code'])}#{A.effect_index(a)}" for a in search_window['snapshot']['actions'] if a['act'] == 'Activate'] if search_window else None,
                   our_choice_at_search_window=(search_window['selected']['act'] + ':' + str(nm(search_window['selected']['code']))) if search_window else None,
                   first_search=dict(outcome=o1['outcome'], opponent_hand=o1['evidence'].get('opponent_hand')),
                   opponent_script=dict(mode=d['opponent_script']['mode'],
                                        blocked=None if not blocked else dict(step=blocked['step'], next_plan_steps=[f"{x['act']}:{nm(x['code'])}" for x in blocked['next_plan_steps']],
                                                                              legal=[f"{a}:{nm(code)}" for a, code, _, _ in blocked['legal'] if code])),
                   opponent_actions=actions(rows, 1, 2)),
        turn3=dict(our_actions=actions(rows, 0, 3)))
C.atomic_json(O / 'summary.json', summary)
for c, v in summary['conditions'].items():
    print('=====', c, '| replay', v['replay']['reproduced'], '| visibility failures', v['visibility']['failures'])
    for k in ['T1', 'T2', 'T3']:
        x = v[k]; print(f"  {k} LP {x['lp']} ours M {x['ours']['monsters']} S {x['ours']['spell_trap']} H {x['ours']['hand']} | opp M {x['opponent']['monsters']} H#{x['opponent']['hand_count']}")
    print('  search window options', v['turn2']['our_options_at_search_window'], '-> chosen', v['turn2']['our_choice_at_search_window'], '| first search', v['turn2']['first_search']['outcome'])
    print('  opp script', v['turn2']['opponent_script']['mode'], v['turn2']['opponent_script']['blocked'])
