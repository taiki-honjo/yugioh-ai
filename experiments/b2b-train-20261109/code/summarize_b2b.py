"""Offline summary of the B2B trial: before/after comparison on g03/g05, training records, retention (no engine)."""
import collections
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-train-20261109'
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402
spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
nm = lambda c: (db.execute('select name from texts where id=?', (int(c),)).fetchone() or ['?'])[0] if c else None
B2B, RUK = 65961304, 13597785
Z = lambda s, c, l: collections.Counter(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == l)
names = lambda counter: sorted(nm(c) for c in counter.elements())


def state(s):
    if not s:
        return None
    return dict(lp=dict(ours=s['lp0'], opponent=s['lp1']), turn=s['turn_count'],
                ours=dict(monsters=[f"{nm(x['code'])}(L{x['level']})" for x in s['cards'] if x['controller'] == 0 and x['location'] == 4],
                          spell_trap=names(Z(s, 0, 8)), hand=names(Z(s, 0, 2)), graveyard=names(Z(s, 0, 16)), banished=names(Z(s, 0, 32))),
                opponent=dict(monsters=[f"{nm(x['code'])}(L{x['level']})" for x in s['cards'] if x['controller'] == 1 and x['location'] == 4],
                              hand_count=sum(Z(s, 1, 2).values()), graveyard=names(Z(s, 1, 16))))


def our_rows(d, turn):
    return [r for r in d['trace'] if r['source'] != 'landing-record' and r['snapshot']['to_play'] == 0 and r['snapshot']['turn_count'] == turn]


def b2b_use(d):
    rows = our_rows(d, 2)
    opportunities = [dict(step=r['step'], message=r['snapshot']['message'], chain=[nm(c['code']) for c in r['snapshot']['public_chain']],
                          chosen=f"{r['selected']['act']}:{nm(r['selected']['code'])}" if r['selected']['code'] else r['selected']['act'])
                     for r in rows if any(a['act'] == 'Activate' and a['code'] == B2B and A.effect_index(a) == 0 for a in r['snapshot']['actions'])]
    act = next((r for r in rows if r['selected']['act'] == 'Activate' and r['selected']['code'] == B2B and A.effect_index(r['selected']) == 0), None)
    out = dict(opportunities=opportunities, activated=act is not None)
    if act:
        after = [r for r in rows if r['step'] > act['step']]
        target = next((r for r in after if r['snapshot']['message'] == 'select_card'), None)
        option = next((r for r in after if r['snapshot']['message'] == 'select_option'), None)
        yesno = next((r for r in after if r['snapshot']['message'] == 'select_yesno' and any(a['code'] == B2B for a in r['snapshot']['actions'])), None)
        resolved = next((r['snapshot'] for r in d['trace'] if r['step'] > act['step'] and r['snapshot']['chain_size'] == 0), d['end'])
        before = act['snapshot']
        out.update(step=act['step'], chain=[nm(c['code']) for c in before['public_chain']],
                   target=dict(offered=[nm(a['code']) for a in target['snapshot']['actions']], chosen=nm(target['selected']['code'])) if target else 'no prompt (single candidate)',
                   option={1190: 'to hand', 1152: 'special summon'}.get(option['selected'].get('effect')) if option else 'no prompt',
                   s_summon=('yes' if yesno['selected']['act'] == 'Activate' else 'no') if yesno else 'not offered',
                   our_change=dict(field=dict(gained=names(Z(resolved, 0, 4) - Z(before, 0, 4)), lost=names(Z(before, 0, 4) - Z(resolved, 0, 4))),
                                   hand=dict(gained=names(Z(resolved, 0, 2) - Z(before, 0, 2)), lost=names(Z(before, 0, 2) - Z(resolved, 0, 2))),
                                   graveyard=dict(gained=names(Z(resolved, 0, 16) - Z(before, 0, 16)), lost=names(Z(before, 0, 16) - Z(resolved, 0, 16)))))
    return out


def actions(d, turn):
    out = []
    for r in our_rows(d, turn):
        a, s = r['selected'], r['snapshot']
        if a['act'] in ('Cancel',) or s['message'] in ('select_place', 'select_position'):
            continue
        if a['act'] == 'None' and not a['code'] and a.get('phase') in (None, 'None'):
            continue
        eff = A.effect_index(a)
        out.append(f"{r['step']}:{a['act'] if a['act'] != 'None' else 'choose'}:{nm(a['code']) or a.get('phase')}" + (f"#{eff}" if eff is not None else ''))
    return out


def battle(d):
    rows = [r for r in our_rows(d, 3) if r['snapshot']['message'] == 'select_battlecmd']
    cmds = [dict(step=r['step'], offered=[f"{a['act']}:{nm(a['code'])}" for a in r['snapshot']['actions'] if a['act'] in ('Attack', 'DirectAttack')],
                 chosen=f"{r['selected']['act']}:{nm(r['selected']['code'])}" if r['selected']['act'] in ('Attack', 'DirectAttack') else r['selected'].get('phase'),
                 lp=(r['snapshot']['lp0'], r['snapshot']['lp1'])) for r in rows]
    attacks = collections.Counter(c['chosen'].split(':', 1)[1] for c in cmds if c['chosen'] and c['chosen'].startswith(('Attack', 'DirectAttack')))
    return dict(commands=cmds, attacks_by_monster=dict(attacks), b2b_attack_offered=any('Ｂ２Ｂ' in o for c in cmds for o in c['offered']),
                b2b_attacked=attacks.get(nm(B2B), 0))


def main():
    conds = ['g03-before', 'g03-after', 'g05-before', 'g05-after']
    out = {}
    for c in conds:
        path = OUT / 'conditions' / f'{c}.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rep = json.loads((OUT / 'replays' / f'{c}.json').read_text()) if (OUT / 'replays' / f'{c}.json').exists() else None
        rows = [r for r in d['trace'] if r['source'] != 'landing-record']
        o1 = A.o1_stages(rows, d['end'])
        t = d['time_points']
        out[c] = dict(engine_run=d['engine_run'], replay=rep and dict(engine_run=rep['engine_run'], reproduced=f"{rep['reproduced_steps']}/{rep['of']}", end_matches=rep['end_matches']),
                      our_model=d['our_model'], failure=d['failure'], lua_errors=d['lua_errors'], duel_ended=d['duel_ended'],
                      visibility_failures=len(d['visibility']['failures']), opponent_script=d['opponent_script'],
                      T1=state(t.get('T1', {}).get('snapshot')), T2=state(t.get('T2', {}).get('snapshot')), T3=state(t.get('T3', {}).get('snapshot')),
                      b2b3=b2b_use(d), opponent_search=o1['outcome'], turn2_actions=actions(d, 2), turn3_actions=actions(d, 3), turn3_battle=battle(d))
    training = None
    if (OUT / 'candidate.json').exists():
        cand = C.read_json(OUT / 'candidate.json')
        rows = [json.loads(l) for l in Path(cand['rehearsal_metrics']).read_text().splitlines()]
        mbs = [mb for r in rows for mb in r['minibatches']]
        b2b_mbs = [mb for mb in mbs if mb['sequences_b2b'] > 0]
        mean = lambda xs: sum(xs) / max(len(xs), 1)
        training = dict(candidate=cand['candidate'], base_sha256=cand['base_sha256'], head_sha256=cand['head_sha256'], updates=len(rows),
                        sampled_source_counts=cand['sampled_source_counts'],
                        loss_steps=dict(old=sum(mb['loss_steps_old'] for mb in mbs), b2b=sum(mb['loss_steps_b2b'] for mb in mbs)),
                        ce=dict(old_mean=mean([mb['ce_old'] for mb in mbs]), b2b_mean_over_b2b_minibatches=mean([mb['ce_b2b'] for mb in b2b_mbs]),
                                b2b_first=b2b_mbs[0]['ce_b2b'] if b2b_mbs else None, b2b_last=b2b_mbs[-1]['ce_b2b'] if b2b_mbs else None),
                        weighted_grad_norm=dict(old_mean=mean([mb['weighted_grad_norm_old'] for mb in mbs]),
                                                b2b_mean_over_b2b_minibatches=mean([mb['weighted_grad_norm_b2b'] for mb in b2b_mbs])),
                        ppo_grad_norm_mean=mean([mb['ppo_grad_norm'] for mb in mbs]), training_result=cand['training_result'])
    retention = C.read_json(OUT / 'retention/retention-summary.json') if (OUT / 'retention/retention-summary.json').exists() else None
    summary = dict(created_jst=C.now(), plan_sha256=C.sha(OUT / 'plan.json'), comparisons=out, training=training, retention=retention,
                   engine_runs=len(A.Ledger(OUT / 'engine-runs.jsonl', limit=12).started()))
    C.atomic_json(OUT / 'summary.json', summary)
    for c, v in out.items():
        print('=====', c, '| replay', v['replay'] and v['replay']['reproduced'], '| ended', bool(v['duel_ended']), '| visibility failures', v['visibility_failures'])
        for k in ['T1', 'T2', 'T3']:
            x = v[k]
            if x:
                print(f"  {k} LP {x['lp']} ours M {x['ours']['monsters']} H {x['ours']['hand']} | opp M {x['opponent']['monsters']}")
        b = v['b2b3']; print('  B2B3 opportunities', len(b['opportunities']), '| activated', b['activated'], {k: b.get(k) for k in ['target', 'option', 's_summon', 'our_change'] if k in b})
        print('  opponent search', v['opponent_search'], '| turn2', v['turn2_actions'][:12])
        print('  turn3', v['turn3_actions'][:24]); print('  battle', v['turn3_battle']['attacks_by_monster'], 'B2B offered', v['turn3_battle']['b2b_attack_offered'])
    if training:
        print('TRAINING', json.dumps({k: training[k] for k in ['sampled_source_counts', 'loss_steps', 'ce', 'weighted_grad_norm', 'ppo_grad_norm_mean']}, ensure_ascii=False))
    if retention:
        print('RETENTION', json.dumps({k: retention[k] for k in ['candidate_passed', 'total', 'lost_case_ids', 'gained_case_ids', 'all_baseline_cases_retained']}))


if __name__ == '__main__':
    main()
