"""Blind free engine rollouts, with model-only actions and no target/plan inputs."""
import argparse,json
from pathlib import Path
from all_openings_common import *
import all_opening_cases as cases
p=argparse.ArgumentParser();p.add_argument('--policy',choices=['before','after'],required=True);p.add_argument('--split',choices=['validation','holdout'],default='holdout');p.add_argument('--checkpoint');args=p.parse_args()
m=json.loads((B/'manifest.json').read_text());spec=m['decks']['ElfnotesKewlTune']
if args.policy=='before':policy=Policy(spec['source_model'],spec['source_head']);checkpoint=spec['source_model']
else:
    chosen=Path(args.checkpoint) if args.checkpoint else Path(json.loads((B/'training/ElfnotesKewlTune/result.json').read_text())['selected_checkpoint'])
    policy=Policy(chosen/'base.flax_model',chosen/'chain-head.flax_model');checkpoint=str(chosen)
suffix=Path(checkpoint).name if args.checkpoint else args.policy
out=B/'evaluation'/args.split/suffix;out.mkdir(parents=True,exist_ok=False);rows=[];seen=set()
for c in spec['cases'][args.split]:
    key=(c['eval_family'],tuple(sorted(c['hand'])),c['seed'])
    if key in seen:continue
    seen.add(key);r,_=v.run(cases.normalize(c),policy=policy);write(out/(c['case_id']+'.json'),r)
    row=dict(case_id=c['case_id'],family=c['eval_family'],passed=r['passed'],board=r['board'],failure=r['failure'],steps=len(r['trace']),
        chosen_variant=r.get('chosen_variant'),recovered_ash_activated=r.get('recovered_ash_activated',False),
        opponent_search_negated=r.get('opponent_search_negated',False),crystal_first_preference_met=r.get('crystal_first_preference_met'),
        normal_summons_diagnostic=r['normal_summons_diagnostic'],lua_errors=r['lua_errors'])
    rows.append(row);print('ROLLOUT',args.policy,args.split,c['eval_family'],c['group_id'],row['passed'],str(row['failure'])[:110],flush=True)
families={f:dict(passed=sum(r['passed'] for r in rows if r['family']==f),total=sum(r['family']==f for r in rows),
    variants={str(x):sum(r['chosen_variant']==x for r in rows if r['family']==f) for x in set(r['chosen_variant'] for r in rows if r['family']==f)}) for f in sorted({r['family'] for r in rows})}
summary=dict(policy=args.policy,split=args.split,checkpoint=checkpoint,passed=sum(r['passed'] for r in rows),total=len(rows),families=families,cases=rows,
    lua_errors=v.native.get_lua_error_count(),model_received_goals_or_plans=False,normal_summon_usage_scored=False,
    scope='Four unseen full-hand compositions per evaluation context. Opponent passes during setup, except the six explicit Rukina interruption arms; recovery tests then face one Rukina search. Not matchup win rate.')
write(out/'summary.json',summary);print('EVALUATION_COMPLETE',args.policy,args.split,summary['passed'],summary['total'],flush=True)
