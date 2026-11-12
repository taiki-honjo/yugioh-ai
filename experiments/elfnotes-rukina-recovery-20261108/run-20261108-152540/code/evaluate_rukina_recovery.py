"""Unseen-hand free rollouts: source and selected own policy, without teacher actions."""
import argparse,collections
from elfnotes_recovery_common import *
p=argparse.ArgumentParser();p.add_argument('--policy',choices=['before','after'],required=True);args=p.parse_args()
manifest=json.loads((B/'manifest.json').read_text());deck=manifest['decks']['ElfnotesKewlTune']
if args.policy=='before':policy=Policy(deck['source_model'],deck['source_head'])
else:
    train=json.loads((B/'training/ElfnotesKewlTune/result.json').read_text());selected=Path(train['selected_checkpoint'])
    policy=Policy(selected/'base.flax_model',selected/'chain-head.flax_model')
out=B/'evaluation'/args.policy;out.mkdir(parents=True,exist_ok=False);rows=[]
for case in deck['cases']['holdout']:
    result,_=v.run(case,policy=policy)
    write(out/(case['case_id']+'.json'),result)
    row={k:result[k] for k in ['passed','board','interruption_verified','turn_completed','failure','normal_summons_diagnostic','interruption_injected','interruption_countered','lua_errors']}
    row.update(case_id=case['case_id'],family=case['family'],steps=len(result['trace']))
    rows.append(row)
    print('ROLLOUT',args.policy,case['case_id'],row['passed'],row['failure'],row['steps'],flush=True)
families={f:dict(passed=sum(r['passed'] for r in rows if r['family']==f),total=sum(r['family']==f for r in rows)) for f in sorted({r['family'] for r in rows})}
summary=dict(policy=args.policy,passed=sum(r['passed'] for r in rows),total=len(rows),families=families,cases=rows,
    lua_errors=v.native.get_lua_error_count(),model_received_goals_or_plans=False,normal_summon_usage_scored=False,
    scope='Four unseen full-hand multisets for each of six Rukina branches and two existing openings. Fixed targeted single interruption; not general matchup win rate.')
write(out/'summary.json',summary);print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)
