import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from endboard_witness import *
tr=json.loads((S/'training/result.json').read_text());assert tr['status']=='complete' and not tr['holdout_used_for_training_or_selection']
selected=Path(tr['selected_checkpoint']);before=Path(json.loads((B/'training/result.json').read_text())['selected_checkpoint'])
assert sha(before/'base.flax_model')==tr['source_model_sha256'] and sha(before/'chain-head.flax_model')==tr['source_head_sha256']
manifest=json.loads((S/'dataset/split-manifest.json').read_text());assert sha(S/'dataset/split-manifest.json')==tr['dataset_manifest_sha256']
out=S/'evaluation';out.mkdir(exist_ok=False);summary={};start=time.perf_counter()
for label,path in [('before',before),('after',selected)]:
    policy=Policy(path/'base.flax_model',path/'chain-head.flax_model');results=[]
    for case in manifest['new_cases']['holdout']:
        d=run(case,policy,case['arm']);d['end_goal']=end_goal(case,d['final'],d['normal_summons_used'],d['ash_injected'])
        p=out/label;p.mkdir(exist_ok=True);write(p/(case['case_id']+'.json'),d);results.append(d)
        write(out/'progress.json',dict(policy=label,completed=len(results),total=36))
    stats={}
    for arm in ['pass','ash_once']:
        rs=[d for d in results if d['arm']==arm]
        stats[arm]=dict(passed=sum(d['end_goal']['passed'] for d in rs),cases=len(rs),
            turn_completed=sum(d['turn_completed'] for d in rs),hand_preserved=sum(d['end_goal']['protected'] for d in rs),
            normal_preserved=sum(d['end_goal']['normal_preserved'] for d in rs),ash_injected=sum(d['ash_injected'] for d in rs),
            families={f:dict(passed=sum(d['end_goal']['passed'] for d in rs if d['case']['id']==f),
                cases=sum(d['case']['id']==f for d in rs)) for f in ['rukina','dina','regina']})
    summary[label]=stats;print('RESULT',label,json.dumps(stats),flush=True)
write(out/'summary.json',dict(status='complete',results=summary,holdout_used_for_selection=False,selected_checkpoint=str(selected),
    goal_scope=manifest['goals'],expert_optimality_claimed=False,elapsed_seconds=time.perf_counter()-start,
    lua_errors=c.native.get_lua_error_count(),resources=c.native.get_env_resource_stats()))
