import sys,json,gc,collections,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_policy import *
from chain_counter_cases import run_counter
S=P/'experiments/endboard-20261108';OUT=S/'short-regression';OUT.mkdir(exist_ok=False)
train=json.loads((S/'training/result.json').read_text());assert train['status']=='complete' and not train['holdout_used_for_training_or_selection']
selected=Path(train['selected_checkpoint']);manifest=json.loads((B/'dataset/split-manifest.json').read_text())
assert sha(S/'dataset/split-manifest.json')==train['dataset_manifest_sha256']
old_cases=json.loads((P/'experiments/combo-benchmark-20261107/evaluation-v1/cases.json').read_text())
specs={'diagnostic24':old_cases,'holdout':manifest['combo_cases']['holdout']}
def combo(case,policy):
    env,obs,info,_=c.create_env(case);policy.reset();trace=[];reason='decision_limit';repeated=collections.Counter()
    try:
        for step in range(128):
            s=c.native.benchmark_snapshot();collect_snapshot_checks(obs,info,s)
            g=c.goal_check(case,s)
            if g['pass_']:reason='goal_reached';break
            if s['done']:reason='engine_ended';break
            if s['turn_player']!=0 or s['phase'] not in [1,2,4]:reason='left_main_phase_1';break
            assert int(info['raw_num_options'][0])<=128
            digest=hashlib.sha256(b''.join(np.asarray(obs[k]).tobytes() for k in ['cards_','global_','actions_','h_actions_'])).hexdigest()
            repeated[digest]+=1
            a=policy.act(obs,info) if s['to_play']==0 else c.passive(s)
            assert 0<=a<len(s['actions'])
            trace.append(dict(step=step,player=s['to_play'],message=s['message'],selected=s['actions'][a],
                              public_chain=s['public_chain'],chain_features=chain_features(info)[0].tolist()))
            obs,_,term,trunc,info=env.step(np.array([a],dtype=np.int32))
            if term[0] or trunc[0]:reason='engine_ended_or_limit';break
        final=c.native.benchmark_snapshot();g=c.goal_check(case,final)
        if g['pass_']:reason='goal_reached'
        assert c.native.get_lua_error_count()==0
        return dict(id=case['case_id'],family=case['id'],passed=g['pass_'],reason=reason,steps=len(trace),
                    goal=g,trace=trace,final=c.snapshot_compact(final),max_repeated_observation=max(repeated.values(),default=0))
    finally:env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
settings=[('baseline',None,None,False),('bc_with_chain',selected/'base.flax_model',selected/'chain-head.flax_model',True),
          ('same_bc_without_head',selected/'base.flax_model',selected/'chain-head.flax_model',False)]
summaries={};start=time.perf_counter()
for label,base,head,enabled in settings:
    policy=Policy(base,head,enabled);stats={}
    for split,cases in specs.items():
        results=[]
        for case in cases:
            d=combo(case,policy);results.append(d)
            out=OUT/label/split;out.mkdir(parents=True,exist_ok=True)
            (out/(case['case_id']+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        stats[split]=dict(passed=sum(d['passed'] for d in results),total=len(results),
                         families={family:dict(passed=sum(d['passed'] for d in results if d['family']==family),
                            total=sum(d['family']==family for d in results)) for family in sorted({d['family'] for d in results})})
        print('COMBO_RESULT',label,split,stats[split]['passed'],len(results),flush=True)
    counter=[]
    for case in manifest['counter_cases']['holdout']:
        d=run_counter(case,policy=policy);d.pop('examples');counter.append(d)
        out=OUT/label/'chain_holdout';out.mkdir(parents=True,exist_ok=True)
        (out/(case['id']+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    stats['chain_holdout']=dict(passed=sum(d['passed'] for d in counter),total=len(counter),
            own=dict(passed=sum(d['passed'] for d in counter if d['expected']=='pass'),total=sum(d['expected']=='pass' for d in counter)),
            opponent=dict(passed=sum(d['passed'] for d in counter if d['expected']=='ash'),total=sum(d['expected']=='ash' for d in counter)),
            cases={kind:dict(passed=sum(d['passed'] for d in counter if d['kind']==kind),total=sum(d['kind']==kind for d in counter)) for kind in sorted({d['kind'] for d in counter})})
    summaries[label]=stats
    (OUT/'progress.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('CHAIN_RESULT',label,json.dumps(stats['chain_holdout']),flush=True)
assert summaries['baseline']['diagnostic24']['passed']==3
(OUT/'summary.json').write_text(json.dumps(dict(status='complete',results=summaries,
          elapsed_seconds=time.perf_counter()-start,selected_checkpoint=str(selected),
          selected_before_holdout=True,goal_definition_unchanged=True,native_lua_errors=c.native.get_lua_error_count(),
          warning='same_bc_without_head is inference-time removal of the jointly-trained head, not separately trained BC ablation.'),indent=2)+'\n')
print('EVALUATION_COMPLETE',json.dumps(summaries),flush=True)
