import sys,json,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_imitation_common import *
case=dict(c.fixture_definitions()[0]);case.update(seed=7100,case_id='rukina_search-hand0')
old=P/'experiments/elfnotes-20261106/expansion-revised-20261106-205114'
model=json.loads((old/'additional-training-result.json').read_text())['trained_checkpoint']
env,obs,info,_=c.create_env(case);agent=c.Learned(obs);agent.reset(model)
import jax
infer=jax.jit(lambda params,state,obs:agent.agent.apply(params,obs,state))
rows=[];bad=None
try:
    for step in range(80):
        before=c.native.benchmark_snapshot();n=collect_snapshot_checks(obs,info,before)
        if before['turn_player']!=0 or before['phase'] not in [1,2,4]:break
        if before['to_play']==0:
            out=infer(agent.params,agent.state,obs);agent.state=out[0]
            logits=np.asarray(out[1])[0];a=int(logits.argmax())
        else:a=c.passive(before);logits=None
        chosen=before['actions'][a]
        record=dict(step=step,player=before['to_play'],message=before['message'],
                    public_chain=before['public_chain'],chain_features=chain_features(info)[0].tolist(),
                    legal_actions=before['actions'],selected=chosen,
                    encoded_selected=np.asarray(obs['actions_'])[0,a].tolist(),
                    logits=logits[:n].tolist() if logits is not None else None)
        obs,_,term,trunc,info=env.step(np.array([a],dtype=np.int32))
        after=c.native.benchmark_snapshot();record['after_public_chain']=after['public_chain'];rows.append(record)
        if chosen['code']==14558127 and chosen['act']=='Activate' and before['to_play']==0:
            assert before['public_chain'][-1]['code']==c.RUK and before['public_chain'][-1]['player']==0
            assert after['public_chain'][-1]['code']==14558127 and after['public_chain'][-1]['player']==0
            bad=record
        if term[0] or trunc[0]:break
    assert bad is not None
    previous=json.loads((P/'experiments/combo-benchmark-20261107/evaluation-v1/model69632-rukina_search-hand0.json').read_text())
    assert [(r['message'],r['selected']) for r in rows]==[(r['message'],r['selected']) for r in previous['trace']]
    result=dict(status='complete',action_card_and_index_checks=True,old_trajectory_reproduced=True,
                no_private_card_id_leak=True,raw_observation_shape_unchanged=True,
                self_ash_reproduced=True,self_ash_step=bad['step'],self_ash_decision=bad,
                model_sha256=sha(model),model_path=model,trace=rows,
                interpretation='No selected-card/index mismatch in this reproduction. Old inputs lack explicit pending-chain owner/card fields; own history and visible board can still provide indirect cues. This does not prove the missing field is the sole cause.')
    (B/'self-ash-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('AUDIT_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['trace','self_ash_decision']}),flush=True)
finally:
    env.close();del env;gc.collect();assert c.native.get_env_resource_stats()['live_duels']==0
