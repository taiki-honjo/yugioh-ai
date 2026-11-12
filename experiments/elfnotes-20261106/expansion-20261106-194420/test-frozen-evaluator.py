import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

project=Path('/home/hatk8/projects/yugioh-ai')
spec=importlib.util.spec_from_file_location('evaluated',project/'env/elfnotes/evaluate_frozen_20261106.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def context(factory, model=False):
    c=m.EvalContext.__new__(m.EvalContext)
    c.num_envs=2; c.env_threads=1; c.env_id='YGOPro-v1'; c.deck_name='fixture'
    c.max_options=128; c.n_history_actions=32; c.max_steps=1000; c.timeout=600
    c.EnvPreprocess=lambda env,skip_mask:env
    c.ygoenv=SimpleNamespace(make=lambda **kw:factory())
    c.jax=SimpleNamespace(device_put=lambda x:x); c.jnp=np
    c.agent=SimpleNamespace(init_rnn_state=lambda n:np.zeros((n,1)))
    c._step_model=lambda params,state,obs,active,done:(state,np.zeros(2,dtype=int))
    return c

class TwoLengths:
    def reset(self):
        self.t=0
        return {}, {'to_play':np.zeros(2,dtype=int),'num_options':np.ones(2,dtype=int)}
    def step(self,actions):
        self.t+=1
        done=np.array([self.t==1,self.t==3])
        return {},done.astype(float),done,np.zeros(2,dtype=bool),{
            'to_play':np.zeros(2,dtype=int),'num_options':np.ones(2,dtype=int),
            'win_reason':np.ones(2,dtype=int),'step_time':np.zeros((2,2)),
            'termination_reason':np.zeros(2,dtype=int)}
    def close(self): pass

random={'kind':'builtin','strategy':'random'}
c=context(TwoLengths)
episodes,timing=c._run_single_wave_seat({'kind':'checkpoint'},random,{'params':True},None,0,1,0)
assert [e['env_steps'] for e in episodes]==[1,3]
assert timing['cand_model_calls']==3 and timing['cand_model_rows']==6
assert timing['env_slot_steps_submitted']==6 and timing['env_steps_total']==4
assert timing['cand_decision_steps_total']==4
assert timing['cand_model_sec']==timing['cand_model_batch_wall_sec']
assert timing['env_step_sec']==timing['env_step_batch_wall_sec']

class Terminal(TwoLengths):
    reason=0
    def step(self,actions):
        obs,reward,term,trunc,info=super().step(actions)
        term[:]=True; reward[:]=1
        info['termination_reason'][:]=self.reason
        return obs,reward,term,trunc,info

class Timeout(Terminal): reason=2
class Limit(Terminal): reason=1
for factory,label in [(Timeout,'engine_timeout'),(Limit,'truncated')]:
    c=context(factory)
    rows,_=c._run_single_wave_seat(random,random,None,None,0,1,0)
    assert all(e['abnormal'] and e['outcome']=='excluded' and e['win']==e['loss']==e['draw']==0 for e in rows)
    assert all(e['reason_label']==label for e in rows)
c=context(Terminal); c.max_steps=1
rows,_=c._run_single_wave_seat(random,random,None,None,0,1,0)
assert all(not e['abnormal'] and e['outcome']=='win' for e in rows), 'Natural win at budget boundary must be preserved with native reason'
low,high,se,n=m.paired_bootstrap_ci95([0.5]*16)
assert (low,high,se,n)==(0.5,0.5,0.0,16)

# A clean half of an incomplete/aborted pair must not enter the paired score.
c=context(Terminal); c.load_params=lambda spec:None; c.warmup_seconds=0
def run_wave(*args, candidate_seat, **kwargs):
    rows=[]
    for i in range(2):
        abnormal=(i==0 and candidate_seat==0)
        rows.append(dict(pair_id=f'p{i}',candidate_seat=candidate_seat,abnormal=abnormal,
                         win=0 if abnormal else 1,loss=0,draw=0,candidate_reward=0 if abnormal else 1,
                         candidate_decisions=1,env_steps=1,cand_engine_step_ms=1,
                         truncated=abnormal,engine_timeout=False,timeout_detection_supported=True,
                         reason_label='truncated' if abnormal else 'normal_lp0'))
    return rows,dict(wave_wall_sec=1,cand_model_sec=0,opp_model_sec=0,env_step_sec=0,
                     cand_model_batch_wall_sec=0,opp_model_batch_wall_sec=0,env_step_batch_wall_sec=0,
                     cand_decision_steps_total=2,opp_decision_steps_total=0,env_steps_total=2,
                     cand_model_calls=0,opp_model_calls=0,env_step_calls=1,cand_model_rows=0,
                     opp_model_rows=0,env_slot_steps_submitted=2)
c._run_single_wave_seat=run_wave
s=c.evaluate_matchup('random','random',project,num_episodes=4)
assert s['results']['total_episodes']==4 and s['results']['scored_episodes']==2
assert s['results']['wins']==2 and s['results']['excluded_from_paired_score']==2
assert s['results']['seed_pairs']['scored_pairs']==1

result={'checks':['full_batch_rows_6_vs_useful_decisions_4','timeout_flag_excluded_without_wall_delay',
                  'step_limit_flag_excluded','natural_boundary_win_preserved',
                  'confidence_resamples_seed_pairs','incomplete_pair_excluded'], 'passed':6}
print(json.dumps(result,indent=2))
(project/'experiments/elfnotes-20261106/evaluation-regressions.json').write_text(json.dumps(result,indent=2))
