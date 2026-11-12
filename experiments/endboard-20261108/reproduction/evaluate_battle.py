import argparse,sys,json,importlib.util,gc,ctypes,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from chain_policy import *
R=B/'runtime';S=P/'experiments/endboard-20261108'
parser=argparse.ArgumentParser();parser.add_argument('--deck',required=True);parser.add_argument('--policy',choices=['baseline','bc_with_chain'],required=True)
args=parser.parse_args()
meta_base=P/'experiments/meta-opponents-20261107'
meta_run=Path((meta_base/'latest-run.txt').read_text().strip());meta_manifest=json.loads((meta_run/'manifest.json').read_text())
training=json.loads((S/'training/result.json').read_text());selected=Path(training['selected_checkpoint']);before=Path(json.loads((B/'training/result.json').read_text())['selected_checkpoint'])
candidate=str((before if args.policy=='baseline' else selected)/'base.flax_model')
opponent=json.loads((meta_run/'training'/args.deck/'phase2-65536/result.json').read_text())['trained_checkpoint']
out=S/'battle'/args.policy/args.deck;out.mkdir(parents=True,exist_ok=False)
scorer=S/'scoring-evaluator.py'
if not scorer.exists():scorer.write_bytes((P/'scripts/evaluate.py').read_bytes())
spec=importlib.util.spec_from_file_location('frozen_score',scorer);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
c.native.set_benchmark_fixture([])
class Wrap:
    def __init__(self,raw,ctx):self.raw=raw;self.ctx=ctx
    def __getattr__(self,name):return getattr(self.raw,name)
    def reset(self,*a,**k):
        data=self.raw.reset(*a,**k);self.ctx.current_info=data[-1];self.ctx.check_info();return data
    def step(self,*a,**k):
        data=self.raw.step(*a,**k);self.ctx.current_info=data[-1];self.ctx.check_info();return data
class Factory:
    def __init__(self,ctx,native):self.ctx=ctx;self.native=native
    def make(self,**kwargs):
        seat=self.ctx.seat
        if seat is not None:
            kwargs['deck1'],kwargs['deck2']=(('ElfnotesKewlTune',args.deck) if seat==0 else (args.deck,'ElfnotesKewlTune'))
            self.ctx.deck_calls.append(dict(seat=seat,deck1=kwargs['deck1'],deck2=kwargs['deck2'],seed=kwargs['seed']))
        return Wrap(self.native.make(**kwargs),self.ctx)
class Context(e.EvalContext):
    def __init__(self):
        self.seat=None;self.current_info=None;self.deck_calls=[];self.progress=[]
        self.raw_options_peak=0;self.head_candidate_batches=0;self.head_opponent_batches=0;self.public_chain_observations=0
        super().__init__(R/'agent/scripts',meta_run/'decks',code_list_file=meta_manifest['vocabulary'],max_options=128,
                         num_envs=8,env_threads=4,max_steps=4000,timeout=60)
        self.deck_name='ElfnotesKewlTune';self.ygoenv=Factory(self,self.ygoenv)
    def check_info(self):
        info=self.current_info
        self.raw_options_peak=max(self.raw_options_peak,int(np.asarray(info['raw_num_options']).max()))
        assert self.raw_options_peak<=128
        self.public_chain_observations+=int((np.asarray(info['public_chain_length'])>0).sum())
    def _run_single_wave_seat(self,*a,**k):
        self.seat=k['candidate_seat']
        try:rows,timing=super()._run_single_wave_seat(*a,**k)
        finally:self.seat=None
        self.progress.extend(rows);gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
        assert c.native.get_lua_error_count()==0
        (out/'progress.json').write_text(json.dumps(dict(completed=len(self.progress),abnormal=sum(r['abnormal'] for r in self.progress)))+'\n')
        print('PROGRESS',args.policy,args.deck,len(self.progress),flush=True)
        return rows,timing
ctx=Context();candidate_params=ctx.load_params(e.resolve_agent_spec(candidate,P))
head_path=(before if args.policy=='baseline' else selected)/'chain-head.flax_model'
head=jax.device_put(flax.serialization.msgpack_restore(head_path.read_bytes()))
@jax.jit
def infer(params,state,obs,active,done,features,enabled):
    state=jax.tree.map(lambda x:jnp.where(done[:,None],0,x),state)
    output=ctx.agent.apply(params,obs,state);next_state=jax.tree.map(lambda n,o:jnp.where(active[:,None],n,o),output[0],state)
    logits=output[1]+HEAD.apply(head,features,obs['actions_'])*enabled
    return next_state,jnp.argmax(logits,axis=-1)
def step_model(params,state,obs,active,done):
    assert ctx.current_info is not None
    features=chain_features(ctx.current_info)
    enabled=np.float32(params is candidate_params)
    if enabled:
        if params is candidate_params:ctx.head_candidate_batches+=1
        else:ctx.head_opponent_batches+=1
    return infer(params,state,obs,active,done,features,enabled)
ctx._step_model=step_model
result=ctx.evaluate_matchup(candidate,opponent,P,num_episodes=64,base_seed=41037,max_steps_per_wave=4000)
for call in ctx.deck_calls:
    assert (call['deck1'],call['deck2'])==(('ElfnotesKewlTune',args.deck) if call['seat']==0 else (args.deck,'ElfnotesKewlTune'))
groups={}
for row in result['episodes']:groups.setdefault(row['pair_id'],[]).append(row)
excluded=[pid for pid,v in groups.items() if any(r['abnormal'] for r in v)]
qualified=[r for pid,v in groups.items() if pid not in excluded for r in v]
scores=[sum(r['win'] for r in v)/2 for pid,v in groups.items() if pid not in excluded]
ci=e.paired_wilson_ci95(scores)
reference=dict(episodes=len(qualified),wins=sum(r['win'] for r in qualified),losses=sum(r['loss'] for r in qualified),
               draws=sum(r['draw'] for r in qualified),win_rate=sum(r['win'] for r in qualified)/len(qualified) if qualified else None,
               ci95=list(ci[:2]),excluded_pairs=excluded,
               scope='reference over complete natural pairs; native cutoff zero reward is not LP adjudication')
metadata=dict(status='complete',policy=args.policy,deck=args.deck,episodes=64,seed=41037,candidate=candidate,
              candidate_sha256=sha(candidate),opponent=opponent,opponent_sha256=sha(opponent),
              head_sha256=sha(head_path),
              native=json.loads((B/'runtime.json').read_text()),evaluator_sha256=sha(scorer),
              vocabulary_sha256=meta_manifest['vocabulary_sha256'],deck_calls=ctx.deck_calls,
              candidate_only_uses_chain_head=True,lua_errors=c.native.get_lua_error_count(),
              raw_options_peak=ctx.raw_options_peak,head_candidate_batches=ctx.head_candidate_batches,
              head_opponent_batches=ctx.head_opponent_batches,public_chain_observations=ctx.public_chain_observations,
              resources=c.native.get_env_resource_stats())
assert metadata['lua_errors']==0 and metadata['resources']['live_duels']==0
assert ctx.head_opponent_batches==0 and ctx.head_candidate_batches>0
(out/'summary.json').write_text(json.dumps(dict(meta=metadata,matchup=result,complete_natural_pairs=reference),indent=2)+'\n')
with (out/'episodes.csv').open('w',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=list(result['episodes'][0]));writer.writeheader();writer.writerows(result['episodes'])
print('COMPLETE',args.policy,args.deck,json.dumps(reference),flush=True)
