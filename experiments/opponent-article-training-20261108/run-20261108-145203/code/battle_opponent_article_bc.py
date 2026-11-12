"""Paired, seat-swapped matches against one fixed Elfnotes policy."""
import argparse,ctypes,gc,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from opponent_article_bc_common import *
p=argparse.ArgumentParser();p.add_argument('--deck',required=True);p.add_argument('--policy',choices=['before','after'],required=True)
p.add_argument('--episodes',type=int,default=64);args=p.parse_args()
manifest=json.loads((B/'manifest.json').read_text());spec=manifest['decks'][args.deck]
source=P/'scripts/evaluate.py';scorer=B/'scoring-evaluator.py'
if not scorer.exists():scorer.write_bytes(source.read_bytes())
assert sha(source)==sha(scorer),'User evaluator changed during this run'
module=importlib.util.spec_from_file_location('pinned_match_scorer',scorer);e=importlib.util.module_from_spec(module);module.loader.exec_module(e)
OUT=B/'battle'/args.deck/args.policy;OUT.mkdir(parents=True,exist_ok=False)
if args.policy=='before':candidate=Path(spec['source_model']);candidate_head=init_head();head_path=None
else:
    selected=Path(json.loads((B/'training'/args.deck/'result.json').read_text())['selected_checkpoint'])
    candidate=selected/'base.flax_model';head_path=selected/'chain-head.flax_model';candidate_head=load(head_path)
fixed_selected=Path(json.loads((P/'experiments/endboard-20261108/training/result.json').read_text())['selected_checkpoint'])
fixed=fixed_selected/'base.flax_model';fixed_head_path=fixed_selected/'chain-head.flax_model';fixed_head=load(fixed_head_path)
v.native.set_benchmark_fixture([])
class Wrap:
    def __init__(self,raw,ctx):self.raw=raw;self.ctx=ctx
    def __getattr__(self,name):return getattr(self.raw,name)
    def reset(self,*a,**k):
        data=self.raw.reset(*a,**k);self.ctx.info=data[-1];self.ctx.check();return data
    def step(self,*a,**k):
        data=self.raw.step(*a,**k);self.ctx.info=data[-1];self.ctx.check();return data
class Factory:
    def __init__(self,native,ctx):self.native=native;self.ctx=ctx
    def make(self,**kwargs):
        if self.ctx.seat is not None:
            kwargs['deck1'],kwargs['deck2']=(args.deck,'ElfnotesKewlTune') if self.ctx.seat==0 else ('ElfnotesKewlTune',args.deck)
            self.ctx.deck_calls.append(dict(seat=self.ctx.seat,seed=kwargs['seed'],deck1=kwargs['deck1'],deck2=kwargs['deck2']))
        return Wrap(self.native.make(**kwargs),self.ctx)
class Context(e.EvalContext):
    def __init__(self):
        self.seat=None;self.info=None;self.rows=[];self.peak_options=0;self.deck_calls=[]
        super().__init__(v.R/'agent/scripts',v.run/'decks',code_list_file=manifest['vocabulary'],max_options=128,
                         num_envs=8,env_threads=4,max_steps=4000,timeout=60)
        self.deck_name=args.deck;self.ygoenv=Factory(self.ygoenv,self)
    def check(self):
        self.peak_options=max(self.peak_options,int(np.asarray(self.info['raw_num_options']).max()))
        assert self.peak_options<=128,'Action truncation invalidates comparison'
    def _run_single_wave_seat(self,*pos,**kw):
        self.seat=kw['candidate_seat']
        try:rows,timing=super()._run_single_wave_seat(*pos,**kw)
        finally:self.seat=None
        self.rows.extend(rows);gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
        assert v.native.get_lua_error_count()==0
        write(OUT/'progress.json',dict(completed=len(self.rows),total=args.episodes,abnormal=sum(x['abnormal'] for x in self.rows)))
        print('MATCH_PROGRESS',args.deck,args.policy,len(self.rows),flush=True)
        return rows,timing
ctx=Context();candidate_params=ctx.load_params(e.resolve_agent_spec(str(candidate),P))
@jax.jit
def infer(params,head,state,obs,active,done,chain):
    state=jax.tree.map(lambda x:jnp.where(done[:,None],0,x),state)
    result=ctx.agent.apply(params,obs,state)
    state=jax.tree.map(lambda n,o:jnp.where(active[:,None],n,o),result[0],state)
    logits=result[1]+HEAD.apply(head,chain,obs['actions_'])
    return state,jnp.argmax(logits,axis=-1)
def step_model(params,state,obs,active,done):
    return infer(params,candidate_head if params is candidate_params else fixed_head,state,obs,active,done,features(ctx.info))
ctx._step_model=step_model
result=ctx.evaluate_matchup(str(candidate),str(fixed),P,num_episodes=args.episodes,base_seed=811037,max_steps_per_wave=4000)
pairs={}
for row in result['episodes']:pairs.setdefault(row['pair_id'],[]).append(row)
excluded=[k for k,rows in pairs.items() if any(x['abnormal'] for x in rows)]
natural=[x for k,rows in pairs.items() if k not in excluded for x in rows]
totals=dict(episodes=len(natural),wins=sum(x['win'] for x in natural),losses=sum(x['loss'] for x in natural),draws=sum(x['draw'] for x in natural),excluded_pairs=excluded)
assert v.native.get_lua_error_count()==0 and v.native.get_env_resource_stats()['live_duels']==0
report=dict(status='complete',deck=args.deck,policy=args.policy,totals=totals,matchup=result,
    candidate=str(candidate),candidate_sha256=sha(candidate),candidate_head=str(head_path) if head_path else None,
    fixed_elfnotes_model=str(fixed),fixed_elfnotes_sha256=sha(fixed),fixed_head_sha256=sha(fixed_head_path),
    seed=811037,evaluator_sha256=sha(scorer),runtime=manifest['runtime'],deck_calls=ctx.deck_calls,
    lua_errors=0,raw_options_peak=ctx.peak_options,scope='Small paired match sample; no claim of statistically established strength improvement.')
write(OUT/'summary.json',report);print('MATCH_COMPLETE',args.deck,args.policy,json.dumps(totals),flush=True)
