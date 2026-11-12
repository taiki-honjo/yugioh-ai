"""Keep the user's evaluator intact; supply explicit, seat-aware deck pairs."""
import argparse,csv,ctypes,gc,hashlib,importlib.util,json,os,sys
from collections import Counter,deque
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
project=Path('/home/hatk8/projects/yugioh-ai')
base=project/'experiments/meta-opponents-20261107'
run=Path((base/'latest-run.txt').read_text().strip())
manifest=json.loads((run/'manifest.json').read_text())
runtime=Path(manifest['runtime'])
sys.path[:0]=[str(runtime/'agent'),str(runtime/'agent/ygoenv')]
JST=timezone(timedelta(hours=9))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--deck',required=True)
    p.add_argument('--opponent',default='random');p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--seed',type=int,default=6037);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--diagnose-wave-seed',type=int)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    evaluator_path=run/'evaluator.py'
    assert sha(evaluator_path)==manifest['evaluator_sha256']
    spec=importlib.util.spec_from_file_location('frozen_current_evaluator',evaluator_path)
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    from ygoenv.ygopro.ygopro_ygoenv import get_lua_error_count,get_env_resource_stats
    class DeckFactory:
        def __init__(self,ctx,native):self.ctx=ctx;self.native=native
        def make(self,**kwargs):
            seat=self.ctx.active_candidate_seat
            if seat is not None:
                kwargs['deck1'],kwargs['deck2']=(('ElfnotesKewlTune',args.deck) if seat==0 else (args.deck,'ElfnotesKewlTune'))
                self.ctx.deck_calls.append(dict(candidate_seat=seat,deck1=kwargs['deck1'],deck2=kwargs['deck2'],seed=kwargs['seed']))
            raw=self.native.make(**kwargs)
            if args.diagnose_wave_seed is None or seat is None:return raw
            class TraceEnv:
                def __getattr__(self,name):return getattr(raw,name)
                def reset(self,*a,**k):return raw.reset(*a,**k)
                def step(self,*a,**k):
                    result=raw.step(*a,**k);info=result[-1]
                    if 'raw_num_options' in info:
                        diag['peak_raw_options_target']=max(diag['peak_raw_options_target'],int(info['raw_num_options'][3]))
                    return result
            return TraceEnv()
    class MultiDeckContext(e.EvalContext):
        def __init__(self):
            self.active_candidate_seat=None;self.deck_calls=[]
            super().__init__(runtime/'agent/scripts',run/'decks',code_list_file=manifest['vocabulary'],
                             max_options=128,num_envs=8,env_threads=4,max_steps=4000,timeout=60)
            self.deck_name='ElfnotesKewlTune'
            self.ygoenv=DeckFactory(self,self.ygoenv)
        def _run_single_wave_seat(self,*pos,**kw):
            self.active_candidate_seat=kw['candidate_seat']
            try:rows,timing=super()._run_single_wave_seat(*pos,**kw)
            finally:self.active_candidate_seat=None
            for row in rows:
                row.update(candidate_deck='ElfnotesKewlTune',opponent_deck=args.deck)
            partial.extend(rows)
            gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
            status=dict(completed=len(partial),requested=args.episodes,
                        abnormal=sum(row['abnormal'] for row in partial),
                        lua_errors=get_lua_error_count(),resources=get_env_resource_stats())
            write(args.out/'progress.json',status);write(args.out/'partial.json',partial)
            print('PROGRESS',json.dumps(status),flush=True)
            if status['lua_errors']:raise RuntimeError('Lua errors: inspect log; do not accept benchmark')
            return rows,timing
    assert sha(manifest['vocabulary'])==manifest['vocabulary_sha256']
    assert sha(manifest['candidate_checkpoint'])==manifest['candidate_sha256']
    assert sha(runtime/'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==manifest['native_sha256']
    if Path(args.opponent).is_file() and args.deck!='ElfnotesKewlTune':
        associated=json.loads((Path(args.opponent).parent/'runtime.json').read_text())
        assert associated['vocabulary_sha256']==manifest['vocabulary_sha256']
        assert associated['deck_sha256']==sha(run/'decks'/(args.deck+'.ydk'))
        assert associated['native_sha256']==manifest['native_sha256']
    partial=[];ctx=MultiDeckContext()
    if args.diagnose_wave_seed is not None:
        import numpy as np
        diag=dict(peak_raw_options_target=0,candidate_calls=0,opponent_calls=0)
        signatures=Counter();last=deque(maxlen=12)
        cand=e.resolve_agent_spec(manifest['candidate_checkpoint'],project)
        opp=e.resolve_agent_spec(args.opponent,project)
        ctx.warmup_jit();cp=ctx.load_params(cand);op=ctx.load_params(opp)
        original_step=ctx._step_model
        def traced(params,state,obs,active,done):
            result=original_step(params,state,obs,active,done)
            if bool(np.asarray(active)[3]):
                who='candidate' if params is cp else 'opponent';diag[who+'_calls']+=1
                action=int(np.asarray(result[1])[3]);global_state=np.asarray(obs['global_'])[3]
                action_state=np.asarray(obs['actions_'])[3,action]
                digest=hashlib.sha256(b''.join(np.asarray(obs[k])[3].tobytes()
                       for k in ['cards_','global_','actions_','h_actions_'])).hexdigest()
                signatures[(who,digest,action)]+=1
                last.append(dict(who=who,call=diag[who+'_calls'],action=action,global_features=global_state.tolist(),
                                 selected_action_features=action_state.tolist(),observation_sha256=digest))
            return result
        ctx._step_model=traced;ctx.max_steps=8000
        episodes,_=ctx._run_single_wave_seat(cand,opp,cp,op,wave_idx=14,
                                            wave_seed=args.diagnose_wave_seed,candidate_seat=1,max_steps_per_wave=8000)
        diag.update(target_episode=episodes[3],last_calls=list(last),
                    top_repeated_observations=[dict(who=k[0],observation_sha256=k[1],action=k[2],count=n)
                                               for k,n in signatures.most_common(8)],max_steps=8000,
                    lua_errors=get_lua_error_count())
        write(args.out/'diagnosis.json',diag)
        print('DIAGNOSIS',json.dumps(diag),flush=True)
        return
    meta=dict(status='running',timestamp_jst=datetime.now(JST).isoformat(),
              evaluator_sha256=manifest['evaluator_sha256'],native_sha256=manifest['native_sha256'],
              vocabulary_sha256=manifest['vocabulary_sha256'],candidate_checkpoint=manifest['candidate_checkpoint'],
              candidate_checkpoint_sha256=manifest['candidate_sha256'],candidate_deck='ElfnotesKewlTune',
              opponent_deck=args.deck,opponent_policy=args.opponent,
              opponent_checkpoint_sha256=sha(args.opponent) if Path(args.opponent).is_file() else None,
              base_seed=args.seed,num_episodes=args.episodes,num_envs=8,env_threads=4,max_steps=4000,timeout=60,
              max_options=128,deck_path=str(run/'decks'),jax_backend=ctx.jax.default_backend(),
              opponent_strength_scope='local learned policy; no expert or ladder strength certification')
    write(args.out/'summary.json',dict(meta=meta,matchups=[]))
    result=ctx.evaluate_matchup(manifest['candidate_checkpoint'],args.opponent,project,
                               num_episodes=args.episodes,base_seed=args.seed,max_steps_per_wave=4000)
    result['config'].update(candidate_deck='ElfnotesKewlTune',opponent_deck=args.deck,
                            candidate_deck_sha256=sha(run/'decks/ElfnotesKewlTune.ydk'),
                            opponent_deck_sha256=sha(run/'decks'/(args.deck+'.ydk')))
    for call in ctx.deck_calls:
        seat=call['candidate_seat'];assert call['deck1']==('ElfnotesKewlTune' if seat==0 else args.deck)
        assert call['deck2']==(args.deck if seat==0 else 'ElfnotesKewlTune')
    assert len(ctx.deck_calls)==args.episodes//8
    assert get_lua_error_count()==0 and get_env_resource_stats()['live_duels']==0
    meta.update(status='complete' if result['terminations']['abnormal_total']==0 else 'requires_review',
                lua_errors=get_lua_error_count(),resources=get_env_resource_stats(),
                completed_jst=datetime.now(JST).isoformat(),deck_calls=ctx.deck_calls,
                jit_warmup_seconds=ctx.warmup_seconds)
    write(args.out/'summary.json',dict(meta=meta,matchups=[result]))
    (args.out/'report.md').write_text(e.format_markdown_report(meta,[result]))
    with (args.out/'episodes.csv').open('w',newline='') as h:
        fields=list(dict.fromkeys(k for row in result['episodes'] for k in row))
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(result['episodes'])
    (args.out/'evaluator.py').write_bytes(evaluator_path.read_bytes())
    (args.out/'driver.py').write_bytes(Path(__file__).read_bytes())
    print('RESULT',json.dumps(result['results']),flush=True)
    if meta['status']!='complete':raise RuntimeError('Abnormal games require review')
if __name__=='__main__':main()
