#!/usr/bin/env python3
"""Frozen-runtime, configurable seat-paired evaluations with durable progress."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
PROJECT=Path('/home/hatk8/projects/yugioh-ai')
BASE=PROJECT/'experiments/elfnotes-20261106'
ENGINE=PROJECT/'env/elfnotes/evaluate_frozen_20261106.py'
JST=timezone(timedelta(hours=9))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num-episodes',type=int,default=16)
    parser.add_argument('--num-envs',type=int,default=4)
    parser.add_argument('--env-threads',type=int,default=2)
    parser.add_argument('--seed',type=int,default=3026)
    parser.add_argument('--train-dir',type=Path)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--baseline-dir',type=Path)
    parser.add_argument('--matchups',nargs='+',choices=['initial-random','trained-random','trained-initial'],
                        default=['initial-random','trained-random','trained-initial'])
    args=parser.parse_args()
    if args.num_envs<1 or args.num_episodes<2*args.num_envs or args.num_episodes%(2*args.num_envs):
        parser.error('--num-episodes must be a positive multiple of 2*num-envs')
    if args.seed in (0,100): parser.error('Use a seed distinct from training and trial evaluation')
    train=(args.train_dir or Path((BASE/'latest-training-path.txt').read_text().strip())).resolve()
    initial=train/'checkpoints/initial.flax_model'
    trained=list((train/'checkpoints').glob('*steps.flax_model'))
    if len(trained)!=1 or not initial.is_file(): parser.error('Expected initial weights and one final checkpoint')
    trained=trained[0]
    subprocess.run([sys.executable,str(PROJECT/'scripts/check-elfnotes-runtime.py'),
                    '--checkpoint',str(trained)],check=True)
    spec=importlib.util.spec_from_file_location('elfnotes_frozen_evaluator',ENGINE)
    evaluator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    import numpy as np
    out=(args.out_dir or BASE/('evaluation-'+datetime.now(JST).strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',started_jst=datetime.now(JST).isoformat(),timestamp_jst=datetime.now(JST).isoformat(),
              base_seed=args.seed,num_episodes=args.num_episodes,num_envs=args.num_envs,env_threads=args.env_threads,
              max_steps=4000,max_options=128,timeout=10,jit_warmup_seconds=0,
              initial_checkpoint=str(initial),initial_checkpoint_sha256=sha(initial),
              trained_checkpoint=str(trained),trained_checkpoint_sha256=sha(trained),
              evaluator_path=str(ENGINE),evaluator_sha256=sha(ENGINE),
              runtime=json.loads((PROJECT/'env/elfnotes/runtime.json').read_text()),
              requested_matchups=args.matchups)
    write_json(out/'summary.json',dict(meta=meta,matchups=[]))
    print('CONFIG',json.dumps({k:meta[k] for k in ['num_episodes','num_envs','base_seed','evaluator_sha256']}),flush=True)
    ctx=evaluator.EvalContext(BASE/'runtime/agent/scripts',PROJECT/'assets/deck/ElfnotesKewlTune.ydk',
                             max_options=128,num_envs=args.num_envs,env_threads=args.env_threads,max_steps=4000,timeout=10)
    from ygoenv.ygopro.ygopro_ygoenv import get_lua_error_count
    summaries=[]
    mapping={'initial-random':(initial,'random'),'trained-random':(trained,'random'),
             'trained-initial':(trained,initial)}
    for key in args.matchups:
        candidate,opponent=mapping[key]
        print('EVALUATING',key,flush=True)
        def progress(data):
            public=dict(matchup=key,**{k:v for k,v in data.items() if k!='episodes'})
            public['lua_errors']=get_lua_error_count()
            print('PROGRESS',json.dumps(public),flush=True)
            write_json(out/'progress.json',public)
            write_json(out/(key+'-partial.json'),dict(progress=public,episodes=data['episodes']))
            if get_lua_error_count(): raise RuntimeError('Lua errors detected')
        ctx.progress_callback=progress
        summary=ctx.evaluate_matchup(str(candidate),str(opponent),PROJECT,num_episodes=args.num_episodes,base_seed=args.seed)
        summary['comparison_key']=key
        summaries.append(summary)
        meta.update(jit_warmup_seconds=ctx.warmup_seconds,lua_errors=get_lua_error_count())
        write_json(out/'summary.json',dict(meta=meta,matchups=summaries))
        (out/'report.md').write_text(evaluator.format_markdown_report(meta,summaries),encoding='utf-8')
        print('RESULT',key,json.dumps(summary['results']),flush=True)

    def scores(summary):
        pairs={}
        for row in summary['episodes']:
            if row['included_in_paired_score']: pairs.setdefault(row['pair_id'],[]).append(row['win'])
        return {key:float(np.mean(values)) for key,values in pairs.items() if len(values)==2}
    after=next((s for s in summaries if s['comparison_key']=='trained-random'),None)
    before=next((s for s in summaries if s['comparison_key']=='initial-random'),None)
    if args.baseline_dir:
        prior=json.loads((args.baseline_dir/'summary.json').read_text())
        for field in ['base_seed','num_episodes','num_envs','max_steps','max_options','evaluator_sha256']:
            if prior['meta'][field]!=meta[field]: raise ValueError(f'Baseline configuration changed: {field}')
        reference=Path(prior['meta']['trained_checkpoint'])
        if sha(reference)!=sha(initial):
            import flax.serialization
            def leaves(x,prefix=''):
                if isinstance(x,dict): return {k:v for name,v in x.items() for k,v in leaves(v,prefix+'/'+name).items()}
                return {prefix:np.asarray(x)}
            a,b=[leaves(flax.serialization.msgpack_restore(p.read_bytes())) for p in [reference,initial]]
            if a.keys()!=b.keys() or not all(np.array_equal(a[k],b[k]) for k in a):
                raise ValueError('Additional training did not start from the baseline checkpoint')
            meta['initial_equals_baseline']='identical arrays'
        else: meta['initial_equals_baseline']='byte identical'
        before=next(s for s in prior['matchups'] if s['comparison_key']=='trained-random')
        meta['baseline_directory']=str(args.baseline_dir)
    if before and after:
        a,b=scores(before),scores(after)
        common=sorted(a.keys()&b.keys())
        differences=[b[key]-a[key] for key in common]
        meta['matched_before_after']=dict(complete_matched_pairs=len(common),
            winrate_change=float(np.mean(differences)) if differences else None,
            ci95=list(evaluator.paired_bootstrap_ci95(differences)[:2]) if len(differences)>1 else None,
            method='percentile bootstrap of paired changes over complete seed pairs',
            pair_changes=[dict(pair_id=key,before=a[key],after=b[key]) for key in common])
    meta.update(status='complete',completed_jst=datetime.now(JST).isoformat(),lua_errors=get_lua_error_count())
    assert get_lua_error_count()==0
    write_json(out/'summary.json',dict(meta=meta,matchups=summaries))
    (out/'report.md').write_text(evaluator.format_markdown_report(meta,summaries),encoding='utf-8')
    episodes=[dict(matchup_index=i+1,comparison_key=s['comparison_key'],**e)
              for i,s in enumerate(summaries) for e in s['episodes']]
    fields=list(dict.fromkeys(k for row in episodes for k in row))
    with (out/'episodes.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(episodes)
    (BASE/'latest-evaluation-path.txt').write_text(str(out)+'\n')
    print('COMPLETE',out,flush=True)
    if 'matched_before_after' in meta:
        print('MATCHED_CHANGE',json.dumps({k:v for k,v in meta['matched_before_after'].items() if k!='pair_changes'}),flush=True)

if __name__=='__main__': main()
