#!/usr/bin/env python3
import csv
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
import subprocess

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
project=Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0,str(project/'scripts'))
import evaluate as evaluator
import numpy as np

base=project/'experiments/elfnotes-20261106'
runtime=base/'runtime'
train=Path((base/'latest-training-path.txt').read_text().strip())
initial=train/'checkpoints/initial.flax_model'
trained=list((train/'checkpoints').glob('*steps.flax_model'))
assert len(trained)==1,trained
trained=trained[0]
subprocess.run([sys.executable,str(project/'scripts/check-elfnotes-runtime.py'),
                '--checkpoint',str(trained)],check=True)
out=base/('evaluation-'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S'))
out.mkdir(exist_ok=True)
ctx=evaluator.EvalContext(runtime/'agent/scripts',project/'assets/deck/ElfnotesKewlTune.ydk',
                          max_options=128,num_envs=4,env_threads=2,max_steps=4000,timeout=10)
summaries=[]
for candidate,opponent in [(initial,'random'),(trained,'random'),(trained,initial)]:
    print('EVALUATING',Path(candidate).name,'vs',Path(opponent).name,flush=True)
    summary=ctx.evaluate_matchup(str(candidate),str(opponent),project,num_episodes=16,base_seed=3026)
    summaries.append(summary)
    print('RESULT',json.dumps(summary['results'],ensure_ascii=False),flush=True)

def pair_scores(summary):
    groups={}
    for e in summary['episodes']:
        if e['included_in_paired_score']:
            groups.setdefault(e['pair_id'],[]).append(e['win'])
    return {key:float(np.mean(values)) for key,values in groups.items()}
before,after=map(pair_scores,summaries[:2])
shared=sorted(before.keys()&after.keys())
diffs=[after[k]-before[k] for k in shared]
delta=dict(complete_matched_pairs=len(shared),winrate_change=float(np.mean(diffs)) if diffs else None,
           ci95=list(evaluator.paired_bootstrap_ci95(diffs)[:2]) if len(diffs)>1 else None,
           method='paired percentile bootstrap of after-minus-before seed-pair win rates',
           caution='Small empirical sample; this is a functional baseline, not proof of improved strength')
from ygoenv.ygopro.ygopro_ygoenv import get_lua_error_count
meta=dict(timestamp_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(),base_seed=3026,
          num_episodes=16,num_envs=4,env_threads=2,jit_warmup_seconds=ctx.warmup_seconds,
          initial_checkpoint=str(initial),trained_checkpoint=str(trained),
          lua_errors=get_lua_error_count(),matched_before_after=delta)
assert get_lua_error_count()==0
report=evaluator.format_markdown_report(meta,summaries)
(out/'summary.json').write_text(json.dumps(dict(meta=meta,matchups=summaries),ensure_ascii=False,indent=2)+'\n')
(out/'report.md').write_text(report,encoding='utf-8')
episodes=[dict(matchup_index=i+1,**e) for i,s in enumerate(summaries) for e in s['episodes']]
with (out/'episodes.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(episodes[0]))
    writer.writeheader(); writer.writerows(episodes)
print('MATCHED_CHANGE',json.dumps(delta),flush=True)
(base/'latest-evaluation-path.txt').write_text(str(out)+'\n')
print('SAVED',out,flush=True)
