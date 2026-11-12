import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['JAX_PLATFORMS']='cpu'
project=Path('/home/hatk8/projects/yugioh-ai')
base=project/'experiments/elfnotes-20261106'
legacy='--legacy' in sys.argv
r=Path('/opt/ygo-ai/current') if legacy else base/'runtime/agent'
sys.path[:0]=[str(r),str(r/'ygoenv')]
os.chdir(r/'scripts')
import ygoenv
from ygoai.rl.env import EnvPreprocess
from ygoai.utils import init_ygopro
spec=importlib.util.spec_from_file_location('evaluator',project/'scripts/evaluate.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
deck=project/'assets/deck'/('BlueEyes.ydk' if legacy else 'ElfnotesKewlTune.ydk')
name=init_ygopro('YGOPro-v1','english',str(deck),'code_list.txt')
c=m.EvalContext.__new__(m.EvalContext)
c.num_envs=2; c.env_threads=1; c.env_id='YGOPro-v1'; c.deck_name=name
c.max_options=128; c.n_history_actions=32; c.max_steps=1; c.timeout=10
c.ygoenv=ygoenv; c.EnvPreprocess=EnvPreprocess
random={'kind':'builtin','strategy':'random'}
rows,_=c._run_single_wave_seat(random,random,None,None,0,3072724,0)
assert all(e['truncated'] and e['reason_label']=='truncated' and e['outcome']=='excluded' and e['candidate_reward']==0 for e in rows)
assert all(e['timeout_detection_supported'] != legacy for e in rows)
label='legacy' if legacy else 'modern'
result=dict(engine=label,passed=True,episodes=rows)
(base/f'{label}-budget-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(engine=label,passed=True,excluded_games=len(rows))))
