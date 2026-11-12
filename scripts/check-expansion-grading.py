import sys,runpy
from pathlib import Path
p=Path('/home/hatk8/projects/yugioh-ai/evaluation/normal-policy-v2/code')
sys.path.insert(0,str(p))
runpy.run_path(str(p/'verify_normal_policy_v2.py'),run_name='__main__')
