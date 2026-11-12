import json
import os
from pathlib import Path
import sys
r=Path('/home/hatk8/projects/yugioh-ai/experiments/elfnotes-20261106/runtime')
sys.path[:0]=[str(r/'agent'),str(r/'agent/ygoenv')]
os.chdir(r/'agent/scripts')
from ygoai.utils import init_ygopro
from ygoenv.ygopro.ygopro_ygoenv import run_effect_checks,get_lua_error_count
init_ygopro('YGOPro-v1','english','/home/hatk8/projects/yugioh-ai/assets/deck/ElfnotesKewlTune.ydk','code_list.txt')
rows=run_effect_checks()
result={'checks':rows,'lua_errors':get_lua_error_count()}
print(json.dumps(result,indent=2),flush=True)
(r.parent/'effect-checks.json').write_text(json.dumps(result,indent=2))
assert all(row['pass'] for row in rows)
assert get_lua_error_count()==0
