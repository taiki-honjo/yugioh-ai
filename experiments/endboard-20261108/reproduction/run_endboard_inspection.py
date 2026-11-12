import json,subprocess
from pathlib import Path
W=Path(__file__).parent
S=Path('/home/hatk8/projects/yugioh-ai/experiments/endboard-20261108')
for arm,policy in [('pass','baseline'),('ash_once','current'),('ash_once','baseline')]:
    if (S/arm/policy/'summary.json').exists():continue
    name=policy+'-'+arm
    print('BEGIN',name,flush=True)
    with (S/(name+'.log')).open('w') as h:
        r=subprocess.run(['python3',str(W/'run_ygo_memory_limited.py'),'--name','endboard-'+name,
            '--record',str(S/('memory-'+name+'.json')),'--limit-mib','5120','--','env','MALLOC_ARENA_MAX=2',
            '/opt/ygo-ai/.venv/bin/python','-u',str(W/'inspect_full_turn.py'),'--arm',arm,'--policy',policy],stdout=h,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Inspection failed: '+name)
    print('DONE',name,flush=True)
(S/'inspection-complete.json').write_text(json.dumps(dict(status='complete',conditions=4,cases_per_condition=24,total=96))+'\n')
