import json,subprocess,time
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');S=P/'experiments/endboard-20261108';W=S/'reproduction'
def job(name,script,*args):
    print('BEGIN',name,flush=True)
    with (S/(name+'.log')).open('w') as h:
        r=subprocess.run(['python3',str(W/'run_ygo_memory_limited.py'),'--name','endboard-'+name,'--record',str(S/('memory-'+name+'.json')),
            '--limit-mib','5120','--','env','MALLOC_ARENA_MAX=2','/opt/ygo-ai/.venv/bin/python','-u',str(W/script),*args],stdout=h,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Validation failed: '+name)
    print('DONE',name,flush=True)
assert (S/'training/result.json').exists()
assert json.loads((S/'training/result.json').read_text())['status']=='complete'
if not (S/'evaluation/summary.json').exists():job('full-evaluation','evaluate_endboard.py')
if not (S/'short-regression/summary.json').exists():job('short-regression','evaluate_short_regression.py')
if not (S/'reinspection/summary.json').exists():job('reinspection','reinspect_endboard.py')
for policy in ['baseline','bc_with_chain']:
    for deck in ['Artmage','ClownTroupe','BrandedDracotail','SkyStriker']:
        if not (S/'battle'/policy/deck/'summary.json').exists():
            job(('battle-'+policy+'-'+deck).lower(),'evaluate_battle.py','--policy',policy,'--deck',deck)
(S/'validation-complete.json').write_text(json.dumps(dict(status='complete',battle_games=512,battle_seed=41037,
    policy_names={'baseline':'previous BC with old head','bc_with_chain':'new endboard BC with new head'},selected_without_holdout=True))+'\n')
print('VALIDATION_COMPLETE',flush=True)
