import sys,subprocess,json
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');B=P/'experiments/chain-imitation-20261108'
W=Path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/work');PY='/opt/ygo-ai/.venv/bin/python'
def job(name,script,*args):
    print('BEGIN',name,flush=True)
    with (B/(name+'.log')).open('w') as h:
        result=subprocess.run(['python3',str(W/'run_ygo_memory_limited.py'),'--name','chain-'+name,
            '--record',str(B/('memory-'+name+'.json')),'--limit-mib','5120','--','env','MALLOC_ARENA_MAX=2',PY,'-u',str(W/script),*args],stdout=h,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect its log')
    print('DONE',name,flush=True)
assert (B/'training/result.json').is_file()
record=json.loads((B/'training/result.json').read_text());assert record['status']=='complete'
if not (B/'normal-mode-regression.json').exists():job('normal-regression','check_chain_runtime_regression.py')
if not (B/'evaluation/summary.json').exists():job('fixed-evaluation','evaluate_chain_imitation.py')
for policy in ['baseline','bc_with_chain']:
    for deck in ['Artmage','ClownTroupe','BrandedDracotail','SkyStriker']:
        if not (B/'battle'/policy/deck/'summary.json').exists():job(('battle-'+policy+'-'+deck).lower(),'eval_chain_battle.py','--policy',policy,'--deck',deck)
(B/'validation-complete.json').write_text(json.dumps(dict(status='complete',battle_cases=8,battle_games=512,
    battle_seed=11037,battle_episode_count_per_case=64,selection='checkpoint fixed by validation CE before all holdout and battle tests'),indent=2)+'\n')
print('VALIDATION_COMPLETE',flush=True)
