import sys,runpy,json,hashlib
from pathlib import Path
p=Path('/home/hatk8/projects/yugioh-ai');b=p/'experiments/chain-imitation-20261108';r=b/'runtime'
sys.path[:0]=[str(r/'agent'),str(r/'agent/ygoenv')]
import ygoenv
from ygoenv.ygopro import ygopro_ygoenv as native
assert str(r) in native.__file__
prior=p/'experiments/elfnotes-20261106/expansion-revised-20261106-205114'
training=json.loads((prior/'additional-training-result.json').read_text())
dest=b/'normal-mode-regression'
sys.argv=['eval_meta_opponents.py','--deck','ElfnotesKewlTune','--opponent',training['source_checkpoint'],
          '--episodes','16','--seed','3026','--out',str(dest)]
runpy.run_path('/mnt/c/Users/hatk8/Documents/Codex/2026-10-27/new-chat/work/eval_meta_opponents.py',run_name='__main__')
actual=json.loads((dest/'summary.json').read_text())['matchups'][0]['episodes']
old=json.loads((prior/'after-256/summary.json').read_text())
expected=next(s for s in old['matchups'] if s['comparison_key']=='trained-initial')['episodes'][:16]
fields=['pair_id','wave_seed','env_slot','candidate_seat','candidate_reward','win','loss','draw','env_steps']
assert len(actual)==len(expected)==16 and all(all(a[k]==e[k] for k in fields) for a,e in zip(actual,expected))
(b/'normal-mode-regression.json').write_text(json.dumps(dict(passed=True,episodes=16,fields=fields,
    fixture_disabled=True,loaded_native=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print('NORMAL_MODE_REGRESSION_PASSED')
