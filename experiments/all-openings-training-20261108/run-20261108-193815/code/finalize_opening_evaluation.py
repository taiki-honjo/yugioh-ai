from pathlib import Path
import json,shutil
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
B=Path((P/'experiments/all-openings-training-20261108/latest-run.txt').read_text().strip())
for name in ['all_opening_witness.py','evaluate_all_openings.py','finalize_opening_evaluation.py']:shutil.copy2(W/name,B/'code'/name)
before=B/'evaluation/holdout/before';archive=B/'evaluation/holdout/before-opponent-search-selection-incomplete'
if before.exists():assert not archive.exists();before.rename(archive)
m=json.loads((B/'manifest.json').read_text());m['evaluation_script_note']='Opponent Rukina search is resolved with a legal Spirit selection when the AI does not negate it; earlier incomplete-script baseline is archived. Teacher setup labels are unchanged.'
(B/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print('EVALUATOR_READY',B)
