from pathlib import Path
import shutil
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
B=Path((P/'experiments/all-openings-training-20261108/latest-run.txt').read_text().strip());f=B/'code/train_all_openings.py';t=f.read_text()
assert not (B/'training/ElfnotesKewlTune').exists()
t=t.replace('probs.append(probabilities)',"if probabilities.shape[1]<d['labels'].shape[1]:probabilities=np.pad(probabilities,((0,0),(0,d['labels'].shape[1]-probabilities.shape[1]),(0,0)),mode='edge')\n        probs.append(probabilities)")
f.write_text(t)
for name in ['prepare_all_opening_training.py','finalize_all_opening_trainer.py','evaluate_all_openings.py']:shutil.copy2(W/name,B/'code'/name)
compile(t,str(f),'exec');print('TRAINER_READY',B)
