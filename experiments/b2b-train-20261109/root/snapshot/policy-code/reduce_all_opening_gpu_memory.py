"""Resume saved weights with single-trajectory batches and rematerialized scan gradients."""
import hashlib,json,os,shutil,signal,time
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai')
B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=B/'training/ElfnotesKewlTune';arch=B/'training-gpu-pressure-stage/ElfnotesKewlTune'
assert old.is_dir() and not arch.exists() and not (old/'result.json').exists()
chosen=read(old/'selected.json');progress=read(old/'progress.json')
assert chosen['step']>=200
matched=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeDecodeError):continue
    if str(B/'code/train_all_openings.py') in cmd:matched.append(int(proc.name))
assert len(matched)<=1
for pid in matched:os.kill(pid,signal.SIGTERM)
for _ in range(200):
    if not any((Path('/proc')/str(pid)).exists() for pid in matched):break
    time.sleep(.1)
assert not any((Path('/proc')/str(pid)).exists() for pid in matched)
arch.parent.mkdir(exist_ok=True);old.rename(arch)
source=arch/f"step-{chosen['step']:04d}"
assert all((source/n).is_file() for n in ['base.flax_model','chain-head.flax_model'])
m=read(B/'manifest.json');spec=m['decks']['ElfnotesKewlTune']
note=dict(reason='GPU allocation near device limit and stalled update progress; reduce activation memory.',
    prior_source={k:spec[k] for k in ['source_model','source_head','source_model_sha256','source_head_sha256']},
    prior_checkpoint_step=chosen['step'],last_completed_progress=progress['step'],checkpoint=str(source),optimizer_state_restarted=True)
spec.update(source_model=str(source/'base.flax_model'),source_model_sha256=sha(source/'base.flax_model'),
    source_head=str(source/'chain-head.flax_model'),source_head_sha256=sha(source/'chain-head.flax_model'))
m['gpu_pressure_continuation']=note
m['training_plan'].update(updates=1400,batch_size=1)
m['memory_controls'].update(batch_size=1,gradient_rematerialization=True)
(B/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
f=B/'code/train_all_openings.py';t=f.read_text()
shutil.copy2(f,arch/'train_all_openings.py')
needle='jax.lax.scan(step,initial,';assert needle in t
t=t.replace(needle,'jax.lax.scan(jax.checkpoint(step),initial,')
t=t.replace("range(0,len(d['labels']),2)","range(0,len(d['labels']),1)").replace("min(i+2,len(d['labels']))","min(i+1,len(d['labels']))")
assert 'jax.checkpoint(step)' in t
compile(t,str(f),'exec');f.write_text(t)
shutil.copy2(Path(__file__),B/'code'/Path(__file__).name)
(arch/'continuation.json').write_text(json.dumps(note,indent=2)+'\n')
print(json.dumps(dict(status='ready',source=str(source),source_step=chosen['step'],last_progress=progress['step'],updates=1400,batch_size=1)))
