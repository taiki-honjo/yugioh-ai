"""Continue from a saved, validation-selected main checkpoint on the expanded corpus."""
import hashlib,json,os,shutil,signal,time
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');W=Path(__file__).parent
main=Path((P/'experiments/all-openings-training-20261108/latest-run.txt').read_text().strip())
B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
chosen=read(main/'training/ElfnotesKewlTune/selected.json');source=Path(chosen['checkpoint'])
assert chosen['step']>=200 and all((source/n).is_file() for n in ['base.flax_model','chain-head.flax_model'])
progress=read(main/'training/ElfnotesKewlTune/progress.json')
matched=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeDecodeError):continue
    if str(main/'code/train_all_openings.py') in cmd:matched.append(int(proc.name))
assert len(matched)<=1
def alive(pid):
    try:return (Path('/proc',str(pid))/'stat').read_text().split()[2]!='Z'
    except OSError:return False
for pid in matched:os.kill(pid,signal.SIGTERM)
for _ in range(200):
    if not any(alive(pid) for pid in matched):break
    time.sleep(.1)
assert not any(alive(pid) for pid in matched)
note=dict(status='continued-from-saved-checkpoint',reason='User approved Baronne alternatives for the two hand-Spirit openings; continue jointly with memory-bounded updates.',
    checkpoint=str(source),checkpoint_step=chosen['step'],last_completed_progress=progress['step'],continuation_run=str(B),optimizer_state_restarted=True)
(main/'training/ElfnotesKewlTune/continuation.json').write_text(json.dumps(note,indent=2)+'\n')
m=read(B/'manifest.json');spec=m['decks']['ElfnotesKewlTune'];assert not (B/'training/ElfnotesKewlTune').exists()
m.setdefault('original_source',{k:spec[k] for k in ['source_model','source_head','source_model_sha256','source_head_sha256']})
spec.update(source_model=str(source/'base.flax_model'),source_model_sha256=sha(source/'base.flax_model'),source_head=str(source/'chain-head.flax_model'),source_head_sha256=sha(source/'chain-head.flax_model'))
m.update(await_main_training_source=False,main_training_run=str(main),main_checkpoint_used=chosen['step'],main_last_completed_progress=progress['step'],
    main_stage_continuation_note=note,memory_controls=dict(synchronous_updates=True,donate_parameters_and_optimizer_state=True,buckets=2))
m['training_plan']['updates']=1400
(B/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for name in ['all_opening_witness.py','evaluate_all_openings.py','resume_all_openings_with_baronne.py']:shutil.copy2(W/name,B/'code'/name)
f=B/'code/train_all_openings.py';t=f.read_text()
t=t.replace('import optax','import optax\nfrom functools import partial')
t=t.replace("assert manifest['status']=='complete'","assert manifest['status']=='complete'\nassert not manifest.get('await_main_training_source',False)")
t=t.replace("/32)*32","/64)*64").replace("/32).astype(int)","/64).astype(int)")
t=t.replace('@jax.jit\ndef update','@partial(jax.jit,donate_argnums=(0,1,2))\ndef update')
t=t.replace("params,st_b,st_h,metrics,norm=update(params,st_b,st_h,batch_of(data['train'],idx))","params,st_b,st_h,metrics,norm=update(params,st_b,st_h,batch_of(data['train'],idx))\n        jax.block_until_ready(params)")
assert 'donate_argnums' in t and 'block_until_ready(params)' in t and '/32' not in t
compile(t,str(f),'exec');f.write_text(t)
print(json.dumps(dict(status='ready',run=str(B),source_checkpoint=str(source),updates=1400,source_base_sha256=spec['source_model_sha256'])))
