import argparse,hashlib,json,os,re,subprocess,sys,threading,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
p=argparse.ArgumentParser();p.add_argument('--deck',required=True);p.add_argument('--phase',type=int,choices=[1,2],required=True)
args=p.parse_args()
project=Path('/home/hatk8/projects/yugioh-ai');base=project/'experiments/meta-opponents-20261107'
run=Path((base/'latest-run.txt').read_text().strip());manifest=json.loads((run/'manifest.json').read_text())
runtime=Path(manifest['runtime']);python='/opt/ygo-ai/.venv/bin/python'
recipe=next(r for r in manifest['recipes'] if r['key']==args.deck)
steps=4096 if args.phase==1 else 65536;updates=steps//128
dest=run/'training'/args.deck/('phase'+str(args.phase)+'-'+str(steps))
dest.mkdir(parents=True,exist_ok=False);ckpt=dest/'checkpoints';ckpt.mkdir()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(timezone(timedelta(hours=9))).isoformat()
assert sha(manifest['vocabulary'])==manifest['vocabulary_sha256']
assert sha(runtime/'agent/ygoenv/ygoenv/ygopro/ygopro_ygoenv.cpython-310-x86_64-linux-gnu.so')==manifest['native_sha256']
source=None
if args.phase==2:
    previous=json.loads((run/'training'/args.deck/'phase1-4096/result.json').read_text())
    assert previous['status']=='complete';source=Path(previous['trained_checkpoint'])
    assert sha(source)==previous['files'][str(source)]['sha256']
metadata=json.loads((project/'env/elfnotes/runtime.json').read_text())
metadata.update(deck=recipe['ydk'],deck_sha256=recipe['ydk_sha256'],vocabulary=manifest['vocabulary'],
                vocabulary_sha256=manifest['vocabulary_sha256'],training_decisions=steps,training_phase=args.phase)
(ckpt/'runtime.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
command=[python,'-u','cleanba.py','--actor-device-ids','0','--learner-device-ids','0',
         '--deck',recipe['ydk'],'--code-list-file',manifest['vocabulary'],
         '--max-options','128','--timeout','10','--local-num-envs','4','--local-env-threads','2',
         '--num-actor-threads','1','--num-steps','32','--num-minibatches','4','--learning-rate','1e-4',
         '--vloss-clip','1.0','--save-interval',str(updates),'--local-eval-episodes','4',
         '--eval-interval',str(updates),'--total-timesteps',str(steps),'--seed','0',
         '--log-frequency','1','--ckpt-dir',str(ckpt),'--tb-dir',str(dest/'tensorboard')]
if source:command+=['--checkpoint',str(source)]
env=dict(os.environ,PYTHONPATH=str(runtime/'agent')+':'+str(runtime/'agent/ygoenv'),
         XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
record=dict(status='running',deck=args.deck,phase=args.phase,decisions=steps,command=command,
            started_jst=stamp(),seed=0,vocabulary_sha256=manifest['vocabulary_sha256'],native_sha256=manifest['native_sha256'])
(dest/'result.json').write_text(json.dumps(record,indent=2)+'\n')
stop=threading.Event();samples=[];start=time.perf_counter()
def monitor():
    with (dest/'gpu.csv').open('w') as h:
        h.write('elapsed_seconds,memory_used_mib,utilization_percent\n')
        while not stop.is_set():
            d=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if d.returncode==0:
                fields=d.stdout.strip().splitlines()[0].split(',');sample=[time.perf_counter()-start,*map(float,fields)]
                samples.append(sample);h.write(','.join(map(str,sample))+'\n');h.flush()
            stop.wait(2)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    with (dest/'train.log').open('w') as h:
        result=subprocess.run(command,cwd=runtime/'agent/scripts',env=env,stdout=h,stderr=subprocess.STDOUT,timeout=3600)
    assert result.returncode==0,f'Training failed: {dest}/train.log'
finally:stop.set();thread.join(timeout=12)
log=(dest/'train.log').read_text();assert 'NATIVE_LUA_ERRORS 0' in log
assert f'actor_update={updates},' in log
assert "actor_devices=['cuda:0']" in log and "learner_devices=['cuda:0']" in log
trained=list(ckpt.glob('*steps.flax_model'));assert len(trained)==1 and trained[0].name.endswith(f'_{steps}steps.flax_model')
trained=trained[0];initial=ckpt/'initial.flax_model'
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import flax.serialization
def flatten(v,prefix=''):
    if isinstance(v,dict):return {k:x for name,child in v.items() for k,x in flatten(child,prefix+'/'+name).items()}
    return {prefix:np.asarray(v)}
first,final=[flatten(flax.serialization.msgpack_restore(x.read_bytes())) for x in [initial,trained]]
assert first.keys()==final.keys()
assert all(first[k].shape==final[k].shape and np.all(np.isfinite(final[k])) for k in first)
reference=source or project/'experiments/elfnotes-20261106/train-20261106-190557-4096/checkpoints/initial.flax_model'
ref=flatten(flax.serialization.msgpack_restore(reference.read_bytes()))
assert ref.keys()==first.keys() and all(np.array_equal(ref[k],first[k]) for k in ref),'Initial weights differ from the required source'
files=[initial,trained]+([source] if source else [])
record.update(status='complete',completed_jst=stamp(),elapsed_seconds=time.perf_counter()-start,
              trained_checkpoint=str(trained),initial_checkpoint=str(initial),source_checkpoint=str(source) if source else None,
              required_initial_exact=True,all_finite=True,gpu_actor_and_learner=True,native_lua_errors=0,
              learner_updates=updates,parameters=sum(int(x.size) for x in final.values()),
              changed_values=sum(int(np.count_nonzero(first[k]!=final[k])) for k in first),
              peak_total_gpu_memory_mib=max(s[1] for s in samples) if samples else None,
              files={str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
assert record['changed_values']>0
(dest/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print('TRAINING_COMPLETE',json.dumps({k:v for k,v in record.items() if k not in ['files','command']}),flush=True)
