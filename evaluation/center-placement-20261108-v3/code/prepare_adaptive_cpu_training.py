"""Use a bounded CPU worker while the desktop is encoding, preserving every prior stage."""
import json,os,shutil,signal,time
from pathlib import Path
P=Path('/home/hatk8/projects/yugioh-ai');B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeDecodeError):continue
    if '/work/probe_cpu_training.py' in cmd:os.kill(int(proc.name),signal.SIGTERM);os.kill(int(proc.name),signal.SIGCONT)
time.sleep(.5)
old=B/'training/ElfnotesKewlTune';arch=B/'training-gpu-stall-stage/ElfnotesKewlTune'
assert (old/'stalled-trial.json').exists() and not (old/'result.json').exists() and not arch.exists()
assert not list(old.glob('step-*'))
arch.parent.mkdir(exist_ok=True);old.rename(arch)
f=B/'code/train_all_openings.py';shutil.copy2(f,arch/'train_all_openings.py');t=f.read_text()
t=t.replace('from all_openings_common import *',"import os\nos.environ['JAX_PLATFORMS']='cpu'\nos.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])\nos.nice(10)\nfrom all_openings_common import *")
needle="history=[];best=float('inf');selected=None;rng=np.random.default_rng(plan['seed'])"
helpers='''def resource_state():
    try:
        state=json.loads((B/'training/resource-state.json').read_text())
        if state['pid']==os.getpid() and time.time()-state['sampled_unix']<12:return state
    except (OSError,ValueError,KeyError):pass
    return dict(training_duty=.85,mode='conservative-monitor-unavailable')
def resource_wait(duration=0):
    state=resource_state()
    while state['training_duty']<=0:
        write(OUT/'heartbeat.json',dict(phase='resource-paused',unix=time.time(),mode=state['mode']))
        time.sleep(1);state=resource_state()
    delay=min(duration,5)*(1/max(float(state['training_duty']),.05)-1)
    until=time.monotonic()+min(delay,30)
    while time.monotonic()<until:
        if resource_state()['training_duty']>=1:break
        time.sleep(min(.2,max(0,until-time.monotonic())))
'''
assert needle in t;t=t.replace(needle,helpers+'\n'+needle)
t=t.replace("for step in range(1,plan['updates']+1):","for step in range(1,plan['updates']+1):\n        resource_wait()\n        write(OUT/'heartbeat.json',dict(phase='computing',step=step,completed_updates=step-1,unix=time.time()))\n        update_started=time.perf_counter()")
t=t.replace('jax.block_until_ready(params)','jax.block_until_ready(params)\n        resource_wait(time.perf_counter()-update_started)')
t=t.replace("gpu_devices=list(map(str,jax.devices())),","gpu_devices=list(map(str,jax.devices())),execution_backend=jax.default_backend(),cpu_affinity=sorted(os.sched_getaffinity(0)),adaptive_resource_control=True,")
compile(t,str(f),'exec');f.write_text(t)
m=json.loads((B/'manifest.json').read_text());m['training_plan']['updates']=400
m['adaptive_resources']=dict(reason='User is screen sharing in Discord and requested dynamic resource use. GPU gradient execution stalled; CPU diagnostic updates completed.',
    worker_backend='cpu',max_cpu_threads=2,nice=10,cooperative=True,source_checkpoint_preserved=True,abandoned_gpu_trial_checkpoint_updates=0,
    stop_signal_not_used_during_active_updates=True,gpu_probe_command_buffer_disabled=True,gpu_probe_launch_blocking=True)
(B/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
policy=json.loads((B/'training/resource-policy.json').read_text());policy.update(cooperative=True,worker_backend='cpu',encoding_duty=.85,heavy_encoding_duty=.5)
(B/'training/resource-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
for name in ['adaptive_training_resources.py','prepare_adaptive_cpu_training.py']:shutil.copy2(Path(__file__).parent/name,B/'code'/name)
print(json.dumps(dict(status='ready',updates=400,backend='cpu',cpu_threads=2,source=m['decks']['ElfnotesKewlTune']['source_model'])))
