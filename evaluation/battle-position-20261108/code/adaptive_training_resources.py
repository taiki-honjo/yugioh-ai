"""Throttle only the pinned training process as encoding/load changes; always resume on exit."""
import argparse,csv,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
P=Path('/home/hatk8/projects/yugioh-ai');B=Path((P/'experiments/all-openings-bar-alternative-20261108/latest-run.txt').read_text().strip())
expected=str(B/'code/train_all_openings.py');out=B/'training';config=out/'resource-policy.json'
defaults=dict(sample_seconds=2.0,normal_duty=1.0,encoding_duty=.5,heavy_encoding_duty=.2,
    encoding_threshold_percent=3,heavy_encoding_threshold_percent=70,encoding_release_seconds=15,
    low_memory_free_mib=1536,critical_memory_free_mib=768,low_memory_duty=.15,critical_memory_duty=0.0,cooperative=False,worker_backend='cuda',
    cpu_affinity_encoding=[0,2],cpu_affinity_normal=[0,2,4,6])
if not config.exists():config.write_text(json.dumps(defaults,indent=2)+'\n')
stopping=False;paused=False;last_encoding=0;last_mode=None
def stop(*_):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def alive():
    try:return expected in (Path('/proc')/str(a.pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeDecodeError):return False
def resume():
    global paused
    if paused and alive():os.kill(a.pid,signal.SIGCONT)
    paused=False
def wait(seconds):
    end=time.monotonic()+seconds
    while not stopping and alive() and time.monotonic()<end:time.sleep(min(.1,max(0,end-time.monotonic())))
log=out/f'resource-monitor-{a.pid}-{time.time_ns()}.csv';assert not log.exists()
try:
    with log.open('w') as h:
        writer=csv.writer(h);writer.writerow(['unix_time','memory_total_mib','memory_used_mib','gpu_util_percent','encoder_util_percent','duty','mode']);h.flush()
        while not stopping and alive():
            try:c={**defaults,**json.loads(config.read_text())}
            except (OSError,ValueError):c=defaults
            result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,utilization.encoder','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            if result.returncode!=0:resume();wait(1);continue
            try:total,used,util,encoder=map(float,result.stdout.strip().splitlines()[0].split(','))
            except ValueError:resume();wait(1);continue
            now=time.time();free=total-used
            if encoder>=c['encoding_threshold_percent']:last_encoding=now
            duty=c['normal_duty'];mode='normal'
            if now-last_encoding<c['encoding_release_seconds']:duty=c['encoding_duty'];mode='encoding-priority'
            if encoder>=c['heavy_encoding_threshold_percent']:duty=min(duty,c['heavy_encoding_duty']);mode='heavy-encoding-priority'
            if c['worker_backend']=='cuda':
                if free<c['low_memory_free_mib']:duty=min(duty,c['low_memory_duty']);mode='memory-pressure'
                if free<c['critical_memory_free_mib']:duty=c['critical_memory_duty'];mode='memory-critical-paused'
            else:
                meminfo={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                if meminfo['MemAvailable']<768*1024:duty=0.;mode='cpu-memory-pressure-paused'
            duty=max(0.,min(float(duty),1.));period=max(.5,min(float(c['sample_seconds']),5.))
            state=dict(pid=a.pid,sampled_unix=now,memory_used_mib=used,memory_free_mib=free,gpu_utilization_percent=util,
                encoder_utilization_percent=encoder,mode=mode,training_duty=duty,
                cooperative=c['cooperative'],worker_backend=c['worker_backend'],note='Duty controls update/submission time, not an exact GPU utilization percentage. Pausing retains existing allocations.')
            if c['worker_backend']=='cpu' and alive():
                affinity=c['cpu_affinity_encoding'] if now-last_encoding<c['encoding_release_seconds'] else c['cpu_affinity_normal']
                for task in (Path('/proc')/str(a.pid)/'task').iterdir():
                    try:os.sched_setaffinity(int(task.name),affinity)
                    except ProcessLookupError:pass
                state['cpu_affinity']=affinity
            temp=out/'resource-state.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'resource-state.json')
            writer.writerow([now,total,used,util,encoder,duty,mode]);h.flush()
            if mode!=last_mode:print('RESOURCE_MODE',json.dumps(state),flush=True);last_mode=mode
            if c['cooperative']:resume();wait(period);continue
            if duty>0:resume();wait(period*duty)
            if duty<1 and alive() and not stopping:os.kill(a.pid,signal.SIGSTOP);paused=True;wait(period*(1-duty))
finally:
    resume()
    (out/'resource-controller-exit.json').write_text(json.dumps(dict(pid=a.pid,ended_unix=time.time(),training_alive=alive(),target_resumed=True),indent=2)+'\n')
