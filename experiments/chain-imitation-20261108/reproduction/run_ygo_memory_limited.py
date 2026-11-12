"""Root supervisor: memory-limit only the requested Yu-Gi-Oh child job."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--name',required=True)
parser.add_argument('--record',type=Path,required=True)
parser.add_argument('--limit-mib',type=int,default=5120)
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
assert os.geteuid()==0
assert re.fullmatch(r'[a-z0-9_-]+',args.name)
assert 1024<=args.limit_mib<=6144
command=args.command[1:] if args.command[:1]==['--'] else args.command
assert command
group=Path('/sys/fs/cgroup')/('codex-ygo-'+args.name)
group.mkdir()
(group/'memory.max').write_text(str(args.limit_mib*1024*1024))
(group/'memory.swap.max').write_text('0')
if (group/'memory.oom.group').exists(): (group/'memory.oom.group').write_text('1')
start=time.perf_counter()
def enter_group(): (group/'cgroup.procs').write_text(str(os.getpid()))
def stop_children(*unused):
    if (group/'cgroup.kill').exists(): (group/'cgroup.kill').write_text('1')
signal.signal(signal.SIGINT,stop_children)
signal.signal(signal.SIGTERM,stop_children)
process=subprocess.Popen(['runuser','-u','hatk8','--',*command],preexec_fn=enter_group)
code=process.wait()
if (group/'cgroup.procs').read_text().strip(): stop_children()
events={key:int(value) for key,value in (line.split() for line in (group/'memory.events').read_text().splitlines())}
peak=int((group/'memory.peak').read_text()) if (group/'memory.peak').exists() else None
record=dict(completed_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(),command=command,
            limit_mib=args.limit_mib,peak_bytes=peak,memory_events=events,exit_code=code,
            elapsed_seconds=time.perf_counter()-start)
args.record.write_text(json.dumps(record,indent=2)+'\n')
for _ in range(20):
    try: group.rmdir(); break
    except OSError: time.sleep(.05)
print('MEMORY_LIMIT_RESULT',json.dumps(record),flush=True)
raise SystemExit(code if code>=0 else 128-code)
