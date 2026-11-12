"""Launch RL-training jobs inside resource limits; the root's STOP is lifted only while the finite job runs.

  rlt_launch.py train                 GPU (one job; refuses if another compute process is on the GPU), job timeout 3,600 s
  rlt_launch.py eval retain NAME...   CPU, job timeout 1,800 s
  rlt_launch.py eval battles          CPU, job timeout 7,000 s (the job stops starting games after 6,600 s)

Limits: memory 5 GiB, no swap, CPU 300 %, nice 10. STOP is written back to "now" when the job ends, fails or is
interrupted. One training run only (train-runs.jsonl).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
RUNTIME = Path('/home/hatk8/projects/yugioh-ai/evaluation/opponent-article-basics-20261108/runtime')
PYTHON = '/opt/ygo-ai/.venv/bin/python'
LIMITS = dict(memory_gib=5, cpu_percent=300, nice=10)
TIMEOUTS = dict(train=3600, retain=1800, battles=7000)
BASE_ENV = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', XLA_PYTHON_CLIENT_PREALLOCATE='false', MALLOC_ARENA_MAX='2',
                PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled', HF_HUB_OFFLINE='1')
JST = timezone(timedelta(hours=9))


def now():
    return datetime.now(JST).isoformat(timespec='seconds')


def gpu_busy():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 'nvidia-smi unavailable'
    return out.strip()


def main(argv):
    kind = argv[0]
    if not (ROOT / 'registration.json').exists():
        raise SystemExit('register first')
    stop = ROOT / 'STOP'
    if stop.read_text().strip() != 'now':
        raise SystemExit('STOP is not "now"; another job may be running')
    env = dict(os.environ, **BASE_ENV)
    run_number = None
    if kind == 'train':
        busy = gpu_busy()
        if busy:
            raise SystemExit(f'GPU has compute processes: {busy}')
        ledger = ROOT / 'train-runs.jsonl'
        started = [json.loads(l) for l in ledger.read_text().splitlines() if '"start"' in l] if ledger.exists() else []
        if started:
            raise SystemExit('a training run was started already; one candidate only')
        run_number = 1
        script, args, log, timeout = ROOT / 'code/rlt_train.py', [], ROOT / 'logs/train.log', TIMEOUTS['train']
        env.pop('JAX_PLATFORMS', None)
    elif kind == 'eval' and argv[1] in ('retain', 'battles'):
        script, args, timeout = ROOT / 'code/rlt_eval.py', argv[1:], TIMEOUTS[argv[1]]
        log = ROOT / 'logs' / f'eval-{argv[1]}.log'
        env['JAX_PLATFORMS'] = 'cpu'
    else:
        raise SystemExit(__doc__)
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={LIMITS['memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={LIMITS['cpu_percent']}%", 'nice', '-n', str(LIMITS['nice']), 'timeout', str(timeout),
           PYTHON, '-B', '-u', str(script)] + args
    log.parent.mkdir(exist_ok=True)
    if kind == 'train':
        with (ROOT / 'train-runs.jsonl').open('a') as h:
            h.write(json.dumps(dict(event='start', run=run_number, time_jst=now(), command=cmd)) + '\n')
    stop.write_text(f'lifted for {kind} {" ".join(argv[1:3])} at {now()}\n')
    code = None
    try:
        with log.open('a') as handle:
            handle.write(f'=== {now()} {" ".join(cmd)}\n')
            handle.flush()
            code = subprocess.run(cmd, cwd=RUNTIME / 'agent/scripts', env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    finally:
        stop.write_text('now\n')
        if kind == 'train':
            with (ROOT / 'train-runs.jsonl').open('a') as h:
                h.write(json.dumps(dict(event='finish', run=run_number, time_jst=now(), exit_code=code)) + '\n')
    keep = ('RESULT', 'SAVED', 'ABORTED', 'DEVICES', 'RT1', 'RETAIN', 'BATTLE', 'WALL', 'LEDGER', 'Traceback', 'Error', 'RuntimeError', 'AssertionError')
    lines = [l for l in log.read_text().splitlines() if l.startswith(keep)]
    print('\n'.join(lines[-40:]))
    print(f'exit={code} log={log} STOP={stop.read_text().strip()}')


if __name__ == '__main__':
    main(sys.argv[1:])
