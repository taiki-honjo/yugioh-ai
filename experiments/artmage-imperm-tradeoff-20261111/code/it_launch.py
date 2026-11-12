"""Launch Imperm trade-off jobs inside resource limits; the root's STOP is lifted only while the finite job runs.

  it_launch.py runs record|replay NAME...   (CPU, engine; scripted functional test, no model, no training)

Limits: memory 5 GiB, no swap, CPU 300 %, 1,800 s, nice 10. CPU only (JAX_PLATFORMS=cpu). STOP is written back to "now"
when the job ends, fails or is interrupted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111')
RUNTIME = Path('/home/hatk8/projects/yugioh-ai/evaluation/opponent-article-basics-20261108/runtime')
PYTHON = '/opt/ygo-ai/.venv/bin/python'
LIMITS = dict(memory_gib=5, cpu_percent=300, timeout_seconds=1800, nice=10)
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
    if kind != 'runs':
        raise SystemExit('only "runs" jobs exist in this probe')
    if True:
        script, args, log = ROOT / 'code/it_runs.py', argv[1:], ROOT / 'logs' / ('runs-' + '-'.join(argv[1:3]) + '.log')
        env['JAX_PLATFORMS'] = 'cpu'
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={LIMITS['memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={LIMITS['cpu_percent']}%", 'nice', '-n', str(LIMITS['nice']), 'timeout', str(LIMITS['timeout_seconds']),
           PYTHON, '-B', '-u', str(script)] + args
    log.parent.mkdir(exist_ok=True)
    if kind == 'train':
        with (ROOT / 'train-runs.jsonl').open('a') as h:
            h.write(json.dumps(dict(event='start', run=run_number, time_jst=now(), command=cmd)) + '\n')
    stop.write_text(f'lifted for {kind} {" ".join(argv[1:])} at {now()}\n')
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
    keep = ('RECORD', 'REPLAY', 'VALIDATE', 'FREE', 'SUBSTITUTE', 'RETAIN', 'RT1', 'LEDGER', 'DIAG', 'BRANCH', 'RESPONSE', 'CURRENT', 'PREPARE', 'SCREENS', 'POLICY', 'RESULT', 'SAVED', 'ABORTED', 'DEVICES', 'Traceback', 'Error', 'RuntimeError', 'AssertionError')
    lines = [l for l in log.read_text().splitlines() if l.startswith(keep)]
    print('\n'.join(lines[-60:]))
    print(f'exit={code} log={log} STOP={stop.read_text().strip()}')


if __name__ == '__main__':
    main(sys.argv[1:])
