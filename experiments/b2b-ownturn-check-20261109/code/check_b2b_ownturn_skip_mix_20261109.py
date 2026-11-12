"""One additional route within the original four-duel budget.

The attempted Clip search was illegal (Clip is level 2 like Mix).
Keep that failed run unchanged. Instead decline Mix's optional effect entirely.
This file only composes the existing isolated diagnostic; no production code edits.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('/home/hatk8/projects/yugioh-ai/experiments/b2b-ownturn-check-20261109')
spec = importlib.util.spec_from_file_location('original_check', BASE / 'code/check_b2b_ownturn_20261109.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_route_case = module.route_case


def route_case(raw, variant):
    assert variant == 'skip-mix-effect'
    case = original_route_case(raw, 'standard')
    jj = next(a for a in case['plan'] if a['act'] == 'None' and a['code'] == module.ASH and a.get('after_location') == 16)
    jj['code'] = module.CUE
    at = next(i for i, a in enumerate(case['plan']) if a['act'] == 'Activate' and a['code'] == module.MIX)
    del case['plan'][at:at + 2]
    case['materials'][module.REMIX] = [[(module.MIX, 'm'), (module.ASH, 'm')]]
    return case


module.route_case = route_case
if '--job' in sys.argv:
    module.main('skip-mix-effect', replay=False)
else:
    cfg = module.C.load_config(module.CONFIG)
    target = BASE / 'code/check_b2b_ownturn_skip_mix_20261109.py'
    assert not target.exists()
    target.write_bytes(Path(__file__).read_bytes())
    module.write(BASE / 'skip-mix-effect-code.json', dict(
        wrapper_sha256=module.C.sha(target), base_code_sha256=module.C.sha(BASE / 'code/check_b2b_ownturn_20261109.py'),
        engine_run_limit=4, change='Do not activate Mix; preserve Ash by releasing Cue to JJ; all cards and starting hands unchanged.'))
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(target), '--job']
    log = BASE / 'skip-mix-effect-record.log'
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=module.C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT, timeout=90)
    print(log.read_text()[-5500:])
    sys.exit(done.returncode)
