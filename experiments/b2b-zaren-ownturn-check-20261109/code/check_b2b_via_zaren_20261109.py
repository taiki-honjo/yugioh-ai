"""User-specified own-turn route: Track 4 + Ash 3 -> Zaren 7; Zaren + Cue 3 -> B2B.

One record and one replay; separate ledger with a hard cap of two duels.
Uses the unchanged, pinned local engine and the existing train hand and seed.
No model, training, production configuration changes, or card-state edits.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-zaren-ownturn-check-20261109'
BASE = P / 'experiments/b2b-ownturn-check-20261109/code/check_b2b_ownturn_20261109.py'
spec = importlib.util.spec_from_file_location('original_check', BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = OUT
ZAREN = 4891376
prepare_original = module.prepare
route_original = module.route_case


def route_case(raw, variant):
    assert variant == 'via-zaren'
    case = route_original(raw, 'standard')
    cost_at = next(i for i, a in enumerate(case['plan']) if a['act'] == 'None' and a['code'] == module.ASH and a.get('after_location') == 16)
    # Keep the line through Cue's recruitment of Ash; omit JJ's tribute effect.
    assert case['plan'][cost_at - 1] == dict(act='Activate', code=14442329)
    case['plan'] = case['plan'][:cost_at - 1] + [dict(act='SpSummon', code=ZAREN), dict(act='SpSummon', code=module.B2B), dict(act='Set', code=module.SYNC)]
    case['materials'][ZAREN] = [[(module.ASH, 'm'), (module.TRACK, 'm')]]
    case['materials'][module.B2B] = [[(module.CUE, 'm'), (ZAREN, 'm')]]
    return case


def prepare():
    cfg, raw, witness, analysis = prepare_original()
    original_ledger = analysis.Ledger

    class TwoRunLedger(original_ledger):
        def __init__(self, path, limit=2):
            super().__init__(path, limit=2)

    analysis.Ledger = TwoRunLedger
    return cfg, raw, witness, analysis


module.prepare = prepare
module.route_case = route_case
if '--job' in sys.argv:
    module.main('via-zaren', replay='--replay' in sys.argv)
else:
    cfg = module.C.load_config(module.CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'code/check_b2b_via_zaren_20261109.py'
    target.parent.mkdir(exist_ok=True)
    source = Path(__file__).read_bytes()
    if target.exists():
        assert target.read_bytes() == source, 'Diagnostic code changed after registration'
    else:
        target.write_bytes(source)
        module.write(OUT / 'plan.json', dict(
            date='2026-11-09', engine_run_limit=2, operations=['record', 'replay'],
            wrapper_sha256=module.C.sha(target), base_driver=str(BASE), base_driver_sha256=module.C.sha(BASE),
            case_id=module.CASE_ID, seed=151080, split='train',
            route='Trackmaker 4 + recruited Ash 3 -> Zaren 7; Zaren 7 + Cue 3 -> B2B 10; set Synchro',
            model_used=False, training=False, production_writes=False,
            master_duel_client_used=False, graveyard_hate_injected=False))
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(target), '--job']
    replay = '--replay' in sys.argv
    if replay:
        cmd.append('--replay')
    log = OUT / ('replay.log' if replay else 'record.log')
    assert not log.exists(), 'Refusing to overwrite the diagnostic log'
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=module.C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT, timeout=90)
    print(log.read_text()[-5500:])
    sys.exit(done.returncode)
