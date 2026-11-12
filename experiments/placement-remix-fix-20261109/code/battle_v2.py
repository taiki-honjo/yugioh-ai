"""Re-run the battle that stopped (A-777 candidate vs adopted model, 32 games, seed 90000555)
through the normal evaluation code, with the v2 adapter pinned in root-v2. Counting wrappers
only observe the adapter; they do not change any decision. Writes only into this directory."""
import collections, json, sys, traceback
from pathlib import Path
from types import SimpleNamespace
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
cfg = C.load_config(OUT / 'configs/root-v2.json')
common = C.backend().policy_common(cfg)
import center_placement as cp
assert Path(cp.__file__).resolve() == (OUT / 'root-v2/snapshot/policy-code/center_placement.py').resolve()
assert C.sha(cp.__file__) == C.sha(P / 'evaluation/placement-remix-fix-20261109/code/center_placement.py')
counts, ended = collections.Counter(), collections.Counter()
original_observe, original_end = cp.PlacementMemory.observe, cp.PlacementMemory.end_remix

def observe(self, msg, act, code, hand=False, chain_length=None):
    before = None if self.remix_targets is None else list(self.remix_targets)
    result = original_observe(self, msg, act, code, hand, chain_length)
    if act == 8 and code == cp.REMIX:
        counts[f'remix_activate_label_msg{msg}'] += 1
    if before is not None and len(before) == 2 and code in before and msg in (3, 9) and self.locked:
        counts['remix_resolved_with_arrival'] += 1
    return result

def end_remix(self, reason, code=0):
    if self.remix_targets is not None:
        ended[reason] += 1
    return original_end(self, reason, code)

cp.PlacementMemory.observe, cp.PlacementMemory.end_remix = observe, end_remix
run = P / 'experiments/opening-rehearsal-ab-20261109/a-ppo'
(OUT / 'battle-v2').mkdir(exist_ok=True)
args = SimpleNamespace(candidate=str(run / 'cycles/256-seed10000777/candidate'), opponent=str(OUT / 'root-v2/models/initial'),
                       episodes=32, seed=90000555, result=str(OUT / 'battle-v2/battle.json'))
outcome = 'completed'
try:
    C.evaluate_job(args, cfg)
except Exception as exc:
    outcome = f'{type(exc).__name__}: {exc}'; traceback.print_exc()
C.atomic_json(OUT / 'battle-v2/remix-tracking.json', dict(outcome=outcome, adapter_sha256=C.sha(cp.__file__), counts=counts,
                                                          tracking_ended=ended, candidate=args.candidate, opponent=args.opponent,
                                                          episodes=32, seed=90000555))
print('BATTLE_V2', outcome, dict(counts), dict(ended), flush=True)
