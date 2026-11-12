"""Reproduce the A-777 battle stop with the UNCHANGED v1 placement adapter and log the
PlacementMemory event sequence. Evaluation only; writes only into this directory."""
import json, sys, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
cfg = C.load_config(P / 'experiments/opening-rehearsal-ab-20261109/configs/a-ppo.json')
common = C.backend().policy_common(cfg)
import center_placement as cp
assert C.sha(cp.__file__) == '805e68213710ad411ae4c5a3fc47bd998a8eb7ed2113a3335386eeed7db00b3c'
failures = []
original_observe, original_reset, original_act = cp.PlacementMemory.observe, cp.PlacementMemory.reset, cp.PlacementPolicy.act

def reset(self):
    original_reset(self); self._hist = []; self._ctx = None

def act(self, obs, info):
    actions = np.asarray(obs['actions_'])[0]; g = np.asarray(obs['global_'])[0]; n = int(info['num_options'][0])
    self.memory._ctx = dict(turn=int(g[4]), phase=int(g[5]), own_turn=bool(g[7]), chain_length=int(info['public_chain_length'][0]),
        chain_card=int(info['public_chain_card_id'][0]), chain_player=int(info['public_chain_player'][0]),
        options=[[self.id_codes.get(int(r[1]) * 256 + int(r[2]), 0), int(r[3]), int(r[4]), str(r[0])] for r in actions[:n]])
    return original_act(self, obs, info)

def observe(self, msg, act, code, hand=False):
    entry = dict(msg=msg, act=act, code=code, hand=bool(hand), context=getattr(self, '_ctx', None),
                 before=dict(incoming=self.incoming, locked=self.locked, remix=None if self.remix_targets is None else list(self.remix_targets)))
    self.__dict__.setdefault('_hist', []).append(entry)
    try:
        return original_observe(self, msg, act, code, hand)
    except Exception as exc:
        failures.append(dict(error=repr(exc), history=self._hist[-30:])); raise

cp.PlacementMemory.reset, cp.PlacementMemory.observe, cp.PlacementPolicy.act = reset, observe, act
run = P / 'experiments/opening-rehearsal-ab-20261109/a-ppo'
args = SimpleNamespace(candidate=str(run / 'cycles/256-seed10000777/candidate'), opponent=str(run / 'models/initial'),
                       episodes=32, seed=90000555, result=str(OUT / 'repro/battle-v1-adapter.json'))
outcome = 'completed'
try:
    C.evaluate_job(args, cfg)
except Exception as exc:
    outcome = f'{type(exc).__name__}: {exc}'; traceback.print_exc()
(OUT / 'repro').mkdir(exist_ok=True)
(OUT / 'repro/remix-crash-v1.json').write_text(json.dumps(dict(outcome=outcome, adapter_sha256=C.sha(cp.__file__),
    candidate=args.candidate, opponent=args.opponent, episodes=32, seed=90000555, failures=failures), indent=1) + '\n')
print('REPRO_OUTCOME', outcome, len(failures), flush=True)
