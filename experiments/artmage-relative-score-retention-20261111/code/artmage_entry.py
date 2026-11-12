"""Isolated Artmage entry: explicit pinned paths and hashes; never reads or writes the common registry.

create_policy(base, head) loads any pinned base + chain-head pair (default: the work-area copies of the
registered start model) and wraps it with the common position adapter loaded by file (same as
scripts/deck_policy.py, but with explicit paths). The engine, vocabulary, deck and model code are the
same pinned files the registered Artmage policy uses. Importing this module imports the pinned
article_witness module, which initialises the local engine (no server access).
"""
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artmage_pins import (P, W, MODEL_CODE, ADAPTER_DIR, RUNTIME, META_RUN, NATIVE, DECK_KEY, PINS,  # noqa: E402,F401
                          sha, verify_pins)

verify_pins()
sys.path.insert(0, str(MODEL_CODE))
from opponent_article_bc_common import Policy, v, np  # noqa: E402

assert Path(v.__file__).resolve() == (MODEL_CODE / 'article_witness.py').resolve(), v.__file__
assert Path(v.run).resolve() == META_RUN.resolve(), ('engine deck run differs', v.run)
assert Path(v.R).resolve() == RUNTIME.resolve(), ('runtime differs', v.R)
_IDS = {i + 1: int(line.split()[0]) for i, line in enumerate(PINS['vocabulary'][0].read_text().splitlines())}
assert _IDS == v.ID_CODES, 'Loaded native vocabulary differs from the pinned vocabulary'


def _adapter():
    spec = importlib.util.spec_from_file_location('artmage_isolated_battle_position', PINS['adapter'][0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_policy(base=None, head=None, positions=True, expected=None):
    """Policy for an explicit base + chain-head pair. expected = (base_sha256, head_sha256) for non-default pairs."""
    base = Path(base or PINS['base'][0])
    head = Path(head or PINS['head'][0])
    want = expected or (PINS['base'][1], PINS['head'][1])
    got = (sha(base), sha(head))
    if got != tuple(want):
        raise RuntimeError(f'Model hash mismatch: {got} != {want}')
    policy = Policy(str(base), str(head))
    if positions:
        policy = _adapter().PositionPolicy(policy, _IDS)
    policy.deck_name = DECK_KEY
    policy.policy_metadata = dict(base=str(base), base_sha256=got[0], head=str(head), head_sha256=got[1],
                                  positions_enabled=positions, registry_read=False)
    return policy


def create_teacher(teacher):
    """Wrap a legal teacher with the same position adapter (positions decided from public information only)."""
    module = _adapter()
    return module.PositionTeacher(teacher, module.load_card_data())


def env_for(case, rows=None):
    """Fresh single env from a fixed-hand fixture (both players get case['hand']); returns env, obs, info."""
    v.native.set_benchmark_fixture(rows if rows is not None else v.fixture(case, DECK_KEY, case['seed']))
    env = v.ygoenv.make('YGOPro-v1', env_type='gymnasium', num_envs=1, num_threads=1, seed=case['seed'], deck1=DECK_KEY,
                        deck2=DECK_KEY, player=-1, play_mode='self', max_options=128, n_history_actions=32, oppo_info=False,
                        async_reset=False, max_steps=2048, max_episode_steps=2048, timeout=10)
    env.num_envs = 1
    env = v.EnvPreprocess(env, skip_mask=True)
    return (env, *env.reset())


def close(env):
    env.close()
