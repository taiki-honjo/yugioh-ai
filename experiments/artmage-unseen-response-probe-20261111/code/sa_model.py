"""Scoped response adapter: frozen base + frozen chain head + a new response adapter (3 files + manifest).

Inference (per own decision):
    state, frozen_logits = Policy.infer(base, head, state, obs, features(info))   # the UNCHANGED original computation
    if gate(obs, features(info)):  logits = frozen_logits + adapter(features(info), obs['actions_'])
    else:                          logits = frozen_logits                           # returned as is
gate = select_chain (actions_[0, 3] == 2) AND public chain length >= 1 AND relative chain role == 2 (the latest
link's player is the opponent of the player to act; features(info) role). No card ids are used by the gate.
The adapter is a NEW ChainHead instance (same class as the existing chain head, independent parameters, zero output
layer => exactly 0 at initialisation); it has no state and does not touch the RNN. The position adapter stays outside.
The adapter only shifts action scores; it is not a "always negate" rule. It corrects the frozen, state-dependent
logits from the public chain and action features only; it does not read the whole board itself.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import artmage_entry as E  # noqa: F401  (pins checked; puts the pinned model code on sys.path)
from opponent_article_bc_common import ChainHead, Policy, features, load  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.serialization as ser  # noqa: E402

TYPE, VERSION = 'artmage-scoped-response-adapter', 1
SELECT_CHAIN = 2  # message id of select_chain in actions_[..., 3] (checked against saved NPZ/snapshots)
ADAPTER = ChainHead()
FROZEN_SHA = ('b9c7751f5c1f7179faa4911e1a52e8603e332fd9700a0f301dd4e759f8208f66', '77191d5d9a9aaa65d63902c93e47c0e3274009e7a91ee1e8e8abb48fcd73c6a2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate(actions, chain):
    """actions: [A, 12] of one observation; chain: [4] = features(info) row (length, role, card, effect)."""
    return bool(int(actions[0, 3]) == SELECT_CHAIN and int(chain[0]) >= 1 and int(chain[1]) == 2)


def init_adapter(seed):
    return ADAPTER.init(jax.random.PRNGKey(seed), jnp.zeros((1, 4), jnp.int32), jnp.zeros((1, 128, 12), jnp.uint8))


@jax.jit
def adapter_delta(adapter, chain, actions):
    return ADAPTER.apply(adapter, chain, actions)


def save_adapter(params, path):
    Path(path).write_bytes(ser.msgpack_serialize(jax.device_get(params)))


def write_manifest(folder, base, head, adapter, extra=None):
    m = dict(type=TYPE, version=VERSION, base=dict(file=Path(base).name, sha256=sha(base)), head=dict(file=Path(head).name, sha256=sha(head)),
             adapter=dict(file=Path(adapter).name, sha256=sha(adapter), module='opponent_article_bc_common.ChainHead (new parameters)'),
             gate='select_chain AND chain length >= 1 AND relative role == 2', **(extra or {}))
    (Path(folder) / 'manifest.json').write_text(json.dumps(m, indent=1) + '\n')
    return m


def read_manifest(folder):
    """Validate the 3-file candidate; refuse missing files, wrong type/version or hash mismatches."""
    folder = Path(folder)
    m = json.loads((folder / 'manifest.json').read_text())
    if (m.get('type'), m.get('version')) != (TYPE, VERSION):
        raise RuntimeError(f"unsupported policy type/version: {m.get('type')} {m.get('version')}")
    paths = {}
    for part in ['base', 'head', 'adapter']:
        p = folder / m[part]['file']
        if not p.exists():
            raise RuntimeError(f'missing {part} file: {p}')
        if sha(p) != m[part]['sha256']:
            raise RuntimeError(f'{part} hash mismatch: {p}')
        paths[part] = p
    if (m['base']['sha256'], m['head']['sha256']) != FROZEN_SHA:
        raise RuntimeError('base/head are not the frozen start pair')
    return m, paths


class AdapterPolicy:
    """Frozen Policy (unchanged object and jit) plus the gated adapter; logs every gated decision."""

    def __init__(self, base_path, head_path, adapter_path):
        self.inner = Policy(str(base_path), str(head_path))
        self.adapter = load(adapter_path)
        self.calls, self.log = 0, []

    def reset(self):
        self.inner.reset()
        self.calls = 0

    def step(self, obs, info):
        """(final logits, gated). The frozen part is exactly Policy.act's computation (same object, same jit)."""
        chain = features(info)
        self.inner.state, logits = self.inner.infer(self.inner.base, self.inner.head, self.inner.state, obs, chain)
        call = self.calls
        self.calls += 1
        actions = np.asarray(obs['actions_'])[0]
        if not gate(actions, chain[0]):
            return logits, False
        delta = adapter_delta(self.adapter, jnp.asarray(chain), jnp.asarray(obs['actions_']))
        final = logits + delta
        f0, f1, d = np.asarray(logits)[0], np.asarray(final)[0], np.asarray(delta)[0]
        legal = actions[:, 3] != 0
        self.log.append(dict(call=call, chain=[int(x) for x in chain[0]], legal=int(legal.sum()), frozen_argmax=int(f0.argmax()), final_argmax=int(f1.argmax()),
                             max_abs_delta_legal=float(np.abs(d[legal]).max()) if legal.any() else 0.0,
                             delta_frozen_choice=float(d[f0.argmax()]), delta_final_choice=float(d[f1.argmax()])))
        return final, True

    def act(self, obs, info):
        logits, _ = self.step(obs, info)
        return int(np.asarray(logits)[0].argmax())


def create_adapter_policy(folder, positions=True):
    m, paths = read_manifest(folder)
    inner = AdapterPolicy(paths['base'], paths['head'], paths['adapter'])
    policy = E._adapter().PositionPolicy(inner, E._IDS) if positions else inner
    policy.deck_name = E.DECK_KEY
    policy.policy_metadata = dict(type=TYPE, version=VERSION, manifest=str(Path(folder) / 'manifest.json'), manifest_sha256=sha(Path(folder) / 'manifest.json'),
                                  base_sha256=m['base']['sha256'], head_sha256=m['head']['sha256'], adapter_sha256=m['adapter']['sha256'],
                                  positions_enabled=positions, registry_read=False)
    policy.adapter_policy = inner
    return policy


class AdapterEntry:
    """Stands in for the isolated entry module when it is passed to the frozen runners: every attribute is the entry's,
    except create_policy, which builds the 3-file policy (the runners pass the frozen base/head pair; it is verified)."""

    def __init__(self, entry, folder):
        self._entry, self.folder, self.created = entry, Path(folder), []

    def __getattr__(self, name):
        return getattr(self._entry, name)

    def create_policy(self, base=None, head=None, positions=True, expected=None):
        if (sha(base), sha(head)) != FROZEN_SHA or tuple(expected) != FROZEN_SHA:
            raise RuntimeError('the runner did not pass the frozen base/head pair')
        policy = create_adapter_policy(self.folder, positions)
        self.created.append(policy)
        return policy

    def last_info(self):
        p = self.created[-1]
        return dict(p.policy_metadata, decisions=p.adapter_policy.calls, adapter_active=p.adapter_policy.log)
