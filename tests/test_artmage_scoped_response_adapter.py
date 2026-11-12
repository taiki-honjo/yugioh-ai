"""Offline tests for the scoped response adapter (no engine duel, CPU)."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import sa_runs as SR  # noqa: E402  (frozen modules from this root)
import sa_model as SM  # noqa: E402
import rl_data as RD  # noqa: E402
import pilot_data as PD  # noqa: E402
import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from opponent_article_bc_common import Policy  # noqa: E402

FROZEN = (ROOT / 'models/frozen/base.flax_model', ROOT / 'models/frozen/chain-head.flax_model')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def folder(adapter):
    d = Path(tempfile.mkdtemp(prefix='sa-test-'))
    for f in FROZEN:
        shutil.copyfile(f, d / f.name)
    SM.save_adapter(adapter, d / 'response-adapter.flax_model')
    SM.write_manifest(d, d / 'base.flax_model', d / 'chain-head.flax_model', d / 'response-adapter.flax_model')
    return d


def nonzero():
    p = flax.core.unfreeze(SM.init_adapter(3))
    p['params']['output']['kernel'] = jax.random.normal(jax.random.PRNGKey(5), p['params']['output']['kernel'].shape)
    return p


def obs_info(s, t):
    obs = {k: s[k][t:t + 1] for k in PD.OBS_KEYS}
    obs['mask_'] = None
    ch = s['chain'][t]
    return obs, dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                     public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                     public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))


class GateTests(unittest.TestCase):
    def test_gate_definition_without_card_ids(self):
        a = np.zeros((128, 12), np.uint8)
        a[0, 3] = 2
        self.assertTrue(SM.gate(a, [1, 2, 43, 0]))
        self.assertTrue(SM.gate(a, [3, 2, 7, 5]))  # any card / effect / length >= 1
        self.assertFalse(SM.gate(a, [1, 1, 43, 0]))  # own chain
        self.assertFalse(SM.gate(a, [0, 0, 0, 0]))
        a[0, 3] = 3
        self.assertFalse(SM.gate(a, [1, 2, 43, 0]))  # select_card

    def test_targets_are_gated_and_the_message_id_matches(self):
        for n in RD.NAMES:
            s, m = RD.build_response(n)
            t = m['target_index']
            self.assertTrue(SM.gate(s['actions_'][t], s['chain'][t]))
            self.assertEqual(int(s['actions_'][t, 0, 3]), SM.SELECT_CHAIN)


class AdapterTests(unittest.TestCase):
    def test_zero_initial_adapter_and_optimizer_scope(self):
        p = SM.init_adapter(11111501)
        d = SM.ADAPTER.apply(p, jnp.asarray([[1, 2, 43, 0]], jnp.int32), jnp.ones((1, 128, 12), jnp.uint8))
        self.assertEqual(float(jnp.abs(d).max()), 0.0)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-5))
        st = opt.init(p)
        shapes = sorted(x.shape for x in jax.tree.leaves(st) if hasattr(x, 'shape') and x.ndim > 0)
        self.assertEqual(shapes, sorted([x.shape for x in jax.tree.leaves(p)] * 2))

    def test_factory_refuses_bad_candidates_and_entry_refuses_other_pairs(self):
        d = folder(nonzero())
        try:
            m = json.loads((d / 'manifest.json').read_text())
            m['adapter']['sha256'] = '0' * 64
            (d / 'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(RuntimeError):
                SM.create_adapter_policy(d, positions=False)
            m['adapter']['sha256'] = sha(d / 'response-adapter.flax_model'); m['type'] = 'other'
            (d / 'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(RuntimeError):
                SM.create_adapter_policy(d, positions=False)
        finally:
            shutil.rmtree(d)
        entry = SM.AdapterEntry(object(), ROOT)
        other = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111/train/candidate')
        with self.assertRaises(RuntimeError):
            entry.create_policy(other / 'base.flax_model', other / 'chain-head.flax_model', expected=(sha(other / 'base.flax_model'), sha(other / 'chain-head.flax_model')))

    def test_non_zero_adapter_is_ignored_off_target_and_used_on_target(self):
        d = folder(nonzero())
        try:
            s, m = RD.build_response('response-S1')
            ap = SM.create_adapter_policy(d, positions=False)
            base = Policy(str(FROZEN[0]), str(FROZEN[1]))
            for t, expect_change in [(0, False), (1, False), (m['target_index'], True)]:
                base.reset(); ap.reset()
                obs, info = obs_info(s, t)
                base.state, l0 = base.infer(base.base, base.head, base.state, obs, SM.features(info))
                l1, gated = ap.step(obs, info)
                self.assertEqual(gated, expect_change)
                self.assertEqual(not np.array_equal(np.asarray(l0), np.asarray(l1)), expect_change)
            self.assertEqual(len(ap.log), 1)
        finally:
            shutil.rmtree(d)


class RunnerTests(unittest.TestCase):
    def test_runner_uses_this_root_and_the_three_file_factory(self):
        self.assertEqual(SR.ROOT, ROOT)
        for m in (SR.F, SR.HR, SR.SM):
            self.assertEqual(Path(m.__file__).resolve().parent, ROOT / 'code')
        for n, digest in SR.FROZEN.items():
            self.assertEqual(sha(ROOT / 'code' / n), digest)
        src = (ROOT / 'code/sa_runs.py').read_text()
        self.assertIn("E = SM.AdapterEntry(E0, ROOT / 'train/candidate')", src)
        self.assertEqual(tuple(sha(f) for f in FROZEN), SM.FROZEN_SHA)


if __name__ == '__main__':
    unittest.main()
