"""Offline tests for the response-coverage study (no engine duel, CPU, synthetic or saved inputs)."""
import hashlib
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
SA = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import cv_runs as CR  # noqa: E402  (frozen modules from this root first)
import cv_train as CT  # noqa: E402
import sa_model as SM  # noqa: E402
import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def synthetic_batch(seed=0):
    rng = np.random.default_rng(seed)
    legal = np.zeros((4, 128), bool)
    legal[:, :5] = True
    actions = np.zeros((4, 128, 12), np.uint8)
    actions[:, :5, 1] = 0
    actions[:, :5, 2] = rng.integers(1, 200, (4, 5))
    actions[:, :5, 3] = 2
    actions[:, :5, 4] = 1
    frozen = rng.normal(0, 3, (4, 128)).astype(np.float32)
    frozen[:, 5:] = 1e6  # illegal options with huge logits must not matter
    ref = np.stack([CT.legal_logsoftmax(frozen[j].astype(np.float64), legal[j]) for j in range(4)]).astype(np.float32)
    return dict(frozen=jnp.asarray(frozen), chain=jnp.asarray(np.array([[1, 2, 43, 0]] * 4, np.int32)), actions=jnp.asarray(actions),
                legal=jnp.asarray(legal), label=jnp.asarray(np.array([1, 2, 0, 0], np.int32)), ref_logp=jnp.asarray(ref),
                is_ce=jnp.asarray(np.array([True, True, False, False])))


def nonzero(seed=3):
    p = flax.core.unfreeze(SM.init_adapter(seed))
    p['params']['output']['kernel'] = jax.random.normal(jax.random.PRNGKey(5), p['params']['output']['kernel'].shape)
    return p


class LossTests(unittest.TestCase):
    def test_reference_has_no_gradient_and_kl_is_legal_only(self):
        b = synthetic_batch()
        p = nonzero()
        g = jax.grad(lambda r: CT.kl_term(p, dict(b, ref_logp=r))[0])(b['ref_logp'])
        self.assertEqual(float(jnp.max(jnp.abs(g))), 0.0)
        kl, aux = CT.kl_term(p, b)
        self.assertTrue(np.isfinite(float(kl)) and float(kl) > 0)
        z, _ = CT.kl_term(SM.init_adapter(CT.SEED), b)
        self.assertLess(float(z), 1e-6)  # zero adapter = the frozen distribution

    def test_coefficient_zero_equals_the_positive_only_update(self):
        b = synthetic_batch(1)
        p = nonzero(7)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-5))
        new0, _, m = CT.make_step(opt, b, coef=0.0)(p, opt.init(p))
        # the scoped study's loss: legal-option CE averaged over the positive screens only
        pos = {k: v[:2] for k, v in b.items()}

        def legal_ce(a, batch):
            delta = SM.ADAPTER.apply(a, batch['chain'], batch['actions'])
            lq = jax.nn.log_softmax(jnp.where(batch['legal'], batch['frozen'] + delta, -1e9), -1)
            return jnp.mean(-jnp.take_along_axis(lq, batch['label'][:, None], -1)[:, 0])
        # (a) with coefficient 0 the KL term contributes exactly nothing: identical to the CE-term-only update
        @jax.jit
        def ce_only(a, o):
            g_ce = jax.grad(lambda x: CT.ce_term(x, b)[0])(a)
            upd_ce, o = opt.update(g_ce, o, a)
            return optax.apply_updates(a, upd_ce)
        exact = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(new0), jax.tree.leaves(ce_only(p, opt.init(p)))))
        self.assertEqual(exact, 0.0)
        # (b) the CE term equals the scoped study's positive-only loss and update up to float rounding (4-row vs 2-row batch shapes)
        g = jax.grad(legal_ce)(p, pos)
        upd, _ = opt.update(g, opt.init(p), p)
        new1 = optax.apply_updates(p, upd)
        diff = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(new0), jax.tree.leaves(new1)))
        self.assertLess(diff, 1e-6)
        self.assertAlmostEqual(float(m['ce']), float(legal_ce(p, pos)), places=5)


class ScreenTests(unittest.TestCase):
    def test_screens_use_npz_indices_and_match_the_records(self):
        rows = CT.screen_rows()
        for k, r in rows.items():
            self.assertNotEqual(r['index'], CT.SCREENS[k]['step'])
            h, i = r['history'], r['index']
            self.assertEqual(len(h['labels']), i + 1)
            self.assertTrue(SM.gate(h['actions_'][i], h['chain'][i]))
        self.assertEqual(rows['S1']['who'], 'probe-R1')
        self.assertEqual({rows[k]['who'] for k in ['S2', 'M', 'P']}, {'model'})


class StructureTests(unittest.TestCase):
    def test_inference_code_is_unchanged_and_the_runner_uses_the_three_file_entry(self):
        self.assertEqual(sha(ROOT / 'code/sa_model.py'), sha(SA / 'code/sa_model.py'))
        self.assertEqual(CR.ROOT, ROOT)
        self.assertEqual(len(CR.B_NAMES), 8)
        self.assertEqual(CR.LIMIT, 18)
        self.assertIn("E = SM.AdapterEntry(E0, ROOT / 'train/candidate')", (ROOT / 'code/cv_runs.py').read_text())
        for m in (CR.F, CR.HR, CR.SM, CT):
            self.assertEqual(Path(m.__file__).resolve().parent, ROOT / 'code')
        self.assertEqual(tuple(sha(f) for f in CT.FROZEN), SM.FROZEN_SHA)


if __name__ == '__main__':
    unittest.main()
