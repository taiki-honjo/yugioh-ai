"""Offline tests for the relative-score retention study (no engine duel, CPU)."""
import hashlib
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111')
CV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rs_probe_runs as PR  # noqa: E402  (frozen modules from this root first)
import rs_train as RT  # noqa: E402
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
    actions[:, :5, 2] = rng.integers(1, 200, (4, 5))
    actions[:, :5, 3] = 2
    actions[:, :5, 4] = 1
    frozen = rng.normal(0, 3, (4, 128)).astype(np.float32)
    frozen[:, 5:] = 1e6
    ref = np.stack([RT.legal_logsoftmax(frozen[j].astype(np.float64), legal[j]) for j in range(4)]).astype(np.float32)
    pass_index = np.array([4, 4, 4, 4], np.int32)
    label = np.array([1, 2, 0, 0], np.int32)
    protect = legal.copy()
    protect[np.arange(4), pass_index] = False
    protect[0, 1] = protect[1, 2] = False  # the taught options of the two CE rows
    return dict(frozen=jnp.asarray(frozen), chain=jnp.asarray(np.array([[1, 2, 43, 0]] * 4, np.int32)), actions=jnp.asarray(actions),
                legal=jnp.asarray(legal), label=jnp.asarray(label), ref_logp=jnp.asarray(ref), is_ce=jnp.asarray(np.array([True, True, False, False])),
                pass_index=jnp.asarray(pass_index), protect=jnp.asarray(protect))


def nonzero(seed=3):
    p = flax.core.unfreeze(SM.init_adapter(seed))
    p['params']['output']['kernel'] = jax.random.normal(jax.random.PRNGKey(5), p['params']['output']['kernel'].shape)
    return p


class RelativeTermTests(unittest.TestCase):
    def test_invariant_to_a_common_shift_and_zero_at_init(self):
        b = synthetic_batch()
        p = nonzero()
        q = flax.core.unfreeze(jax.tree.map(lambda x: x, p))
        q['params']['output']['bias'] = q['params']['output']['bias'] + 7.0  # adds 7 to every option's delta
        r1, _ = RT.rel_term(p, b)
        r2, _ = RT.rel_term(q, b)
        self.assertAlmostEqual(float(r1), float(r2), places=3)
        self.assertGreater(float(r1), 0.0)
        r0, _ = RT.rel_term(SM.init_adapter(RT.SEED), b)
        self.assertEqual(float(r0), 0.0)

    def test_taught_option_and_illegal_options_are_not_protected(self):
        b = synthetic_batch()
        p = nonzero()
        g = jax.grad(lambda a: RT.rel_term(a, b)[0])(p)
        # the gradient w.r.t. the action embedding row of the taught option of row 0 comes only from other rows sharing the card id
        base, aux = RT.rel_term(p, b)
        b2 = dict(b, protect=b['protect'].at[0, 1].set(True))
        self.assertNotAlmostEqual(float(RT.rel_term(p, b2)[0]), float(base), places=6)  # protecting the taught option would change the term
        b3 = dict(b, legal=b['legal'], protect=b['protect'])
        self.assertTrue(bool(jnp.all(~b3['protect'] | b3['legal'])))
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g)))

    def test_coefficient_zero_equals_the_coverage_update(self):
        b = synthetic_batch(1)
        p = nonzero(7)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-5))
        new0, _, m = RT.make_step(opt, b, coef=1.0, coef_rel=0.0)(p, opt.init(p))

        @jax.jit
        def coverage_step(a, o):  # the coverage objective: mean CE + 1.0 * mean KL (same jit path)
            g_ce = jax.grad(lambda x: RT.ce_term(x, b)[0])(a)
            g_kl = jax.grad(lambda x: RT.kl_term(x, b)[0])(a)
            g = jax.tree.map(lambda u, v: u + 1.0 * v, g_ce, g_kl)
            upd, o = opt.update(g, o, a)
            return optax.apply_updates(a, upd)
        diff = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(jax.tree.leaves(new0), jax.tree.leaves(coverage_step(p, opt.init(p)))))
        self.assertEqual(diff, 0.0)

    def test_ce_and_kl_code_is_the_coverage_code(self):
        import inspect
        src = (CV / 'code/cv_train.py').read_text()
        for fn in [RT.log_q, RT.ce_term, RT.kl_term, RT.legal_logsoftmax]:
            self.assertIn(inspect.getsource(fn), src)


class SetupTests(unittest.TestCase):
    def test_files_and_entries(self):
        self.assertEqual(sha(ROOT / 'code/sa_model.py'), sha(CV / 'code/sa_model.py'))
        self.assertEqual(tuple(sha(f) for f in RT.FROZEN), SM.FROZEN_SHA)
        self.assertEqual(PR.CAND, ROOT / 'train/candidate')
        self.assertIn('H1-USAGI-CAND', PR.COMPARISON)
        o = PR.Opponent('USAGI', None)
        self.assertEqual(o.code, PR.USAGI)
        for m in (PR.F, PR.SM, RT):
            self.assertEqual(Path(m.__file__).resolve().parent, ROOT / 'code')

    def test_cache_masks_when_prepared(self):
        if not RT.CACHE.exists():
            self.skipTest('cache not prepared yet')
        z = np.load(RT.CACHE)
        for k in RT.ORDER:
            prot, legal, pi = z[f'{k}__protect'], z[f'{k}__legal'], int(z[f'{k}__pass'])
            self.assertTrue(legal[pi] and not prot[pi])
            self.assertTrue(np.all(~prot | legal))
            if RT.SCREENS[k]['kind'] == 'ce':
                self.assertFalse(prot[int(z[f'{k}__label'])])
            else:
                self.assertEqual(int(prot.sum()), int(legal.sum()) - 1)


if __name__ == '__main__':
    unittest.main()
