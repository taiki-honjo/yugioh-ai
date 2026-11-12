"""Offline tests for the context-retention study (no engine duel; CPU; small truncated batches)."""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
PREV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rl_runs as RR  # noqa: E402  (frozen modules from this root first)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import batch_model as BM  # noqa: E402
import cr_data as CD  # noqa: E402
import cr_model as CM  # noqa: E402
import pilot_data as PD  # noqa: E402
import pilot_model as PM  # noqa: E402
import rl_data as RD  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Shared:
    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            responses, metas = {}, {}
            for n in RD.NAMES:
                responses[n], metas[n] = RD.build_response(n)
            params, bstats, _ = PM.load_pair(ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
            cls.built = (responses, metas, params, bstats)
        return cls.built


def small_batch(T=8):
    """S1/S2 truncated to T steps; loss on the last step (its recorded label), KL mask = preserve mask before it."""
    responses, metas, params, bstats = Shared.get()
    items, keeps = [], []
    for n in RD.NAMES:
        s = {k: v[:T].copy() for k, v in responses[n].items()}
        s['loss'] = np.zeros(T, np.float32); s['loss'][-1] = 1.0
        keep = CD.preserve_mask(responses[n], metas[n])[:T].copy(); keep[-1] = 0.0
        items.append(s); keeps.append(keep)
    b = PD.batch(items, length=T)
    lp = np.asarray(CM.legal_logp(params, bstats, {k: jnp.asarray(v) for k, v in b.items()}))
    b.update(ref_logp=np.where(b['actions_'][..., 3] != 0, lp, CD.MASKED).astype(np.float32), legal=b['actions_'][..., 3] != 0,
             kl_mask=np.stack(keeps, 1).astype(np.float32))
    return b, params, bstats


class MaskTests(unittest.TestCase):
    def test_window_position_and_single_legal_steps_are_not_preserved(self):
        responses, metas, _, _ = Shared.get()
        for n in RD.NAMES:
            s, m = responses[n], metas[n]
            keep = CD.preserve_mask(s, m)
            self.assertEqual(keep[m['target_index']], 0.0)
            self.assertTrue((keep[s['message'] == PD.POSITION_MESSAGE] == 0).all())
            self.assertTrue((keep[(s['actions_'][:, :, 3] != 0).sum(1) < 2] == 0).all())
            self.assertGreater(keep.sum(), 10)

    def test_s2_history_contains_a_forced_step_kept_as_a_distribution_not_a_label(self):
        responses, metas, _, _ = Shared.get()
        ms = CD.step_meanings('response-S2')
        forced = [i for i, x in enumerate(ms) if x['who'] == 'branch-control']
        self.assertEqual(len(forced), 1)
        i = forced[0]
        self.assertEqual(ms[i]['actions'][ms[i]['executed']]['code'], 1122030)  # 襲撃 executed by the experiment
        self.assertEqual(CD.preserve_mask(responses['response-S2'], metas['response-S2'])[i], 1.0)


class LossTests(unittest.TestCase):
    def test_kl_is_zero_at_the_reference_and_ignores_executed_labels(self):
        b, params, bstats = small_batch()
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        kl, aux = CM.kl_context(params, bstats, jb)
        self.assertLess(float(kl), 1e-5)
        jb2 = dict(jb, labels=jnp.zeros_like(jb['labels']))
        self.assertEqual(float(CM.kl_context(params, bstats, jb2)[0]), float(kl))

    def test_kl_uses_legal_options_only_and_stays_finite(self):
        b, params, bstats = small_batch()
        uniform = np.where(b['legal'], 0.0, CD.MASKED).astype(np.float32)
        uniform = uniform - np.log(np.maximum(b['legal'].sum(-1, keepdims=True), 1))
        b2 = dict(b, ref_logp=np.where(b['legal'], uniform, CD.MASKED).astype(np.float32))
        kl, _ = CM.kl_context(params, bstats, {k: jnp.asarray(v) for k, v in b2.items()})
        self.assertTrue(np.isfinite(float(kl)))
        self.assertGreater(float(kl), 0.0)

    def test_reference_receives_no_gradient(self):
        b, params, bstats = small_batch()
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        g = jax.grad(lambda r: CM.kl_context(params, bstats, dict(jb, ref_logp=r))[0])(jb['ref_logp'])
        self.assertEqual(float(jnp.max(jnp.abs(g))), 0.0)

    def test_coefficient_zero_equals_the_original_update(self):
        b, params, bstats = small_batch(T=6)
        jb = {k: jnp.asarray(v) for k, v in b.items()}
        opt = BM.make_optimizer()
        p0, o0, _ = BM.update(params, opt.init(params), bstats, jb, opt)
        p1, o1, m1 = CM.update_response(params, opt.init(params), bstats, jb, opt, coef=0.0)
        diff = max(float(jnp.max(jnp.abs(a - c))) for a, c in zip(jax.tree.leaves(p0), jax.tree.leaves(p1)))
        self.assertEqual(diff, 0.0)
        self.assertGreater(float(m1['kl_grad_norm']), -1.0)


class SamplerAndEvaluationTests(unittest.TestCase):
    def test_draw_order_equals_the_response_study(self):
        existing, names, pairs, stats = RD.HD.load()
        s = RD.BlockSampler8(existing, pairs, 11111301)
        prev = [json.loads(l) for l in (PREV / 'train/updates.jsonl').read_text().splitlines()]
        for u in prev:
            kind, v = s.next()
            used = list(v) if kind == 'response' else [f'{g}:{i}' for g, i in v] if kind == 'existing' else None
            self.assertEqual(kind, u['kind'])
            if used is not None:
                self.assertEqual(used, u['sequences'])

    def test_evaluation_reads_this_root(self):
        self.assertEqual(RR.ROOT, ROOT)
        for m in (RR.F, RR.HR, CD, CM, RD):
            self.assertEqual(Path(m.__file__).resolve().parent, ROOT / 'code')
        for n, digest in RR.FROZEN.items():
            self.assertEqual(sha(ROOT / 'code' / n), digest)
        self.assertEqual(RD.MANIFEST, PREV / 'data/manifest.json')


if __name__ == '__main__':
    unittest.main()
