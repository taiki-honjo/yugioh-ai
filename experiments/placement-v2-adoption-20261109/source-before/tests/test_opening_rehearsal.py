"""Opening rehearsal: data isolation, loss masking, path parity, gradients, wiring.

Uses the real adopted model / train.npz read-only. No GPU requirement, no APIs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # parity is algorithmic; keep the GPU free for jobs
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(P / 'scripts'))
sys.dont_write_bytecode = True
import continuous_learning as C  # noqa: E402
import continuous_opening_backend as b  # noqa: E402
import opening_rehearsal as R  # noqa: E402
import opening_rehearsal_ab as AB  # noqa: E402

CFG = json.loads((P / 'config/continuous-elfnotes.json').read_text())
PROD = Path(CFG['root'])
MANIFEST = Path(CFG['opening_manifest'])
DATA = R.dataset_directory(MANIFEST)
SETTINGS = dict(enabled=True, split='train', lambda_bc=1.0, lambda_kl=0.0, sequences_per_minibatch=1, seed=7,
                sha256=json.loads(MANIFEST.read_text())['decks']['ElfnotesKewlTune']['train']['sha256'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic():
    """2 sequences, T=4. seq0: len 3 (t1 is a select_place adapter step); seq1: len 4."""
    n, T = 2, 4
    a = np.zeros((n, T, 128, 12), np.uint8)
    targets = np.zeros((n, T, 128), np.uint8)
    labels = np.zeros((n, T), np.int32)
    for t, (msg, legal, tg) in enumerate([(1, 3, [0, 2]), (11, 2, [1]), (2, 2, [1, 5]), (2, 2, [1, 5])]):
        a[0, t, :legal, 3] = msg
        targets[0, t, tg] = 1
        labels[0, t] = tg[0]
    for t in range(T):
        a[1, t, :2, 3] = 3
        targets[1, t, 0] = 1
    lengths = np.array([3, 4], np.int32)
    weights = (np.arange(T)[None] < lengths[:, None]).astype(np.float32)
    return dict(cards_=np.zeros((n, T, 160, 41), np.uint8), global_=np.zeros((n, T, 23), np.uint8), actions_=a,
                h_actions_=np.zeros((n, T, 32, 14), np.uint8), chain=np.zeros((n, T, 4), np.int64),
                labels=labels, targets=targets, lengths=lengths, weights=weights)


class DataAndLossTests(unittest.TestCase):
    def test_acceptable_set_ce_masks_padding_adapter_and_illegal(self):
        import jax.numpy as jnp
        prepared = R.prepare(synthetic())
        s = prepared['stats']
        self.assertEqual((s['loss_steps'], s['adapter_steps_excluded'], s['padding_steps_excluded']), (6, 1, 1))
        self.assertEqual((s['multi_target_loss_steps'], s['illegal_targets_dropped']), (1, 1))
        steps = R.batch(prepared, [0, 1], 4)
        np.testing.assert_array_equal(steps['loss_mask'].T, [[1, 0, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(int(steps['targets'][2, 0, 5]), 0)  # illegal option 5 dropped from the acceptable set
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(4, 2, 128)).astype(np.float32)
        logits[steps['actions_'][..., 3] == 0] = np.finfo(np.float32).min
        loss, metrics = R.acceptable_action_ce(jnp.asarray(logits), steps)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        rows = [(0, 0, [0, 2]), (2, 0, [1]), (0, 1, [0]), (1, 1, [0]), (2, 1, [0]), (3, 1, [0])]
        expected = np.mean([-np.log(p[t, k, tg].sum()) for t, k, tg in rows])
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertEqual(float(metrics['loss_steps']), 6)
        # Changing logits on padding or on the adapter step must not change the loss.
        changed = logits.copy()
        changed[3, 0, :2] += 50
        changed[1, 0, :2] -= 50
        self.assertAlmostEqual(float(R.acceptable_action_ce(jnp.asarray(changed), steps)[0]), expected, places=5)

    def test_inconsistent_data_is_rejected(self):
        bad = synthetic()
        bad['targets'][1, 2] = 0
        bad['targets'][1, 2, 7] = 1  # only an illegal option is marked acceptable
        bad['labels'][1, 2] = 7
        with self.assertRaisesRegex(ValueError, 'no legal acceptable action'):
            R.prepare(bad)
        bad = synthetic()
        bad['weights'][0, 3] = 1  # padding weighted
        with self.assertRaisesRegex(ValueError, 'padding'):
            R.prepare(bad)

    def test_only_the_train_split_is_accepted(self):
        for split in ['validation', 'holdout']:
            with self.assertRaisesRegex(ValueError, 'train split only'):
                R.validate_settings(dict(SETTINGS, split=split))
            with self.assertRaisesRegex(ValueError, 'train split only'):
                R.load_train_split(MANIFEST, DATA, split)
        with self.assertRaisesRegex(ValueError, 'lambda_kl'):
            R.validate_settings(dict(SETTINGS, lambda_kl=0.1))
        prepared = R.load_train_split(MANIFEST, DATA, expected_sha256=SETTINGS['sha256'])
        s = prepared['stats']
        self.assertEqual((s['sequences'], s['decisions'], s['hand_overlap_with_validation'], s['hand_overlap_with_holdout']),
                         (416, 25802, 0, 0))
        self.assertEqual(s['adapter_steps_excluded'], 2784 + 3840)
        with tempfile.TemporaryDirectory() as tmp:
            # validation data renamed as train.npz is refused by checksum.
            shutil.copyfile(DATA / 'validation.npz', Path(tmp) / 'train.npz')
            shutil.copyfile(DATA / 'train-witnesses.json', Path(tmp) / 'train-witnesses.json')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                R.load_train_split(MANIFEST, tmp)
        with self.assertRaisesRegex(ValueError, 'pinned'):
            R.load_train_split(MANIFEST, DATA, expected_sha256='0' * 64)
        manifest = json.loads(MANIFEST.read_text())
        leaked = copy.deepcopy(manifest)
        cases = leaked['decks']['ElfnotesKewlTune']['cases']
        cases['train'][0] = dict(cases['train'][0], hand=list(reversed(cases['validation'][0]['hand'])))
        with self.assertRaisesRegex(ValueError, 'shares'):
            R.check_split_isolation(leaked)
        with self.assertRaisesRegex(ValueError, 'holdout'):
            AB.load_split_for_diagnostics(MANIFEST, DATA, 'holdout')

    def test_sampler_is_deterministic_and_bounded(self):
        prepared = R.prepare(synthetic())
        a, ia = R.Sampler(prepared, 4, 2, 11).next()
        c, ic = R.Sampler(prepared, 4, 2, 11).next()
        np.testing.assert_array_equal(ia, ic)
        self.assertEqual(a['cards_'].shape, (4, 4, 2, 160, 41))
        self.assertEqual(a['chain'].dtype, np.int32)


class ModelPathTests(unittest.TestCase):
    """Real adopted model; compares the rehearsal unroll with both runtime paths."""

    @classmethod
    def setUpClass(cls):
        import jax
        cls.common = b.policy_common(CFG)  # production snapshot policy-code, read-only
        cls.agent = b.CompositeAgent(cls.common.AGENT, cls.common.HEAD)
        model = PROD / 'models/initial'
        cls.hashes = {n: sha(model / n) for n in ['base.flax_model', 'chain-head.flax_model']}
        base, head = cls.common.load(model / 'base.flax_model'), cls.common.load(model / 'chain-head.flax_model')
        cls.variables = {'params': {'base': base['params'], 'head': head['params']}, 'batch_stats': base.get('batch_stats', {})}
        cls.prepared = R.load_train_split(MANIFEST, DATA)
        lengths = cls.prepared['lengths']
        chained = [i for i in np.argsort(lengths, kind='stable')
                   if any(cls.prepared['flat']['chain'][cls.prepared['offsets'][i] + t, 0] > 0
                          and cls.prepared['flat']['loss_mask'][cls.prepared['offsets'][i] + t] > 0 for t in range(lengths[i]))]
        cls.indices = [int(chained[0]), int(chained[1])]
        cls.length = int(lengths[cls.indices].max())
        cls.steps = R.batch(cls.prepared, cls.indices, cls.length)
        cls.logits = np.asarray(jax.jit(lambda v, s: R.unroll_logits(cls.agent, v, s))(cls.variables, cls.steps))

    def legal(self, t, k):
        return self.steps['actions_'][t, k, :, 3] != 0

    def test_unroll_matches_evaluation_policy_with_reset_per_sequence(self):
        policy = self.common.Policy(PROD / 'models/initial/base.flax_model', PROD / 'models/initial/chain-head.flax_model')
        for k, i in enumerate(self.indices):
            policy.reset()
            for t in range(int(self.prepared['lengths'][i])):
                obs = {key: self.steps[key][t, k][None] for key in R.OBS_KEYS}
                policy.state, logits = policy.infer(policy.base, policy.head, policy.state, obs, self.steps['chain'][t, k][None])
                legal = self.legal(t, k)
                np.testing.assert_allclose(np.asarray(logits)[0][legal], self.logits[t, k][legal], atol=2e-4, rtol=1e-4)

    def test_unroll_matches_ppo_multistep_path(self):
        import jax.numpy as jnp
        k, i = 0, self.indices[0]
        n = int(self.prepared['lengths'][i])
        obs = {key: jnp.asarray(self.steps[key][:n, k]) for key in R.OBS_KEYS}
        obs.update(mask_=None, chain_=jnp.asarray(self.steps['chain'][:n, k]))
        state = self.agent.init_rnn_state(1)
        out = self.agent.apply(self.variables, obs, (state, state), jnp.zeros(n, bool), jnp.ones(n, bool))
        logits = np.asarray(out[1])
        for t in range(n):
            legal = self.legal(t, k)
            np.testing.assert_allclose(logits[t][legal], self.logits[t, k][legal], atol=2e-4, rtol=1e-4)

    def test_gradient_reaches_base_and_head_and_is_added_scaled(self):
        import jax
        params = self.variables['params']
        ppo = jax.tree.map(lambda x: np.full(x.shape, 0.25, np.float32), params)
        total, stats = jax.jit(lambda p, g, s: R.combine(self.agent, p, {}, g, s, 3.0))(params, ppo, self.steps)
        _, _, grads = R.rehearsal_gradient(self.agent, params, {}, self.steps)
        stats = dict(zip(R.METRIC_NAMES, np.asarray(stats)))
        self.assertGreater(stats['rehearsal_base_grad_norm'], 0)
        self.assertGreater(stats['rehearsal_head_grad_norm'], 0)
        for path in [('base', 'FiLMActor_0'), ('head', 'output')]:
            got = jax.tree.leaves(total[path[0]][path[1]])[0]
            want = 0.25 + 3.0 * jax.tree.leaves(grads[path[0]][path[1]])[0]
            np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-6)
        for name, value in self.hashes.items():
            self.assertEqual(sha(PROD / 'models/initial' / name), value)


class WiringTests(unittest.TestCase):
    def test_disabled_path_is_byte_identical_and_enabled_path_is_hooked(self):
        text = b.patched_trainer((Path(CFG['runtime']) / 'agent/scripts/cleanba.py').read_text())
        self.assertEqual(text, (PROD / 'snapshot/train.py').read_text())
        patched = R.patch_trainer(text)
        self.assertEqual(patched.count("~jnp.reshape(obs['adapter_controlled_'], mask.shape)"), 2)
        self.assertIn('OPENING_REHEARSAL.combine(', patched)
        self.assertIn('OPENING_REHEARSAL.next_device_batches(args, learner_devices)', patched)
        self.assertIn('install_rehearsal(globals())', patched)
        with self.assertRaises(RuntimeError):
            R.patch_trainer(patched)

    def test_initialize_pins_train_only_in_a_new_root(self):
        before = AB.production_fingerprint()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(CFG, root=str(Path(tmp) / 'arm'), rehearsal=SETTINGS)
            path = Path(tmp) / 'cfg.json'
            path.write_text(json.dumps(cfg))
            cfg = C.load_config(path)
            state = C.backend().initialize(cfg)
            snap = Path(cfg['root']) / 'snapshot'
            hashes = json.loads((snap / 'hashes.json').read_text())
            self.assertEqual(hashes['rehearsal/train.npz'], SETTINGS['sha256'])
            self.assertFalse([n for n in hashes if n.endswith(('validation.npz', 'holdout.npz'))])
            self.assertEqual(hashes['opening_rehearsal.py'], sha(P / 'scripts/opening_rehearsal.py'))
            self.assertIn('install_rehearsal(globals())', (snap / 'train.py').read_text())
            self.assertEqual(state['opening_rehearsal']['split'], 'train')
            self.assertEqual(state['status'], 'stopped')
            for name in ['base.flax_model', 'chain-head.flax_model']:
                self.assertEqual(sha(Path(cfg['root']) / 'models/initial' / name), sha(PROD / 'models/initial' / name))
        self.assertEqual(AB.production_fingerprint(), before)

    def test_runner_refuses_production_root(self):
        for path in [PROD, PROD / 'cycles/x', PROD.parent]:
            with self.assertRaises(RuntimeError):
                AB.assert_isolated(path)
        AB.assert_isolated(P / 'experiments/opening-rehearsal-ab-20261109/a-ppo')


if __name__ == '__main__':
    unittest.main(verbosity=2)
