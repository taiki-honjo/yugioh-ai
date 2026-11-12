"""B2B rehearsal extension: old path unchanged, separate context/loss masks, fixed 14/1/1 schedule, per-source split.

Real adopted model, real train.npz and the built B2B demo data, read only. CPU. No engine, no training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(P / 'scripts'))
sys.dont_write_bytecode = True
import continuous_opening_backend as b  # noqa: E402
import opening_rehearsal as R0  # noqa: E402  (unchanged reference module)
spec = importlib.util.spec_from_file_location('b2b_opening_rehearsal', P / 'experiments/b2b-train-20261109/code/b2b_opening_rehearsal.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

CFG = json.loads((P / 'config/continuous-elfnotes.json').read_text())
PROD = Path(CFG['root'])
MANIFEST = Path(CFG['opening_manifest'])
DATA = R0.dataset_directory(MANIFEST)
DEMO = P / 'experiments/b2b-train-20261109/data'
OLD = dict(enabled=True, split='train', lambda_bc=5.0, lambda_kl=0.0, sequences_per_minibatch=2, seed=20261109,
           sha256=json.loads(MANIFEST.read_text())['decks']['ElfnotesKewlTune']['train']['sha256'])
B2B = dict(enabled=True, sha256=R.sha(DEMO / 'b2b-demos.npz'), manifest_sha256=R.sha(DEMO / 'b2b-demos-manifest.json'),
           block_sequences=16, old_per_block=14, demos_per_block={'1': 1, '2': 1}, schedule_seed=20261110)


def demo_arrays():
    with np.load(DEMO / 'b2b-demos.npz') as z:
        return {k: z[k] for k in z.files}


class OldPathTests(unittest.TestCase):
    def test_settings_prepare_batch_and_sampler_are_unchanged_without_b2b(self):
        self.assertEqual(R.validate_settings(OLD), R0.validate_settings(OLD))
        old, new = R0.load_train_split(MANIFEST, DATA), R.load_train_split(MANIFEST, DATA)
        for k in old['flat']:
            np.testing.assert_array_equal(old['flat'][k], new['flat'][k])
        self.assertEqual(old['stats'], new['stats'])
        a, c = R0.Sampler(old, 4, 2, np.random.SeedSequence([1, 2])), R.Sampler(new, 4, 2, np.random.SeedSequence([1, 2]))
        for _ in range(3):
            (sa, ia), (sc, ic) = a.next(), c.next()
            np.testing.assert_array_equal(ia, ic)
            self.assertEqual(sorted(sa), sorted(sc))           # no 'source' key on the old path
            for k in sa:
                np.testing.assert_array_equal(sa[k], sc[k])

    def test_trainer_patch_is_identical(self):
        self.assertEqual(R.TRAINER_PATCHES, R0.TRAINER_PATCHES)


class DemoMaskTests(unittest.TestCase):
    def test_context_and_loss_masks_are_separate(self):
        arrays = demo_arrays()
        prepared = R.prepare_demos(arrays)
        manifest = json.loads((DEMO / 'b2b-demos-manifest.json').read_text())
        self.assertEqual(prepared['stats']['loss_steps_per_sequence'], [d['loss_steps'] for d in manifest['demos']])
        for i, demo in enumerate(manifest['demos']):
            L = int(arrays['lengths'][i])
            start = prepared['offsets'][i]
            loss = prepared['flat']['loss_mask'][start:start + L]
            self.assertEqual(sorted(np.where(loss > 0)[0].tolist()), sorted(x['position'] for x in demo['labelled']))
            self.assertEqual(int(L), demo['decisions'])                    # turn-1 context kept, not trimmed as padding
            self.assertGreater(demo['own_turn_context_steps'], 0)
            self.assertTrue(np.all(loss[:demo['own_turn_context_steps']] == 0))
        self.assertGreater(prepared['stats']['adapter_steps_in_context'], 0)

    def test_inconsistent_demo_data_is_rejected(self):
        base = demo_arrays()
        cases = []
        bad = copy.deepcopy(base); bad['loss_weights'][0, -1] = 1; bad['weights'][0, -1] = 0; cases.append(bad)    # label in padding area
        bad = copy.deepcopy(base); t = int(np.where(bad['actions_'][0, :, 0, 3] == 11)[0][0]); bad['loss_weights'][0, t] = 1
        bad['targets'][0, t, bad['labels'][0, t]] = 1; cases.append(bad)                                            # adapter step labelled
        bad = copy.deepcopy(base); t = int(np.where(bad['loss_weights'][0] == 0)[0][0]); bad['targets'][0, t, 0] = 1; cases.append(bad)
        bad = copy.deepcopy(base); t = int(np.where(bad['loss_weights'][0] > 0)[0][0])
        bad['targets'][0, t] = 0; bad['targets'][0, t, (bad['labels'][0, t] + 1) % 2] = 1; cases.append(bad)       # label not acceptable
        for i, arrays in enumerate(cases):
            with self.subTest(i):
                with self.assertRaises(ValueError):
                    R.prepare_demos(arrays)

    def test_demo_loading_checks_hashes_split_and_manifest(self):
        merged = R.merge(R.load_train_split(MANIFEST, DATA), R.load_demo_split(MANIFEST, DEMO, R.validate_b2b(B2B, 2)))
        self.assertEqual((merged['old_count'], merged['stats']['sequences'], merged['stats']['max_length']), (416, 418, 120))
        with self.assertRaises(ValueError):
            R.load_demo_split(MANIFEST, DEMO, dict(R.validate_b2b(B2B, 2), sha256='0' * 64))
        with tempfile.TemporaryDirectory() as tmp:
            for name in R.B2B_FILES:
                shutil.copy(DEMO / name, Path(tmp) / name)
            m = json.loads((Path(tmp) / R.B2B_FILES[1]).read_text())
            m['demos'][0]['case_id'] = 'not-a-train-case'
            (Path(tmp) / R.B2B_FILES[1]).write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                R.load_demo_split(MANIFEST, Path(tmp), dict(R.validate_b2b(B2B, 2), manifest_sha256=R.sha(Path(tmp) / R.B2B_FILES[1])))

    def test_b2b_settings_composition_is_checked(self):
        self.assertEqual(R.validate_settings(dict(OLD, b2b=B2B))['b2b']['old_per_block'], 14)
        for change in [dict(old_per_block=13), dict(block_sequences=15, old_per_block=13), dict(sha256='x')]:
            with self.assertRaises(ValueError):
                R.validate_settings(dict(OLD, b2b=dict(B2B, **change)))


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merged = R.merge(R.load_train_split(MANIFEST, DATA), R.load_demo_split(MANIFEST, DEMO, R.validate_b2b(B2B, 2)))

    def sampler(self, seed=(20261110, 10021110)):
        return R.MixedSampler(self.merged, 4, 2, list(seed), R.validate_b2b(B2B, 2))

    def test_every_sixteen_sequences_hold_fourteen_old_and_one_of_each_demo(self):
        s = self.sampler()
        idx = [s.next()[1].ravel() for _ in range(32)]           # 32 PPO updates = 256 sequences
        flat = np.concatenate(idx)
        sources = self.merged['sources'][flat]
        for block in range(16):
            part = sources[block * 16:(block + 1) * 16]
            self.assertEqual((int((part == 0).sum()), int((part == 1).sum()), int((part == 2).sum())), (14, 1, 1))
        self.assertEqual(s.counts(), {'0': 224, '1': 16, '2': 16})
        self.assertTrue(np.all(flat[sources == 0] < 416))
        positions = {tuple(np.where(sources[b * 16:(b + 1) * 16] > 0)[0]) for b in range(16)}
        self.assertGreater(len(positions), 1)                    # demo slots are shuffled, not fixed
        again = self.sampler()
        np.testing.assert_array_equal(np.concatenate([again.next()[1].ravel() for _ in range(32)]), flat)

    def test_demo_slots_return_the_demo_sequence_with_its_masks(self):
        s = self.sampler()
        steps, idx = s.next()
        k = np.argwhere(self.merged['sources'][idx] == 1)
        if len(k) == 0:
            steps, idx = s.next(); k = np.argwhere(self.merged['sources'][idx] == 1)
        m, j = k[0]
        demo = R.load_demo_split(MANIFEST, DEMO, R.validate_b2b(B2B, 2))
        L = int(demo['lengths'][0])
        np.testing.assert_array_equal(steps['loss_mask'][m, :L, j], demo['flat']['loss_mask'][:L])
        np.testing.assert_array_equal(steps['cards_'][m, :L, j], demo['flat']['cards_'][:L])
        self.assertTrue(np.all(steps['source'][m, :L, j] == 1))
        self.assertTrue(np.all(steps['loss_mask'][m, L:, j] == 0))


class MixedGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls._cpu = jax.default_device(jax.devices('cpu')[0]); cls._cpu.__enter__()
        cls.common = b.policy_common(CFG)
        cls.agent = b.CompositeAgent(cls.common.AGENT, cls.common.HEAD)
        model = PROD / 'models/initial'
        base, head = cls.common.load(model / 'base.flax_model'), cls.common.load(model / 'chain-head.flax_model')
        cls.params = {'base': base['params'], 'head': head['params']}
        merged = R.merge(R.load_train_split(MANIFEST, DATA), R.load_demo_split(MANIFEST, DEMO, R.validate_b2b(B2B, 2)))
        old = int(np.argsort(merged['lengths'][:416])[0])
        cls.steps = R.batch(merged, [old, merged['demo_indices'][1]], int(max(merged['lengths'][old], merged['lengths'][merged['demo_indices'][1]])))

    @classmethod
    def tearDownClass(cls):
        cls._cpu.__exit__(None, None, None)

    def test_split_gradient_equals_the_original_objective(self):
        import jax
        ppo = jax.tree.map(lambda x: np.zeros(x.shape, np.float32), self.params)
        total, stats = R.combine_mixed(self.agent, self.params, {}, ppo, self.steps, 5.0)
        plain = {k: v for k, v in self.steps.items() if k != 'source'}
        reference, ref_stats = R0.combine(self.agent, self.params, {}, ppo, plain, 5.0)
        for a, c in zip(jax.tree.leaves(total), jax.tree.leaves(reference)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=2e-4, atol=1e-6)
        s = dict(zip(R.METRIC_NAMES_B2B, np.asarray(stats))); r = dict(zip(R0.METRIC_NAMES, np.asarray(ref_stats)))
        self.assertAlmostEqual(s['ce'], r['ce'], places=4)
        self.assertEqual(s['loss_steps'], r['loss_steps'])
        self.assertEqual(s['loss_steps_old'] + s['loss_steps_b2b'], s['loss_steps'])
        self.assertEqual((s['sequences_old'], s['sequences_b2b']), (1.0, 1.0))
        self.assertEqual(s['loss_steps_b2b'], 9.0)
        self.assertGreater(s['weighted_grad_norm_b2b'], 0)
        self.assertGreater(s['weighted_grad_norm_old'], 0)


if __name__ == '__main__':
    unittest.main()
