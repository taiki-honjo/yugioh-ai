"""Offline tests for the response study (no engine duel; no training)."""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
HB = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import rl_runs as RR  # noqa: E402  (imports the frozen modules from this root first)
import rl_data as RD  # noqa: E402
import batch_model as BM  # noqa: E402
import numpy as np  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ResponseDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {n: RD.build_response(n) for n in RD.NAMES}

    def test_only_the_window_action_carries_loss_and_the_whole_context_is_kept(self):
        for n, (seq, meta) in self.built.items():
            rec = json.loads((RD.SOURCES[n]['dir'] / 'record.json').read_text())
            own_until = [r['step'] for r in rec['trace'] if r['snapshot']['to_play'] == 0 and r['step'] <= meta['target_step']]
            self.assertEqual(len(seq['labels']), len(own_until))
            self.assertEqual(meta['context_steps'], len(own_until))
            self.assertEqual(float(seq['loss'].sum()), 1.0)
            self.assertEqual(int(np.flatnonzero(seq['loss'])[0]), meta['target_index'])
            self.assertEqual(meta['target_index'], len(seq['labels']) - 1)
            with np.load(RD.SOURCES[n]['dir'] / 'observations.npz') as z:
                for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain', 'labels']:
                    self.assertTrue(np.array_equal(seq[k], z[k][:len(own_until)]), k)

    def test_labels_are_the_registered_legal_responses_to_the_opponent_chain(self):
        want = {'response-S1': (27184601, 10011, 149), 'response-S2': (34541940, 10011, 169)}
        for n, (seq, meta) in self.built.items():
            t = meta['target_index']
            self.assertEqual((meta['label']['code'], meta['label']['effect'], meta['target_step']), want[n])
            self.assertNotEqual(int(seq['actions_'][t, seq['labels'][t], 3]), 0)
            self.assertEqual(int(seq['chain'][t][1]), 2)  # chain owner: opponent
            self.assertIsNotNone(meta['cancel_index'])

    def test_window_detection_separates_opponent_and_own_medius(self):
        base = dict(to_play=0, turn_count=2, turn_player=1, phase=4, message='select_chain')
        self.assertTrue(RD.is_medius_window(dict(base, public_chain=[dict(code=RD.MEDIUS, player=1, description=RD.MEDIUS * 16)])))
        self.assertFalse(RD.is_medius_window(dict(base, public_chain=[dict(code=RD.MEDIUS, player=0, description=RD.MEDIUS * 16)])))
        self.assertFalse(RD.is_medius_window(dict(base, public_chain=[dict(code=RD.MEDIUS, player=1, description=RD.MEDIUS * 16 + 1)])))
        self.assertTrue(RR.is_medius_window(dict(base, public_chain=[dict(code=RD.MEDIUS, player=1, description=RD.MEDIUS * 16)])))


class SamplerTests(unittest.TestCase):
    def test_block_counts_over_256_updates(self):
        existing, names, pairs, stats = RD.HD.load()
        sampler = RD.BlockSampler8(existing, pairs, 11111301)
        kinds = [sampler.next()[0] for _ in range(256)]
        for b in range(32):
            block = kinds[8 * b:8 * b + 8]
            self.assertEqual((block.count('existing'), block.count('pair'), block.count('response')), (3, 3, 2))
        c = sampler.counts()
        self.assertEqual((c['existing_minibatches'], c['pair_minibatches'], c['response_minibatches']), (96, 96, 64))
        self.assertEqual(c['responses'], {'response-S1': 64, 'response-S2': 64})
        self.assertEqual(c['sequence_presentations'], dict(old=384, new=128))

    def test_config(self):
        self.assertEqual((BM.CONFIG['updates'], BM.CONFIG['diagnostics_at'], BM.CONFIG['learning_rate'], BM.CONFIG['adam_eps'],
                          BM.CONFIG['max_grad_norm'], BM.CONFIG['batch_sequences']), (256, [0, 256], 1.5e-5, 1e-5, 1.0, 2))


class EvaluationIdentityTests(unittest.TestCase):
    def test_frozen_modules_come_from_this_root(self):
        for m in (RR.F, RR.HR, RD, BM):
            self.assertEqual(Path(m.__file__).resolve().parent, ROOT / 'code')
        for n, digest in RR.FROZEN.items():
            self.assertEqual(sha(ROOT / 'code' / n), digest)
        hb_reg = json.loads((HB / 'amendment-1.json').read_text())['code']
        self.assertEqual(sha(ROOT / 'code/hb_runs.py'), hb_reg['hb_runs.py'])

    def test_start_model_retention_records_are_the_start_model(self):
        for n in RR.C_NAMES:
            rec = json.loads((HB / 'eval/retain/candidate' / f'{n}.json').read_text())
            self.assertEqual(list(rec['model_sha256']), RD.MODEL_SHA, n)
        self.assertEqual(len(RR.C_NAMES), 13)

    def test_a_sites_and_b_names_are_the_comparison_conditions(self):
        for site, (old, window) in RR.A_SITES.items():
            rec = json.loads((RR.FLC / 'runs' / old / 'record.json').read_text())
            self.assertEqual(next(r['step'] for r in rec['trace'] if RR.is_medius_window(r['snapshot'])), window)
            self.assertEqual(list(rec['model_sha256']), RD.MODEL_SHA)
        self.assertEqual(sorted(RR.B_NAMES), sorted(['BV1-1cost-VARNISH-OM', 'BV1-2cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OM', 'BV1-2cost-VARNISH-OM']))


if __name__ == '__main__':
    unittest.main()
