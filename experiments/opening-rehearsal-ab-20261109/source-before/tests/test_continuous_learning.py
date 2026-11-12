"""Meaningful guard, checkpoint recovery, and retention tests; no GPU or APIs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import flax.serialization

PROJECT = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('continuous', PROJECT / 'scripts/continuous_learning.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ContinuousLearningTests(unittest.TestCase):
    def test_virtualenv_python_path_is_preserved(self):
        cfg=m.load_config(PROJECT/'config/continuous-elfnotes.json')
        self.assertEqual(cfg['python'],'/opt/ygo-ai/.venv/bin/python')

    def summary(self, **changes):
        result = dict(total_episodes=128, verified_episodes=128, requires_rerun_or_exclusion=False,
                      win_rate_ci95=[.55,.8])
        result.update(changes)
        return dict(results=result, terminations=dict(abnormal_total=0))

    def test_promotion_needs_clean_evidence_of_improvement(self):
        self.assertTrue(m.promotion_ok(self.summary(), 128))
        for change in [dict(verified_episodes=126), dict(total_episodes=126),
                       dict(requires_rerun_or_exclusion=True), dict(win_rate_ci95=[.5,.8])]:
            self.assertFalse(m.promotion_ok(self.summary(**change),128))
        summary = self.summary()
        summary['terminations']['abnormal_total'] = 1
        self.assertFalse(m.promotion_ok(summary,128))

    def checkpoint(self, path, values):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(flax.serialization.to_bytes({'params':{'x':np.array(values,dtype=np.float32)}}))

    def test_invalid_weights_and_wrong_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.checkpoint(p/'source',[1,2]);self.checkpoint(p/'initial',[1,2]);self.checkpoint(p/'trained',[2,3])
            self.assertEqual(m.validate_weights(p/'source',p/'initial',p/'trained')['changed_values'],2)
            self.checkpoint(p/'trained',[np.nan,3])
            with self.assertRaises(RuntimeError):m.validate_weights(p/'source',p/'initial',p/'trained')
            self.checkpoint(p/'trained',[2,3]);self.checkpoint(p/'initial',[0,2])
            with self.assertRaises(RuntimeError):m.validate_weights(p/'source',p/'initial',p/'trained')

    def test_recovery_skips_partial_file_and_keeps_committed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cycle=root/'cycles/000001';ck=cycle/'checkpoints'
            self.checkpoint(cycle/'source.flax_model',[1,2])
            self.checkpoint(ck/'initial.flax_model',[1,2])
            self.checkpoint(ck/'1_4096steps.flax_model',[2,3])
            (ck/'1_8192steps.flax_model').write_bytes(b'incomplete')
            state=dict(pending=dict(phase='training',directory=str(cycle),source=str(cycle/'source.flax_model'),starting_added_steps=128),
                       latest_checkpoint=str(root/'models/latest.flax_model'))
            cfg=dict(root=str(root))
            m.interrupted_candidate(cfg,state)
            self.assertEqual(state['added_steps'],4224)
            self.assertIsNone(state['pending'])
            self.assertEqual(m.sha(root/'models/latest.flax_model'),m.sha(ck/'1_4096steps.flax_model'))

    def test_retention_preserves_best_latest_and_active_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(3):
                self.checkpoint(root/f'cycles/{i:06d}/checkpoints/1_4096steps.flax_model',[i])
                self.checkpoint(root/f'cycles/{i:06d}/source.flax_model',[i])
            self.checkpoint(root/'models/best.flax_model',[9])
            active=root/'cycles/000001'
            m.prune_checkpoints(dict(root=str(root),keep_cycle_checkpoints=1),dict(pending=dict(directory=str(active))))
            self.assertFalse((root/'cycles/000000/source.flax_model').exists())
            self.assertTrue((active/'source.flax_model').exists())
            self.assertTrue((root/'models/best.flax_model').exists())
            self.assertTrue((root/'cycles/000002/source.flax_model').exists())

    def test_atomic_state_is_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.json'
            for i in range(3):m.atomic_json(p,dict(cycle=i))
            self.assertEqual(json.loads(p.read_text()),dict(cycle=2))
            self.assertFalse(list(p.parent.glob('*.tmp-*')))


if __name__=='__main__':unittest.main(verbosity=2)
