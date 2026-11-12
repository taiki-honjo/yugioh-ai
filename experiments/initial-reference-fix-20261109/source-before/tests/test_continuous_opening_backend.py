import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import flax.serialization as ser

P=Path('/home/hatk8/projects/yugioh-ai');sys.path.insert(0,str(P/'scripts'))
import continuous_opening_backend as b
import continuous_learning as C

class OpeningBackendTests(unittest.TestCase):
    def test_base_and_head_round_trip_preserves_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'source';source.mkdir()
            base={'params':{'x':np.arange(6,dtype=np.float32).reshape(2,3)}}
            head={'params':{'y':np.array([4.,5.],dtype=np.float32)}}
            (source/'base.flax_model').write_bytes(ser.to_bytes(base))
            (source/'chain-head.flax_model').write_bytes(ser.to_bytes(head))
            b.pack_bundle(source,p/'combined')
            b.unpack_bundle(p/'combined',p/'out',source)
            metadata=json.loads((p/'out/metadata.json').read_text())
            self.assertEqual(metadata['head_sha256'],hashlib.sha256((p/'out/chain-head.flax_model').read_bytes()).hexdigest())
            for name,key in [('base.flax_model','x'),('chain-head.flax_model','y')]:
                actual=ser.msgpack_restore((p/'out'/name).read_bytes())['params'][key]
                expected=(base if key=='x' else head)['params'][key]
                np.testing.assert_array_equal(actual,expected)

    def test_regression_rejects_losing_one_case_even_if_count_matches(self):
        baseline=dict(total=3,passed_case_ids=['a','b'])
        candidate=dict(status='complete',total=3,lua_errors=0,normal_summon_usage_scored=False,
                       holdout_used=False,passed_case_ids=['a','b'])
        self.assertTrue(b.regression_ok(baseline,candidate))
        candidate['passed_case_ids']=['a','c']
        self.assertFalse(b.regression_ok(baseline,candidate))
        candidate['passed_case_ids']=['a','b'];candidate['holdout_used']=True
        self.assertFalse(b.regression_ok(baseline,candidate))
        candidate['holdout_used']=False;candidate['lua_errors']=1
        self.assertFalse(b.regression_ok(baseline,candidate))

    def test_patched_trainer_masks_adapter_steps_at_both_loss_sites(self):
        cfg=json.loads((P/'config/continuous-elfnotes.json').read_text())
        text=(Path(cfg['runtime'])/'agent/scripts/cleanba.py').read_text()
        patched=b.patched_trainer(text)
        self.assertEqual(patched.count("~jnp.reshape(obs['adapter_controlled_'], mask.shape)"),2)
        self.assertIn('OpeningAdapterBank(args.local_num_envs)',patched)
        self.assertIn('opening_adapters.apply(cached_next_obs, info',patched)
        self.assertIn('n_valids = jnp.maximum(jnp.sum(mask), 1)',patched)
        self.assertIn('jax.block_until_ready(device_params)',patched)

    def test_current_config_is_paused_and_uses_a_pair(self):
        cfg=json.loads((P/'config/continuous-elfnotes.json').read_text())
        self.assertEqual(cfg['backend'],'verified_opening')
        root=Path(cfg['root']);self.assertTrue((root/'STOP').exists())
        state=json.loads((root/'state.json').read_text())
        self.assertEqual(state['status'],'stopped')
        for key in ['latest_checkpoint','best_checkpoint']:
            for name in ['base.flax_model','chain-head.flax_model']:
                self.assertTrue((Path(state[key])/name).is_file())

    def test_failed_opening_candidate_rolls_latest_back_to_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cycles').mkdir()
            initial=root/'initial';initial.mkdir();(initial/'base.flax_model').write_bytes(b'initial')
            candidate=root/'candidate';candidate.mkdir();(candidate/'base.flax_model').write_bytes(b'candidate')
            state=dict(status='ready',completed_cycles=0,cycle=0,added_steps=0,best_added_steps=0,
                       latest_checkpoint=str(initial),best_checkpoint=str(initial),pending=None)
            cfg=dict(root=str(root),backend='verified_opening',minimum_free_gib=0,bootstrap_steps=128,
                     steps_per_cycle=128,training_seed_base=100,eval_seed_base=1000)
            result=dict(clean=False,promoted=False,opening_regression=dict(passed=False))
            with patch.object(C,'initialize',return_value=state),patch.object(C,'runtime_check'),\
                 patch.object(C,'train_cycle',return_value=candidate),patch.object(C,'evaluate_candidate',return_value=result),\
                 patch.object(C,'prune_checkpoints'),patch.object(C,'interrupted_candidate'),patch.object(C.signal,'signal'):
                C.run_loop(cfg,once=True)
            self.assertEqual(state['latest_checkpoint'],str(initial))
            self.assertEqual(state['best_checkpoint'],str(initial))
            self.assertEqual(state['last_decision'],'opening_regression_rejected')
            self.assertEqual(state['status'],'stopped')

    def test_production_config_snapshot_and_command_agree(self):
        """Training is never started: the command is captured before any job runs."""
        cfg=C.load_config(P/'config/continuous-elfnotes.json');root=Path(cfg['root'])
        policy=json.loads(Path(cfg['verified_policy_config']).read_text())
        state=json.loads((root/'state.json').read_text())
        self.assertEqual((root/'STOP').read_text().strip(),'now');self.assertEqual(state['status'],'stopped')
        for key in ['best_checkpoint','latest_checkpoint']:
            for name,field in [('base.flax_model','base_sha256'),('chain-head.flax_model','head_sha256')]:
                self.assertEqual(C.sha(Path(state[key])/name),policy[field])
        hashes=json.loads((root/'snapshot/hashes.json').read_text())
        self.assertEqual(hashes['train.py'],hashlib.sha256(b.expected_trainer_text(cfg).encode()).hexdigest())
        self.assertFalse([n for n in hashes if n.endswith(('validation.npz','holdout.npz'))])
        rehearsal=cfg.get('rehearsal') or {}
        if rehearsal.get('enabled'):
            import opening_rehearsal as R
            pinned=json.loads((root/'snapshot/rehearsal/settings.json').read_text())
            self.assertEqual(pinned['settings'],R.validate_settings(rehearsal))
            manifest=json.loads(Path(cfg['opening_manifest']).read_text())
            self.assertEqual(hashes['rehearsal/train.npz'],rehearsal['sha256'])
            self.assertEqual(rehearsal['sha256'],manifest['decks']['ElfnotesKewlTune']['train']['sha256'])
        B=C.backend();captured=[]
        def run_job(cfg,state,command,directory,name):
            captured.append([str(x) for x in command]);raise InterruptedError('captured; no training')
        B.control.run_job=run_job
        with tempfile.TemporaryDirectory() as tmp:
            for i,steps in enumerate([cfg['bootstrap_steps'],cfg['steps_per_cycle']]):
                d=Path(tmp)/f'{i}-{steps}'
                d.mkdir()
                with self.assertRaises(InterruptedError):
                    B.train_cycle(cfg,{},dict(directory=str(d),source=state['best_checkpoint'],steps=steps,training_seed=1))
                command=captured[-1];arg=lambda f:command[command.index(f)+1]
                self.assertEqual(float(arg('--learning-rate')),cfg['learning_rate'])
                self.assertEqual(int(arg('--total-timesteps')),steps)
                self.assertEqual(int(arg('--save-interval')),min(int(cfg.get('save_interval_updates',32)),steps//128))
                self.assertEqual(Path(command[command.index('-u')+1]),root/'snapshot/train.py')

if __name__=='__main__':unittest.main(verbosity=2)
