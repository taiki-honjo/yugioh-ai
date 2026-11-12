"""Experiment runner safety: no result reuse across different conditions, and every exit
path leaves experiment arms stopped. Temporary directories only; production is never written."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(P / 'scripts'))
sys.dont_write_bytecode = True
import continuous_learning as C  # noqa: E402
import opening_rehearsal_ab as AB  # noqa: E402

SEED = 10000555


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value if isinstance(value, str) else json.dumps(value))


class Experiment:
    """A miniature experiment root shaped like the real one."""

    def __init__(self, tmp, name='a-ppo'):
        self.exp, self.name = Path(tmp) / 'exp', name
        self.root = self.exp / name
        self.plan = dict(experiment_root=str(self.exp), base_config=str(P / 'config/continuous-elfnotes.json'),
                         steps=256, seeds=[SEED], battle_episodes=32, battle_seed=90000555,
                         arms={name: {'rehearsal': None}}, _file=str(self.exp / 'plan.json'))
        for part, data in [('base.flax_model', b'initial-base'), ('chain-head.flax_model', b'initial-head')]:
            write(self.root / 'models/initial' / part, data)
        write(self.root / 'snapshot/hashes.json', {'train.py': '0' * 64})
        self.cfg = AB.arm_config(self.plan, name, self.plan['arms'][name])  # the same config path run() uses

    def completed_run(self, seed=SEED, steps=256, episodes=32, battle_seed=90000555):
        plan = dict(self.plan, steps=steps, battle_episodes=episodes, battle_seed=battle_seed)
        d = self.root / 'cycles' / f'{steps}-seed{seed}'
        write(d / 'run-conditions.json', AB.requested_conditions(plan, self.name, plan['arms'][self.name], self.cfg, seed))
        for part in ['base.flax_model', 'chain-head.flax_model']:
            write(d / 'source-model' / part, (self.root / 'models/initial' / part).read_bytes())
            write(d / 'candidate' / part, b'candidate-' + part.encode())
        write(d / 'training-result.json', {})
        write(d / 'train-command.json', {'command': ['python', 'train.py', '--total-timesteps', str(steps), '--seed', str(seed),
                                                    '--checkpoint', str(d / 'source-combined.flax_model')]})
        write(d / 'battle-command.json', {'command': ['python', 'continuous_learning.py', 'job-eval', '--candidate', str(d / 'candidate'),
                                                     '--opponent', str(self.root / 'models/initial'),
                                                     '--episodes', str(episodes), '--seed', str(battle_seed)]})
        result = dict(arm=self.name, seed=seed, steps=steps, base_sha256=C.sha(d / 'candidate/base.flax_model'),
                      head_sha256=C.sha(d / 'candidate/chain-head.flax_model'), opening_passed=65, opening_total=66,
                      regression_passed=True, lost_case_ids=[], new_case_ids=[],
                      teacher_forced={'train': {'ce': .06}, 'validation': {'ce': .08}}, resources={},
                      battle=dict(status='complete', terminations={'abnormal_total': 0},
                                  results=dict(total_episodes=episodes, verified_episodes=episodes, wins=16, losses=16,
                                               draws=0, win_rate=.5)))
        write(d / 'run-result.json', result)
        return d, result


class ReuseTests(unittest.TestCase):
    def test_identical_conditions_are_reused_without_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            _, result = e.completed_run()
            self.assertEqual(AB.run_one(None, e.plan, e.name, e.plan['arms'][e.name], e.cfg, {}, SEED), result)

    def test_changed_battle_plan_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            e.completed_run()
            changed = dict(e.plan, battle_episodes=8, battle_seed=123)
            with self.assertRaisesRegex(RuntimeError, r"battle_episodes.*battle_seed"):
                AB.run_one(None, changed, e.name, changed['arms'][e.name], e.cfg, {}, SEED)

    def test_changed_rehearsal_or_initial_model_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            e.completed_run()
            rehearsal = dict(enabled=True, split='train', lambda_bc=5.0, lambda_kl=0.0, sequences_per_minibatch=2, seed=1,
                             sha256='0' * 64)
            with self.assertRaisesRegex(RuntimeError, 'rehearsal'):
                AB.run_one(None, e.plan, e.name, {'rehearsal': rehearsal}, e.cfg, {}, SEED)
            (e.root / 'models/initial/base.flax_model').write_bytes(b'another-start')
            with self.assertRaisesRegex(RuntimeError, 'initial_base_sha256'):
                AB.run_one(None, e.plan, e.name, e.plan['arms'][e.name], e.cfg, {}, SEED)

    def test_run_without_recorded_conditions_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            d, _ = e.completed_run()
            (d / 'run-conditions.json').unlink()
            with self.assertRaisesRegex(RuntimeError, 'no recorded run conditions'):
                AB.run_one(None, e.plan, e.name, e.plan['arms'][e.name], e.cfg, {}, SEED)

    def test_evidence_that_disagrees_with_recorded_conditions_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            d, _ = e.completed_run()
            command = json.loads((d / 'battle-command.json').read_text())
            command['command'][command['command'].index('--episodes') + 1] = '8'
            write(d / 'battle-command.json', command)
            with self.assertRaisesRegex(RuntimeError, 'battle episodes in battle-command.json'):
                AB.run_one(None, e.plan, e.name, e.plan['arms'][e.name], e.cfg, {}, SEED)
            e.completed_run()  # restore, then swap the candidate weights behind the result
            (d / 'candidate/base.flax_model').write_bytes(b'swapped')
            with self.assertRaisesRegex(RuntimeError, 'candidate base.flax_model hash'):
                AB.run_one(None, e.plan, e.name, e.plan['arms'][e.name], e.cfg, {}, SEED)

    def test_summary_refuses_runs_outside_the_current_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            e.completed_run()  # 256 steps, seed 10000555
            later = dict(e.plan, steps=4096, seeds=[10000999], battle_episodes=8, battle_seed=123)
            with patch.object(AB, 'load_plan', return_value=later), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'Refusing to summarize'):
                    AB.summarize(SimpleNamespace(plan=later['_file']))
            self.assertFalse((e.exp / 'summary.json').exists())

    def test_summary_keeps_matching_runs_and_lists_missing_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Experiment(tmp)
            e.completed_run()
            plan = dict(e.plan, seeds=[SEED, 10000777])
            with patch.object(AB, 'load_plan', return_value=plan), patch('builtins.print'):
                AB.summarize(SimpleNamespace(plan=plan['_file']))
            summary = json.loads((e.exp / 'summary.json').read_text())
            self.assertEqual([r['seed'] for r in summary['runs']], [SEED])
            self.assertEqual((summary['missing_runs'], summary['complete'], summary['conditions_verified']),
                             ([f'{e.name}/10000777'], False, True))


class StopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e = Experiment(self.tmp.name)
        self.production = Path(self.tmp.name) / 'fake-production'
        baseline = dict(passed=65, total=66, passed_case_ids=['known'], lua_errors=0)
        write(self.production / 'opening-baseline.json', baseline)
        root = self.e.root
        write(root / 'STOP', 'now')  # completion marker left by an earlier run of this runner
        write(root / 'arm-setup.json', {})
        write(root / 'opening-baseline.json', baseline)
        write(root / 'baseline/teacher-forced-initial.json', {})
        write(root / 'state.json', dict(status='stopped'))
        self.state = dict(status='stopped')
        self.patches = [patch.object(AB, 'load_plan', return_value=self.e.plan),
                        patch.object(AB, 'production_root', return_value=self.production),
                        patch.object(AB, 'production_fingerprint', return_value={'unchanged': True}),
                        patch.object(AB, 'resident_learners', return_value=[]),
                        patch.object(AB, 'check_snapshot', return_value={}),
                        patch.object(AB, 'raise_on_termination_signal'),
                        patch.object(AB, 'LEFTOVER_WAIT_SECONDS', 0.5),
                        patch('builtins.print')]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def backend(self, job_error=None):
        def run_job(*args, **kwargs):
            self.assertFalse((self.e.root / 'STOP').exists(), 'jobs run only while the marker is lifted')
            raise job_error
        return SimpleNamespace(initialize=lambda cfg: self.state, control=SimpleNamespace(run_job=run_job))

    def final_state(self):
        return json.loads((self.e.root / 'state.json').read_text())

    def test_job_failure_restores_stop_and_records_failure(self):
        with patch.object(AB, 'backend', return_value=self.backend()), \
             patch.object(AB, 'run_one', side_effect=RuntimeError('simulated finite-job failure')):
            with self.assertRaisesRegex(RuntimeError, 'simulated finite-job failure'):
                AB.run(SimpleNamespace(plan='unused', arms=None, seeds=None))
        self.assertEqual((self.e.root / 'STOP').read_text(), 'now')
        state = self.final_state()
        self.assertEqual((state['status'], state['last_decision']), ('stopped', 'finite_ab_experiment_failed_not_adopted'))
        self.assertIn('simulated finite-job failure', state['last_error'])

    def test_setup_failure_after_the_stop_marker_was_lifted(self):
        (self.e.root / 'opening-baseline.json').unlink()  # setup must run the baseline job, which fails
        with patch.object(AB, 'backend', return_value=self.backend(RuntimeError('baseline job failed'))):
            with self.assertRaisesRegex(RuntimeError, 'baseline job failed'):
                AB.run(SimpleNamespace(plan='unused', arms=None, seeds=None))
        self.assertEqual((self.e.root / 'STOP').read_text(), 'now')
        self.assertIn('baseline job failed', self.final_state()['last_error'])

    def test_interrupt_also_restores_stop(self):
        with patch.object(AB, 'backend', return_value=self.backend()), \
             patch.object(AB, 'run_one', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                AB.run(SimpleNamespace(plan='unused', arms=None, seeds=None))
        self.assertTrue((self.e.root / 'STOP').exists())

    def test_successful_run_leaves_stop_and_complete_state(self):
        _, result = self.e.completed_run()
        with patch.object(AB, 'backend', return_value=self.backend()), patch.object(AB, 'summarize') as summarize:
            AB.run(SimpleNamespace(plan='unused', arms=None, seeds=None))
        summarize.assert_called_once()
        self.assertTrue((self.e.root / 'STOP').exists())
        self.assertEqual(self.final_state()['last_decision'], 'finite_ab_experiment_complete_not_adopted')

    def test_diagnostic_context_restores_stop_on_error(self):
        with self.assertRaises(ValueError):
            with AB.arm_active(self.e.cfg):
                self.assertFalse((self.e.root / 'STOP').exists())
                raise ValueError('diagnostic job failed')
        self.assertEqual((self.e.root / 'STOP').read_text(), 'now')

    def test_closing_an_arm_terminates_a_job_left_running(self):
        job = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', str(self.e.root / 'cycles/x/train.py')])
        try:
            terminated = AB.close_arm(self.e.cfg, dict(status='running'), 'failed', RuntimeError('x'))
            self.assertIn(job.pid, terminated)
            self.assertIsNotNone(job.wait(timeout=10))
            self.assertEqual(self.final_state()['leftover_jobs_terminated'], terminated)
        finally:
            job.kill()

    def test_production_root_is_never_an_arm(self):
        with patch.object(AB, 'production_root', return_value=self.e.root):
            with self.assertRaises(RuntimeError):
                AB.close_arm(self.e.cfg, {}, 'failed')


class SignalTest(unittest.TestCase):
    def test_sigterm_runs_finally_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'finally-ran'
            script = textwrap.dedent(f'''
                import os, signal, sys
                sys.path.insert(0, {str(P / 'scripts')!r})
                import opening_rehearsal_ab as AB
                AB.raise_on_termination_signal()
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    import time; time.sleep(5)
                finally:
                    open({str(marker)!r}, 'w').write('yes')
            ''')
            done = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=120)
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual(marker.read_text(), 'yes')


if __name__ == '__main__':
    unittest.main(verbosity=2)
