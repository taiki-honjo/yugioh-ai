"""evaluate_candidate(): which checkpoints the scout, reference and confirmation battles really load.

The real continuous_learning.evaluate_candidate() runs against a temporary root. Only the job runner is
replaced: each battle job is executed in-process through the real evaluate_job() (the verified_opening
path mapping) and, for verified_opening, the real backend loader (install_evaluation's load, which reads
<dir>/base.flax_model and <dir>/chain-head.flax_model). The evaluator placed in the temporary snapshot
resolves agents with scripts/evaluate.py's own resolve_agent_spec() and loads them, but plays no games:
it returns a scripted summary. No training, no battles, no production files. CPU only."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import jax
import numpy as np
import flax.serialization as ser

P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import continuous_opening_backend as B

FAKE_EVALUATOR = '''
import importlib.util, json, os
from pathlib import Path
import numpy as np
import flax.serialization as ser
_spec = importlib.util.spec_from_file_location('real_evaluator_for_reference_test', %(real)r)
_real = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_real)


def _marker(params):
    p = params['params']
    if 'base' in p:  # verified_opening loader: base + chain-head
        return [float(np.asarray(p['base']['m'])[0]), float(np.asarray(p['head']['m'])[0])]
    return [float(np.asarray(p['m'])[0])]


class EvalContext:
    """Same agent resolution and loading entry points as scripts/evaluate.py EvalContext; plays no games."""
    def __init__(self, scripts_dir, deck, **kw):
        self.num_envs = kw['num_envs']; self.agent = object(); self.EnvPreprocess = object(); self._opening_infos = None

    def _run_single_wave_seat(self, **kw):
        raise AssertionError('no games are played in this test')

    def load_params(self, spec):  # single-file checkpoints, like scripts/evaluate.py
        return ser.msgpack_restore(Path(spec['path']).read_bytes())

    def evaluate_matchup(self, candidate_str, opponent_str, proj_dir, num_episodes=32, base_seed=1000, max_steps_per_wave=4000):
        cand_spec = _real.resolve_agent_spec(candidate_str, proj_dir)  # FileNotFoundError for a missing checkpoint
        opp_spec = _real.resolve_agent_spec(opponent_str, proj_dir)
        cand, opp = self.load_params(cand_spec), self.load_params(opp_spec)
        kind = os.environ['TEST_EVAL_KIND']
        log = Path(os.environ['TEST_EVAL_LOG'])
        calls = json.loads(log.read_text()) if log.exists() else []
        calls.append(dict(kind=kind, candidate=candidate_str, opponent=opponent_str, episodes=num_episodes, seed=base_seed,
                          candidate_marker=_marker(cand), opponent_marker=_marker(opp)))
        log.write_text(json.dumps(calls))
        scripted = json.loads(Path(os.environ['TEST_EVAL_PLAN']).read_text())[kind]
        verified = scripted.get('verified', num_episodes)
        return dict(results=dict(total_episodes=num_episodes, verified_episodes=verified, requires_rerun_or_exclusion=False,
                                 win_rate=scripted['win_rate'], win_rate_ci95=scripted['ci']),
                    terminations=dict(abnormal_total=0))
'''

WIN = dict(win_rate=.66, ci=[.56, .75])
EVEN = dict(win_rate=.5, ci=[.41, .59])
BASE_SEED = 80004985


def save(path, marker):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ser.to_bytes({'params': {'m': np.array([marker], dtype=np.float32)}}))


def pair(directory, marker):
    save(directory / 'base.flax_model', marker)
    save(directory / 'chain-head.flax_model', marker + .5)
    return directory


class EvaluateCandidateReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cpu = jax.default_device(jax.devices('cpu')[0]); cls._cpu.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._cpu.__exit__(None, None, None)

    def setUp(self):
        self._path = list(sys.path)
        self.new_tmp()

    def new_tmp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)

    def tearDown(self):
        sys.path[:] = self._path

    # ---- temporary roots
    def make_root(self, verified):
        t = self.t; root = t / 'root'
        (root / 'snapshot').mkdir(parents=True)
        (root / 'snapshot/evaluator.py').write_text(FAKE_EVALUATOR % dict(real=str(P / 'scripts/evaluate.py')))
        cfg = dict(project=str(t), root=str(root), runtime=str(t / 'runtime'), python='/opt/ygo-ai/.venv/bin/python',
                   deck=str(t / 'deck.ydk'), initial_checkpoint=str(t / 'initial'), runtime_manifest=str(t / 'runtime.json'),
                   steps_per_cycle=4096, bootstrap_steps=4096, eval_envs=4, eval_episodes=128, reference_episodes=64,
                   confirmation_episodes=256, max_eval_steps=4000, vocabulary=str(t / 'code_list.txt'))
        directory = root / 'cycles/000005-test'
        if verified:
            cfg['backend'] = 'verified_opening'
            pair(root / 'models/initial', 1.0)
            best = pair(root / 'cycles/000003-best/candidate', 3.0)
            candidate = pair(directory / 'candidate', 5.0)
            C.atomic_json(root / 'opening-baseline.json', dict(total=2, passed=2, lua_errors=0, passed_case_ids=['a', 'b']))
        else:  # previous single-model layout
            for name, marker in [('initial', 1.0), ('latest', 1.0)]:
                save(root / f'models/{name}.flax_model', marker)
            best = root / 'models/best.flax_model'; save(best, 3.0)
            candidate = directory / 'candidate.flax_model'; save(candidate, 5.0)
        C.atomic_json(t / 'config.json', cfg)
        cfg = C.load_config(t / 'config.json')
        state = dict(status='running', phase='evaluating', best_checkpoint=str(best), latest_checkpoint=str(candidate),
                     completed_cycles=4, added_steps=20480)
        pending = dict(cycle=5, directory=str(directory), steps=4096, phase='evaluating', candidate=str(candidate),
                       training_seed=10000005, eval_seed=BASE_SEED)
        return cfg, state, pending, candidate

    def evaluate(self, verified, plan, opening_passed=('a', 'b')):
        cfg, state, pending, candidate = self.make_root(verified)
        jobs = []

        def run_job(cfg_, state_, command, directory, name):  # what the job subprocess would do, in-process
            jobs.append(dict(name=name, command=[str(x) for x in command]))
            arg = lambda flag: str(command[command.index(flag) + 1])
            if name == 'opening-regression':
                C.atomic_json(arg('--result'), dict(status='complete', total=2, passed=len(opening_passed), lua_errors=0,
                                                   normal_summon_usage_scored=False, holdout_used=False,
                                                   passed_case_ids=list(opening_passed)))
                return
            self.assertEqual(command[command.index('job-eval') - 1], str(P / 'scripts/continuous_learning.py'))
            os.environ['TEST_EVAL_KIND'] = name
            C.evaluate_job(argparse.Namespace(candidate=arg('--candidate'), opponent=arg('--opponent'),
                                              episodes=int(arg('--episodes')), seed=int(arg('--seed')),
                                              result=Path(arg('--result'))), C.load_config(arg('--config')))

        lua = types.ModuleType('ygoenv.ygopro.ygopro_ygoenv'); lua.get_lua_error_count = lambda: 0
        modules = {'ygoenv': types.ModuleType('ygoenv'), 'ygoenv.ygopro': types.ModuleType('ygoenv.ygopro'),
                   'ygoenv.ygopro.ygopro_ygoenv': lua}
        log, plan_path = self.t / 'calls.json', self.t / 'plan.json'
        plan_path.write_text(json.dumps(plan))
        common = types.SimpleNamespace(HEAD=None, features=None)
        with patch.object(C, 'run_job', run_job), patch.object(C, 'runtime_check', lambda cfg: None), \
             patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {'TEST_EVAL_LOG': str(log), 'TEST_EVAL_PLAN': str(plan_path), 'TEST_EVAL_KIND': ''}), \
             patch.object(B, 'policy_common', lambda cfg: common), \
             patch.object(B, 'CompositeAgent', lambda agent, head: agent), \
             patch.object(B, 'augmented_preprocess', lambda pre, features, ctx: pre), \
             patch.object(B, 'AdapterBank', lambda cfg, n: object()):
            decision = C.evaluate_candidate(cfg, state, pending, Path(candidate))
        calls = {c['kind']: c for c in (json.loads(log.read_text()) if log.exists() else [])}
        self.assertEqual(json.loads((Path(pending['directory']) / 'decision.json').read_text()), json.loads(json.dumps(decision)))
        return cfg, decision, [j['name'] for j in jobs], {j['name']: j['command'] for j in jobs}, calls

    # ---- two-model (verified_opening) layout
    def test_verified_reference_battle_loads_the_initial_base_and_head(self):
        cfg, decision, names, commands, calls = self.evaluate(True, dict(evaluation=WIN, reference=EVEN, confirmation=WIN))
        root = Path(cfg['root'])
        self.assertEqual(names, ['opening-regression', 'evaluation', 'reference', 'confirmation'])
        reference = commands['reference'][commands['reference'].index('--opponent') + 1]
        self.assertEqual(reference, str(root / 'models/initial'))
        self.assertTrue(Path(reference).is_dir())
        self.assertTrue((Path(reference) / 'base.flax_model').is_file() and (Path(reference) / 'chain-head.flax_model').is_file())
        # what the loader actually read: base and chain-head of each side
        self.assertEqual(calls['reference']['opponent_marker'], [1.0, 1.5])  # initial pair
        self.assertEqual(calls['reference']['candidate_marker'], [5.0, 5.5])
        self.assertEqual(calls['evaluation']['opponent_marker'], [3.0, 3.5])  # current best, not the initial pair
        self.assertEqual(calls['confirmation']['opponent_marker'], [3.0, 3.5])
        self.assertEqual(calls['evaluation']['candidate_marker'], [5.0, 5.5])
        self.assertTrue(decision['promoted'] and decision['clean'])
        self.assertIn('reference_evidence', decision)
        self.assertEqual(decision['opening_regression']['lost_case_ids'], [])

    def test_verified_battle_conditions_and_seeds_are_unchanged(self):
        _, _, _, _, calls = self.evaluate(True, dict(evaluation=WIN, reference=EVEN, confirmation=WIN))
        self.assertEqual({k: (c['episodes'], c['seed']) for k, c in calls.items()},
                         dict(evaluation=(128, BASE_SEED), reference=(64, BASE_SEED), confirmation=(256, BASE_SEED + 1)))

    def test_verified_promotion_gates_are_not_bypassed(self):
        cases = [
            ('scout CI lower bound must exceed .5', dict(evaluation=EVEN, reference=EVEN, confirmation=WIN), False, False),
            ('reference win rate must be at least .5', dict(evaluation=WIN, reference=dict(win_rate=.45, ci=[.33, .58]),
                                                            confirmation=WIN), False, False),
            ('reference must be clean', dict(evaluation=WIN, reference=dict(EVEN, verified=62), confirmation=WIN), False, False),
            ('confirmation decides', dict(evaluation=WIN, reference=EVEN, confirmation=dict(win_rate=.55, ci=[.49, .61])), True, False),
        ]
        for label, plan, confirmed, promoted in cases:
            with self.subTest(label):
                self.new_tmp()
                _, decision, names, _, _ = self.evaluate(True, plan)
                self.assertEqual('confirmation' in names, confirmed)
                self.assertEqual(decision['promoted'], promoted)
                self.assertIn('reference', names)

    def test_verified_lost_opening_still_stops_before_any_battle(self):
        _, decision, names, _, calls = self.evaluate(True, dict(evaluation=WIN, reference=EVEN, confirmation=WIN),
                                                     opening_passed=('a',))
        self.assertEqual(names, ['opening-regression'])
        self.assertEqual(calls, {})
        self.assertEqual((decision['promoted'], decision['reason'], decision['opening_regression']['lost_case_ids']),
                         (False, 'opening_regression', ['b']))

    # ---- previous single-model layout keeps models/initial.flax_model
    def test_single_model_reference_battle_keeps_initial_flax_model(self):
        cfg, decision, names, commands, calls = self.evaluate(False, dict(evaluation=WIN, reference=EVEN, confirmation=WIN))
        root = Path(cfg['root'])
        self.assertEqual(names, ['evaluation', 'reference', 'confirmation'])
        reference = commands['reference'][commands['reference'].index('--opponent') + 1]
        self.assertEqual(reference, str(root / 'models/initial.flax_model'))
        self.assertTrue(Path(reference).is_file())
        self.assertEqual(calls['reference']['opponent_marker'], [1.0])
        self.assertEqual(calls['evaluation']['opponent_marker'], [3.0])
        self.assertEqual({k: (c['episodes'], c['seed']) for k, c in calls.items()},
                         dict(evaluation=(128, BASE_SEED), reference=(64, BASE_SEED), confirmation=(256, BASE_SEED + 1)))
        self.assertTrue(decision['promoted'])


if __name__ == '__main__':
    unittest.main()
