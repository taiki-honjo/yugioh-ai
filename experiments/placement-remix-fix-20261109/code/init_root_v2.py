"""Create a NEW root whose snapshot pins the v2 placement adapter, and measure the adopted
model's 66 validation openings through the normal job path. No training, no adoption.
The root is left with STOP on every exit path."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
root = OUT / 'root-v2'
production = Path(C.read_json(P / 'config/continuous-elfnotes.json')['root'])
assert not root.resolve().is_relative_to(production.resolve())
config = OUT / 'configs/root-v2.json'
if not config.exists():
    C.atomic_json(config, dict(C.read_json(P / 'config/continuous-elfnotes.json'), root=str(root),
                               verified_policy_config=str(P / 'config/verified-opening-policy-placement-remix-fix-20261109.json'),
                               experiment_note='placement adapter v2 check; evaluation only; resident operation not allowed',
                               resident_operation_allowed=False))
cfg = C.load_config(config)
B = C.backend()
try:
    state = B.initialize(cfg)
    ours, theirs = C.read_json(root / 'snapshot/hashes.json'), C.read_json(production / 'snapshot/hashes.json')
    differing = sorted(n for n in theirs if ours.get(n) != theirs.get(n))
    expected = {'policy-code/center_placement.py', 'policy-code/verify_opponent_article_basics.py',  # v2 / root path
                'opening-policy.json', 'continuous_opening_backend.py', 'meta-run-manifest.json'}
    unexpected = [n for n in differing if n not in expected]
    assert ours['policy-code/center_placement.py'] == C.sha(P / 'evaluation/placement-remix-fix-20261109/code/center_placement.py')
    assert ours['train.py'] == theirs['train.py'], 'trainer must be the unchanged production trainer'
    for name in ['base.flax_model', 'chain-head.flax_model']:
        assert C.sha(root / 'models/initial' / name) == C.sha(production / 'models/initial' / name)
    baseline = root / 'opening-baseline.json'
    if not baseline.exists():
        C.run_job(cfg, state, [cfg['python'], '-B', '-u', str(P / 'scripts/continuous_learning.py'), 'job-openings', '--config',
                               cfg['_file'], '--candidate', str(root / 'models/initial'), '--result', str(baseline)],
                  root / 'baseline', 'opening-baseline')
    ours_baseline, reference = C.read_json(baseline), C.read_json(production / 'opening-baseline.json')
    result = dict(snapshot_files_differing_from_production=differing, unexpected_differences=unexpected,
                  v2_center_placement_sha256=ours['policy-code/center_placement.py'], trainer_identical_to_production=True,
                  initial_model_identical_to_adopted=True, baseline_passed=ours_baseline['passed'], baseline_total=ours_baseline['total'],
                  baseline_lua_errors=ours_baseline['lua_errors'],
                  same_passed_case_ids_as_production_v1_baseline=ours_baseline['passed_case_ids'] == reference['passed_case_ids'],
                  same_per_case_results=[(c['case_id'], c['passed']) for c in ours_baseline['cases']] ==
                                        [(c['case_id'], c['passed']) for c in reference['cases']])
    C.atomic_json(OUT / 'root-v2-check.json', result)
    print('ROOT_V2', json.dumps(result), flush=True)
finally:
    (root / 'STOP').write_text('now')
    if (root / 'state.json').exists():
        C.save_state(cfg, C.read_json(root / 'state.json'), status='stopped', phase='idle', pid=None, child_pid=None,
                     last_decision='evaluation_only_not_adopted')
