"""Create the v2 production root (STOP=now from the first moment), pin v2 in a NEW snapshot,
measure the adopted model's 66 validation openings, and compare with the v1 production root.
Does NOT switch config/continuous-elfnotes.json; switch_config.py does that after this passes.
No training, no resident operation, no adoption of new weights."""
import json, subprocess, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C

production_config = C.read_json(P / 'config/continuous-elfnotes.json')
old_root = Path(production_config['root'])
new_root = P / 'experiments/continuous-elfnotes-openings-v2-20261109'
assert old_root.name == 'continuous-elfnotes-openings-20261109', 'production already switched?'
staged = A / 'config-staged/continuous-elfnotes.json'
if not staged.exists():
    C.atomic_json(staged, dict(production_config, root=str(new_root),
                               verified_policy_config=str(P / 'config/verified-opening-policy-placement-remix-fix-20261109.json'),
                               placement_adapter_version='v2-remix-tracking-20261109',
                               previous_production_root=str(old_root),
                               previous_verified_policy_config=production_config['verified_policy_config']))
cfg = C.load_config(staged)
new_root.mkdir(parents=True, exist_ok=True)
(new_root / 'STOP').write_text('now')  # before any other file exists in the new root
B = C.backend()
state = B.initialize(cfg)
assert (new_root / 'STOP').read_text() == 'now' and state['status'] == 'stopped'

ours, theirs = C.read_json(new_root / 'snapshot/hashes.json'), C.read_json(old_root / 'snapshot/hashes.json')
differing = sorted(n for n in set(ours) | set(theirs) if ours.get(n) != theirs.get(n))
expected = {'policy-code/center_placement.py',            # v1 -> v2 (the change being adopted)
            'policy-code/verify_opponent_article_basics.py',  # embeds the root's own snapshot path
            'opening-policy.json',                         # v2 policy config
            'continuous_opening_backend.py'}                # module-origin guard added on 2026-11-09
assert set(differing) <= expected, differing
assert ours['policy-code/center_placement.py'] == '3e96241095487fb48426099d774b7813cee348fc282dacbde0ced9af2d7abbba'
assert ours['train.py'] == theirs['train.py'] == '70cd047ef8acb7b22f1a82191c1dd03d99770ee7185f88f018c4a2ebfd9399bd'
for name in ['base.flax_model', 'chain-head.flax_model']:
    assert C.sha(new_root / 'models/initial' / name) == C.sha(old_root / 'models/initial' / name)

baseline = new_root / 'opening-baseline.json'
if not baseline.exists():
    # Run the finite evaluation job directly so the root's STOP=now stays in place throughout
    # (run_job() would correctly refuse to start anything while STOP=now exists).
    log = A / 'v2-root-baseline.log'
    runtime = Path(cfg['runtime']) / 'agent'
    with log.open('w') as handle:
        done = subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G",
                               '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10',
                               cfg['python'], '-B', '-u', str(P / 'scripts/continuous_learning.py'), 'job-openings',
                               '--config', str(staged), '--candidate', str(new_root / 'models/initial'), '--result', str(baseline)],
                              cwd=runtime / 'scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT, timeout=1800)
    assert done.returncode == 0, f'baseline job failed; see {log}'
mine, reference = C.read_json(baseline), C.read_json(old_root / 'opening-baseline.json')
result = dict(new_root=str(new_root), old_root=str(old_root), staged_config=str(staged),
              snapshot_files_differing_from_v1_root=differing, trainer_sha256=ours['train.py'],
              v2_center_placement_sha256=ours['policy-code/center_placement.py'],
              base_sha256=C.sha(new_root / 'models/initial/base.flax_model'),
              head_sha256=C.sha(new_root / 'models/initial/chain-head.flax_model'),
              baseline_passed=mine['passed'], baseline_total=mine['total'], baseline_lua_errors=mine['lua_errors'],
              same_passed_case_ids=mine['passed_case_ids'] == reference['passed_case_ids'],
              same_per_case_results=[(c['case_id'], c['passed']) for c in mine['cases']] ==
                                    [(c['case_id'], c['passed']) for c in reference['cases']],
              stop=(new_root / 'STOP').read_text(), status=C.read_json(new_root / 'state.json')['status'])
assert result['baseline_total'] == 66 and result['baseline_lua_errors'] == 0
assert result['same_passed_case_ids'] and result['same_per_case_results'] and result['stop'] == 'now'
C.save_state(cfg, C.read_json(new_root / 'state.json'), placement_adapter_version='v2-remix-tracking-20261109',
             previous_production_root=str(old_root), opening_validation_baseline=mine['passed'], opening_regression_total=mine['total'],
             normal_summon_usage_scored=False, last_decision='created_stopped_v2_placement_adapter_same_weights')
C.atomic_json(A / 'v2-root-check.json', result)
print('V2_ROOT', json.dumps(result), flush=True)
