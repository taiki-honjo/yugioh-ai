"""Create the production root for the manually adopted cumulative-16384 model (STOP=now first),
pin it with placement adapter v2 in a NEW snapshot, and re-measure the 66 validation openings.
Does NOT switch config/continuous-elfnotes.json (switch_config.py does that after this passes).
No training, no resident operation."""
import json, subprocess, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C

production_config = C.read_json(P / 'config/continuous-elfnotes.json')
old_root = Path(production_config['root'])
assert old_root.name == 'continuous-elfnotes-openings-v2-20261109', 'production root is not the expected v2 root'
new_root = P / 'experiments/continuous-elfnotes-openings-d16384-20261109'
policy = P / 'config/verified-opening-policy-d16384-20261109.json'
staged = A / 'config-staged/continuous-elfnotes.json'
if not staged.exists():
    C.atomic_json(staged, dict(production_config, root=str(new_root), verified_policy_config=str(policy),
                               adopted_model_version='d16384-20261109-manual',
                               previous_production_root=str(old_root),
                               previous_verified_policy_config=production_config['verified_policy_config']))
cfg = C.load_config(staged)
new_root.mkdir(parents=True, exist_ok=True)
(new_root / 'STOP').write_text('now')  # before any other file exists in the new root
state = C.backend().initialize(cfg)
assert (new_root / 'STOP').read_text() == 'now' and state['status'] == 'stopped'

ours, theirs = C.read_json(new_root / 'snapshot/hashes.json'), C.read_json(old_root / 'snapshot/hashes.json')
differing = sorted(n for n in set(ours) | set(theirs) if ours.get(n) != theirs.get(n))
expected = {'opening-policy.json',                          # the new policy config (new model, same adapters)
            'policy-code/verify_opponent_article_basics.py',  # embeds the root's own snapshot path
            'continuous_opening_backend.py'}                  # optional checkpoint settings added after the v2 root
assert set(differing) <= expected, differing
assert ours['policy-code/center_placement.py'] == '3e96241095487fb48426099d774b7813cee348fc282dacbde0ced9af2d7abbba'
assert ours['train.py'] == theirs['train.py'] == '70cd047ef8acb7b22f1a82191c1dd03d99770ee7185f88f018c4a2ebfd9399bd'
model = Path(C.read_json(policy)['selected_checkpoint'])
for name in ['base.flax_model', 'chain-head.flax_model']:
    assert C.sha(new_root / 'models/initial' / name) == C.sha(model / name)

baseline = new_root / 'opening-baseline.json'
if not baseline.exists():
    log = A / 'root-baseline.log'
    with log.open('w') as handle:  # direct job run: the new root's STOP=now stays in place throughout
        done = subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G",
                               '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10',
                               cfg['python'], '-B', '-u', str(P / 'scripts/continuous_learning.py'), 'job-openings',
                               '--config', str(staged), '--candidate', str(new_root / 'models/initial'), '--result', str(baseline)],
                              cwd=Path(cfg['runtime']) / 'agent/scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT,
                              timeout=1800)
    assert done.returncode == 0, f'baseline job failed; see {log}'
mine = C.read_json(baseline)
experiment = C.read_json(P / 'experiments/opening-d16384-20261109/d-lr1.5e-5-l5/cycles/12288-seed10016384/opening-regression.json')
previous = C.read_json(old_root / 'opening-baseline.json')
result = dict(new_root=str(new_root), old_root=str(old_root), staged_config=str(staged), policy_config=str(policy),
              policy_config_sha256=C.sha(policy), snapshot_files_differing_from_v2_root=differing,
              trainer_sha256=ours['train.py'], center_placement_sha256=ours['policy-code/center_placement.py'],
              base_sha256=C.sha(new_root / 'models/initial/base.flax_model'), head_sha256=C.sha(new_root / 'models/initial/chain-head.flax_model'),
              baseline_passed=mine['passed'], baseline_total=mine['total'], baseline_lua_errors=mine['lua_errors'],
              same_per_case_as_experiment_final=[(c['case_id'], c['passed']) for c in mine['cases']] ==
                                                [(c['case_id'], c['passed']) for c in experiment['cases']],
              same_passed_ids_as_previous_adopted=mine['passed_case_ids'] == previous['passed_case_ids'],
              stop=(new_root / 'STOP').read_text(), status=C.read_json(new_root / 'state.json')['status'])
assert result['baseline_total'] == 66 and result['baseline_lua_errors'] == 0 and result['same_per_case_as_experiment_final']
assert result['stop'] == 'now'
C.save_state(cfg, C.read_json(new_root / 'state.json'), adopted_model_version='d16384-20261109-manual',
             adoption_record=str(A), previous_production_root=str(old_root), placement_adapter_version='v2-remix-tracking-20261109',
             opening_validation_baseline=mine['passed'], opening_regression_total=mine['total'], normal_summon_usage_scored=False,
             last_decision='created_stopped_manual_adoption_d16384_same_adapters')
C.atomic_json(A / 'root-check.json', result)
print('ROOT', json.dumps(result), flush=True)
