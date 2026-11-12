"""Switch config/continuous-elfnotes.json to the manually adopted cumulative-16384 model root.
The previous root (v2 adapter, previous weights), its config and the retention-failure records stay
untouched. Writes a rollback record. No training, no resident operation."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
config = P / 'config/continuous-elfnotes.json'
backup = A / 'source-before/config/continuous-elfnotes.json'
check = C.read_json(A / 'root-check.json')
current, staged = C.read_json(config), C.read_json(A / 'config-staged/continuous-elfnotes.json')
assert C.sha(config) == C.sha(backup), 'production config changed since the backup'
assert current['root'] == check['old_root'] and staged['root'] == check['new_root']
for root in [Path(check['old_root']), Path(check['new_root'])]:
    state = C.read_json(root / 'state.json')
    assert (root / 'STOP').read_text().strip() == 'now' and state['status'] == 'stopped' and state['pid'] is None
assert check['same_per_case_as_experiment_final'] and check['baseline_lua_errors'] == 0
retention_records = {str(P / 'experiments/opening-d16384-20261109/battles/battle-result.json'):
                     'd9d6aee2c3606789a9ab5aa49ed7d91b422fdc598a2d352d9d96e8f811c23e0e',
                     str(P / 'experiments/opening-d16384-20261109/fixed-test-comparison.json'):
                     'e4877ccfb371270ee3f303b0b08f52fdbdd7b18cae81d94ecaf93fd38b5fca79'}
assert all(C.sha(p) == h for p, h in retention_records.items()), 'retention-failure records must stay unchanged'
before = C.sha(config)
C.atomic_json(config, staged)
record = dict(
    switched_jst=C.now(), adoption='MANUAL by user decision; cumulative-16384 D candidate. Automatic (pre-registered) retention rule failed and stays on record.',
    config=str(config), config_sha256_before=before, config_sha256_after=C.sha(config),
    old_root=check['old_root'], new_root=check['new_root'],
    old_verified_policy_config=current['verified_policy_config'], old_verified_policy_config_sha256=C.sha(current['verified_policy_config']),
    new_verified_policy_config=staged['verified_policy_config'], new_verified_policy_config_sha256=C.sha(staged['verified_policy_config']),
    old_base_sha256='966cc8b1c3f0c092e824cc52dffcaf9c71f0c9373ece0fd07831d4a9df4f7cd1',
    old_head_sha256='1aaf569daf53a4453ef2f2e450c924af014aea83b3e4f98fea2644b795b52dc8',
    new_base_sha256=check['base_sha256'], new_head_sha256=check['head_sha256'],
    placement_adapter_sha256=check['center_placement_sha256'], retention_failure_records_unchanged=retention_records,
    both_roots_stopped=True, resident_operation_resumed=False, windows_side_touched=False, training_performed=False,
    rollback=f'copy {backup} to {config}: both inference entries then load the previous pair (966cc8b1/1aaf569d) with placement v2 from the unchanged v2 root')
C.atomic_json(A / 'switch-record.json', record)
print('SWITCHED', json.dumps(record), flush=True)
