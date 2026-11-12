"""Switch config/continuous-elfnotes.json to the verified v2 root. Old root, its STOP/state/
snapshot/models and the v1 policy config are left untouched. Writes a rollback record."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
config = P / 'config/continuous-elfnotes.json'
backup = A / 'source-before/config/continuous-elfnotes.json'
check = C.read_json(A / 'v2-root-check.json')
current, staged = C.read_json(config), C.read_json(A / 'config-staged/continuous-elfnotes.json')
assert C.sha(config) == C.sha(backup), 'production config changed since the backup'
assert current['root'] == check['old_root'] and staged['root'] == check['new_root']
for root in [Path(check['old_root']), Path(check['new_root'])]:
    state = C.read_json(root / 'state.json')
    assert (root / 'STOP').read_text().strip() == 'now' and state['status'] == 'stopped' and state['pid'] is None
assert check['same_per_case_results'] and check['baseline_lua_errors'] == 0
before = C.sha(config)
C.atomic_json(config, staged)
record = dict(
    switched_jst=C.now(), adopted='placement adapter v2 only; neural weights unchanged (same base + chain-head pair)',
    config=str(config), config_sha256_before=before, config_sha256_after=C.sha(config),
    old_root=check['old_root'], new_root=check['new_root'],
    old_verified_policy_config=current['verified_policy_config'], old_verified_policy_config_sha256=C.sha(current['verified_policy_config']),
    new_verified_policy_config=staged['verified_policy_config'], new_verified_policy_config_sha256=C.sha(staged['verified_policy_config']),
    v1_center_placement_sha256='805e68213710ad411ae4c5a3fc47bd998a8eb7ed2113a3335386eeed7db00b3c',
    v2_center_placement_sha256=check['v2_center_placement_sha256'], base_sha256=check['base_sha256'], head_sha256=check['head_sha256'],
    both_roots_stopped=True, resident_operation_resumed=False, windows_side_touched=False,
    rollback=f'copy {backup} to {config} (both inference entries then load v1 again); the v1 root was never modified')
C.atomic_json(A / 'switch-record.json', record)
print('SWITCHED', json.dumps(record), flush=True)
