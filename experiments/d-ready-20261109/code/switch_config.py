"""Switch config/continuous-elfnotes.json to the verified, STOPPED condition-D root. The adopted pair
and placement adapter v2 do not change. The previous root/config stay untouched; a rollback record
is written. No training, no resident start."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
config = P / 'config/continuous-elfnotes.json'
backup = A / 'source-before/config/continuous-elfnotes.json'
check = C.read_json(A / 'root-check.json')
current, staged = C.read_json(config), C.read_json(A / 'config-staged/continuous-elfnotes.json')
assert C.sha(config) == C.sha(backup) == 'b50810a34352f00a3f14f4bb79154f2c41b60e3d918c512aeeb23c8152b07d15', 'production config changed since the backup'
assert C.sha(A / 'config-staged/continuous-elfnotes.json') == check['staged_config_sha256']
assert current['root'] == check['old_root'] and staged['root'] == check['new_root']
assert current['verified_policy_config'] == staged['verified_policy_config']  # same adopted pair and adapters
for root in [Path(check['old_root']), Path(check['new_root'])]:
    state = C.read_json(root / 'state.json')
    assert (root / 'STOP').read_text().strip() == 'now' and state['status'] == 'stopped' and state['pid'] is None and state['pending'] is None
assert check['same_per_case_results'] and check['baseline_lua_errors'] == 0 and check['trainer_matches_effective_config']
records = {str(P / 'experiments/opening-d16384-20261109/battles/battle-result.json'): 'd9d6aee2c3606789a9ab5aa49ed7d91b422fdc598a2d352d9d96e8f811c23e0e',
           str(P / 'experiments/opening-d16384-20261109/fixed-test-comparison.json'): 'e4877ccfb371270ee3f303b0b08f52fdbdd7b18cae81d94ecaf93fd38b5fca79',
           str(P / 'experiments/adoption-d16384-20261109/switch-record.json'): '46e3995c4a3314c8be6b8eff21c459a1e20127a5bdb0dacfd5e103203bfbc53e'}
assert all(C.sha(p) == h for p, h in records.items()), 'retention-failure / manual-adoption records must stay unchanged'
before = C.sha(config)
C.atomic_json(config, staged)
record = dict(
    switched_jst=C.now(), change='training method only: condition D (lr 1.5e-5, train-split rehearsal lambda 5), bootstrap/cycle 4096; '
                                 'adopted pair and placement adapter v2 unchanged',
    config=str(config), config_sha256_before=before, config_sha256_after=C.sha(config),
    old_root=check['old_root'], new_root=check['new_root'],
    old_training=dict(learning_rate=current['learning_rate'], rehearsal=current.get('rehearsal'),
                      bootstrap_steps=current['bootstrap_steps'], steps_per_cycle=current['steps_per_cycle']),
    new_training=dict(learning_rate=staged['learning_rate'], rehearsal=staged['rehearsal'],
                      bootstrap_steps=staged['bootstrap_steps'], steps_per_cycle=staged['steps_per_cycle']),
    verified_policy_config=staged['verified_policy_config'], base_sha256=check['base_sha256'], head_sha256=check['head_sha256'],
    placement_adapter_sha256=check['center_placement_sha256'], new_trainer_sha256=check['trainer_sha256'],
    records_unchanged=records, both_roots_stopped=True, resident_operation_resumed=False, windows_side_touched=False,
    training_performed=False, battles_performed=False,
    rollback=f'copy {backup} to {config}: production returns to the previous stopped root (same adopted pair, plain PPO lr 3e-5, '
             'no rehearsal, cycle 65536); the previous root was never modified')
C.atomic_json(A / 'switch-record.json', record)
print('SWITCHED', json.dumps({k: record[k] for k in ['config_sha256_before', 'config_sha256_after', 'new_root', 'new_training', 'rollback']}), flush=True)
