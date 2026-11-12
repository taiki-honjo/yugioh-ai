"""Create the stopped production root for condition D (PPO lr 1.5e-5 + train-split rehearsal lambda 5)
with the adopted cumulative-16384 pair and placement adapter v2. STOP=now is the first file.
Verification only: NO training, NO resident start. Does not switch the production config."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal as R

D = dict(enabled=True, split='train', lambda_bc=5.0, lambda_kl=0.0, sequences_per_minibatch=2, seed=20261109,
         sha256='0ef4700c92ddb6f26c3e3f5dc04d4893d2cb02c096304af0bbdc546f8070e4da')
ADOPTED = ('86b4358dcfec696cb4b3cc4c19781c34292b2509bf037b9e4f44d9ed8ec59d1c', '6ca3221f1828eecc5cea817003c8df5e8802b75fabef0f4aa33105ad0f979fce')
production_config_path = P / 'config/continuous-elfnotes.json'
production_config = C.read_json(production_config_path)
old_root = Path(production_config['root'])
assert old_root.name == 'continuous-elfnotes-openings-d16384-20261109', 'unexpected production root'
policy = C.read_json(production_config['verified_policy_config'])
assert (policy['base_sha256'], policy['head_sha256']) == ADOPTED
new_root = P / 'experiments/continuous-elfnotes-openings-d16384-rehearsal-ready-20261109'
staged = A / 'config-staged/continuous-elfnotes.json'
if not staged.exists():
    C.atomic_json(staged, dict(production_config, root=str(new_root), learning_rate=1.5e-5, rehearsal=D,
                               bootstrap_steps=4096, steps_per_cycle=4096,
                               initial_checkpoint=policy['selected_checkpoint'],
                               initial_model_note='Adopted cumulative-16384 D model (manual adoption 2026-11-09, base 86b4358d.../head 6ca3221f...). '
                                                  'Training in this root starts from it; decisions added here are counted from 0.',
                               initial_training_steps_note='initial_training_steps=69632 is the historical RL-origin count of the original base, '
                                                           'not decisions added in this root and not the 16384 D decisions',
                               training_condition='D: PPO learning_rate 1.5e-5 + train-split acceptable-action rehearsal lambda_bc 5 (no KL, not CLEAR)',
                               previous_production_root=str(old_root),
                               previous_production_config_sha256=C.sha(production_config_path)))
cfg = C.load_config(staged)
R.validate_settings(cfg['rehearsal'])
assert (cfg['learning_rate'], cfg['bootstrap_steps'], cfg['steps_per_cycle']) == (1.5e-5, 4096, 4096)
assert not cfg.get('keep_all_checkpoints') and 'save_interval_updates' not in cfg  # experiment-only settings stay off
manifest = C.read_json(cfg['opening_manifest'])
assert manifest['decks']['ElfnotesKewlTune']['train']['sha256'] == D['sha256']

new_root.mkdir(parents=True, exist_ok=True)
(new_root / 'STOP').write_text('now')  # before any other file exists in the new root
B = C.backend()
state = B.initialize(cfg)
assert (new_root / 'STOP').read_text() == 'now' and state['status'] == 'stopped' and state['pid'] is None
C.runtime_check(cfg)

# ---- snapshot, trainer and data
ours, theirs = C.read_json(new_root / 'snapshot/hashes.json'), C.read_json(old_root / 'snapshot/hashes.json')
differing = sorted(n for n in set(ours) | set(theirs) if ours.get(n) != theirs.get(n))
expected_diff = {'train.py', 'continuous_opening_backend.py', 'policy-code/verify_opponent_article_basics.py',
                 'opening_rehearsal.py', 'rehearsal/train.npz', 'rehearsal/train-witnesses.json', 'rehearsal/settings.json'}
assert set(differing) <= expected_diff, differing
assert ours['policy-code/center_placement.py'] == '3e96241095487fb48426099d774b7813cee348fc282dacbde0ced9af2d7abbba'
trainer_text = (new_root / 'snapshot/train.py').read_text()
assert trainer_text == B.expected_trainer_text(cfg) and 'install_rehearsal(globals())' in trainer_text
assert trainer_text.count("~jnp.reshape(obs['adapter_controlled_'], mask.shape)") == 2
assert ours['opening_rehearsal.py'] == C.sha(P / 'scripts/opening_rehearsal.py') and ours['rehearsal/train.npz'] == D['sha256']
assert not [n for n in ours if n.endswith(('validation.npz', 'holdout.npz'))]
pinned = C.read_json(new_root / 'snapshot/rehearsal/settings.json')
assert pinned['settings'] == R.validate_settings(D) and pinned['validation_or_holdout_loaded'] is False
for name, digest in zip(['base.flax_model', 'chain-head.flax_model'], ADOPTED):
    assert C.sha(new_root / 'models/initial' / name) == digest

# ---- trainer-side wiring exactly as train.py would do it (module from the snapshot, data checks); no training
probe = ("import json,sys;sys.path.insert(0,%r);import opening_rehearsal as R;ns={};R.install_rehearsal(ns);"
         "o=ns['OPENING_REHEARSAL'];print('PROBE',json.dumps(dict(module=R.__file__,settings=o.settings,stats=o.prepared['stats'])))"
         % str(new_root / 'snapshot'))
done = subprocess.run([cfg['python'], '-B', '-c', probe], capture_output=True, text=True, timeout=600,
                      env=dict(os.environ, YGO_CONTINUOUS_CONFIG=str(staged), JAX_PLATFORMS='cpu'))
line = next(l for l in done.stdout.splitlines() if l.startswith('PROBE '))
wiring = json.loads(line[6:])
assert Path(wiring['module']).resolve() == (new_root / 'snapshot/opening_rehearsal.py').resolve()
assert wiring['settings'] == R.validate_settings(D)
assert (wiring['stats']['sequences'], wiring['stats']['decisions'], wiring['stats']['loss_steps']) == (416, 25802, 19178)

# ---- the command a bootstrap/cycle WOULD run, captured before any job starts
captured = []
def capture(cfg_, state_, command, directory, name):
    captured.append([str(x) for x in command]); raise InterruptedError('captured; no training')
B.control.run_job = capture
with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, 'planned').mkdir()
    try:
        B.train_cycle(cfg, {}, dict(directory=tmp + '/planned', source=state['best_checkpoint'], steps=cfg['bootstrap_steps'], training_seed=0))
    except InterruptedError:
        pass
command = captured[-1]
flag = lambda f: command[command.index(f) + 1]
assert Path(command[command.index('-u') + 1]) == new_root / 'snapshot/train.py'
assert (float(flag('--learning-rate')), int(flag('--total-timesteps')), int(flag('--save-interval'))) == (1.5e-5, 4096, 32)
B = C.backend()  # restore the real job runner

# ---- 66 validation openings with the adopted pair in the new root (direct job; STOP=now stays)
baseline = new_root / 'opening-baseline.json'
if not baseline.exists():
    log = A / 'root-baseline.log'
    with log.open('w') as handle:
        done = subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G",
                               '-p', 'MemorySwapMax=0', '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10',
                               cfg['python'], '-B', '-u', str(P / 'scripts/continuous_learning.py'), 'job-openings',
                               '--config', str(staged), '--candidate', str(new_root / 'models/initial'), '--result', str(baseline)],
                              cwd=Path(cfg['runtime']) / 'agent/scripts', env=C.job_env(cfg), stdout=handle, stderr=subprocess.STDOUT,
                              timeout=1800)
    assert done.returncode == 0, f'baseline job failed; see {log}'
mine, reference = C.read_json(baseline), C.read_json(old_root / 'opening-baseline.json')
result = dict(new_root=str(new_root), old_root=str(old_root), staged_config=str(staged), staged_config_sha256=C.sha(staged),
              snapshot_hashes_sha256=C.sha(new_root / 'snapshot/hashes.json'), snapshot_files_differing_from_previous_root=differing,
              trainer_sha256=ours['train.py'], trainer_matches_effective_config=True, rehearsal_hooked=True,
              rehearsal_module_sha256=ours['opening_rehearsal.py'], rehearsal_data_sha256=ours['rehearsal/train.npz'],
              rehearsal_witnesses_sha256=ours['rehearsal/train-witnesses.json'], rehearsal_settings=pinned['settings'],
              rehearsal_data=pinned['data'], trainer_side_wiring=wiring,
              planned_bootstrap_command=command, base_sha256=ADOPTED[0], head_sha256=ADOPTED[1],
              center_placement_sha256=ours['policy-code/center_placement.py'],
              baseline_passed=mine['passed'], baseline_total=mine['total'], baseline_lua_errors=mine['lua_errors'],
              same_passed_case_ids_as_previous_root=mine['passed_case_ids'] == reference['passed_case_ids'],
              same_per_case_results=[(c['case_id'], c['passed']) for c in mine['cases']] == [(c['case_id'], c['passed']) for c in reference['cases']],
              stop=(new_root / 'STOP').read_text(), status=C.read_json(new_root / 'state.json')['status'], training_performed=False)
assert result['baseline_total'] == 66 and result['baseline_passed'] == 65 and result['baseline_lua_errors'] == 0
assert result['same_passed_case_ids_as_previous_root'] and result['same_per_case_results'] and result['stop'] == 'now'
C.save_state(cfg, C.read_json(new_root / 'state.json'), training_condition='D', learning_rate=cfg['learning_rate'],
             opening_rehearsal=pinned['settings'], bootstrap_steps=cfg['bootstrap_steps'], steps_per_cycle=cfg['steps_per_cycle'],
             adopted_model_version='d16384-20261109-manual', previous_production_root=str(old_root),
             placement_adapter_version='v2-remix-tracking-20261109', opening_validation_baseline=mine['passed'],
             opening_regression_total=mine['total'], normal_summon_usage_scored=False,
             last_decision='created_stopped_d_ready_no_training')
C.atomic_json(A / 'root-check.json', result)
print('ROOT', json.dumps({k: v for k, v in result.items() if k not in ('planned_bootstrap_command', 'rehearsal_data', 'trainer_side_wiring')}), flush=True)
