"""Isolated training root for the B2B small-training trial: build + verify (no training), then exactly one 4,096-decision
chunk from the adopted pair. Refuses to train twice. No production change, no adoption, no battles.

  train_b2b.py build     # STOP=now first; standard initialize(); pin the extended rehearsal module and B2B demo data
  train_b2b.py verify    # trainer text, pinned hashes, trainer-side loading of the mixed data, 14/1/1 schedule (no training)
  train_b2b.py train     # one chunk (4,096 decisions, registered seed), candidate recorded in candidate.json
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-train-20261109'
ROOT = OUT / 'root'
CONFIG_OUT = OUT / 'config.json'
PLAN = json.loads((OUT / 'plan.json').read_text())
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402

REH = PLAN['training']['rehearsal']
SETTINGS = dict(enabled=True, split='train', lambda_bc=REH['lambda_bc'], lambda_kl=REH['lambda_kl'],
                sequences_per_minibatch=REH['sequences_per_minibatch'], seed=REH['seed'], sha256=REH['old_data_sha256'],
                b2b=dict(enabled=True, **{k: REH['b2b'][k] for k in ['sha256', 'manifest_sha256', 'block_sequences', 'old_per_block',
                                                                     'demos_per_block', 'schedule_seed']}))


def config():
    prod = json.loads((P / 'config/continuous-elfnotes.json').read_text())
    cfg = dict(prod, root=str(ROOT), learning_rate=PLAN['training']['learning_rate'], rehearsal=SETTINGS,
               bootstrap_steps=PLAN['training']['decisions'], steps_per_cycle=PLAN['training']['decisions'],
               initial_checkpoint=PLAN['start_model']['path'],
               training_condition='B2B trial: D (lr 1.5e-5, train rehearsal lambda 5) + 2 train-only B2B demonstrations at 12.5% of rehearsal sequences',
               experiment_note='isolated experiment root; not production; never run the loop here (STOP=now)')
    for key in ['previous_production_root', 'previous_production_config_sha256', 'keep_all_checkpoints', 'save_interval_updates']:
        cfg.pop(key, None)
    return cfg


def build():
    assert not ROOT.exists(), 'root already built'
    cfg = config()
    C.atomic_json(CONFIG_OUT, cfg)
    ROOT.mkdir(parents=True)
    (ROOT / 'STOP').write_text('now')  # before anything else exists in the root
    cfg = C.load_config(CONFIG_OUT)
    B = C.backend()
    B.initialize(cfg)  # standard snapshot: policy code, train.npz, the production rehearsal trainer patch, adopted pair
    snap = ROOT / 'snapshot'
    module = OUT / 'code/b2b_opening_rehearsal.py'
    assert C.sha(module) == PLAN['code']['b2b_opening_rehearsal_sha256'], 'extended module changed after registration'
    shutil.copyfile(module, snap / 'opening_rehearsal.py')
    for name in ['b2b-demos.npz', 'b2b-demos-manifest.json']:
        shutil.copyfile(OUT / 'data' / name, snap / 'rehearsal' / name)
    sys.path.insert(0, str(snap))
    import importlib.util
    spec = importlib.util.spec_from_file_location('pinned_rehearsal', snap / 'opening_rehearsal.py')
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    settings = R.validate_settings(cfg['rehearsal'])
    old = R.load_train_split(snap / 'opening-manifest.json', snap / 'rehearsal', expected_sha256=settings['sha256'])
    demos = R.load_demo_split(snap / 'opening-manifest.json', snap / 'rehearsal', settings['b2b'])
    merged = R.merge(old, demos)
    settings_path = snap / 'rehearsal/settings.json'
    previous = json.loads(settings_path.read_text())
    (snap / 'rehearsal/settings.json').write_text(json.dumps(dict(
        previous, settings=settings, data=old['stats'], b2b_data=demos['stats'], merged=dict(sequences=merged['stats']['sequences'], max_length=merged['stats']['max_length']),
        objective='ppo_loss + lambda_bc * acceptable_action_ce(train old 416 + 2 B2B demonstrations, fixed 14/1/1 per 16 sequences)',
        module='experiments/b2b-train-20261109/code/b2b_opening_rehearsal.py pinned as opening_rehearsal.py'), ensure_ascii=False, indent=2) + '\n')
    C.atomic_json(snap / 'hashes.json', {str(p.relative_to(snap)): C.sha(p) for p in snap.rglob('*') if p.is_file() and p.name != 'hashes.json'})
    C.runtime_check(cfg)
    print('BUILT', json.dumps(dict(root=str(ROOT), config_sha256=C.sha(CONFIG_OUT), snapshot_hashes_sha256=C.sha(snap / 'hashes.json'),
                                   trainer_sha256=C.sha(snap / 'train.py'), module_sha256=C.sha(snap / 'opening_rehearsal.py'))))


def verify():
    cfg = C.load_config(CONFIG_OUT)
    C.runtime_check(cfg)
    snap = ROOT / 'snapshot'
    B = C.backend()
    trainer = (snap / 'train.py').read_text()
    prod_trainer = (Path(json.loads((P / 'config/continuous-elfnotes.json').read_text())['root']) / 'snapshot/train.py').read_text()
    assert trainer == B.expected_trainer_text(cfg) == prod_trainer, 'trainer differs from the production rehearsal trainer'
    assert (ROOT / 'STOP').read_text() == 'now'
    state = C.read_json(ROOT / 'state.json')
    for name, key in [('base.flax_model', 'base_sha256'), ('chain-head.flax_model', 'head_sha256')]:
        assert C.sha(ROOT / 'models/initial' / name) == PLAN['start_model'][key]
    prod_hashes = C.read_json(Path(json.loads((P / 'config/continuous-elfnotes.json').read_text())['root']) / 'snapshot/hashes.json')
    ours = C.read_json(snap / 'hashes.json')
    differing = sorted(n for n in set(ours) | set(prod_hashes) if ours.get(n) != prod_hashes.get(n))
    probe = ("import json,sys,numpy as np;sys.path.insert(0,%r);import opening_rehearsal as R;ns={};R.install_rehearsal(ns);o=ns['OPENING_REHEARSAL'];"
             "s=R.MixedSampler(o.prepared,4,o.settings['sequences_per_minibatch'],[o.settings['b2b']['schedule_seed'],%d],o.settings['b2b']);"
             "idx=np.concatenate([s.next()[1].ravel() for _ in range(32)]);src=o.prepared['sources'][idx];"
             "blocks=[[int((src[b*16:(b+1)*16]==k).sum()) for k in (0,1,2)] for b in range(16)];"
             "print('PROBE',json.dumps(dict(module=R.__file__,settings=o.settings,old=o.prepared['stats']['old']['sequences'],b2b=o.prepared['stats']['b2b'],"
             "counts=s.counts(),blocks_ok=all(b==[14,1,1] for b in blocks))))") % (str(snap), PLAN['training']['training_seed'])
    done = subprocess.run([cfg['python'], '-B', '-c', probe], capture_output=True, text=True, timeout=900,
                          env=dict(os.environ, YGO_CONTINUOUS_CONFIG=str(CONFIG_OUT), JAX_PLATFORMS='cpu'))
    line = next((l for l in done.stdout.splitlines() if l.startswith('PROBE ')), None)
    assert line, done.stdout[-2000:] + done.stderr[-2000:]
    wiring = json.loads(line[6:])
    assert Path(wiring['module']).resolve() == (snap / 'opening_rehearsal.py').resolve()
    assert wiring['counts'] == {'0': 224, '1': 16, '2': 16} and wiring['blocks_ok'] and wiring['old'] == 416
    result = dict(verified_jst=C.now(), trainer_sha256=C.sha(snap / 'train.py'), trainer_identical_to_production=True,
                  snapshot_files_differing_from_production=differing, trainer_side_wiring=wiring, start_model=PLAN['start_model'],
                  stop=(ROOT / 'STOP').read_text(), status=state['status'], training_performed=False)
    C.atomic_json(OUT / 'root-check.json', result)
    print('VERIFIED', json.dumps(dict(differing=differing, counts=wiring['counts'], b2b_loss_steps=wiring['b2b']['loss_steps'],
                                      b2b_context_steps=wiring['b2b']['context_steps'])))


def train():
    assert not (OUT / 'candidate.json').exists(), 'one candidate only: refusing to train again'
    check = C.read_json(OUT / 'root-check.json')
    assert check['trainer_identical_to_production'] and check['training_performed'] is False
    cfg = C.load_config(CONFIG_OUT)
    C.runtime_check(cfg)
    state = C.read_json(ROOT / 'state.json')
    directory = ROOT / 'cycles' / '000001-b2b-trial'
    directory.mkdir(parents=True)
    pending = dict(cycle=1, directory=str(directory), source=str(ROOT / 'models/initial'), steps=PLAN['training']['decisions'],
                   training_seed=PLAN['training']['training_seed'], phase='training')
    C.atomic_json(OUT / 'training-started.json', dict(time_jst=C.now(), pending=pending))
    B = C.backend()
    # run_job() terminates a job whenever the root holds STOP=now (attempt 1 was aborted that way before it started).
    # Lift this experiment root's STOP only while the single job runs, and always restore it.
    assert (ROOT / 'STOP').read_text() == 'now'
    (ROOT / 'STOP').unlink()
    try:
        candidate = B.train_cycle(cfg, state, pending)
    finally:
        (ROOT / 'STOP').write_text('now')
        C.save_state(cfg, C.read_json(ROOT / 'state.json'), child_pid=None)
    result = C.read_json(directory / 'training-result.json')
    log = (directory / 'train.log').read_text(errors='replace')
    sources = next((json.loads(l.split(' ', 1)[1]) for l in log.splitlines() if l.startswith('OPENING_REHEARSAL_SOURCES ')), None)
    record = dict(time_jst=C.now(), candidate=str(candidate), base_sha256=C.sha(Path(candidate) / 'base.flax_model'),
                  head_sha256=C.sha(Path(candidate) / 'chain-head.flax_model'), training_result=result, sampled_source_counts=sources and sources['counts'],
                  rehearsal_metrics=str(directory / 'rehearsal-metrics.jsonl'), training_seed=pending['training_seed'], decisions=pending['steps'],
                  start_model=PLAN['start_model'])
    C.atomic_json(OUT / 'candidate.json', record)
    C.save_state(cfg, C.read_json(ROOT / 'state.json'), status='stopped', phase='idle', pid=None, child_pid=None,
                 last_decision='b2b_trial_trained_not_evaluated_not_adopted')
    print('TRAINED', json.dumps(dict(candidate=record['candidate'], base=record['base_sha256'][:16], head=record['head_sha256'][:16],
                                     sources=record['sampled_source_counts'])))


if __name__ == '__main__':
    dict(build=build, verify=verify, train=train)[sys.argv[1]]()
