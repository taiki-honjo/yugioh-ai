"""registration.json (before prepare / training / engine runs) and amendment-1.json (cache + prechecks fixed, before training).

  cv_register.py register
  cv_register.py fix-cache
"""
import filecmp
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import cv_runs as CR  # noqa: E402
import cv_train as CT  # noqa: E402
import sa_model as SM  # noqa: E402
import artmage_entry as E  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
SA = P / 'experiments/artmage-scoped-response-adapter-20261111'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    pins = {k: v for k, v in E.verify_pins().items()}
    sa_reg = json.loads((SA / 'registration.json').read_text())
    assert pins == json.loads((P / 'experiments/artmage-response-learning-20261111/registration.json').read_text())['pins']  # the scoped study asserted the same
    frozen = {n: dict(sha256=sha(ROOT / 'models/frozen' / n), byte_identical_to_source=filecmp.cmp(ROOT / 'models/frozen' / n, HB / 'train/candidate' / n, shallow=False))
              for n in ['base.flax_model', 'chain-head.flax_model']}
    assert all(v['byte_identical_to_source'] for v in frozen.values())
    same_code = {n: sha(ROOT / 'code' / n) == sa_reg['code'][n] for n in ['sa_model.py', 'artmage_entry.py', 'artmage_pins.py', 'flc_runs.py', 'hb_runs.py']}
    assert all(same_code.values()), same_code
    # C: reuse without new runs (the scoped study's 13 trajectories: no gated screen, identical to the start model)
    sa_sum = json.loads((SA / 'summary.json').read_text())
    c_evidence = {}
    for n in CR.C_NAMES:
        side = json.loads((SA / 'eval/retain/candidate' / f'{n}.policy.json').read_text())['policy']
        row = sa_sum['C']['rt1'].get(n) or sa_sum['C']['f'].get(n)
        c_evidence[n] = dict(scoped_record_sha256=sha(SA / 'eval/retain/candidate' / f'{n}.json'), start_record_sha256=sha(HB / 'eval/retain/candidate' / f'{n}.json'),
                             scoped_adapter_activations=len(side['adapter_active']), scoped_identical_to_start=row['identical_to_start'])
    assert all(v['scoped_adapter_activations'] == 0 and v['scoped_identical_to_start'] for v in c_evidence.values())
    screens = CT.screen_rows()
    reg = dict(
        status='registered_before_prepare_training_and_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-response-coverage-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-response-coverage-plan-2026-11-11.md')),
        frozen=frozen, pins=pins, unchanged_from_scoped_study=same_code,
        policy=dict(type=SM.TYPE, version=SM.VERSION, structure='unchanged (sa_model.py identical): frozen base + frozen chain head + response adapter, '
                    'same gate (select_chain, chain length >= 1, role 2), same inputs; no card / hand / deck conditions added'),
        screens={k: dict(note=CT.SCREENS[k]['note'], kind=CT.SCREENS[k]['kind'], source=str(CT.SCREENS[k]['dir']), step=CT.SCREENS[k]['step'],
                         npz_index=screens[k]['index'], recorded_action=list(CT.SCREENS[k]['want']), who=screens[k]['who'], files=screens[k]['files'])
                 for k in CT.ORDER},
        keep_screens_note='M and P keep the frozen base+head distribution; neither is claimed optimal (M did not stop 獄神獣; P is not a proven-correct pass). '
                          'From this study on M and P are training / repair cases (earlier splits are not rewritten).',
        training=dict(loss='mean(CE_S1, CE_S2) + 1.0 * mean(KL_M, KL_P); KL(p_frozen base+head || p_frozen logits + adapter), temperature 1, legal options only, '
                           'reference stop_gradient; no loss on other screens or the old 42 sequences; no gradient into the context',
                      updates=512, optimizer='fresh Adam 1e-3, eps 1e-5', clip='global norm 1.0 on the adapter', seed=11111501, init='the scoped study\'s init (zero output), not 405d2987',
                      device='CPU', coefficient=CT.COEF, no=['coefficient tuning', 'more candidates', 'intermediate selection', 'extension', 'other seeds'],
                      logged='per update: CE and KL separately, their pre-clip gradient norms, the combined norm, clip flag, update norm, max |delta|; '
                             'before / after: each screen\'s first choice, probabilities, CE / KL, per-option delta'),
        prechecks=['zero adapter == the original entry on the 42 sequences + the 8 comparison NPZ (logits, RNN states, post-position actions)',
                   'after training: the same off-target identity; reload equality on the 4 screens; the 3-file entry uses the adapter',
                   'frozen base/head, sa_model.py, gate, position adapter identical to the scoped study'],
        evaluation=dict(
            new_runs='the 8 functional-comparison conditions (BV1 1/2-cost x 改変/襲撃 x O-M/O-I) with the new candidate: record + replay = 16; cap 18 (2 spare '
                     'only for a harness fix that keeps the conditions)',
            runner='cv_runs.py (code-changes-cv_runs.diff), the 3-file entry passed explicitly; every record carries the 3 hashes and the adapter log',
            C=dict(new_runs=0, reuse='the 13 trajectories have no gated screen (scoped study: 0 activations, identical to the start model; Codex audit 62bad1de); '
                                     'the frozen inference and the gate are unchanged and only the never-activated adapter weights differ, so the trajectories are '
                                     'the same; shown offline by the off-target identity checks', evidence=c_evidence),
            A=dict(new_runs=0, reference='A-S1 / A-S2 are referenced from the B runs BV1-1cost-VARNISH-OM / BV1-2cost-ASSAULT-OM only if, up to the first '
                                         'opponent-chain window, the B trace (snapshots, action meanings) and NPZ inputs equal the old start-model record and the '
                                         'adapter acted 0 times before it; not counted as independent runs'),
            compared=dict(start_model={n: sha(FLC / 'runs' / n / 'record.json') for n in CR.B_NAMES},
                          scoped_adapter={n: sha(SA / 'runs' / f'B-{n}' / 'record.json') for n in CR.B_NAMES if (SA / 'runs' / f'B-{n}').exists()},
                          scoped_adapter_not_run=[n for n in CR.B_NAMES if not (SA / 'runs' / f'B-{n}').exists()]),
            items=['1. S1 stop by ディアクトロス② kept; S2 and 2-cost 改変 stops not lost',
                   '2. M (1cost 襲撃 O-M): the original first response and follow-up regained? a different response is recorded as a change from the keep '
                   'target with its resolution / resources / LP, not called a bad move',
                   '3. P (2cost 改変 O-I): the original pass kept? (not a proof that pass is optimal)',
                   '4. the 3 other O-I response screens (not trained): how the choice and the result change; not fed back into training',
                   '5. the path before the response equals the original; changes after a different response are separated'],
            items_note='observation points of this experiment, not production acceptance conditions'),
        engine=dict(cap=18, plan='8 records + 8 replays = 16', failures_count=True, previous_unused_not_carried=True),
        not_done=['PPO', 'normal battles', 'adoption', 'acceptance changes', 'keep-vs-reshape re-test', 'input / gate changes', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        code_changes=dict(cv_runs=dict(file='code-changes-cv_runs.diff', sha256=sha(ROOT / 'code-changes-cv_runs.diff')),
                          training='cv_train.py (new): 4 screens, CE + KL; sa_model.py unchanged'),
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


def fix_cache():
    path = ROOT / 'amendment-1.json'
    if path.exists():
        raise SystemExit('amendment-1 exists')
    if (ROOT / 'train/updates.jsonl').exists():
        raise SystemExit('training started')
    reg = json.loads((ROOT / 'registration.json').read_text())
    prep = json.loads((ROOT / 'cache/prepare.json').read_text())
    if not prep['all_prechecks_passed']:
        raise SystemExit('prechecks failed: report before continuing')
    assert prep['cache']['sha256'] == sha(CT.CACHE)
    am = dict(amendment=1, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              before_training=True, change='none to code or conditions; fixes the 4-screen cache and the precheck results',
              cache=dict(file=str(CT.CACHE), sha256=sha(CT.CACHE), source=prep['cache']['source'], backend=prep['backend']),
              prechecks=dict(file=str(ROOT / 'cache/prepare.json'), sha256=sha(ROOT / 'cache/prepare.json'), passed=True), code=reg['code'])
    path.write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    print('AMENDMENT', sha(path))


if __name__ == '__main__':
    {'register': register, 'fix-cache': fix_cache}[sys.argv[1]]()
