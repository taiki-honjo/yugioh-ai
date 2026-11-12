"""registration.json (before prepare / training / any engine run) and amendment-1.json (cache + prechecks fixed, before training).
Amendment-2 (the runs still needed) is written by rs_check.py after the offline full-action check.

  rs_register.py register
  rs_register.py fix-cache
"""
import filecmp
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rs_probe_runs as PR  # noqa: E402
import rs_train as RT  # noqa: E402
import sa_model as SM  # noqa: E402
import artmage_entry as E  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
CV = P / 'experiments/artmage-response-coverage-20261111'
UP = P / 'experiments/artmage-unseen-response-probe-20261111'
SCRIPTS = P / 'evaluation/opponent-article-basics-20261108/runtime/card-scripts'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    pins = {k: v for k, v in E.verify_pins().items()}
    assert pins == json.loads((P / 'experiments/artmage-response-learning-20261111/registration.json').read_text())['pins']
    frozen = {n: dict(sha256=sha(ROOT / 'models/frozen' / n), byte_identical_to_source=filecmp.cmp(ROOT / 'models/frozen' / n, HB / 'train/candidate' / n, shallow=False))
              for n in ['base.flax_model', 'chain-head.flax_model']}
    assert all(v['byte_identical_to_source'] for v in frozen.values())
    cv_reg = json.loads((CV / 'registration.json').read_text())
    same = {n: sha(ROOT / 'code' / n) == cv_reg['code'][n] for n in ['sa_model.py', 'artmage_entry.py', 'artmage_pins.py', 'flc_runs.py', 'rl_data.py']}
    assert all(same.values()), same
    reg = dict(
        status='registered_before_prepare_training_and_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-relative-score-retention-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-relative-score-retention-plan-2026-11-11.md')),
        frozen=frozen, pins=pins, unchanged_from_coverage=same,
        comparison_candidate=dict(folder=str(CV / 'train/candidate'), adapter_sha256=json.loads((CV / 'train/result.json').read_text())['saved']['adapter_sha256']),
        policy='unchanged structure / inputs / gate / position adapter / 3-file entry (sa_model.py identical); new adapter from the same seed and zero output',
        screens='the coverage study\'s 4 screens (S1 / S2 CE, M / P KL to the frozen base+head); the cache is recomputed the same way and must equal the coverage cache',
        objective=dict(formula='loss = mean(CE_S1, CE_S2) + 1.0 * mean(KL_M, KL_P) + 1.0 * relative_retention',
                       relative_retention='mean over screens of mean over protected options of (delta(option) - delta(pass))^2; target 0 '
                                          '(= keeping each option-vs-pass logit difference of the frozen model)',
                       protected='S1: legal minus pass minus every option with the taught meaning (ディアクトロス②); S2: legal minus pass minus ファインメルト③; '
                                 'M / P: legal minus pass; pass = the snapshot Cancel checked against the NPZ legal row; illegal / padding never counted; '
                                 'no special rule for G (it is one of the untaught legal options where legal)',
                       not_goals=['making the raw pass delta 0', 'shrinking all |delta|']),
        training=dict(device='CPU', updates=512, optimizer='fresh Adam 1e-3, eps 1e-5', clip='global norm 1.0 on the adapter', seed=11111501,
                      coefficients=dict(kl=1.0, relative=1.0), no=['coefficient search', 'intermediate selection', 'extension', 'other seeds', 'more candidates'],
                      logged='CE, KL, relative term, each pre-clip gradient norm, combined norm, clip flag, update norm; before / after every legal option: '
                             'frozen logit, delta, delta minus delta(pass), new margin, probability; protected max |relative delta|'),
        target_threshold='training screens: the maximum |relative delta| over protected options <= 1.0 point (an experiment guide, not an acceptance '
                         'criterion; a smaller original margin can still flip)',
        existing_conditions=dict(
            flc8=dict(records={n: str(CV / 'runs' / f'B-{n}') for n in RT.FLC8}),
            probe4=dict(records={f'{h}-{o}': str(UP / 'runs' / f'{h}-{o}-CAND') for h in ['H1', 'H2'] for o in ['ASH', 'GAMMA']},
                        base_records={f'{h}-{o}': str(UP / 'runs' / f'{h}-{o}-BASE') for h in ['H1', 'H2'] for o in ['ASH', 'GAMMA']}),
            reuse_rule='the new candidate is fed every own observation of the saved NPZ from the start (RNN reset, each once, position adapter on); '
                       'reused only if EVERY own action index equals the recorded one, the record replayed exactly and its hashes are pinned; one differing '
                       'action -> record + replay with the engine; reused results are referenced, not reported as new runs nor as independent samples',
            C='the 13 retention cases are reused (no gated screen; frozen inference and gate unchanged)',
            A='referenced from the corresponding reused / new B records (S1 / S2) after checking the pre-window inputs and 0 earlier activations'),
        new_condition=dict(name='H1-USAGI', policies=['BASE (frozen base + head)', 'CAND (this study\'s 3-file candidate)'], runs='record + replay each = 4',
                           opponent='as in the unseen-response probe H1, but at the target the opponent activates 幽鬼うさぎ 59438930 (already in its fixed hand); '
                                    'other optional effects pass; same forced-choice rules; end at the first decision after the target chain resolved',
                           usagi_script=dict(file='c59438930.lua', sha256=sha(SCRIPTS / 'c59438930.lua'),
                                             reading='condition: the activating card is on the field and it is a monster effect (or a face-up S/T effect); cost: send '
                                                     'itself to the GY; operation: DESTROY the activating card (no negation)'),
                           not_established='if the effect is illegal or the target is not reached, recorded as such; no other hand is searched',
                           coverage_column='the coverage candidate only as an offline score computation on the BASE saved inputs, never as a played result'),
        engine=dict(cap=30, plan='non-matching existing conditions record + replay (at most 24) + H1-USAGI 4 = at most 28; 2 spare for a harness fix only',
                    failures_count=True, previous_unused_not_carried=True),
        flc_to_run=[], probe_to_run=[],
        items=['1. S1 / S2 improvements, the 2-cost 改変 O-M stop, M / P kept',
               '2. training screens: protected max |relative delta| <= 1.0',
               '3. ASH / GAMMA 4 screens and the new USAGI screen: G-vs-pass relative delta (also whether within 1.0); not fed back into training',
               '4. untrained O-I responses and their results (e.g. 無限泡影 negated by ディアクトロス②) with resources; using an effect is not called an improvement'],
        items_note='a same choice under a 33-35 margin is not counted as success; the relative-delta change is the main check',
        not_done=['PPO', 'normal battles', 'adoption', 'acceptance changes', 'input / structure / gate changes', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        code_changes={f: sha(ROOT / f) for f in ['code-changes-rs_train.diff', 'code-changes-rs_probe_runs.py.diff', 'code-changes-rs_flc_runs.py.diff']},
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
    assert prep['cache']['sha256'] == sha(RT.CACHE)
    am = dict(amendment=1, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              before_training=True, change='none to code or conditions; fixes the 4-screen cache (+ pass index, protected masks) and the precheck results',
              cache=dict(file=str(RT.CACHE), sha256=sha(RT.CACHE), coverage_arrays_equal=prep['coverage_cache']['arrays_equal']),
              prechecks=dict(file=str(ROOT / 'cache/prepare.json'), sha256=sha(ROOT / 'cache/prepare.json'), passed=True),
              flc_to_run=[], probe_to_run=[], code=reg['code'])
    path.write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    print('AMENDMENT', sha(path))


if __name__ == '__main__':
    {'register': register, 'fix-cache': fix_cache}[sys.argv[1]]()
