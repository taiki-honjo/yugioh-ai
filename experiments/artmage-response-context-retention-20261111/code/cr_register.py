"""Write registration.json once (before the reference, training and any engine run), and later amendment-1.json that
fixes the reference arrays by hash (still before training). No duel here.

  cr_register.py register
  cr_register.py fix-reference
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rl_runs as RR  # noqa: E402
import rl_data as RD  # noqa: E402
import batch_model as BM  # noqa: E402
import cr_data as CD  # noqa: E402
import cr_model as CM  # noqa: E402
import artmage_entry as E  # noqa: E402  (pins checked; engine library initialised, no duel)

P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
PREV = P / 'experiments/artmage-response-learning-20261111'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    assert Path(E.__file__).resolve().parent == ROOT / 'code'
    pins = {k: v for k, v in E.verify_pins().items()}
    hb_code = json.loads((HB / 'amendment-1.json').read_text())['code']
    prev_reg = json.loads((PREV / 'registration.json').read_text())
    pins_same = dict(artmage_pins_py_equal_to_hand_branch=sha(ROOT / 'code/artmage_pins.py') == hb_code['artmage_pins.py'],
                     artmage_entry_py_equal_to_hand_branch=sha(ROOT / 'code/artmage_entry.py') == hb_code['artmage_entry.py'],
                     verified_now=True, equal_to_response_study_registration=pins == prev_reg['pins'])
    assert all(pins_same.values()), pins_same
    prev_result = json.loads((PREV / 'train/result.json').read_text())
    prev_files = {}
    for site in RR.A_SITES:
        d = PREV / 'runs' / f'A-{site}'
        prev_files[f'A-{site}'] = {f: sha(d / f) for f in ['record.json', 'observations.npz', 'replay.json']}
    for n in RR.B_NAMES:
        d = PREV / 'runs' / f'B-{n}'
        prev_files[f'B-{n}'] = {f: sha(d / f) for f in ['record.json', 'observations.npz', 'replay.json']}
    for n in RR.C_NAMES:
        prev_files[f'C-{n}'] = sha(PREV / 'eval/retain/candidate' / f'{n}.json')
    start_files = {f'A/B-{n}': sha(FLC / 'runs' / n / 'record.json') for n in RR.B_NAMES}
    start_files.update({f'C-{n}': sha(HB / 'eval/retain/candidate' / f'{n}.json') for n in RR.C_NAMES})
    reg = dict(
        status='registered_before_reference_training_and_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-response-context-retention-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-response-context-retention-plan-2026-11-11.md')),
        start_model=dict(base_sha256=sha(ROOT / 'models/start/base.flax_model'), head_sha256=sha(ROOT / 'models/start/chain-head.flax_model'),
                         source='experiments/artmage-hand-branch-20261110/train/candidate (unadopted); the same start as the response study'),
        comparison_candidate=dict(base_sha256=prev_result['saved']['candidate_base_sha256'], head_sha256=prev_result['saved']['candidate_head_sha256'],
                                  root=str(PREV), report='docs/artmage-response-learning-2026-11-11-report.md', frozen=True),
        pins=pins, pins_identity=pins_same,
        data=dict(manifest=str(RD.MANIFEST), sha256=sha(RD.MANIFEST), note='the response study manifest (old 40 + 2 response sequences), unchanged'),
        retention=dict(
            formula='response minibatch loss = CE_response + 1.0 * KL_context; old minibatches: CE only (unchanged)',
            kl='KL(p_start_legal || p_current_legal), temperature 1, legal options only; mean over the preserved steps of each sequence, then mean over the 2 sequences',
            preserved_steps='own steps strictly before the response window, message != select_position, >= 2 legal options; window, padding excluded',
            reference='the START model\'s own legal-normalised distribution along the recorded history (executed actions only advance the history, '
                      'never labels); base + head final logits; computed by cr_train.py reference on the GPU (matmul precision highest) in the response '
                      'minibatch layout; the CPU difference is recorded by cr_train.py refcheck; fixed by hash in amendment-1 before training',
            reference_gradient='stop_gradient (constant input)', candidate_rnn='the candidate\'s own weights along the same observations',
            backprop='as in the response study (no extra stop-gradient into the context)',
            coefficient=CM.COEF, coefficient_note='fixed for this trial; not tuned; no other coefficients'),
        training=dict(sampling_seed=11111301, retention_coefficient=CM.COEF, updates=BM.CONFIG['updates'], batch_sequences=2, learning_rate=BM.CONFIG['learning_rate'],
                      adam_eps=BM.CONFIG['adam_eps'], max_grad_norm=BM.CONFIG['max_grad_norm'], optimizer='fresh Adam state; base + chain head',
                      blocks='8 minibatches: 3 old GroupSampler + 3 old pair + 2 response, the response study\'s draw order (checked in the job)',
                      previous_updates_sha256=sha(PREV / 'train/updates.jsonl'),
                      planned=dict(old_group_minibatches=96, old_pair_minibatches=96, response_minibatches=64, response_s1=64, response_s2=64,
                                   sequence_presentations=dict(old=384, new=128)),
                      logged='per update: CE and KL separately, pre-clip gradient norm of each term, combined norm, clip flag, base / head update after Adam',
                      diagnostics='updates 0 and 256 on the GPU: 42-sequence teacher-forced CE / top-1, response probabilities, branch detail, and on the '
                                  'preserved steps: mean / max KL, argmax changes (card / effect / zone), start top-1 minus top-2',
                      not_used=['PPO', 'value loss', 'entropy'], no=['intermediate selection', 'other seeds', 'extension', 'coefficient tuning'],
                      after='save, reload through the isolated entry, logits + post-adapter actions on 42 sequences equal to memory, weights finite'),
        evaluation=dict(
            same_as='docs/artmage-response-learning-plan-2026-11-11.md (A 4, B 8, C 13); conditions, seeds, opponent, no normal draws unchanged',
            runner='rl_runs.py of the response study with ROOT rebound to this study (code-changes-rl_runs.diff); frozen flc_runs.py / hb_runs.py unchanged',
            compared=dict(start_model=dict(files=start_files, note='functional comparison B records (also the A old paths) and hand-branch retention records'),
                          no_retention_candidate=dict(files=prev_files, runner_sha256=prev_reg['code']['rl_runs.py'],
                                                      note='same runner semantics; this study\'s rl_runs.py differs only in ROOT (hash differs)')),
            checks=['1. A: S1 response improvement and S2 stop kept (choice of the demonstrated effect and functional stop reported separately)',
                    '2. B: the two 2-cost conditions the start model stopped are not lost, and the S1 improvement appears on the free path',
                    '3. C: no start-model success lost (RT1 g06-g11 and F0/F1/F2/F4/F5/F6-V; F3-V is NOT in this denominator); new successes counted '
                    'separately; F4-V recovery and F3-V continuation both tabled'],
            checks_note='local observation points of this experiment, not production acceptance conditions; misses are reported, no on-the-spot changes'),
        engine=dict(cap=28, plan='A 4 + B 8 + C 13 = 25', spare='3 only for re-running after a harness fix that keeps the conditions', failures_count=True,
                    previous_unused_not_carried=True),
        not_done=['PPO', 'normal battles', 'adoption', 'acceptance / reward changes', 'keep-vs-reshape re-test', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        code_changes=dict(rl_runs=dict(file='code-changes-rl_runs.diff', sha256=sha(ROOT / 'code-changes-rl_runs.diff')),
                          batch_model='identical to the response study copy (256 updates, diagnostics 0 / 256)'),
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


def fix_reference():
    path = ROOT / 'amendment-1.json'
    if path.exists():
        raise SystemExit('amendment-1 exists')
    reg = json.loads((ROOT / 'registration.json').read_text())
    if (ROOT / 'train/updates.jsonl').exists():
        raise SystemExit('training started; the reference can no longer be fixed')
    meta = json.loads(CD.REF.with_suffix('.json').read_text())
    cpu = json.loads((ROOT / 'reference/cpu-check.json').read_text())
    assert meta['sha256'] == sha(CD.REF) == cpu['reference_sha256']
    am = dict(amendment=1, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              before_training=True, change='none to code or conditions; fixes the reference arrays computed after registration',
              reference=dict(file=str(CD.REF), sha256=sha(CD.REF), meta_sha256=sha(CD.REF.with_suffix('.json')), backend=meta['backend'],
                             sequences=meta['sequences']),
              cpu_check=dict(file=str(ROOT / 'reference/cpu-check.json'), sha256=sha(ROOT / 'reference/cpu-check.json'), result=cpu['sequences']),
              code=reg['code'])
    path.write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    print('AMENDMENT', sha(path))


if __name__ == '__main__':
    {'register': register, 'fix-reference': fix_reference}[sys.argv[1]]()
