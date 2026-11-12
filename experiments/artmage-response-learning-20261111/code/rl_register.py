"""Fix the data manifest and write registration.json once, before training and before any engine run (no duel here)."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rl_runs as RR  # noqa: E402
import rl_data as RD  # noqa: E402
import batch_model as BM  # noqa: E402
import artmage_entry as E  # noqa: E402  (pins checked; engine library initialised, no duel)

P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
BATCH = P / 'experiments/artmage-basic-openings-batch-20261110'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    assert Path(E.__file__).resolve().parent == ROOT / 'code'
    manifest = RD.build_manifest() if not RD.MANIFEST.exists() else json.loads(RD.MANIFEST.read_text())
    pins = {k: v for k, v in E.verify_pins().items()}
    hb_code = json.loads((HB / 'amendment-1.json').read_text())['code']
    flc_pins = json.loads((P / 'experiments/artmage-functional-landing-compare-20261110/registration.json').read_text())['pins']
    pins_same = dict(artmage_pins_py_equal_to_hand_branch=sha(ROOT / 'code/artmage_pins.py') == hb_code['artmage_pins.py'],
                     artmage_entry_py_equal_to_hand_branch=sha(ROOT / 'code/artmage_entry.py') == hb_code['artmage_entry.py'],
                     verified_now=True, equal_to_comparison_registration=pins == flc_pins)
    pins_ok = all(pins_same.values())
    hb_ledger = [json.loads(l) for l in (HB / 'eval-runs.jsonl').read_text().splitlines()]
    retain_status = {}
    for i, e in enumerate(hb_ledger):
        if e.get('event') == 'start' and e['purpose'].startswith('retain '):
            retain_status[e['case_id']] = dict(run=e['run'], status=hb_ledger[i + 1].get('status'))
    baseline = {}
    for n in RR.C_NAMES:
        f = HB / 'eval/retain/candidate' / f'{n}.json'
        rec = json.loads(f.read_text())
        baseline[n] = dict(file=str(f), sha256=sha(f), model_sha256=rec['model_sha256'], engine_run=retain_status.get(n),
                           criterion='RT1: passed (goal reached in the opponent turn)' if n.startswith('Artmage-g')
                           else f'F-V: all monsters of the validation script turn-1 end present at own turn-1 end ({BATCH / "validation" / (n + ".json")})',
                           script_sha256=None if n.startswith('Artmage-g') else sha(BATCH / 'validation' / f'{n}.json'))
    reg = dict(
        status='registered_before_training_and_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-response-learning-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-response-learning-plan-2026-11-11.md')),
        start_model=dict(base=str(ROOT / 'models/start/base.flax_model'), head=str(ROOT / 'models/start/chain-head.flax_model'),
                         base_sha256=sha(ROOT / 'models/start/base.flax_model'), head_sha256=sha(ROOT / 'models/start/chain-head.flax_model'),
                         source='experiments/artmage-hand-branch-20261110/train/candidate (unadopted)'),
        pins=pins, pins_identity=pins_same,
        data=dict(manifest=str(RD.MANIFEST), sha256=sha(RD.MANIFEST), old_manifest_sha256=manifest['old']['sha256'],
                  responses={e['name']: dict(target_step=e['target_step'], target_index=e['target_index'], context_steps=e['context_steps'],
                                             label=e['label'], record_sha256=e['record_sha256'], observations_sha256=e['observations_sha256'])
                             for e in manifest['responses']},
                  promotion=manifest['promotion']),
        training=dict(sampling_seed=11111301, updates=BM.CONFIG['updates'], batch_sequences=2, learning_rate=BM.CONFIG['learning_rate'],
                      adam_eps=BM.CONFIG['adam_eps'], max_grad_norm=BM.CONFIG['max_grad_norm'], optimizer='fresh Adam state; base + chain head',
                      loss='per-sequence mean CE over its loss steps, averaged over the 2 sequences; coefficient 1',
                      blocks='8 minibatches: 3 old GroupSampler (24 basic sequences, its own rules, seed + 1) + 3 old pair (16 paired sequences) + '
                             '2 response (S1 + S2 together); slot order shuffled per block by the seed',
                      planned=dict(old_group_minibatches=96, old_pair_minibatches=96, response_minibatches=64, response_s1=64, response_s2=64,
                                   sequence_presentations=dict(old=384, new=128)),
                      not_used=['PPO', 'value loss', 'entropy', 'KL'], diagnostics_at=BM.CONFIG['diagnostics_at'],
                      no=['intermediate checkpoint selection', 'automatic extension', 'other seeds / candidates', 'changing ratios or lr after seeing logs'],
                      device='GPU (one job), matmul precision highest', position_choices='not a direct loss target (adapter-owned)',
                      after='save, reload through the isolated entry, logits + post-adapter actions on 42 sequences equal to memory, weights finite'),
        evaluation=dict(
            baseline_model='start model b9c7751f / 77191d5d only (no older model mixed in)',
            A=dict(sites={s: dict(old_record=str(RR.FLC / 'runs' / o / 'record.json'), old_record_sha256=sha(RR.FLC / 'runs' / o / 'record.json'), window_step=w)
                          for s, (o, w) in RR.A_SITES.items()},
                   rule='replay the old D0R0 own decisions BY MEANING (full snapshot compared at every step) up to the first opponent-Medius① window; the '
                        'candidate reads every observation once; from the window the candidate decides freely (no D1 / R1) to own turn-3 end',
                   runs='2 records + 2 replays'),
            B=dict(conditions=RR.B_NAMES, rule='flc_runs.play (frozen 8af7cbae) with flc_runs.MODEL rebound to the candidate: prefix replay + Graphrea '
                                               'branch control as in the comparison; turn 1 rest, opponent turn and turn 3 are the candidate',
                   old_records={n: sha(RR.FLC / 'runs' / n / 'record.json') for n in RR.B_NAMES}, runs='4 records + 4 replays',
                   note='known diagnostic hands (the two trained hands and their other branch), not a final test on unseen hands'),
            C=dict(cases=RR.C_NAMES, rule='hb_runs.rt1 / hb_runs.free_run (frozen 88e69412) with hb_runs.ROOT rebound to this study (candidate + output)',
                   baseline_reuse=dict(records=baseline, evaluation_code_sha256=sha(ROOT / 'code/hb_runs.py'),
                                       hand_branch_registered_hb_runs=json.loads((HB / 'amendment-1.json').read_text())['code']['hb_runs.py'],
                                       rt1_cases_manifest_sha256=sha(RR.HR.OLD_MANIFEST), batch_plans_sha256=sha(ROOT / 'code/batch_plans.py'),
                                       pins_identity=pins_same),
                   comparison='sets: start-model successes lost / new successes; identical action sequences are not required; turn-3 attacks, '
                              'resources and LP recorded for reference only (no new acceptance condition)', runs='13 candidate runs')),
        engine=dict(cap=28, plan='A 4 + B 8 + C 13 = 25', spare='3 only for re-running after a harness fix that keeps the conditions; not for '
                    'preparation duels, condition search or extra hands/seeds', failures_count=True),
        not_done=['PPO', 'normal battles', 'adoption', 'acceptance / reward changes', 'B2B', 'production / registry changes', 'keep-vs-reshape re-test'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        code_changes=dict(batch_model=dict(file='code-changes-batch_model.diff', sha256=sha(ROOT / 'code-changes-batch_model.diff'))),
    )
    assert pins_ok, pins_same
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path), 'manifest', sha(RD.MANIFEST))


if __name__ == '__main__':
    main()
