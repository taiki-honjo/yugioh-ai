"""Write registration.json once, before any engine run of the hand-branch study (no engine duel here)."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import batch_model as BM  # noqa: E402  (engine library initialised, no duel)
import batch_plans as BP  # noqa: E402
import hb_plans as HP  # noqa: E402
import artmage_teacher as T  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    manifest = json.loads((P / 'experiments/opponent-article-training-20261108/run-20261108-145203/manifest.json').read_text())['decks']['Artmage']['cases']
    known = {}
    for split, cases in manifest.items():
        for c in cases:
            known[HP.hand_key(c['hand'])] = f"old {split} {c['case_id']}"
    for n in ['C1', 'C2']:
        known[HP.hand_key(T.CONDITIONS[n]['hand'])] = f'pilot demo {n}'
    for n, h in BP.PAST_EVALUATION_HANDS.items():
        known[HP.hand_key(h)] = f'pilot evaluation {n}'
    for n, x in list(BP.TRAIN.items()) + list(BP.VALIDATION.items()):
        known[HP.hand_key(x['hand'])] = f'basic batch {n}'
    split, seen = {}, {}
    for n, x in HP.HANDS.items():
        k = HP.hand_key(x['hand'])
        split[n] = dict(overlaps=known.get(k), overlaps_new=seen.get(k))
        assert known.get(k) is None and seen.get(k) is None, (n, known.get(k), seen.get(k))
        seen[k] = n
    for p in HP.PAIRS:
        a_, b_ = list(p['hands'].values())
        assert sorted(a_[:-1]) == sorted(b_[:-1]) and a_[-1] != b_[-1]
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-hand-branch-plan-2026-11-10.md', sha256=sha(ROOT / 'spec/artmage-hand-branch-plan-2026-11-10.md'),
                  report='a separate *-report.md; the spec copy is read-only'),
        start_model=dict(base_sha256=sha(ROOT / 'models/start/base.flax_model'), head_sha256=sha(ROOT / 'models/start/chain-head.flax_model'),
                         source='experiments/artmage-basic-openings-batch-20261110/train/candidate (unadopted research candidate)'),
        substitution=dict(names=list(HP.SUBSTITUTE), meaning=HP.SUBSTITUTE, model='start',
                          rule='replay the start model on the same fixture; earlier choices must equal the saved free run; at the first choice that differs '
                               'from the validation script the model still reads the observation, then the script\'s choice is executed; afterwards the model plays freely '
                               'to the end of own turn 3. Local comparison only; these hands never move into train.'),
        pairs={p['pair']: dict(group=p['group'], split=p['split'], seed=p['seed'], hands={v: h for v, h in p['hands'].items()},
                               differing=p['differing'], routes={v: HP.ROUTE_LABEL[v] for v in p['hands']}) for p in HP.PAIRS},
        split_check=dict(rule='complete-hand multiset vs old train/validation/holdout, pilot C1/C2 and C1v/C2v, basic batch train/validation, and each other',
                         result=split, passed=True),
        plans={k: json.loads(json.dumps(v)) for k, v in HP.PLANS.items()},
        teacher=['BatchTeacher defaults (identical copies; Medius② cost = drawn non-engine monster else FILLER_PRIORITY; 学都 summon method)',
                 'BranchTeacher: a cost-discard selection of only spell/trap costs in hand not covered by the plan follows COST_ORDER (無限泡影, 三戦の才, 聖王の粉砕)',
                 'no count rule or count feature is given to the model'],
        audit_at_branch='hand, payable costs, used effects and remaining resources at each branch are read later from the saved snapshots (audit only)',
        engine=dict(cap=72, plan=dict(substitute=3, train_record=16, train_replay=16, validation_script=8, free_start=8, free_candidate=8, retain=13),
                    order=['substitute F1-V, F3-V, F4-V', 'record + replay the 16 train hands', 'validate the 8 validation hands', 'fix data/manifest.json',
                           'offline diagnostics', 'free runs of the start model on the 8 validation hands', 'training',
                           'free runs of the candidate on the 8 validation hands', 'retention 13 (RT1 g06-g11 + F0-V..F6-V)'],
                    failures_count=True, no_replay_budget_for_free_runs=True,
                    retention_baseline='saved records of the start model (basic batch: eval/rt1/candidate and eval/free/candidate), same weights/code/adapter/hands/seeds'),
        data_rules=['a pair is used only if both hands completed their script and both replays matched', 'existing 24 sequences reused unchanged',
                    'validation pairs are JSON only and never listed in the manifest'],
        training=dict(updates=BM.CONFIG['updates'], batch_sequences=2, learning_rate_base=BM.CONFIG['learning_rate'], learning_rate_head=BM.CONFIG['learning_rate'],
                      adam_eps=BM.CONFIG['adam_eps'], max_grad_norm=BM.CONFIG['max_grad_norm'], ppo=False, value_loss=False, entropy_loss=False, kl=0.0,
                      sampling_seed=30131001, seed_label=10131001,
                      sampling='blocks of 4 minibatches: 2 existing (each slot: one of the 7 basic groups uniformly, then a sequence uniformly) + 2 paired '
                               '(group A/B uniformly, then a complete pair uniformly; the two hands of the pair form the minibatch); slot order shuffled per block',
                      loss='mean over the batch sequences of the per-sequence mean CE over loss steps (same as the basic batch)',
                      padding='batch length = longest sequence in the data; padding repeats the last step with zero loss',
                      diagnostics_at=BM.CONFIG['diagnostics_at'], selection='none; only the state after 1,024 updates is evaluated',
                      stop_on='non-finite loss, gradient or weights; STOP=now', device='GPU, matmul precision highest'),
        not_done=['PPO', 'battles', 'adoption', 'B2B', 'common registry / Elfnotes changes', 'count-based rules'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, gpu_jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=float) + '\n')
    print('REGISTERED', sha(path))


if __name__ == '__main__':
    main()
