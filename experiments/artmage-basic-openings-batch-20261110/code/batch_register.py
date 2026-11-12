"""Write registration.json once, before any engine run of the batch (no engine duel here)."""
import collections
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import batch_plans as BP  # noqa: E402
import batch_model as BM  # noqa: E402  (imports pilot_model → engine library initialised, no duel)
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
            known[BP.hand_key(c['hand'])] = f'old {split} {c["case_id"]}'
    for name in ['C1', 'C2']:
        known[BP.hand_key(T.CONDITIONS[name]['hand'])] = f'pilot demo {name} (train)'
    for name, hand in BP.PAST_EVALUATION_HANDS.items():
        known[BP.hand_key(hand)] = f'pilot evaluation {name}'
    split = {}
    seen = {}
    for name, spec in list(BP.TRAIN.items()) + list(BP.VALIDATION.items()):
        k = BP.hand_key(spec['hand'])
        split[name] = dict(group=list(k), overlaps=known.get(k), overlaps_new=seen.get(k))
        assert known.get(k) is None and seen.get(k) is None, (name, known.get(k), seen.get(k))
        seen[k] = name
    plans = {c: (x['plan'] if isinstance(x['plan'], str) else json.loads(json.dumps(x['plan']))) for c, x in BP.CONDITIONS.items()}
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec='C:/Users/hatk8/Documents/Codex/2026-10-27/new-chat/outputs/artmage-basic-openings-batch-2026-11-10.md',
        start_model=dict(base_sha256=sha(ROOT / 'models/initial/base.flax_model'), head_sha256=sha(ROOT / 'models/initial/chain-head.flax_model'),
                         source='step-0200 (not the unadopted pilot candidate)'),
        conditions={c: dict(label=x['label'], reference=x['reference'], train_reuse=x['train_reuse']) for c, x in BP.CONDITIONS.items()},
        plans=plans,
        train_hands={n: dict(condition=x['condition'], hand=x['hand'], seed=x['seed']) for n, x in BP.TRAIN.items()},
        validation_hands={n: dict(condition=x['condition'], hand=x['hand'], seed=x['seed']) for n, x in BP.VALIDATION.items()},
        split_check=dict(rule='complete-hand multiset; permutations are the same group; compared with old train/validation/holdout, pilot C1/C2, pilot evaluation C1v/C2v and each other',
                         result=split, passed=True),
        out_of_scope=['メディウス＋モンスターコスト (heading only)', 'Fainmelt alone, Nerved and other supplementary lines', 'hand-trap responses'],
        engine=dict(cap=48, planned=47, ledger='eval-runs.jsonl',
                    plan=dict(train_record=10, train_replay=10, validation_script=7, free_start=7, free_candidate=7, rt1_candidate=6),
                    reserve='1 run, only for confirming a failure, with its purpose and code version written into an amendment before use; never for choosing hands or candidates',
                    order=['record + replay the 10 train demos', 'validate the 7 validation hands (script)', 'fix data/manifest.json',
                           'offline diagnostics (start, pilot candidate)', 'free runs of the start model (7)', 'training',
                           'free runs of the candidate (7)', 'RT1 candidate g06-g11 (6)'],
                    demo_stop='first decision of turn 3', free_stop='first decision of turn 4 (end of own turn 3), game over from env.step returns, or 600 decisions',
                    player1='common pass policy every turn; first legal option as a logged fallback', fixture='same five cards for both players; no normal draws (native draw count 0)',
                    rt1_start='reuse: g06/g07 from the pilot start runs and g08-g11 from the artmage-parallel regression (same start weights, engine, adapter, cases)'),
        demo_rules=['A demo is training data only if its script completed (opponent-turn entries may stay unexecuted and are reported) and its replay matched every step, the end state and the observations.',
                    'If a condition cannot be completed, the manifest is fixed with the gap before any model result; no other hand is searched for.',
                    'BatchTeacher defaults: identical copies (same code and zone) resolve to the first; Medius② cost = drawn non-engine monster, else FILLER_PRIORITY.'],
        training=dict(updates=BM.CONFIG['updates'], batch_sequences=BM.CONFIG['batch_sequences'], learning_rate_base=BM.CONFIG['learning_rate'],
                      learning_rate_head=BM.CONFIG['learning_rate'], adam_eps=BM.CONFIG['adam_eps'], max_grad_norm=BM.CONFIG['max_grad_norm'],
                      ppo=False, value_loss=False, entropy_loss=False, kl=0.0, environment_decisions=0, seed_sampling=30121001, sampling_seed=30121001,
                      seed_note='10121001 reserved as the training seed label; the run is deterministic given the sampling seed (no dropout, no sampling of actions)',
                      sampling='each of the 2 sequence slots: a group uniformly among the available groups, then a train sequence uniformly within the group',
                      loss='mean over the batch sequences of (mean CE of the demonstrated label over that sequence\'s loss steps); differs from the pilot (pooled over steps)',
                      masks='context = every player-0 decision; loss = own-turn non-position + scripted opponent-turn non-position; position steps are adapter-owned; select_place stays in the loss',
                      diagnostics_at=BM.CONFIG['diagnostics_at'], selection='none: only the state after 1,024 updates is evaluated',
                      stop_on='non-finite loss, gradient or weights; STOP=now', device='GPU, matmul precision highest',
                      checks_after_save=['reload through the isolated entry with saved hashes', 'logits on every train history equal', 'position-adapter final actions equal', 'saved weights finite', 'base and head both changed']),
        not_done=['PPO / self-play', 'battles', 'adoption', 'B2B', 'server posts', 'common registry / Elfnotes changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, gpu_jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=float) + '\n')
    print('REGISTERED', sha(path))


if __name__ == '__main__':
    main()
