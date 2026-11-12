"""Write registration.json once, before any engine run of the pilot (no engine duel here)."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import pilot_data as D  # noqa: E402
import pilot_eval as EV  # noqa: E402
import artmage_teacher as T  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(hand):
    return tuple(sorted(hand))


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    import artmage_entry  # noqa: F401  (pins checked, engine library initialised, no duel)
    import pilot_model as M
    _, data_stats = D.build()
    manifest = json.loads(EV.OLD_MANIFEST.read_text())['decks']['Artmage']['cases']
    old_groups = {key(c['hand']): split for split, cases in manifest.items() for c in cases}
    demo_groups = {key(T.CONDITIONS[c]['hand']): c for c in T.CONDITIONS}
    variants = {}
    for name, spec in EV.VARIANTS.items():
        variants[name] = dict(hand=spec['hand'], seed=spec['seed'], cost_card=spec['cost_card'], plan=f"registered {spec['base']} plan",
                              filler_priority_for_medius_cost=spec['filler_priority'],
                              shares_group_with_old_split=old_groups.get(key(spec['hand'])), shares_group_with_demo=demo_groups.get(key(spec['hand'])))
        assert variants[name]['shares_group_with_old_split'] is None and variants[name]['shares_group_with_demo'] is None
    evaluation_hands = {c['case_id']: c['hand'] for split in ['validation', 'holdout'] for c in manifest[split] if c['case_id'] in EV.RT1_CASES}
    evaluation_hands.update({c: EV.NT1_CONDITIONS[c]['hand'] for c in EV.NT1_CONDITIONS})
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec='C:/Users/hatk8/Documents/Codex/2026-10-27/new-chat/outputs/artmage-first-learning-decision-2026-11-10.md',
        start_model=dict(base=str(ROOT / 'models/initial/base.flax_model'), base_sha256=sha(ROOT / 'models/initial/base.flax_model'),
                         head=str(ROOT / 'models/initial/chain-head.flax_model'), head_sha256=sha(ROOT / 'models/initial/chain-head.flax_model'),
                         source='experiments/opponent-article-training-20261108/run-20261108-145203/training/Artmage/step-0200'),
        training=dict(
            candidates=1, seed=10111001, rehearsal_seed=30111001, first_learner_seat=0,
            seed_check='10111001, 30111001, 1110003, 1110004: no occurrence in any json/jsonl/py/md/txt/log file under experiments, scripts, config, docs, evaluation before this registration',
            environment_decisions_cap=4096, counting='learner and frozen decisions are logged separately; the cap is on their total',
            collection='natural duels (normal dealing and draws; fresh env per episode; player 0 moves first), learner seat alternates per episode starting with seat 0; frozen side = start model; both sides sample from their own logits; common position adapter for both sides',
            window='closes at the first learner decision or episode end after >= 128 environment decisions, or at the cap',
            update='per window: one PPO epoch, 4 contiguous chunks (<= 32 learner decisions, stored rollout RNN state at chunk start); each chunk is one minibatch with 2 rehearsal sequences; one Adam step per minibatch',
            ppo=M.CONFIG, device='GPU (one GPU job; jax_default_matmul_precision=highest)',
            loss_exclusions='select_position (adapter-owned) steps of the learner are history only',
            rehearsal=dict(per_block='14 old / 1 C1 / 1 C2 per 16 sequences, slot order shuffled by rehearsal_seed; old slots uniform with replacement from the 12 old train sequences',
                           sequences_per_minibatch=2, lambda_=5.0, kl=0.0, loss='CE of the demonstrated label (single acceptable action), mean over the minibatch loss steps',
                           data=data_stats, files=dict(old=dict(path=str(D.OLD), sha256=D.OLD_SHA), C1=D.DEMO_SHA['C1'], C2=D.DEMO_SHA['C2'])),
            candidate='state after the last window; no checkpoint selection; not adopted',
            checks_after_save=['reload through the isolated entry with the saved hashes', 'logits on the C1/C2 histories: memory vs saved equal',
                               'neural argmax and position-adapter final actions checked separately', 'base and head both changed'],
            exposure_check='each episode records each player\'s own hand at its first decision (exact 5 cards when that is in turn 1, otherwise a superset after the draw); compared with the evaluation hands by multiset; overlaps are reported, never replaced'),
        evaluation=dict(
            cap=18, ledger=str(EV.LEDGER),
            order=['variant C1v, C2v (scripted, 2)', 'RT1 start g06, g07 (2)', 'NT1 start C1, C2, C1v, C2v (4)', 'training', 'RT1 candidate g06-g11 (6)', 'NT1 candidate C1, C2, C1v, C2v (4)'],
            rt1=dict(cases=EV.RT1_CASES, baseline='start-model passing set: g06/g07 from this pilot, g08-g11 reused from experiments/artmage-parallel-20261110/regression (identical to the original records)',
                     retention='candidate keeps every baseline-passing case; new passes on baseline failures allowed; never judged by counts',
                     stop='identical to check_other_positions (goal_reached / opponent_main_phase_passed / two_turns_passed / episode_finished, 320 decisions)'),
            nt1=dict(conditions={c: dict(hand=x['hand'], seed=x['seed'], reference=x['reference']) for c, x in EV.NT1_CONDITIONS.items()},
                     variants=variants, stop='first decision of turn 4 (end of own turn 3), game over from env.step returns, or 600 environment decisions (cap is not success)',
                     player1='common pass policy every turn; first legal option as a logged fallback when it cannot decide',
                     model='free, position adapter on; no scripts or goals in its inputs',
                     fixture_property='both players hold the same 5 cards; no normal draws in fixture duels (native draw count 0); turn 3 is observed without a draw',
                     records=['own turn 1 end / opponent turn 2 end / own turn 3 end boards with LP', 'every action with its legal alternatives', 'offered vs used vs resolved effects', 'divergence from the scripted route', 'legacy Assault goal as a diagnostic only']),
            evaluation_hands=evaluation_hands,
            variant_failure_rule='if a variant script fails, record it and revise the plan before training or viewing other results; no search for another hand'),
        not_done=['battles', 'adoption', 'B2B', 'server posts', 'common registry / Elfnotes changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=float) + '\n')
    print('REGISTERED', sha(path))


if __name__ == '__main__':
    main()
