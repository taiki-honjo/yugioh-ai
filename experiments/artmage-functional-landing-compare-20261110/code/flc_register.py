"""Write registration.json once, before any engine run of the functional comparison (no engine duel here)."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402  (pins checked; engine library initialised, no duel)
import flc_runs as F  # noqa: E402
import hb_plans as HP  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    prefixes = {}
    for cost, hand_name in F.HANDS.items():
        prefix, source = F.prefix_of(hand_name)
        prefixes[hand_name] = dict(source, own_decisions_before_branch=len(prefix), decisions=[(p['message'], list(p['meaning'])) for p in prefix])
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-functional-landing-compare-plan-2026-11-10.md', sha256=sha(ROOT / 'spec/artmage-functional-landing-compare-plan-2026-11-10.md')),
        model=dict(base_sha256=sha(F.MODEL[0]), head_sha256=sha(F.MODEL[1]), source='experiments/artmage-hand-branch-20261110/train/candidate (unadopted)'),
        pins={k: v for k, v in E.verify_pins().items()},
        own_hands={n: dict(hand=HP.HANDS[n]['hand'], seed=HP.HANDS[n]['seed']) for n in F.HANDS.values()},
        opponent_hand=F.OPPONENT_HAND,
        fixture='article_witness.fixture order with a separate hand per player; player 0 deck order equals the earlier BV1 runs (same seed, player 0 first)',
        conditions=F.conditions(),
        branch=dict(prompt='turn-1 select_card while Graphrea is on the public chain, offering 改変 and 襲撃 and no trap', choices=dict(VARNISH=F.VARNISH, ASSAULT=F.ASSAULT),
                    prefix='own turn-1 decisions of the candidate free run on the same hand, matched BY MEANING (act, code, effect, zone spec, place, position, phase, finish); '
                           'the model reads every observation once (its own choice is logged, the recorded one is executed); after the branch the model plays freely',
                    prefixes=prefixes),
        opponent=dict(OM='own main phase: normal summon Medius, activate Medius①, choose 獄神獣, special summon (1152); everything else pass',
                      OI='own main phase with an empty field: activate 無限泡影 from the hand; target priority 乙女 > ディアクトロス > ファインメルト > 獄神使 > other, same name lowest zone; everything else pass',
                      hand_traps='never used', turn3='pass', fallback='first legal option only when the pass policy cannot decide (logged)'),
        stop='first decision of turn 4 (end of own turn 3), game over from env.step returns, or 600 decisions',
        engine=dict(cap=16, plan='8 records + 8 replays', failures_count=True),
        not_done=['training', 'adoption', 'battles', 'RT1 or other re-evaluation', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, device='CPU'),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


if __name__ == '__main__':
    main()
