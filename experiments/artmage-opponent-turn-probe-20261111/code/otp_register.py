"""Write registration.json once, before any engine run of the opponent-turn probe (no engine duel here)."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import otp_runs as O  # noqa: E402
import artmage_entry as E  # noqa: E402  (pins checked; engine library initialised, no duel)

P = Path('/home/hatk8/projects/yugioh-ai')
SCRIPTS = P / 'evaluation/opponent-article-basics-20261108/runtime/card-scripts'
CDB = P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    assert Path(E.__file__).resolve().parent == ROOT / 'code'
    reuse = json.loads((ROOT / 'reuse.json').read_text())
    db = sqlite3.connect(CDB)
    texts = {c: db.execute('select name, str1, str2 from texts where id=?', (c,)).fetchone() for c in [O.FINEMELT, O.DIAKTOROS, O.F.MEDIUS]}
    conditions = {}
    for site in O.SITES:
        for cond in ['D0R0', 'D0R1', 'D1R0', 'D1R1']:
            name = f'{site}-{cond}'
            r = reuse['sites'][site]['conditions'][cond]
            conditions[name] = dict(old_condition=O.SITES[site], draw_phase='D1 (decline all optional own activations)' if cond[1] == '1' else 'model',
                                    first_medius_response='R1 substitution' if cond[3] == '1' else 'model',
                                    plan='new record + replay' if name in O.NEW else 'reuse the old record (offline decision-sequence identity, reuse.json)',
                                    offline_first_divergence=r['first_divergence'])
            assert (name in O.NEW) == (not r['reuse_old_record']), name
    hands = {site: dict(hand=O.F.HP.HANDS[O.F.parse(old)['hand_name']]['hand'], seed=O.F.HP.HANDS[O.F.parse(old)['hand_name']]['seed'],
                        hand_name=O.F.parse(old)['hand_name'], graphrea_set=O.F.parse(old)['branch']) for site, old in O.SITES.items()}
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-opponent-turn-probe-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-opponent-turn-probe-plan-2026-11-11.md')),
        model=dict(base_sha256=sha(O.MODEL[0]), head_sha256=sha(O.MODEL[1]), source='experiments/artmage-hand-branch-20261110/train/candidate (unadopted), read-only copy'),
        pins={k: v for k, v in E.verify_pins().items()},
        card_scripts={f'c{c}.lua': sha(SCRIPTS / f'c{c}.lua') for c in [O.FINEMELT, O.DIAKTOROS, O.F.MEDIUS, O.F.INFINITE]},
        cards_cdb_sha256=sha(CDB),
        effect_numbers=dict(
            finemelt_3=dict(code=O.FINEMELT, effect=10011, script='e3 QUICK_O CATEGORY_DISABLE, SetDescription(aux.Stringid(id,1))', cdb_str2=texts[O.FINEMELT][2]),
            diaktoros_2=dict(code=O.DIAKTOROS, effect=10011, script='e2 QUICK_O EVENT_CHAINING CATEGORY_NEGATE+DESTROY, SetDescription(aux.Stringid(id,1))', cdb_str2=texts[O.DIAKTOROS][2]),
            medius_1=dict(code=O.F.MEDIUS, chain_description=O.MEDIUS_EFFECT1, script='e1 TRIGGER_O EVENT_SUMMON_SUCCESS, SetDescription(aux.Stringid(id,0))', cdb_str1=texts[O.F.MEDIUS][1])),
        sites=dict(S1=dict(hands['S1'], old_condition=O.SITES['S1']), S2=dict(hands['S2'], old_condition=O.SITES['S2'])),
        opponent_hand=O.F.OPPONENT_HAND,
        opponent='same as the comparison O-M: own main phase: normal summon Medius, activate Medius①, choose 獄神獣, special summon (1152); everything else pass; '
                 'hand traps never used; turn 3 pass; fallback = first legal option only when the pass policy cannot decide (logged)',
        turn1='prefix replay by meaning + Graphrea set target as in the old condition; the rest of turn 1 is the model. Turn-1 rows and model inputs are '
              'compared with the old record of the same site after the run (offline)',
        rules=dict(
            D1='turn 2 (opponent), DRAW phase only, own select_chain / select_effectyn / select_yesno: execute the legal Cancel. A prompt without a '
               'legal Cancel (or any other message) is left to the model and logged as forced. Standby phase on: model. This suppresses ALL optional '
               'draw-phase activations together; it is not a per-effect (乙女② alone / 覚醒① alone) comparison',
            R1='the FIRST own select_chain in turn 2, opponent main phase 1, whose public chain is exactly one link = player 1 Medius① '
               f'(code {O.F.MEDIUS}, description {O.MEDIUS_EFFECT1}): activate once ファインメルト③ ({O.FINEMELT}, 10011), else ディアクトロス② '
               f'({O.DIAKTOROS}, 10011); same card: zone order (spec letter, number, index). Neither legal: recorded "not established", the model '
               'choice is executed, no other effect is substituted. Every later decision: model',
            model='the model reads every own observation exactly once, also when overridden; its own choice is logged with every override'),
        conditions=conditions,
        reuse=dict(file='reuse.json', sha256=sha(ROOT / 'reuse.json'),
                   summary='S1-D0R0, S2-D0R0: the old records themselves; S2-D0R1: the rule executes the action the model chose at old step 169 '
                           '(ファインメルト③) and nothing else, so the decision sequence is identical; these are not counted as new runs'),
        checkpoints=['own_turn1_end', 'opp_draw_phase_end (first decision snapshot of turn 2 after the draw phase)',
                     'medius_response_window (first R1-type window, candidates listed; recorded in R0 conditions too)',
                     'after_medius_chain (first snapshot with an empty chain after player 1 Medius was on the chain)', 'opp_turn2_end', 'own_turn3_end or game over'],
        stop='first decision of turn 4 (end of own turn 3), game over from env.step returns, or 600 decisions',
        engine=dict(cap=12, plan='5 new conditions x (record + replay) = 10', spare='2 episodes only for redoing a record+replay after a harness fix '
                    'that does not change the decision rules (fixed by an amendment); not for new hands/seeds/conditions/models', failures_count=True),
        not_done=['training', 'model update', 'adoption', 'normal battles', 'reward / adoption-condition changes', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, device='CPU', jobs=1),
        play_diff=dict(file='play-vs-comparison.diff', sha256=sha(ROOT / 'play-vs-comparison.diff')),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


if __name__ == '__main__':
    main()
