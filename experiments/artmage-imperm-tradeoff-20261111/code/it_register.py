"""registration.json, written once before any engine run (no duel here). Includes the offline reuse check for A-I-N.

  it_register.py register
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import it_runs as IT  # noqa: E402
import artmage_entry as E  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
CV = P / 'experiments/artmage-response-coverage-20261111'
SCRIPTS = P / 'evaluation/opponent-article-basics-20261108/runtime/card-scripts'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_a_i_n():
    """The coverage record of BV1-1cost-VARNISH-OI against the A-I-N rule, cut at the first turn-3 decision."""
    cov = CV / 'runs/B-BV1-1cost-VARNISH-OI'
    rec, rep = json.loads((cov / 'record.json').read_text()), json.loads((cov / 'replay.json').read_text())
    old = json.loads((IT.FLC / 'runs/BV1-1cost-VARNISH-OI/record.json').read_text())
    replay_ok = rep['mismatch'] is None and rep['failure'] is None and rep['observations_match'] and rep['checkpoints_equal']
    opp, own = IT.Opponent('I', E.v), IT.Own(IT.parse('A-I-N'))  # the same pass policy the runner uses
    window = None
    for i, r in enumerate(rec['trace']):
        s = dict(r['snapshot'], _step=r['step'])
        if s['turn_count'] >= 3:
            return dict(record=str(cov), record_sha256=sha(cov / 'record.json'), replay_ok=replay_ok, window_step=window, cut_at_step=r['step'],
                        equal=replay_ok and window is not None, why='identical to the A-I-N rule up to the first turn-3 decision')
        if s['to_play'] == 1:
            o, _ = opp(s)
            if o != r['selected']['index']:
                return dict(equal=False, why=f"opponent script differs at step {r['step']}")
            if window is None and old['trace'][i]['snapshot'] != r['snapshot']:
                return dict(equal=False, why=f"pre-window path differs from the start-model record at step {r['step']}")
            continue
        if window is None and not IT.is_imperm_window(s):
            if r['snapshot'] != old['trace'][i]['snapshot'] or r['selected']['index'] != old['trace'][i]['selected']['index']:
                return dict(equal=False, why=f"pre-window path differs from the start-model record at step {r['step']}")
            continue
        if window is None:
            window = r['step']
        idx, why = own(s)
        if idx != r['selected']['index']:
            return dict(equal=False, why=f"own rule ({why}) differs at step {r['step']}")
    return dict(equal=False, why='turn 3 not reached in the record')


def target_history():
    """Our own check of the Codex audit: the targeted Maiden (zone 0) activated its ② at step 119 of the same turn."""
    old = json.loads((IT.FLC / 'runs/BV1-1cost-VARNISH-OI/record.json').read_text())
    rows = {r['step']: r for r in old['trace']}
    act = rows[119]['selected']
    tgt = rows[148]['selected']
    cont = all(any(c['controller'] == 0 and c['location'] == 4 and c['sequence'] == 0 and c['code'] == IT.MAIDEN for c in rows[st]['snapshot']['cards'])
               for st in range(119, 149) if st in rows)
    return dict(step119=dict(act=act['act'], code=act['code'], effect=act.get('effect'), spec=act.get('spec'), turn=rows[119]['snapshot']['turn_count']),
                step148_target=dict(code=tgt['code'], spec=tgt.get('spec')), zone0_maiden_present_119_to_148=cont,
                maiden_script=dict(file='c74631897.lua', sha256=sha(SCRIPTS / 'c74631897.lua'),
                                   reading='e5 = ② (QUICK_O, EVENT_FREE_CHAIN, Stringid(id,1)) with SetCountLimit(1)'),
                codex_audit=dict(file='spec/artmage-imperm-target-history-audit-2026-11-11.json', sha256=sha(ROOT / 'spec/artmage-imperm-target-history-audit-2026-11-11.json')))


def main():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    pins = {k: v for k, v in E.verify_pins().items()}
    reuse = reuse_a_i_n()
    th = target_history()
    assert th['step119']['code'] == IT.MAIDEN and th['step119']['effect'] == 10011 and th['step119']['spec'] == 'm1' and th['step148_target']['spec'] == 'om1'
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-imperm-tradeoff-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-imperm-tradeoff-plan-2026-11-11.md')),
        pins=pins, model='none: scripted functional test; no training, no free model decisions (the current adapter c110f964... is untouched)',
        target_history=th,
        scripts={f'c{c}.lua': sha(SCRIPTS / f'c{c}.lua') for c in [IT.MAIDEN, IT.DIAKTOROS, IT.INFINITE, IT.MEDIUS, IT.FINEMELT, IT.LITERA]},
        fixture=dict(hands={k: IT.F.HP.HANDS[v['hand']] for k, v in IT.START.items()}, opponent_hand=IT.F.OPPONENT_HAND, seed=1112103,
                     note='the functional comparison fixture (no normal draws; effect draws occur)'),
        starts=dict(A='record BV1-1cost-VARNISH-OI replayed by meaning with snapshot equality (opponent script re-run and checked) up to OUR first response to '
                      '無限泡影 (target: zone-0 Maiden which used its ② at step 119); Diaktoros② unused at that point',
                    B='record BV1-2cost-ASSAULT-OI replayed for turn 1; from turn 2 every optional own activation declined in every phase until 無限泡影; '
                      'target: the turn-1 Maiden (the only Maiden; same priority rule, checked legal)'),
        conditions={'A-I-N': 'Diaktoros② on Imperm; opponent then does nothing', 'A-I-P': 'pass on Imperm; opponent then does nothing',
                    'A-IM-N': 'Diaktoros② on Imperm; then opponent normal summons Medius + ① (獄神獣 SS if it resolves); at our Medius① window Diaktoros② if legal else pass',
                    'A-IM-P': 'pass on Imperm; then the same Medius sequence and rule',
                    'B-N': 'Diaktoros② on Imperm; then the original Maiden② once (EX Maiden; Finemelt + Litera)', 'B-P': 'pass on Imperm; then the same Maiden② attempt'},
        own_rule='only the named actions; every other optional own activation declined (legal Cancel) for the whole comparison; forced prompts: select_place first, '
                 'select_position face-up attack else first, select_unselect_card finish when offered, otherwise first option (logged); one-candidate auto prompts not skipped',
        opponent_rule='passes except: main phase 1 of turn 2 with an empty field, 無限泡影 from the hand (target priority 乙女 > ディアクトロス > ファインメルト > 獄神使 '
                      '> other, lowest zone); IM only: after the Imperm chain resolved, normal summon Medius, activate ①, choose 獄神獣, special summon; no attack',
        stop=dict(A='the first decision of turn 3', B='the first decision after the Maiden② chain resolved (or turn 3 if ② never offered)', cap='600 decisions or game over'),
        not_a_label='whether Medius follows is the opponent\'s unpublished plan at the Imperm window; the A-I vs A-IM difference measures the opportunity cost, '
                    'it is not a label for the Imperm window',
        reuse=dict(rule='an existing record is reused only if, over the cut range, the fixture, the rules, every executed action and the end point are the same',
                   A_I_N=reuse),
        to_run=[n for n in IT.CONDITIONS if not (n == 'A-I-N' and reuse['equal'])],
        engine=dict(cap=14, plan='6 conditions x (record + replay) = 12, minus reuse; 2 spare for a harness fix only', failures_count=True),
        record_items=['target Maiden status column (disabled or forbidden) before / after Imperm and whether its ② is offered', 'Diaktoros② use and its legality at Medius①',
                      'Medius effect resolution, 獄神獣 arrival (not confused with destruction)', 'own / opponent field, hand, GY, banished, LP; consumed once-per-turn uses',
                      'B: ② offered / activated / fusion selection shown / materials moved / new fusion monster (classified separately)'],
        not_done=['training', 'model decisions', 'adoption', 'battles', 'input / structure changes', 'B2B', 'production / registry changes'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path), json.dumps(reuse, ensure_ascii=False), 'to_run', reg['to_run'], json.dumps(th, ensure_ascii=False)[:400])


if __name__ == '__main__':
    {'register': main}[sys.argv[1]]()
