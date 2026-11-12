"""registration.json (before any engine run) and amendment-1.json (control reuse decided offline after the comparison runs).

  up_register.py register
  up_register.py controls     (offline: which controls equal an existing comparison record; writes amendment-1.json)
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-unseen-response-probe-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import up_runs as U  # noqa: E402
import sa_model as SM  # noqa: E402
import artmage_entry as E  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
SCRIPTS = P / 'evaluation/opponent-article-basics-20261108/runtime/card-scripts'
CDB = P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb'
CV = P / 'experiments/artmage-response-coverage-20261111'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise SystemExit('registration exists; not rewritten')
    pins = {k: v for k, v in E.verify_pins().items()}
    m, paths = SM.read_manifest(U.CAND)
    cv_result = json.loads((CV / 'train/result.json').read_text())
    assert m['adapter']['sha256'] == cv_result['saved']['adapter_sha256']
    db = sqlite3.connect(CDB)
    deck = E.v.deck_cards('Artmage')
    reg = dict(
        status='registered_before_engine_runs', registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(copy='spec/artmage-unseen-response-probe-plan-2026-11-11.md', sha256=sha(ROOT / 'spec/artmage-unseen-response-probe-plan-2026-11-11.md')),
        diagnosis_only=True, training_runs=0, model_updates=0, pins=pins,
        policies=dict(
            BASE=dict(entry='artmage_entry.create_policy (2-file original entry)', base=str(U.BASE[0]), base_sha256=sha(U.BASE[0]), head=str(U.BASE[1]), head_sha256=sha(U.BASE[1])),
            CAND=dict(entry='sa_model.create_adapter_policy (3-file entry)', folder=str(U.CAND), manifest_sha256=sha(U.CAND / 'manifest.json'),
                      base_sha256=m['base']['sha256'], head_sha256=m['head']['sha256'], adapter_sha256=m['adapter']['sha256'],
                      source=str(CV / 'train/candidate'), byte_copies=True),
            frozen_pairs_equal=(sha(U.BASE[0]), sha(U.BASE[1])) == (m['base']['sha256'], m['head']['sha256']) == SM.FROZEN_SHA,
            scope='unchanged: the adapter acts on select_chain with the latest chain link player = opponent (own turn included); no card-id gate added'),
        hands=dict(H1=U.HANDS['H1'], H2=U.HANDS['H2'], note='the two existing BV1 hands (not a new-hand test)'),
        opponent_hand=dict(codes=U.OPPONENT_HAND, names=[E.v.names.get(c) for c in U.OPPONENT_HAND],
                           driver_in_opponent_deck=True, deck_counts=dict(main=len(deck['main']), extra=len(deck['extra']), gamma=deck['main'].count(U.GAMMA),
                                                                         driver=deck['main'].count(U.DRIVER), ash=deck['main'].count(U.ASH))),
        scripts={f'c{c}.lua': sha(SCRIPTS / f'c{c}.lua') for c in [U.ASH, U.GAMMA, U.G, U.MEDIUS]} | {'c49036338.lua': 'none (PSY-Frame Driver is a Normal Monster, type 17)'},
        cards_cdb_sha256=sha(CDB),
        script_reading=dict(gamma='SpecialSummonStep(Driver) + SpecialSummonStep(Gamma) + one SpecialSummonComplete, then NegateActivation and destroy',
                            g='during chain resolution each EVENT_SPSUMMON_SUCCESS registers one flag; at EVENT_CHAIN_SOLVED it draws the flag count; '
                              'the actual draw count is measured, not assumed', ash='negates an effect that searches / special summons / sends from the deck; no special summon'),
        own_side='the compared policy decides from the start of turn 1; nothing forced (no Graphrea branch control)',
        target='the opponent\'s first chance to respond to OUR first Medius① (code 97556336, description Stringid(id,0)) when that Medius① is chain link 1; '
               'identified by card / effect / activating player / public chain, never by a step number; if our first Medius① is not chain link 1, '
               'or the specified opponent effect is illegal, the condition is recorded as not established (no other hand is searched)',
        opponent='passes until the target; at the target activates Ash (ASH) or Gamma (GAMMA); for Gamma chooses Driver, face-up attack position '
                 '(else the first option), the first zone; no other optional effect afterwards; first legal option only if the pass policy cannot decide (logged)',
        stop='the first decision after the target chain fully resolved (chain size 0 after the opponent activation), or game over from env.step, or 600 decisions',
        conditions=U.COMPARISON,
        controls=dict(names=U.CONTROLS, rule='H1 only: base policy before the target; at OUR first response after the opponent activation exactly one '
                                               'substitution (G activation 23434538 / pass); illegal -> not established, the run stops; afterwards every '
                                               'optional own activation is declined (legal Cancel), forced choices are the base policy\'s; same stop rule',
                      reuse='a control is not run if an existing comparison record executes exactly the actions the control rule implies from the start '
                            'to the stop (pre-target path, the substitution, every later own decision, the same stop); checked offline before any control run; '
                            'reused controls are not counted as runs'),
        record=['own public board / hand / chain and the opponent card + effect at each gated screen', 'legal options, frozen logits, adapter delta, final logits, '
                'first choice and pass probability before / after', 'relative_delta(action) = delta(action) - delta(pass); new_margin = old_margin + relative_delta '
                '(cards with the same meaning separated by card / effect / controller / zone)', 'effects used, resolution, G use and measured draws, hand / deck / field, '
                'Medius / Gamma / Driver state', 'first point where BASE and CAND differ, adapter activations', 'NPZ, action list, replay check, the 3 hashes, the opponent script log'],
        engine=dict(cap=26, plan='comparison 8 x (record + replay) = 16; controls up to 4 x 2 = 8; total at most 24; 2 spare only for a harness fix',
                    failures_count=True, previous_unused_not_carried=True),
        not_done=['training', 'model / input / scope changes', 'adoption', 'normal battles', 'acceptance changes', 'B2B', 'production / registry changes',
                  'the known 8 conditions or the 13 retention cases'],
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, job_seconds=1800, nice=10, jobs=1),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
    )
    assert reg['policies']['frozen_pairs_equal']
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1, default=str) + '\n')
    print('REGISTERED', sha(path))


def implied_by_control(rec, base_rec, control):
    """Does the record execute exactly what the control rule implies? (pre-target identical to the base record; first own
    response = G / pass; later own: Cancel when a pass message offers it, otherwise the record must equal the base policy,
    which is only checkable when the record IS a base-policy record)."""
    tr = rec['trace']
    act_step = next((e['step'] for e in rec['opponent_events'] if e['what'].startswith('target:') and 'activated' in e['what']), None)
    if act_step is None:
        return False, 'target not established in the record'
    pre_ok = all(x['snapshot'] == y['snapshot'] and x['selected']['index'] == y['selected']['index'] for x, y in zip(tr[:act_step + 1], base_rec['trace'][:act_step + 1]))
    if not pre_ok:
        return False, 'pre-target path differs from the base record'
    first = True
    for r in tr[act_step + 1:]:
        s, a = r['snapshot'], r['selected']
        if s['to_play'] != 0:
            continue
        if first:
            first = False
            want = (a['act'] == 'Activate' and a['code'] == U.G) if control == 'G' else (a['act'] == 'Cancel' and s['message'] in U.PASS_MESSAGES)
            if not want:
                return False, f"first response is {a['act']} {a['code']}"
            continue
        has_cancel = s['message'] in U.PASS_MESSAGES and any(x['act'] == 'Cancel' for x in s['actions'])
        if has_cancel and a['act'] != 'Cancel':
            return False, f"later own optional activation at step {r['step']}"
        if not has_cancel and rec['policy_kind'] != 'BASE':
            return False, f"later forced choice at step {r['step']} made by a non-base policy"
    if first:
        return False, 'no own response after the opponent activation'
    return True, 'identical actions under the control rule; same stop rule'


def controls():
    path = ROOT / 'amendment-1.json'
    if path.exists():
        raise SystemExit('amendment-1 exists')
    if any((ROOT / 'runs' / n / 'record.json').exists() for n in U.CONTROLS):
        raise SystemExit('a control already ran')
    reg = json.loads((ROOT / 'registration.json').read_text())
    decisions, to_run = {}, []
    for n in U.CONTROLS:
        c = U.parse(n)
        base_rec = json.loads((ROOT / 'runs' / f"H1-{c['opp']}-BASE" / 'record.json').read_text())
        found = None
        for cand in [f"H1-{c['opp']}-BASE", f"H1-{c['opp']}-CAND"]:
            rec = json.loads((ROOT / 'runs' / cand / 'record.json').read_text())
            ok, why = implied_by_control(rec, base_rec, c['control'])
            decisions.setdefault(n, []).append(dict(record=cand, record_sha256=sha(ROOT / 'runs' / cand / 'record.json'), equal=ok, why=why))
            if ok and found is None:
                found = cand
        if found is None:
            to_run.append(n)
        decisions[n] = dict(checks=decisions[n], reused_record=found)
    am = dict(amendment=1, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              change='control reuse decided offline (no engine); conditions unchanged', controls=decisions, controls_to_run=to_run, code=reg['code'])
    path.write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    print('AMENDMENT', sha(path), json.dumps({n: d['reused_record'] for n, d in decisions.items()}), 'to run:', to_run)


if __name__ == '__main__':
    {'register': register, 'controls': controls}[sys.argv[1]]()
