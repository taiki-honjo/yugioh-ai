"""Offline summary of the basic-opening batch (no engine): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import batch_plans as BP  # noqa: E402

_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
RT1_BASE = {'Artmage-g06-p0': P / 'experiments/artmage-learning-pilot-20261110/eval/rt1/start/Artmage-g06-p0.json',
            'Artmage-g07-p0': P / 'experiments/artmage-learning-pilot-20261110/eval/rt1/start/Artmage-g07-p0.json',
            **{f'Artmage-g{i:02d}-p0': P / f'experiments/artmage-parallel-20261110/regression/Artmage-g{i:02d}-p0.json' for i in range(8, 12)}}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(c):
    return NAMES.get(c, str(c))


def meaning(a):
    return (a['act'], a['code'], a.get('effect'))


def moves(trace, turn):
    out = []
    for r in trace:
        s = r['snapshot']
        if s['to_play'] != 0 or s['turn_count'] != turn:
            continue
        a = r['selected']
        if a['act'] == 'Cancel' or s['message'] in ('select_place', 'select_position', 'select_disfield'):
            continue
        if s['message'] in ('select_chain', 'select_effectyn', 'select_yesno') and a['act'] != 'Activate':
            continue
        out.append(('Phase', 0, a.get('phase')) if s['message'] == 'select_idlecmd' and a['act'] == 'None' else meaning(a))
    return out


def fmt(m):
    return f'Phase:{m[2]}' if m[0] == 'Phase' else f'{m[0]}:{name(m[1])}' + (f'[{m[2]}]' if m[2] not in (None, -1, 0) else '')


def used(trace, turn):
    out = []
    for r in trace:
        s = r['snapshot']
        if s['to_play'] == 0 and s['turn_count'] == turn:
            a = r['selected']
            if a['act'] in ('Activate', 'SpSummon', 'Summon', 'Set', 'MSet', 'SSet') or 'Attack' in a['act']:
                out.append(fmt(meaning(a)))
    return out


def names_of(codes):
    return [name(c) for c in codes] if codes is not None else None


def compact_board(b):
    if not b:
        return None
    return dict(monsters=names_of(b['monsters']), spell_trap=names_of(b['spell_trap']), hand=names_of(b['hand']),
                graveyard=len(b['graveyard']), banished=len(b['banished']), extra_face_up=names_of(b['extra_face_up']), lp0=b['lp0'], lp1=b['lp1'])


def free_row(which, n):
    d = read(ROOT / 'eval/free' / which / f'{n}.json')
    script = read(ROOT / 'validation' / f'{n}.json')
    m, r = moves(d['trace'], 1), moves(script['trace'], 1)
    div = next((i for i, (x, y) in enumerate(zip(m, r)) if x != y), None if len(m) == len(r) else min(len(m), len(r)))
    t1 = d['checkpoints'].get('own_turn1_end')
    script_end = script['checkpoints'].get('own_turn_end')
    landing = None
    if t1 and script_end:
        miss = collections.Counter(script_end['monsters']) - collections.Counter(t1['monsters'])
        landing = dict(script_monsters=names_of(script_end['monsters']), missing=names_of(sorted(miss.elements())), all_present=not miss)
    return dict(model=which, name=n, condition=d['condition'], reason=d['reason'], game_over=d['game_over'], failure=d['failure'],
                turn1_moves=[fmt(x) for x in m], script_moves=len(r), first_divergence=div,
                script_move_at_divergence=fmt(r[div]) if div is not None and div < len(r) else None,
                model_move_at_divergence=fmt(m[div]) if div is not None and div < len(m) else None,
                own_turn1_end=compact_board(t1), landing_vs_script=landing, turn2_own=used(d['trace'], 2),
                opp_turn2_end=compact_board(d['checkpoints'].get('opp_turn2_end')), turn3_own=used(d['trace'], 3),
                own_turn3_end=compact_board(d['checkpoints'].get('own_turn3_end')), visible=len(d['disclosed']), lua=d['lua_errors'],
                fallbacks=d['opponent_fallbacks'])


def main():
    reg, am = read(ROOT / 'registration.json'), read(ROOT / 'amendment-1.json')
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    starts = [r for r in ledger if r['event'] == 'start']
    kinds = collections.Counter(r['purpose'].split()[0] for r in starts)
    demos = {}
    for n, spec in BP.TRAIN.items():
        rec, rep = read(ROOT / 'demos' / n / 'record.json'), read(ROOT / 'demos' / n / 'replay.json')
        demos[n] = dict(condition=spec['condition'], completed=rec['script_completed'], replay_ok=not rep['mismatch'] and rep['observations_match'] and rep['final_equal'],
                        decisions=rec['decisions'], own_turn_end=compact_board(rec['checkpoints'].get('own_turn_end')),
                        next_turn=compact_board(rec['checkpoints'].get('next_own_turn_start')),
                        opponent_turn_steps=[r['why'].get('kind') + ':' + fmt(meaning(r['selected'])) for r in rec['trace']
                                             if r['snapshot']['to_play'] == 0 and r['snapshot']['turn_count'] == 2 and r['why'].get('kind') not in ('default-pass',)],
                        visible=len(rec['disclosed']), record_sha256=sha(ROOT / 'demos' / n / 'record.json'))
    failed_a1 = {n: dict(failure=x['failure'], plan_head=x['plan_head']) for n, x in am['failed_records'].items()}
    validation = {n: dict(completed=read(ROOT / 'validation' / f'{n}.json')['script_completed'],
                          own_turn_end=compact_board(read(ROOT / 'validation' / f'{n}.json')['checkpoints'].get('own_turn_end')))
                  for n in BP.VALIDATION}
    free = {f'{w}/{n}': free_row(w, n) for w in ['start', 'candidate'] for n in BP.VALIDATION}
    rt1 = []
    for cid, base in RT1_BASE.items():
        c, b = read(ROOT / 'eval/rt1/candidate' / f'{cid}.json'), read(base)
        same = len(b['trace']) == len(c['trace']) and all(meaning(x['selected']) == meaning(y['selected']) for x, y in zip(b['trace'], c['trace']))
        rt1.append(dict(case_id=cid, baseline_passed=b['passed'], candidate_passed=c['passed'], reason=c['reason'], baseline_steps=len(b['trace']),
                        candidate_steps=len(c['trace']), same_action_sequence=same, visible=len(c['disclosed'])))
    train = read(ROOT / 'train/result.json')
    diags = [json.loads(l) for l in (ROOT / 'train/diagnostics.jsonl').read_text().splitlines()]
    updates = [json.loads(l) for l in (ROOT / 'train/updates.jsonl').read_text().splitlines()]
    branch_track = collections.defaultdict(dict)
    for d in diags:
        for g, v in d['groups'].items():
            for b in v.get('branches', []):
                branch_track[f"{b['name']}@{b['step']} {b['branch']}"][d['update']] = round(b['p_label'], 4)
    pre = read(ROOT / 'diag/pre-training.json')
    summary = dict(
        registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'), manifest_sha256=sha(ROOT / 'data/manifest.json'),
        engine=dict(used=len(starts), cap=53, by_kind=dict(kinds), statuses=collections.Counter(r['status'] for r in ledger if r['event'] == 'finish')),
        demos=demos, failed_before_amendment=failed_a1, validation=validation, free=free, rt1=rt1,
        training=dict(updates=train['updates_done'], sampled=train['sampled'], clipped_updates=train['clipped_updates'], change=train['parameter_change_norm'],
                      saved=train['saved'], data=train['data'], seconds=train['seconds'],
                      diagnostics={d['update']: {g: (round(v['ce_pooled'], 4), round(v['top1'], 4)) for g, v in d['groups'].items()} for d in diags},
                      branches=dict(branch_track), grad_norm_mean=sum(u['grad_norm'] for u in updates) / len(updates)),
        pre_training=dict(conflicts=pre['conflicts'], start={g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in pre['start'].items()},
                          pilot={g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in pre['pilot_candidate'].items()},
                          branch_values_start=pre['branch_values_start'], branch_values_pilot=pre['branch_values_pilot_candidate']),
        visibility_total=sum(d['visible'] for d in demos.values()) + sum(f['visible'] for f in free.values()) + sum(r['visible'] for r in rt1)
        + sum(len(read(ROOT / 'validation' / f'{n}.json')['disclosed']) for n in BP.VALIDATION),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'], 'visible', summary['visibility_total'])
    for n, d in demos.items():
        print('DEMO', n, d['completed'], d['replay_ok'], d['decisions'], d['own_turn_end']['monsters'], d['opponent_turn_steps'][:8])
    for k, f in free.items():
        print('FREE', k, f['reason'], 'div', f['first_divergence'], 'script:', f['script_move_at_divergence'], 'model:', f['model_move_at_divergence'],
              '| landing', (f['landing_vs_script'] or {}).get('all_present'), 'missing', (f['landing_vs_script'] or {}).get('missing'))
        print('    t1', (f['own_turn1_end'] or {}).get('monsters'), '| t2 own', f['turn2_own'][:4], '| t3', f['turn3_own'][:8], '| lp1', (f['own_turn3_end'] or {}).get('lp1'), f['game_over'])
    for r in rt1:
        print('RT1', r)
    print('BRANCHES')
    for k, v in branch_track.items():
        print('  ', k, v)


if __name__ == '__main__':
    main()
