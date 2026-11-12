"""Offline summary of the hand-branch study (no engine): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
BATCH = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import hb_plans as HP  # noqa: E402

_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
STUDIED = {HP.NERVED, HP.BEAST, HP.VARNISH, HP.ASSAULT}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(c):
    return NAMES.get(c, str(c))


def names(cs):
    return [name(c) for c in cs] if cs is not None else None


def branch_choices(trace):
    """Studied choices in turn 1 from the snapshots: ネルヴェド/獄神獣 and Graphrea's 改変/襲撃, with an audit of the hand."""
    out = []
    for r in trace:
        s = r['snapshot']
        if s['to_play'] != 0 or s['turn_count'] != 1 or s['message'] != 'select_card':
            continue
        offered = {a['code'] for a in s['actions']}
        kind = 'A' if {HP.NERVED, HP.BEAST} <= offered else 'B' if ({HP.VARNISH, HP.ASSAULT} <= offered and not offered & {HP.AWAKENING, 44654994}) else None
        if not kind:
            continue
        hand = sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == 2)
        out.append(dict(step=r['step'], kind=kind, chosen=name(r['selected']['code']),
                        audit=dict(hand=names(hand), costs_in_hand=sum(1 for c in hand if c in HP.COST_CARDS),
                                   spell_trap_in_hand=len([c for c in s['cards'] if c['controller'] == 0 and c['location'] == 2 and c['code'] in HP.COST_CARDS | {HP.ASSAULT, HP.VARNISH, HP.AWAKENING, 36709484}]),
                                   field=names(sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] in (4, 8))),
                                   graveyard=len([c for c in s['cards'] if c['controller'] == 0 and c['location'] == 16]))))
    return out


def board(b):
    if not b:
        return None
    return dict(monsters=names(b['monsters']), spell_trap=names(b['spell_trap']), hand=names(b['hand']), graveyard=len(b['graveyard']),
                banished=len(b['banished']), lp0=b['lp0'], lp1=b['lp1'])


def landing(free, script_end):
    t1 = free['checkpoints'].get('own_turn1_end')
    if not (t1 and script_end):
        return None
    miss = collections.Counter(script_end['monsters']) - collections.Counter(t1['monsters'])
    return dict(all_present=not miss, missing=names(sorted(miss.elements())))


def run_row(d, script):
    cp = d['checkpoints']
    return dict(reason=d['reason'], game_over=d['game_over'], branch=branch_choices(d['trace']),
                landing=landing(d, script['checkpoints'].get('own_turn_end') if script else None),
                t1=board(cp.get('own_turn1_end')), t2=board(cp.get('opp_turn2_end')), t3=board(cp.get('own_turn3_end')),
                lp1_t3=(cp.get('own_turn3_end') or {}).get('lp1'), visible=len(d['disclosed']), lua=d['lua_errors'], fallbacks=d['opponent_fallbacks'])


def main():
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    starts = [r for r in ledger if r['event'] == 'start']
    subs = {}
    for n in HP.SUBSTITUTE:
        d = read(ROOT / 'eval/substitute/start' / f'{n}.json')
        base = read(BATCH / 'eval/free/candidate' / f'{n}.json')
        script = read(BATCH / 'validation' / f'{n}.json')
        sb = d['substitute']
        subs[n] = dict(step=sb['step'], substituted=[name(sb['meaning'][1])], model_choice=sb.get('model_choice', {}).get('name'),
                       prefix_mismatch=sb.get('prefix_mismatch'), after=run_row(d, script), before=run_row(base, script))
    demos = {}
    for n, x in HP.HANDS.items():
        if x['split'] != 'train':
            continue
        rec, rep = read(ROOT / 'demos' / n / 'record.json'), read(ROOT / 'demos' / n / 'replay.json')
        demos[n] = dict(completed=rec['script_completed'], replay_ok=not rep['mismatch'] and rep['observations_match'] and rep['final_equal'],
                        decisions=rec['decisions'], branch=branch_choices(rec['trace']), end=board(rec['checkpoints'].get('own_turn_end')))
    validation = {n: dict(completed=read(ROOT / 'validation' / f'{n}.json')['script_completed'],
                          branch=branch_choices(read(ROOT / 'validation' / f'{n}.json')['trace']),
                          end=board(read(ROOT / 'validation' / f'{n}.json')['checkpoints'].get('own_turn_end')))
                  for n, x in HP.HANDS.items() if x['split'] == 'validation'}
    free = {}
    for n, x in HP.HANDS.items():
        if x['split'] != 'validation':
            continue
        script = read(ROOT / 'validation' / f'{n}.json')
        for w in ['start', 'candidate']:
            free[f'{w}/{n}'] = run_row(read(ROOT / 'eval/free' / w / f'{n}.json'), script)
    retain = []
    for cid in [f'Artmage-g{i:02d}-p0' for i in range(6, 12)]:
        c, b = read(ROOT / 'eval/retain/candidate' / f'{cid}.json'), read(BATCH / 'eval/rt1/candidate' / f'{cid}.json')
        same = len(b['trace']) == len(c['trace']) and all((x['selected']['act'], x['selected']['code'], x['selected'].get('effect')) ==
                                                          (y['selected']['act'], y['selected']['code'], y['selected'].get('effect')) for x, y in zip(b['trace'], c['trace']))
        retain.append(dict(case=cid, baseline_passed=b['passed'], candidate_passed=c['passed'], same_action_sequence=same, steps=(len(b['trace']), len(c['trace']))))
    retain_f = {}
    for n in ['F0-V', 'F1-V', 'F2-V', 'F3-V', 'F4-V', 'F5-V', 'F6-V']:
        script = read(BATCH / 'validation' / f'{n}.json')
        retain_f[n] = dict(baseline=run_row(read(BATCH / 'eval/free/candidate' / f'{n}.json'), script),
                           candidate=run_row(read(ROOT / 'eval/retain/candidate' / f'{n}.json'), script))
    train = read(ROOT / 'train/result.json')
    diags = [json.loads(l) for l in (ROOT / 'train/diagnostics.jsonl').read_text().splitlines()]
    track = collections.defaultdict(dict)
    for d in diags:
        for b in d['branches']:
            track[f"{b['name']}@{b['step']} {b['branch']} label={name(b['label_card'])}"][d['update']] = b['p_label']
    pre = read(ROOT / 'diag/pre-training.json')
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'), manifest_sha256=sha(ROOT / 'data/manifest.json'),
                   engine=dict(used=len(starts), cap=72, by_kind=dict(collections.Counter(r['purpose'].split()[0] for r in starts)),
                               statuses=dict(collections.Counter(r['status'] for r in ledger if r['event'] == 'finish'))),
                   substitution=subs, demos=demos, validation=validation, free=free, retain_rt1=retain, retain_f=retain_f,
                   training=dict(updates=train['updates_done'], sampled=train['sampled'], clipped=train['clipped_updates'], change=train['parameter_change_norm'],
                                 saved=train['saved'], data=train['data'], seconds=train['seconds'],
                                 diagnostics={d['update']: {g: (round(v['ce_pooled'], 4), round(v['top1'], 4)) for g, v in d['groups'].items()} for d in diags},
                                 branch_track=dict(track)),
                   pre_training=dict(conflicts=pre['conflicts'], start={g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in pre['start'].items()}),
                   visibility_total=sum(r['visible'] for r in free.values()) + sum(s['after']['visible'] for s in subs.values())
                   + sum(r['candidate']['visible'] for r in retain_f.values()),
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'], 'visible', summary['visibility_total'])
    for n, s in subs.items():
        print('SUB', n, 'step', s['step'], 'model chose', s['model_choice'], '-> substituted', s['substituted'], 'prefix mismatch', s['prefix_mismatch'],
              '| before landing', (s['before']['landing'] or {}).get('all_present'), '| after landing', s['after']['landing'], 'branch after', [(b['kind'], b['chosen']) for b in s['after']['branch']], 'lp1', s['after']['lp1_t3'], s['after']['game_over'])
    for n, d in demos.items():
        print('DEMO', n, d['completed'], d['replay_ok'], [(b['kind'], b['chosen'], b['audit']['costs_in_hand']) for b in d['branch']])
    for k, r in free.items():
        print('FREE', k, r['reason'], 'landing', (r['landing'] or {}).get('all_present'), (r['landing'] or {}).get('missing'), '| branch', [(b['kind'], b['chosen'], b['audit']['costs_in_hand']) for b in r['branch']],
              '| t1', (r['t1'] or {}).get('monsters'), '| lp1', r['lp1_t3'], (r['game_over'] or {}).get('winner'))
    for r in retain:
        print('RT1', r)
    for n, r in retain_f.items():
        b, c = r['baseline'], r['candidate']
        print('RETAIN', n, 'baseline landing', (b['landing'] or {}).get('all_present'), [(x['kind'], x['chosen']) for x in b['branch']], 'lp1', b['lp1_t3'], (b['game_over'] or {}).get('winner'),
              '| candidate landing', (c['landing'] or {}).get('all_present'), [(x['kind'], x['chosen']) for x in c['branch']], 'lp1', c['lp1_t3'], (c['game_over'] or {}).get('winner'))
    print('TRACK (new pairs)')
    for k, v in track.items():
        if k.split('@')[0][:2] in ('A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4'):
            print('  ', k, {u: round(p, 3) for u, p in v.items()})


if __name__ == '__main__':
    main()
