"""Offline summary of the RL training (no engine, no model): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
P = Path('/home/hatk8/projects/yugioh-ai')
HB = P / 'experiments/artmage-hand-branch-20261110'
BATCH = P / 'experiments/artmage-basic-openings-batch-20261110'
sys.path.insert(0, str(ROOT / 'code'))
import rlt_stats as S  # noqa: E402

_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
SOURCES = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'pairs-A', 'pairs-B')
RETAIN = [f'Artmage-g{i:02d}-p0' for i in range(6, 12)] + [f'F{i}-V' for i in range(7)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def names(cs):
    return [NAMES.get(c, str(c)) for c in cs]


def quarter_means(rows, keys):
    out = []
    for part in np.array_split(np.arange(len(rows)), 4):
        if not len(part):
            continue
        sel = [rows[i] for i in part]
        out.append(dict(updates=f'{int(part[0])}-{int(part[-1])}', **{k: float(np.mean([r[k] for r in sel])) for k in keys}))
    return out


def training():
    res = read(ROOT / 'train/result.json') if (ROOT / 'train/result.json').exists() else None
    eps = jsonl(ROOT / 'train/episodes.jsonl')
    wins = jsonl(ROOT / 'train/windows.jsonl')
    mbs = jsonl(ROOT / 'train/minibatches.jsonl')
    vis = jsonl(ROOT / 'train/visibility-events.jsonl')
    ends = collections.Counter((e['end'] or {}).get('kind', 'cut:' + str(e.get('cut_by'))) for e in eps)
    by_seat = {}
    for seat in (0, 1):
        es = [e for e in eps if e['learner_seat'] == seat and e['end'] and e['end']['kind'] == 'natural']
        by_seat['first' if seat == 0 else 'second'] = dict(natural=len(es), learner_wins=sum(e['end']['learner_reward'] > 0 for e in es),
                                                           learner_losses=sum(e['end']['learner_reward'] < 0 for e in es))
    halves = []
    for part in np.array_split(np.arange(len(eps)), 4):
        es = [eps[i] for i in part if eps[i]['end'] and eps[i]['end']['kind'] == 'natural']
        if len(part):
            halves.append(dict(episodes=f'{int(part[0])}-{int(part[-1])}', natural=len(es), learner_wins=sum(e['end']['learner_reward'] > 0 for e in es)))
    keys = ['pg', 'v', 'ent', 'approx_kl', 'clipfrac', 'reh_ce', 'reh_acc', 'ppo_grad', 'reh_grad_weighted', 'combined_grad', 'ppo_reh_cosine',
            'base_update', 'head_update']
    per_source = np.zeros((len(SOURCES), 3))
    counts = np.zeros(len(SOURCES))
    for m in mbs:
        ps = np.asarray(m['per_source'])
        per_source[:, 0] += ps[:, 0]
        for k in range(len(SOURCES)):
            if ps[k, 0] > 0:
                per_source[k, 1] += ps[k, 1] * ps[k, 0]
                per_source[k, 2] += ps[k, 2]
                counts[k] += 1
    src = {s: dict(loss_steps=int(per_source[k, 0]), ce=float(per_source[k, 1] / per_source[k, 0]) if per_source[k, 0] else None,
                   minibatches=int(counts[k]), mean_weighted_grad=float(per_source[k, 2] / counts[k]) if counts[k] else None)
           for k, s in enumerate(SOURCES)}
    lengths = [e['decisions'] for e in eps if e['end'] and e['end']['kind'] == 'natural']
    promo = read(ROOT / 'train/promotion.json') if (ROOT / 'train/promotion.json').exists() else None
    return dict(result={k: v for k, v in (res or {}).items() if k not in ('data',)}, promotion=promo, episodes=len(eps), ends=dict(ends), by_learner_seat=by_seat,
                by_quarter=halves, natural_game_decisions=dict(mean=float(np.mean(lengths)) if lengths else None, max=max(lengths) if lengths else None),
                windows=len(wins), bootstrap_kinds=dict(collections.Counter(w['bootstrap'] for w in wins)),
                minibatches=len(mbs), quarter_means=quarter_means(mbs, keys) if mbs else [], rehearsal_per_source=src,
                visibility=dict(events=len(vis), cards=sum(len(v['cards']) for v in vis), unverified=sum(c['origin'] == 'unverified' for v in vis for c in v['cards']),
                                stopped=[v for v in vis if v['stopped']]))


def battles():
    games = jsonl(ROOT / 'eval/battles/games.jsonl')
    if not games:
        return None
    ev = S.evaluate_module()
    summ = S.pair_summary(games, ev)
    dec = collections.defaultdict(float)
    inf = collections.defaultdict(float)
    for g in games:
        for r, n in g['decisions'].items():
            dec[r] += n
        for r, t in g['inference_seconds'].items():
            inf[r] += t
    summ.update(reasons=dict(collections.Counter((g['result'] or {}).get('reason_label', 'error') for g in games)),
                errors=[(g['pair'], g['candidate_seat'], g['error']) for g in games if g['error']],
                last_turn=dict(mean=float(np.mean([g['last_turn'] for g in games])), max=max(g['last_turn'] for g in games)),
                env_steps=dict(mean=float(np.mean([g['env_steps'] for g in games])), max=max(g['env_steps'] for g in games)),
                inference_ms_per_decision={r: 1000 * inf[r] / dec[r] for r in dec}, decisions=dict(dec),
                visible=sum(g['visible'] for g in games), visible_unverified=sum(g['visible_unverified'] for g in games),
                lua_errors=max(g['lua_errors'] for g in games), position_changes={r: sum(g['position_adapter'][r]['changed'] for g in games) for r in ('candidate', 'start')},
                winner_turn_counts=dict(collections.Counter(g['last_turn'] for g in games if g['result'] and g['result']['outcome'] == 'win')),
                loser_turn_counts=dict(collections.Counter(g['last_turn'] for g in games if g['result'] and g['result']['outcome'] == 'loss')))
    return summ


def landing(free, script):
    t1 = free['checkpoints'].get('own_turn1_end')
    end = script['checkpoints'].get('own_turn_end')
    if not (t1 and end):
        return None
    miss = collections.Counter(end['monsters']) - collections.Counter(t1['monsters'])
    return dict(all_present=not miss, missing=names(sorted(miss.elements())), turn1_monsters=names(t1['monsters']))


def own_key(r):
    a = r['selected']
    return (a['act'], a['code'], a.get('effect'), a.get('spec'), a.get('place'), a.get('position'))


def first_divergence(new, old, own=lambda r: r['snapshot']['to_play'] == 0):
    a = [r for r in new['trace'] if own(r)]
    b = [r for r in old['trace'] if own(r)]
    for x, y in zip(a, b):
        if x['snapshot'] != y['snapshot'] or own_key(x) != own_key(y):
            return dict(step=x['step'], turn=x['snapshot']['turn_count'], message=x['snapshot']['message'], same_snapshot=x['snapshot'] == y['snapshot'],
                        candidate=[x['selected']['act'], NAMES.get(x['selected']['code'], x['selected']['code']), x['selected'].get('effect')],
                        start=[y['selected']['act'], NAMES.get(y['selected']['code'], y['selected']['code']), y['selected'].get('effect')])
    return None if len(a) == len(b) else dict(length=[len(a), len(b)])


def retention():
    out = {}
    for n in RETAIN:
        cp = ROOT / 'eval/retain/candidate' / f'{n}.json'
        if not cp.exists():
            out[n] = None
            continue
        cand, base = read(cp), read(HB / 'eval/retain/candidate' / f'{n}.json')
        if n.startswith('Artmage-g'):
            out[n] = dict(kind='RT1', start=base['passed'], candidate=cand['passed'], start_reason=base['reason'], candidate_reason=cand['reason'],
                          candidate_monsters=names((cand['board'] or {}).get('monsters', [])), start_monsters=names((base['board'] or {}).get('monsters', [])),
                          first_divergence=first_divergence(cand, base), visible=len(cand['disclosed']), lua=cand['lua_errors'])
        else:
            script = read(BATCH / 'validation' / f'{n}.json')
            ls, lc = landing(base, script), landing(cand, script)
            out[n] = dict(kind='F-V', start=bool(ls and ls['all_present']), candidate=bool(lc and lc['all_present']), start_landing=ls, candidate_landing=lc,
                          candidate_failure=cand['failure'], candidate_reason=cand['reason'], game_over=cand['game_over'],
                          first_divergence=first_divergence(cand, base), visible=len(cand['disclosed']), lua=cand['lua_errors'])
    done = {n: x for n, x in out.items() if x}
    start_ok = sorted(n for n, x in done.items() if x['start'])
    cand_ok = sorted(n for n, x in done.items() if x['candidate'])
    return dict(cases=out, evaluated=len(done), start_successes=start_ok, candidate_successes=cand_ok,
                lost=sorted(set(start_ok) - set(cand_ok)), gained=sorted(set(cand_ok) - set(start_ok)),
                same_path=sorted(n for n, x in done.items() if x['first_divergence'] is None))


def main():
    ledger = jsonl(ROOT / 'eval-runs.jsonl')
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), seeds_sha256=sha(ROOT / 'data/seeds.json'),
                   engine=dict(used=len(starts), cap=141, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e)),
                               by_kind=dict(collections.Counter(e['purpose'] for e in starts))),
                   training=training(), battles=battles(), retention=retention(),
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    t = summary['training']
    print('TRAIN', json.dumps({k: t['result'].get(k) for k in ['status', 'total', 'episodes', 'windows', 'minibatches', 'stop_reason', 'error', 'seconds']}))
    print('ENDS', t['ends'], 'SEATS', t['by_learner_seat'], 'BOOT', t['bootstrap_kinds'])
    for q in t['quarter_means']:
        print('Q', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}))
    if summary['battles']:
        b = summary['battles']
        print('BATTLES', json.dumps({k: b[k] for k in ['verified_pairs', 'verified_games', 'wins', 'losses', 'draws', 'win_rate', 'ci95_pairs', 'pair_kinds', 'seats', 'reasons']}, ensure_ascii=False))
    r = summary['retention']
    print('RETAIN', r['evaluated'], 'start', len(r['start_successes']), 'cand', len(r['candidate_successes']), 'lost', r['lost'], 'gained', r['gained'], 'same', len(r['same_path']))
    print('ENGINE', summary['engine'])


if __name__ == '__main__':
    main()
