"""Offline summary of the response-coverage study (no engine, no model): this candidate vs the start model (and the scoped adapter
where it was run), the A reference check, the C reuse evidence, training detail."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
SA = P / 'experiments/artmage-scoped-response-adapter-20261111'
VOCAB = P / 'experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt'
NAMES8 = ['BV1-1cost-VARNISH-OM', 'BV1-1cost-VARNISH-OI', 'BV1-1cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OI',
          'BV1-2cost-VARNISH-OM', 'BV1-2cost-VARNISH-OI', 'BV1-2cost-ASSAULT-OM', 'BV1-2cost-ASSAULT-OI']
TRAINED = {'BV1-1cost-VARNISH-OM': 'S1 (trained, CE)', 'BV1-2cost-ASSAULT-OM': 'S2 (trained, CE)', 'BV1-1cost-ASSAULT-OM': 'M (trained, keep)',
           'BV1-2cost-VARNISH-OI': 'P (trained, keep)'}
MEDIUS, BEAST, INFINITE = 97556336, 23829452, 10045474
_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NM = {c: n for c, n in _db.execute('select id,name from texts')}
CODES = [int(line.split()[0]) for line in VOCAB.read_text().splitlines()]
LABEL = {(74631897, 10011): '乙女②', (27184601, 10011): 'ディアクトロス②', (34541940, 10011): 'ファインメルト③', (97434754, 10011): 'リテラ③',
         (23829452, 10010): '獄神獣②', (23829452, 10011): '獄神獣③', (31822037, 10011): '獄神使②', (23599634, 10011): '覚醒②', (23599634, 10010): '覚醒①',
         (53589300, 10010): 'ネルヴァ②', (44654994, 10010): '奪還①', (23434538, 10010): '増殖するG', (59438930, 0): '幽鬼うさぎ'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def label(a):
    if a['act'] != 'Activate':
        return a['act']
    return LABEL.get((a['code'], a.get('effect'))) or f"{NM.get(a['code'], a['code'])}:e{a.get('effect')}"


def gated(s):
    ch = s['public_chain']
    return bool(s['message'] == 'select_chain' and ch and ch[-1]['player'] != s['to_play'])


def names(cs):
    return [NM.get(c, str(c)) for c in cs]


def board(b):
    if not b:
        return None
    return dict(lp=[b['lp0'], b['lp1']], monsters=names(b['monsters']), spell_trap=names(b['spell_trap']), hand=names(b['hand']),
                opponent_monsters=names(b['opponent_monsters']), opponent_graveyard=names(b['opponent_graveyard']))


def status_flag(d, rec, step, controller, code, sequence=None):
    """NPZ cards_ col 11 (STATUS_DISABLED | STATUS_FORBIDDEN) of a field monster at the first own decision >= step."""
    z = np.load(d / 'observations.npz')
    i = int(np.searchsorted(z['steps'], step))
    if i >= len(z['steps']):
        return None
    s = next(r['snapshot'] for r in rec['trace'] if r['step'] == int(z['steps'][i]))
    phys = [c for c in s['cards'] if c['controller'] == controller and c['location'] == 4 and c['code'] == code and (sequence is None or c['sequence'] == sequence)]
    if len(phys) != 1:
        return dict(step=int(z['steps'][i]), on_field=len(phys))
    rel = 0 if controller == 0 else 1
    rows = [x for x in z['cards_'][i] if x[2] == 3 and x[3] == phys[0]['sequence'] + 1 and x[4] == rel and CODES[((int(x[0]) << 8) | int(x[1])) - 1] == code]
    return dict(step=int(z['steps'][i]), value=int(rows[0][11]) if len(rows) == 1 else None)


def first_window(rec, d, opp):
    tr = rec['trace']
    k = next((i for i, r in enumerate(tr) if r['snapshot']['to_play'] == 0 and gated(r['snapshot'])), None)
    if k is None:
        return None
    w = tr[k]
    end = next((j for j in range(k, len(tr)) if tr[j]['snapshot']['chain_size'] == 0), None)
    after = tr[end]['snapshot'] if end is not None else None
    resp = [label(tr[j]['selected']) for j in range(k, end or len(tr)) if tr[j]['snapshot']['to_play'] == 0 and tr[j]['selected']['act'] == 'Activate']
    out = dict(step=w['step'], chain=[NM.get(c['code'], c['code']) for c in w['snapshot']['public_chain']],
               candidates=[label(a) for a in w['snapshot']['actions']], executed=label(w['selected']), responses_in_chain=resp)
    if after is not None:
        opp_mon = [c['code'] for c in after['cards'] if c['controller'] == 1 and c['location'] == 4]
        opp_gy = [c['code'] for c in after['cards'] if c['controller'] == 1 and c['location'] == 16]
        if opp == 'OM':
            out.update(beast_arrived=BEAST in opp_mon, medius_on_field=MEDIUS in opp_mon,
                       medius_status=status_flag(d, rec, tr[end]['step'], 1, MEDIUS) if MEDIUS in opp_mon else None)
        else:
            tgt = next((e for e in rec['opponent_events'] if e['what'] == '無限泡影 target'), None)
            pick = next((r for r in tr if tgt and r['step'] == tgt['step']), None)
            spec = (pick or {}).get('selected', {}).get('spec', '')
            seq = int(spec[2:]) - 1 if spec.startswith('om') else None
            tcode = tgt['chosen'] if tgt else None
            out.update(infinite_target=NM.get(tcode, tcode), infinite_in_opponent_gy=INFINITE in opp_gy,
                       target_on_field=any(c['controller'] == 0 and c['location'] == 4 and c['code'] == tcode and c['sequence'] == seq for c in after['cards']),
                       target_status=status_flag(d, rec, tr[end]['step'], 0, tcode, seq))
    return out


def summary_of(rec, d, opp):
    cp = rec['checkpoints']
    pol = rec.get('policy')
    return dict(window=first_window(rec, d, opp), t2_end=board(cp.get('opp_turn2_end')), t3_end=board(cp.get('own_turn3_end')),
                t3_attacks=[f"{r['selected']['act']}:{NM.get(r['selected']['code'])}" for r in rec['trace'] if r['snapshot']['turn_count'] == 3
                            and r['snapshot']['to_play'] == 0 and 'Attack' in r['selected']['act']],
                game_over=rec['game_over'], adapter=pol and [dict(call=x['call'], changed=x['frozen_argmax'] != x['final_argmax']) for x in pol['adapter_active']],
                visible=len(rec['disclosed']), lua=rec['lua_errors'])


def first_divergence(a, b):
    key = lambda r: (r['selected']['act'], r['selected']['code'], r['selected'].get('effect'), r['selected'].get('spec'), r['selected'].get('place'), r['selected'].get('position'))
    for x, y in zip(a['trace'], b['trace']):
        if x['snapshot'] != y['snapshot'] or key(x) != key(y):
            return dict(step=x['step'], turn=x['snapshot']['turn_count'], gated=gated(x['snapshot']), candidate=label(x['selected']), start_model=label(y['selected']))
    return None if len(a['trace']) == len(b['trace']) else dict(length=[len(a['trace']), len(b['trace'])])


def main():
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    res, prep, upd = read(ROOT / 'train/result.json'), read(ROOT / 'cache/prepare.json'), [json.loads(l) for l in (ROOT / 'train/updates.jsonl').read_text().splitlines()]
    rows = {}
    for n in NAMES8:
        d = ROOT / 'runs' / f'B-{n}'
        rec, rep, old = read(d / 'record.json'), read(d / 'replay.json'), read(FLC / 'runs' / n / 'record.json')
        opp = n.split('-')[-1]
        div = first_divergence(rec, old)
        sa = SA / 'runs' / f'B-{n}'
        pol = rec['policy']
        rows[n] = dict(role=TRAINED.get(n, 'not trained (O-I diagnostic)' if opp == 'OI' else 'not trained (O-M)'), identical_to_start=div is None, first_divergence=div,
                       policy_hashes=dict(base=pol['base_sha256'][:12], head=pol['head_sha256'][:12], adapter=pol['adapter_sha256'][:12], manifest=pol['manifest_sha256'][:12]),
                       candidate=summary_of(rec, d, opp), start=summary_of(old, FLC / 'runs' / n, opp),
                       scoped=summary_of(read(sa / 'record.json'), sa, opp) if sa.exists() else 'not run in the scoped study',
                       replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']})
    # A reference: B traces up to the first window equal the start-model records (and the NPZ), no activation before
    a_ref = {}
    for site, n in [('S1', 'BV1-1cost-VARNISH-OM'), ('S2', 'BV1-2cost-ASSAULT-OM')]:
        rec, old = read(ROOT / 'runs' / f'B-{n}' / 'record.json'), read(FLC / 'runs' / n / 'record.json')
        w = next(r['step'] for r in rec['trace'] if r['snapshot']['to_play'] == 0 and gated(r['snapshot']))
        same_rows = all(x['snapshot'] == y['snapshot'] and (x['selected']['act'], x['selected']['code'], x['selected'].get('effect'), x['selected'].get('spec')) ==
                        (y['selected']['act'], y['selected']['code'], y['selected'].get('effect'), y['selected'].get('spec')) for x, y in zip(rec['trace'][:w], old['trace'][:w]))
        za, zb = np.load(ROOT / 'runs' / f'B-{n}' / 'observations.npz'), np.load(FLC / 'runs' / n / 'observations.npz')
        k = int(np.searchsorted(za['steps'], w))
        same_obs = all(np.array_equal(za[x][:k + 1], zb[x][:k + 1]) for x in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain']) and np.array_equal(za['labels'][:k], zb['labels'][:k])
        before = [x for x in rec['policy']['adapter_active'] if x['call'] < k]
        a_ref[site] = dict(window_step=w, rows_equal_before_window=same_rows, inputs_equal_to_window=bool(same_obs), activations_before_window=len(before),
                           referenced=same_rows and same_obs and not before)
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'),
                   candidate_manifest_sha256=sha(ROOT / 'train/candidate/manifest.json'),
                   engine=dict(used=len(starts), cap=18, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e))),
                   training=dict(screens=res['screens'], clipped=res['clipped_updates'], post=res['post_training']['comparison'] | {'on_target': res['post_training']['comparison']['on_target']},
                                 position_off=res['post_training']['position_actions_off_target_differences'], reload=res['saved']['reload'],
                                 ce_first_last=[upd[0]['ce'], upd[-1]['ce']], kl_first_last=[upd[0]['kl'], upd[-1]['kl']], kl_max=max(u['kl'] for u in upd),
                                 ce_grad_norm=dict(mean=float(np.mean([u['ce_grad_norm'] for u in upd])), max=max(u['ce_grad_norm'] for u in upd)),
                                 kl_grad_norm=dict(mean=float(np.mean([u['kl_grad_norm'] for u in upd])), max=max(u['kl_grad_norm'] for u in upd)),
                                 update_norm_mean=float(np.mean([u['update_norm'] for u in upd])), max_abs_delta_last=upd[-1]['max_abs_delta']),
                   prechecks={k: prep[k] for k in ['zero_adapter', 'gate_vs_snapshot', 'gated_npz_indices_in_the_8_runs', 'screens', 'all_prechecks_passed']},
                   B=rows, A_reference=a_ref, C=read(ROOT / 'registration.json')['evaluation']['C'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'])
    print('A_REF', a_ref)
    for n, r in rows.items():
        c, s = r['candidate'], r['start']
        cw, sw = c['window'] or {}, s['window'] or {}
        print('=====', n, r['role'], '| identical to start:', r['identical_to_start'], '| div', r['first_divergence'])
        for lab, x, w in [('cand ', c, cw), ('start', s, sw)]:
            print('  ', lab, 'window', w.get('step'), w.get('candidates'), '->', w.get('executed'), w.get('responses_in_chain'),
                  {k: w[k] for k in w if k in ('beast_arrived', 'medius_on_field', 'medius_status', 'infinite_target', 'infinite_in_opponent_gy', 'target_on_field', 'target_status')})
            print('  ', lab, 't2', (x['t2_end'] or {}).get('monsters'), 'hand', (x['t2_end'] or {}).get('hand'), 'opp', (x['t2_end'] or {}).get('opponent_monsters'),
                  '| t3', x['t3_attacks'], (x['t3_end'] or {}).get('lp'), x['game_over'] and 'game over')
        if isinstance(r['scoped'], dict):
            print('   scoped window ->', (r['scoped']['window'] or {}).get('executed'), 't3', (r['scoped']['t3_end'] or {}).get('lp'))
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
