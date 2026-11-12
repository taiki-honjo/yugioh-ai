"""Offline summary of the Imperm / Diaktoros② trade-off (no engine). The reused A-I-N is read from the coverage record, cut at
the first turn-3 decision. Status column = NPZ cards_ col 11 (STATUS_DISABLED | STATUS_FORBIDDEN), tracked by controller / zone / code."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
CV = P / 'experiments/artmage-response-coverage-20261111'
VOCAB = P / 'experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt'
CODES = [int(line.split()[0]) for line in VOCAB.read_text().splitlines()]
_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NM = {c: n for c, n in _db.execute('select id,name from texts')}
MEDIUS, INFINITE, BEAST, MAIDEN, DIAKTOROS, FINEMELT, LITERA = 97556336, 10045474, 23829452, 74631897, 27184601, 34541940, 97434754
LABEL = {(27184601, 10011): 'ディアクトロス②', (74631897, 10011): '乙女②', (74631897, 10010): '乙女①', (23829452, 10010): '獄神獣②', (23829452, 10011): '獄神獣③',
         (23434538, 10010): 'G', (59438930, 0): 'うさぎ', (97434754, 10011): 'リテラ③', (23599634, 10011): '覚醒②', (31822037, 10011): '獄神使②',
         (34541940, 10011): 'ファインメルト③', (53589300, 10010): 'ネルヴァ②', (44654994, 10010): '奪還①'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def lab(a):
    if a['act'] != 'Activate':
        return a['act']
    return LABEL.get((a['code'], a.get('effect')), f"{NM.get(a['code'], a['code'])}:e{a.get('effect')}")


def names(cs):
    return [NM.get(c, str(c)) for c in cs]


def board(b):
    if not b:
        return None
    return dict(lp=[b['lp0'], b['lp1']], monsters=names(b['monsters']), spell_trap=names(b['spell_trap']), hand=names(b['hand']), graveyard=names(b['graveyard']),
                banished=names(b['banished']), opponent_monsters=names(b['opponent_monsters']), opponent_graveyard=names(b.get('opponent_graveyard', [])))


def status_at(z, rec, step, zone, code=MAIDEN):
    """Own MZONE zone `zone` card status column at the first own decision >= step (None if the card is not there)."""
    i = int(np.searchsorted(z['steps'], step))
    if i >= len(z['steps']):
        return None
    st = int(z['steps'][i])
    snap = next(r['snapshot'] for r in rec['trace'] if r['step'] == st)
    here = [c for c in snap['cards'] if c['controller'] == 0 and c['location'] == 4 and c['sequence'] == zone]
    if not here or here[0]['code'] != code:
        return dict(step=st, card_in_zone=NM.get(here[0]['code']) if here else None)
    rows = [x for x in z['cards_'][i] if x[2] == 3 and x[3] == zone + 1 and x[4] == 0 and CODES[((int(x[0]) << 8) | int(x[1])) - 1] == code]
    return dict(step=st, value=int(rows[0][11]) if len(rows) == 1 else None)


def chain_end_after(trace, start_step):
    seen = False
    for r in trace:
        if r['step'] < start_step:
            continue
        if r['snapshot']['chain_size'] >= 1:
            seen = True
        elif seen:
            return r['step']
    return None


def analyse(rec, z, cut=None):
    tr = [r for r in rec['trace'] if cut is None or r['step'] < cut]
    win = next(r for r in tr if r['snapshot']['to_play'] == 0 and r['snapshot']['public_chain'] and r['snapshot']['public_chain'][-1]['player'] == 1
               and r['snapshot']['public_chain'][-1]['code'] == INFINITE)
    tgt_ev = next(e for e in rec['opponent_events'] if e['what'] == '無限泡影 target') if rec.get('opponent_events') else None
    tgt_row = next(r for r in rec['trace'] if r['step'] < win['step'] and r['snapshot']['to_play'] == 1 and r['selected'].get('spec', '').startswith('om')
                   and r['snapshot']['public_chain'] and r['snapshot']['public_chain'][-1]['code'] == INFINITE)
    zone = int(tgt_row['selected']['spec'][2:]) - 1
    end = chain_end_after(tr, win['step'])
    after = next((r for r in tr if r['step'] == end), None)
    used = [dict(step=r['step'], turn=r['snapshot']['turn_count'], phase=r['snapshot']['phase'], effect=lab(r['selected']), spec=r['selected'].get('spec'), who=r['who'])
            for r in tr if r['snapshot']['to_play'] == 0 and r['selected']['act'] == 'Activate' and r['step'] >= win['step']]
    maiden_offered_later = sorted({r['step'] for r in tr if r['step'] > win['step'] and r['snapshot']['to_play'] == 0 and r['snapshot']['turn_count'] == 2
                                   and any(a['act'] == 'Activate' and a['code'] == MAIDEN and a.get('effect') == 10011 and a.get('spec') == f'm{zone + 1}' for a in r['snapshot']['actions'])})
    d2_used_before = [r['step'] for r in tr if r['step'] < win['step'] and r['snapshot']['turn_count'] == 2 and r['snapshot']['to_play'] == 0
                      and r['selected']['act'] == 'Activate' and r['selected']['code'] == DIAKTOROS and r['selected'].get('effect') == 10011]
    med = next((r for r in tr if r['snapshot']['to_play'] == 0 and len(r['snapshot']['public_chain']) == 1 and r['snapshot']['public_chain'][0]['player'] == 1
                and r['snapshot']['public_chain'][0]['code'] == MEDIUS), None)
    med_info = None
    if med is not None:
        mend = chain_end_after(tr, med['step'])
        ma = next((r['snapshot'] for r in tr if r['step'] == mend), None)
        med_info = dict(step=med['step'], legal=[lab(a) for a in med['snapshot']['actions']], executed=lab(med['selected']),
                        diaktoros2_legal=any(a['code'] == DIAKTOROS and a.get('effect') == 10011 for a in med['snapshot']['actions']),
                        beast_arrived=ma is not None and any(c['controller'] == 1 and c['location'] == 4 and c['code'] == BEAST for c in ma['cards']),
                        medius_on_opponent_field=ma is not None and any(c['controller'] == 1 and c['location'] == 4 and c['code'] == MEDIUS for c in ma['cards']),
                        medius_in_opponent_gy=ma is not None and any(c['controller'] == 1 and c['location'] == 16 and c['code'] == MEDIUS for c in ma['cards']))
    return dict(window_step=win['step'], window_legal=[lab(a) for a in win['snapshot']['actions']], response=lab(win['selected']), target_zone=zone,
                diaktoros2_used_before_window=d2_used_before,
                infinite_in_opponent_gy_after=after is not None and any(c['controller'] == 1 and c['location'] == 16 and c['code'] == INFINITE for c in after['snapshot']['cards']),
                target_status_before=status_at(z, rec, win['step'], zone), target_status_after=status_at(z, rec, end, zone) if end else None,
                target_maiden2_offered_later_in_turn2=maiden_offered_later, own_activations_from_window=used, medius=med_info)


def classify_fusion(rec):
    """B: offered / activated / fusion selection shown / materials moved / new fusion Maiden (separately)."""
    log = rec['own_rule_log']
    act = next((x for x in log if x['rule'] == 'original Maiden② (fusion)'), None)
    out = dict(activated=act is not None, activation_step=act and act['step'],
               selection_shown=any(x['rule'] == 'fusion target: EX Maiden' for x in log), materials_chosen=[x['rule'] for x in log if x['rule'].startswith('fusion material')])
    zone = rec.get('maiden_zone_B')
    offered = [r['step'] for r in rec['trace'] if r['snapshot']['to_play'] == 0 and r['snapshot']['turn_count'] == 2
               and any(a['act'] == 'Activate' and a['code'] == MAIDEN and a.get('effect') == 10011 and a.get('spec') == f'm{zone + 1}' for a in r['snapshot']['actions'])]
    out['offered_steps'] = offered
    b0 = rec['checkpoints'].get('after_imperm_chain') or {}
    b1 = rec['checkpoints'].get('after_fusion_chain') or rec['checkpoints'].get('first_turn3_decision') or {}
    c0, c1 = collections.Counter(b0.get('monsters', [])), collections.Counter(b1.get('monsters', []))
    out.update(materials_left_field=[NM[c] for c in [FINEMELT, LITERA] if c0[c] > c1[c]], maiden_count=[c0[MAIDEN], c1[MAIDEN]],
               new_fusion_maiden=c1[MAIDEN] > c0[MAIDEN])
    return out


def main():
    reg = read(ROOT / 'registration.json')
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()] if (ROOT / 'eval-runs.jsonl').exists() else []
    out = {}
    for n in ['A-I-N', 'A-I-P', 'A-IM-N', 'A-IM-P', 'B-N', 'B-P']:
        if n == 'A-I-N' and n not in reg['to_run']:
            d = CV / 'runs/B-BV1-1cost-VARNISH-OI'
            rec = read(d / 'record.json')
            cut = reg['reuse']['A_I_N']['cut_at_step']
            snap = next(r['snapshot'] for r in rec['trace'] if r['step'] == cut)
            import sys
            sys.path.insert(0, str(ROOT / 'code'))
            end_board = dict(lp=[snap['lp0'], snap['lp1']],
                             monsters=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 4)),
                             spell_trap=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 8)),
                             hand=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 2)),
                             graveyard=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 16)),
                             banished=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 32)),
                             opponent_monsters=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 1 and c['location'] == 4)),
                             opponent_graveyard=names(sorted(c['code'] for c in snap['cards'] if c['controller'] == 1 and c['location'] == 16)))
            out[n] = dict(source=f'reused: {d} cut at step {cut}', analysis=analyse(rec, np.load(d / 'observations.npz'), cut=cut), end=end_board)
            continue
        d = ROOT / 'runs' / n
        rec = read(d / 'record.json')
        z = np.load(d / 'observations.npz')
        end = rec['checkpoints'].get('first_turn3_decision') or rec['checkpoints'].get('after_fusion_chain')
        out[n] = dict(source='new record + replay', reason=rec['reason'], failure=rec['failure'], analysis=analyse(rec, z), end=board(end),
                      after_imperm=board(rec['checkpoints'].get('after_imperm_chain')), opponent=[e['what'] for e in rec['opponent_events']],
                      replay=read(d / 'replay.json'), fusion=classify_fusion(rec) if n.startswith('B') else None, visible=len(rec['disclosed']), lua=rec['lua_errors'])
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), engine=dict(used=len(starts), cap=14,
                   statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e))), conditions=out)
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'])
    for n, v in out.items():
        a = v['analysis']
        print('=====', n, v['source'][:40], v.get('reason'))
        print('   window', a['window_step'], a['window_legal'], '->', a['response'], '| D2 used before', a['diaktoros2_used_before_window'],
              '| target zone', a['target_zone'], 'status', a['target_status_before'], '->', a['target_status_after'], '| Imperm in opp GY', a['infinite_in_opponent_gy_after'],
              '| target ② offered later', a['target_maiden2_offered_later_in_turn2'])
        print('   own activations', [(x['step'], x['effect'], x['who']) for x in a['own_activations_from_window']])
        if a['medius']:
            print('   medius', a['medius'])
        if v.get('fusion'):
            print('   fusion', v['fusion'])
        e = v['end'] or {}
        print('   end', e.get('lp'), 'own', e.get('monsters'), 'ST', e.get('spell_trap'), 'hand', e.get('hand'), '| opp', e.get('opponent_monsters'), 'oppGY', e.get('opponent_graveyard'))
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
