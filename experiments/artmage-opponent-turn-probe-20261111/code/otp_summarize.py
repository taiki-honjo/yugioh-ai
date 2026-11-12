"""Offline summary of the opponent-turn probe (no engine, no model): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
OLD = P / 'experiments/artmage-functional-landing-compare-20261110'
VOCAB = P / 'experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt'
sys.path.insert(0, str(ROOT / 'code'))
import otp_runs as O  # noqa: E402

_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
CODES = [int(line.split()[0]) for line in VOCAB.read_text().splitlines()]
LABEL = {(74631897, 10011): '乙女②', (74631897, 10010): '乙女①', (27184601, 10011): 'ディアクトロス②', (34541940, 10011): 'ファインメルト③',
         (34541940, 10010): 'ファインメルト①', (97434754, 10011): 'リテラ③', (23829452, 10010): '獄神獣②', (23829452, 10011): '獄神獣③',
         (31822037, 10011): '獄神使②', (23599634, 10010): '覚醒①', (23599634, 10011): '覚醒②', (97556336, 221): 'メディウス①(誘発の確認)',
         (53589300, 10010): 'ネルヴァ②', (44654994, 10010): '奪還①', (23434538, 10010): '増殖するG', (59438930, 0): '幽鬼うさぎ'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(c):
    return NAMES.get(c, str(c))


def label(a):
    if a['act'] != 'Activate':
        return a['act'] + (f":{name(a['code'])}" if a.get('code') else '')
    return LABEL.get((a['code'], a.get('effect'))) or f"{name(a['code'])}:e{a.get('effect')}"


def names(codes):
    return [name(c) for c in codes]


def board(b):
    if not b:
        return None
    return dict(lp=[b['lp0'], b['lp1']], monsters=names(b['monsters']), spell_trap=names(b['spell_trap']), hand=names(b['hand']),
                graveyard=names(b['graveyard']), banished=names(b['banished']), opponent_monsters=names(b['opponent_monsters']),
                opponent_graveyard=names(b['opponent_graveyard']))


def load(cond):
    if cond in O.NEW:
        d = ROOT / 'runs' / cond
        return json.loads((d / 'record.json').read_text()), d, json.loads((d / 'replay.json').read_text()), 'new'
    d = OLD / 'runs' / O.SITES[cond.split('-')[0]]
    return json.loads((d / 'record.json').read_text()), d, json.loads((d / 'replay.json').read_text()), 'reused old record'


def probe_checkpoints(rec):
    """Old records lack the probe checkpoints: recompute them from the stored snapshots with the same Probe."""
    if 'after_medius_chain' in rec['checkpoints']:
        return rec['checkpoints']
    pr = O.Probe()
    for r in rec['trace']:
        pr(dict(r['snapshot'], _step=r['step']))
    return dict(rec['checkpoints'], **pr.cp)


def turn1_identity(rec, d, old_rec, old_dir):
    rows = [r for r in rec['trace'] if r['snapshot']['turn_count'] == 1]
    old_rows = [r for r in old_rec['trace'] if r['snapshot']['turn_count'] == 1]
    same_rows = len(rows) == len(old_rows) and all(a['snapshot'] == b['snapshot'] and O.F.meaning_key(a['selected']) == O.F.meaning_key(b['selected'])
                                                   for a, b in zip(rows, old_rows))
    z, zo = np.load(d / 'observations.npz'), np.load(old_dir / 'observations.npz')
    k, ko = int((z['turns'] == 1).sum()), int((zo['turns'] == 1).sum())
    same_obs = k == ko and all(np.array_equal(z[x][:k], zo[x][:k]) for x in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain', 'labels'])
    return dict(rows=len(rows), old_rows=len(old_rows), rows_identical=same_rows, own_observations=k, observations_identical=same_obs)


def status_of_opponent_card(d, rec, step, code):
    """NPZ cards_ column 11 (STATUS_DISABLED | STATUS_FORBIDDEN) of the opponent's face-up monster `code` at the first own decision >= step."""
    z = np.load(d / 'observations.npz')
    i = int(np.searchsorted(z['steps'], step))
    if i >= len(z['steps']):
        return None
    s = rec['trace'][int(z['steps'][i])]['snapshot']
    phys = [c for c in s['cards'] if c['controller'] == 1 and c['location'] == 4 and c['code'] == code]
    if len(phys) != 1:
        return dict(step=int(z['steps'][i]), on_field=len(phys))
    cards = z['cards_'][i]
    rows = [r for r in cards if r[2] == 3 and r[3] == phys[0]['sequence'] + 1 and r[4] == 1 and CODES[((int(r[0]) << 8) | int(r[1])) - 1] == code]
    return dict(step=int(z['steps'][i]), zone=phys[0]['sequence'], status_disabled_or_forbidden=int(rows[0][11]) if len(rows) == 1 else None)


def chain_responses(rec, window_step):
    out, i = [], next(k for k, r in enumerate(rec['trace']) if r['step'] == window_step)
    for r in rec['trace'][i:]:
        if r['snapshot']['chain_size'] == 0:
            break
        if r['snapshot']['to_play'] == 0 and r['selected']['act'] == 'Activate':
            out.append(dict(step=r['step'], effect=label(r['selected']), who=r['who']))
    return out


def turn2_own(rec, phases):
    return [dict(step=r['step'], phase=r['snapshot']['phase'], action=label(r['selected']), who=r['who']) for r in rec['trace']
            if r['snapshot']['turn_count'] == 2 and r['snapshot']['to_play'] == 0 and r['snapshot']['phase'] in phases and r['selected']['act'] == 'Activate']


def turn3(rec):
    acts, attacks = [], []
    for r in rec['trace']:
        s = r['snapshot']
        if s['turn_count'] != 3 or s['to_play'] != 0:
            continue
        a = r['selected']
        if a['act'] in ('Activate', 'Summon', 'SpSummon', 'Set', 'MSet', 'SSet'):
            acts.append(label(a) if a['act'] == 'Activate' else f"{a['act']}:{name(a['code'])}")
        if 'Attack' in a['act']:
            attacks.append(f"{a['act']}:{name(a['code'])}")
    return dict(actions=acts, attacks=attacks)


def d1_vs_d0(a, b):
    """Same site and R: compare the D1 run (a) with the D0 run (b) from the Medius window on."""
    (ra, da, _, _), (rb, db, _, _) = load(a), load(b)
    wa, wb = probe_checkpoints(ra).get('medius_response_window'), probe_checkpoints(rb).get('medius_response_window')
    if not wa or not wb:
        return dict(window='not reached in one of the runs')
    ta, tb = ra['trace'], rb['trace']
    ia = next(i for i, r in enumerate(ta) if r['step'] == wa['step'])
    ib = next(i for i, r in enumerate(tb) if r['step'] == wb['step'])
    pa, pb = ta[ia:], tb[ib:]
    key = lambda r: (r['selected']['act'], r['selected']['code'], r['selected'].get('effect'))
    first_action_diff = next((x['step'] for x, y in zip(pa, pb) if key(x) != key(y)), None)
    cards = lambda s: {(c['controller'], c['location'], c['sequence'], c['code'], c['position']) for c in s['cards']}
    za, zb = np.load(da / 'observations.npz'), np.load(db / 'observations.npz')
    ka, kb = int(np.searchsorted(za['steps'], wa['step'])), int(np.searchsorted(zb['steps'], wb['step']))
    return dict(window_step=[wa['step'], wb['step']], rows_from_window=[len(pa), len(pb)],
                window_snapshot_equal=ta[ia]['snapshot'] == tb[ib]['snapshot'],
                window_cards_only_in_d1=[[name(x[3]), dict(controller=x[0], location=x[1], zone=x[2])] for x in sorted(cards(ta[ia]['snapshot']) - cards(tb[ib]['snapshot']))],
                window_cards_only_in_d0=[[name(x[3]), dict(controller=x[0], location=x[1], zone=x[2])] for x in sorted(cards(tb[ib]['snapshot']) - cards(ta[ia]['snapshot']))],
                first_action_difference_ignoring_zone_spec=first_action_diff,
                rows_identical_from_window=len(pa) == len(pb) and all(x['snapshot'] == y['snapshot'] and O.F.meaning_key(x['selected']) == O.F.meaning_key(y['selected']) for x, y in zip(pa, pb)),
                model_inputs_equal_from_window={k: bool(np.array_equal(za[k][ka:], zb[k][kb:])) for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain', 'labels']},
                game_over_equal=ra['game_over'] == rb['game_over'],
                turn3_end_equal=board(ra['checkpoints'].get('own_turn3_end')) == board(rb['checkpoints'].get('own_turn3_end')))


def main():
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    out = {}
    for site in O.SITES:
        old_rec, old_dir, _, _ = load(f'{site}-D0R0')
        for cond in ['D0R0', 'D0R1', 'D1R0', 'D1R1']:
            n = f'{site}-{cond}'
            rec, d, rep, source = load(n)
            cp = probe_checkpoints(rec)
            win, after = cp.get('medius_response_window'), cp.get('after_medius_chain')
            wrow = next((r for r in rec['trace'] if win and r['step'] == win['step']), None)
            iv = rec.get('interventions')
            if iv is None:  # reused: reconstruct the decision log the registered rules produce on this record
                reuse = json.loads((ROOT / 'reuse.json').read_text())['sites'][site]['conditions'][cond]
                iv = reuse['rule_log']
            out[n] = dict(
                source=source, record_sha256=sha(d / 'record.json'), observations_sha256=sha(d / 'observations.npz'),
                replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match', 'reproduced_steps', 'of']},
                turn1_vs_old=turn1_identity(rec, d, old_rec, old_dir) if source == 'new' else 'is the old record',
                interventions=[{k: x[k] for k in x if k not in ('legal',)} | dict(model_choice=label(dict(x['model_choice'], act=x['model_choice']['act'])))
                               for x in iv],
                draw_phase_own=turn2_own(rec, (1,)), standby_phase_own=turn2_own(rec, (2,)),
                at_opp_draw_phase_end=board(cp.get('opp_draw_phase_end')),
                medius_window=dict(step=win['step'], board=board(win), candidates=[label(a) for a in win['candidates'] if a['act'] == 'Activate'],
                                   executed=label(wrow['selected']), who=wrow['who']) if win else 'not reached',
                medius_chain_responses=chain_responses(rec, win['step']) if win else None,
                after_medius_chain=dict(step=after['step'], beast_arrived=after['beast_arrived'], medius_on_field=after['medius_on_field'],
                                        board=board(after)) if after else None,
                medius_status_after_chain=status_of_opponent_card(d, rec, after['step'], O.F.MEDIUS) if after and after['medius_on_field'] else None,
                opp_turn2_end=board(cp.get('opp_turn2_end')), turn3=turn3(rec), own_turn3_end=board(cp.get('own_turn3_end')),
                game_over=rec['game_over'], reason=rec['reason'], visible=len(rec['disclosed']), lua=rec['lua_errors'])
    # D1 vs D0 with the same R: are the executed own decisions after the draw phase the same sequence?
    same_after_draw = {f'{s}-R{r}': d1_vs_d0(f'{s}-D1R{r}', f'{s}-D0R{r}') for s in O.SITES for r in '01'}
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), reuse_sha256=sha(ROOT / 'reuse.json'),
                   engine=dict(used=len(starts), cap=O.LIMIT, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e))),
                   d1_vs_d0_from_medius_window=same_after_draw, conditions=out,
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'])
    for k, x in same_after_draw.items():
        print('D1vsD0', k, json.dumps(x, ensure_ascii=False))
    for n, c in out.items():
        print('=====', n, c['source'], 'turn1', c['turn1_vs_old'] if isinstance(c['turn1_vs_old'], str) else (c['turn1_vs_old']['rows_identical'], c['turn1_vs_old']['observations_identical']))
        print('  interventions', [(x['rule'], x['step'], x['executed'], x['model_choice']) for x in c['interventions']])
        print('  draw', [x['action'] for x in c['draw_phase_own']], '| standby', [x['action'] for x in c['standby_phase_own']])
        w = c['medius_window']
        print('  window', w if isinstance(w, str) else (w['step'], w['board']['monsters'], w['candidates'], '->', w['executed'], w['who']))
        print('  chain', c['medius_chain_responses'], '| after', {k: c['after_medius_chain'][k] for k in ['beast_arrived', 'medius_on_field']},
              'oppGY', c['after_medius_chain']['board']['opponent_graveyard'], '| status', c['medius_status_after_chain'])
        print('  T2end own', c['opp_turn2_end']['monsters'], 'ST', c['opp_turn2_end']['spell_trap'], 'hand', c['opp_turn2_end']['hand'], '| opp', c['opp_turn2_end']['opponent_monsters'])
        print('  T3', c['turn3']['attacks'], '| end', (c['own_turn3_end'] or {}).get('lp'), (c['own_turn3_end'] or {}).get('monsters'), (c['own_turn3_end'] or {}).get('hand'),
              '| game_over', c['game_over'], '| replay', c['replay']['mismatch'] is None and c['replay']['observations_match'], 'visible', c['visible'], 'lua', c['lua'])
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
