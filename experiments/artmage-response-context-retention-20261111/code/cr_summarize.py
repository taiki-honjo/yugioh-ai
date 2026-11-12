"""Offline summary of the context-retention study (no engine, no model): start model / no-retention candidate / this candidate."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
PREV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
HB = P / 'experiments/artmage-hand-branch-20261110'
BATCH = P / 'experiments/artmage-basic-openings-batch-20261110'
VOCAB = P / 'experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt'
MEDIUS, BEAST = 97556336, 23829452
B_NAMES = ['BV1-1cost-VARNISH-OM', 'BV1-2cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OM', 'BV1-2cost-VARNISH-OM']
C_NAMES = [f'Artmage-g{i:02d}-p0' for i in range(6, 12)] + [f'F{i}-V' for i in range(7)]
_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
CODES = [int(line.split()[0]) for line in VOCAB.read_text().splitlines()]
LABEL = {(74631897, 10011): '乙女②', (74631897, 10010): '乙女①', (27184601, 10011): 'ディアクトロス②', (27184601, 10010): 'ディアクトロス①',
         (34541940, 10011): 'ファインメルト③', (34541940, 10010): 'ファインメルト①', (97434754, 10011): 'リテラ③', (97434754, 10010): 'リテラ①',
         (23829452, 10010): '獄神獣②', (23829452, 10011): '獄神獣③', (31822037, 10011): '獄神使②', (31822037, 10010): '獄神使①',
         (23599634, 10010): '覚醒①', (23599634, 10011): '覚醒②', (97556336, 10011): 'メディウス②', (53589300, 10010): 'ネルヴァ②',
         (44654994, 10010): '奪還①', (23434538, 10010): '増殖するG', (59438930, 0): '幽鬼うさぎ', (60946049, 10010): 'グラフレア①',
         (60946049, 10011): 'グラフレア③', (17473466, 10012): 'ネルヴェド(P)①', (17473466, 10013): 'ネルヴェド②'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def name(c):
    return NAMES.get(c, str(c))


def names(cs):
    return [name(c) for c in cs]


def label(a):
    if a['act'] != 'Activate':
        return a['act'] + (f":{name(a['code'])}" if a.get('code') else '')
    return LABEL.get((a['code'], a.get('effect'))) or f"{name(a['code'])}:e{a.get('effect')}"


def is_medius_window(s):
    ch = s['public_chain']
    return (s['to_play'] == 0 and s['turn_count'] == 2 and s['turn_player'] == 1 and s['phase'] == 4 and s['message'] == 'select_chain'
            and len(ch) == 1 and ch[0]['player'] == 1 and ch[0]['code'] == MEDIUS and ch[0]['description'] == MEDIUS * 16)


def board(b):
    if not b:
        return None
    return dict(lp=[b['lp0'], b['lp1']], monsters=names(b['monsters']), spell_trap=names(b['spell_trap']), hand=names(b['hand']),
                opponent_monsters=names(b.get('opponent_monsters', [])), opponent_graveyard=names(b.get('opponent_graveyard', [])))


def snap_board(s):
    z = lambda c, loc: names(sorted(x['code'] for x in s['cards'] if x['controller'] == c and x['location'] == loc))
    return dict(monsters=z(0, 4), spell_trap=z(0, 8), hand=z(0, 2), opponent_monsters=z(1, 4), opponent_graveyard=z(1, 16))


def medius_analysis(rec, obs_path):
    """Window (first own select_chain with the opponent's Medius① alone), executed response, the chain result and the
    opponent Medius status column (STATUS_DISABLED | STATUS_FORBIDDEN) at the first own decision after the chain."""
    tr = rec['trace']
    w = next((i for i, r in enumerate(tr) if is_medius_window(r['snapshot'])), None)
    medius_act = next((i for i, r in enumerate(tr) if r['snapshot']['turn_count'] == 2 and r['snapshot']['to_play'] == 1
                       and r['selected']['act'] == 'Activate' and r['selected']['code'] == MEDIUS), None)
    out = dict(medius_activated=medius_act is not None, window=None)
    if medius_act is None:
        return out
    start = next((k for k in range(medius_act + 1, len(tr)) if tr[k]['snapshot']['chain_size'] >= 1), None)
    end = next((k for k in range(start, len(tr)) if tr[k]['snapshot']['chain_size'] == 0), None) if start is not None else None
    if w is not None:
        r = tr[w]
        out['window'] = dict(step=r['step'], board=snap_board(r['snapshot']), candidates=[label(a) for a in r['snapshot']['actions'] if a['act'] == 'Activate'],
                             executed=label(r['selected']), who=r['who'])
    responses = [dict(step=tr[k]['step'], effect=label(tr[k]['selected'])) for k in range(medius_act + 1, end or len(tr))
                 if tr[k]['snapshot']['to_play'] == 0 and tr[k]['selected']['act'] == 'Activate']
    after = tr[end]['snapshot'] if end is not None else None
    out.update(responses_in_chain=responses, after_step=tr[end]['step'] if end is not None else None,
               beast_arrived=after is not None and any(c['controller'] == 1 and c['location'] == 4 and c['code'] == BEAST for c in after['cards']),
               medius_on_field=after is not None and any(c['controller'] == 1 and c['location'] == 4 and c['code'] == MEDIUS for c in after['cards']),
               opponent_script=[e['what'] for e in rec['opponent_events']])
    if after is not None and out['medius_on_field'] and obs_path.exists():
        z = np.load(obs_path)
        i = int(np.searchsorted(z['steps'], tr[end]['step']))
        if i < len(z['steps']):
            s = next(r['snapshot'] for r in tr if r['step'] == int(z['steps'][i]))
            phys = [c for c in s['cards'] if c['controller'] == 1 and c['location'] == 4 and c['code'] == MEDIUS]
            rows = [x for x in z['cards_'][i] if len(phys) == 1 and x[2] == 3 and x[3] == phys[0]['sequence'] + 1 and x[4] == 1
                    and CODES[((int(x[0]) << 8) | int(x[1])) - 1] == MEDIUS]
            out['medius_status_after'] = dict(step=int(z['steps'][i]), value=int(rows[0][11]) if len(rows) == 1 else None)
    return out


def turn_actions(rec, turn, phases=None):
    return [dict(step=r['step'], phase=r['snapshot']['phase'], action=label(r['selected'])) for r in rec['trace']
            if r['snapshot']['turn_count'] == turn and r['snapshot']['to_play'] == 0 and r['selected']['act'] in ('Activate', 'Summon', 'SpSummon', 'Set', 'MSet', 'SSet')
            and (phases is None or r['snapshot']['phase'] in phases)]


def attacks(rec):
    return [f"{r['selected']['act']}:{name(r['selected']['code'])}" for r in rec['trace']
            if r['snapshot']['turn_count'] == 3 and r['snapshot']['to_play'] == 0 and 'Attack' in r['selected']['act']]


def first_own_divergence(new, old):
    a = [r for r in new['trace'] if r['snapshot']['to_play'] == 0]
    b = [r for r in old['trace'] if r['snapshot']['to_play'] == 0]
    key = lambda r: (r['selected']['act'], r['selected']['code'], r['selected'].get('effect'), r['selected'].get('spec'), r['selected'].get('place'), r['selected'].get('position'))
    for x, y in zip(a, b):
        if x['snapshot'] != y['snapshot'] or key(x) != key(y):
            return dict(step=x['step'], turn=x['snapshot']['turn_count'], phase=x['snapshot']['phase'], message=x['snapshot']['message'],
                        same_snapshot=x['snapshot'] == y['snapshot'], candidate=label(x['selected']), start_model=label(y['selected']))
    return None


def run_summary(rec, d, old=None):
    cp = rec['checkpoints']
    out = dict(reason=rec['reason'], failure=rec['failure'], game_over=rec['game_over'],
               t1=board(cp.get('own_turn1_end')), draw_standby=turn_actions(rec, 2, (1, 2)), medius=medius_analysis(rec, d / 'observations.npz'),
               t2_end=board(cp.get('opp_turn2_end')), t3_actions=turn_actions(rec, 3), t3_attacks=attacks(rec), t3_end=board(cp.get('own_turn3_end')),
               visible=len(rec['disclosed']), lua=rec['lua_errors'])
    if old is not None:
        out['first_divergence_from_start_model'] = first_own_divergence(rec, old)
    return out


def landing(free, script):
    t1 = free['checkpoints'].get('own_turn1_end')
    end = script['checkpoints'].get('own_turn_end')
    if not (t1 and end):
        return None
    miss = collections.Counter(end['monsters']) - collections.Counter(t1['monsters'])
    return dict(all_present=not miss, missing=names(sorted(miss.elements())))


def main():
    reg = read(ROOT / 'registration.json')
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    diags = [json.loads(l) for l in (ROOT / 'train/diagnostics.jsonl').read_text().splitlines()]
    upd = [json.loads(l) for l in (ROOT / 'train/updates.jsonl').read_text().splitlines()]
    prev_upd = [json.loads(l) for l in (PREV / 'train/updates.jsonl').read_text().splitlines()]
    result = read(ROOT / 'train/result.json')
    resp_upd = [u for u in upd if u['kind'] == 'response']
    first_low = {}
    for u in resp_upd:
        for n, ce in zip(u['sequences'], u['per_seq_ce']):
            if ce < np.log(2) and n not in first_low:
                first_low[n] = u['update']
    def kinds(rows, key):
        return {k: dict(mean=float(np.mean([u[key] for u in rows if u['kind'] == k])), max=float(np.max([u[key] for u in rows if u['kind'] == k])))
                for k in ['existing', 'pair', 'response']}
    training = dict(
        at={d['update']: dict(groups={g: (v['ce_pooled'], v['top1']) for g, v in d['groups'].items()},
                              responses={n: dict({k: r[k] for k in ['p_label', 'p_pass', 'argmax_is_label']}, options=[(label(o), round(o['p'], 6)) for o in r['options']])
                                         for n, r in d['responses'].items()},
                              context=d['context'], branches=d['branches']) for d in diags},
        updates=dict(n=len(upd), clipped=sum(1 for u in upd if u['clipped']), clipped_by_kind=dict(collections.Counter(u['kind'] for u in upd if u['clipped'])),
                     grad_norm_by_kind=kinds(upd, 'grad_norm'), response_terms=dict(
                         ce_first_last=[resp_upd[0]['ce'], resp_upd[-1]['ce']], kl_first_last=[resp_upd[0]['kl'], resp_upd[-1]['kl']],
                         kl_mean=float(np.mean([u['kl'] for u in resp_upd])), kl_max=float(np.max([u['kl'] for u in resp_upd])),
                         ce_grad_norm=dict(mean=float(np.mean([u['ce_grad_norm'] for u in resp_upd])), max=float(np.max([u['ce_grad_norm'] for u in resp_upd]))),
                         kl_grad_norm=dict(mean=float(np.mean([u['kl_grad_norm'] for u in resp_upd])), max=float(np.max([u['kl_grad_norm'] for u in resp_upd]))),
                         combined_grad_norm=dict(mean=float(np.mean([u['grad_norm'] for u in resp_upd])), max=float(np.max([u['grad_norm'] for u in resp_upd]))),
                         ce_below_ln2_first_update=first_low),
                     base_update_mean=float(np.mean([u['base_update'] for u in upd])), head_update_mean=float(np.mean([u['head_update'] for u in upd])),
                     previous=dict(clipped=sum(1 for u in prev_upd if u['clipped']), clipped_by_kind=dict(collections.Counter(u['kind'] for u in prev_upd if u['clipped'])),
                                   grad_norm_by_kind=kinds(prev_upd, 'grad_norm')),
                     same_draw_order_as_previous=[(u['kind'], u['sequences']) for u in upd] == [(u['kind'], u['sequences']) for u in prev_upd]),
        result={k: result[k] for k in ['status', 'updates_done', 'sampled', 'parameter_change_norm', 'both_parts_changed', 'clipped_updates', 'saved', 'seconds']},
        reference=read(ROOT / 'amendment-1.json')['reference'], cpu_check=read(ROOT / 'reference/cpu-check.json'))
    # A
    a = {}
    for site, (old_name, window) in [('S1', ('BV1-1cost-VARNISH-OM', 149)), ('S2', ('BV1-2cost-ASSAULT-OM', 169))]:
        d = ROOT / 'runs' / f'A-{site}'
        rec, rep = read(d / 'record.json'), read(d / 'replay.json')
        old = read(FLC / 'runs' / old_name / 'record.json')
        ag = rec['replayed_agreement']
        a[site] = dict(run_summary(rec, d), window_info=rec['window'], replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']},
                       replayed_prefix=dict(own_decisions=len(ag), candidate_would_differ=[dict(step=x['step'], turn=x['turn'], candidate=x['candidate'][:3], replayed=x['replayed'][:3]) for x in ag if not x['candidate_agrees']]),
                       start_model_same_history=run_summary(old, FLC / 'runs' / old_name),
                       no_retention=run_summary(read(PREV / 'runs' / f'A-{site}' / 'record.json'), PREV / 'runs' / f'A-{site}'),
                       no_retention_window=read(PREV / 'runs' / f'A-{site}' / 'record.json')['window'])
    # B
    b = {}
    for n in B_NAMES:
        d = ROOT / 'runs' / f'B-{n}'
        rec, rep = read(d / 'record.json'), read(d / 'replay.json')
        old = read(FLC / 'runs' / n / 'record.json')
        prev = read(PREV / 'runs' / f'B-{n}' / 'record.json')
        b[n] = dict(candidate=run_summary(rec, d, old), start_model=run_summary(old, FLC / 'runs' / n),
                    no_retention=run_summary(prev, PREV / 'runs' / f'B-{n}', old), no_retention_branch_model_choice=prev['branch']['model_choice'],
                    prefix=f"{rec['prefix_replayed']}/{rec['prefix_length']}", branch_executed=rec['branch']['executed'], branch_model_choice=rec['branch']['model_choice'],
                    replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']})
    # C
    c_rt1, c_f = {}, {}
    for n in C_NAMES:
        cand, base = read(ROOT / 'eval/retain/candidate' / f'{n}.json'), read(HB / 'eval/retain/candidate' / f'{n}.json')
        prevc = read(PREV / 'eval/retain/candidate' / f'{n}.json')
        if n.startswith('Artmage-g'):
            same = len(base['trace']) == len(cand['trace']) and all((x['selected']['act'], x['selected']['code'], x['selected'].get('effect')) ==
                                                                    (y['selected']['act'], y['selected']['code'], y['selected'].get('effect')) for x, y in zip(base['trace'], cand['trace']))
            c_rt1[n] = dict(start=base['passed'], no_retention=prevc['passed'], candidate=cand['passed'], same_action_sequence=same, visible=len(cand['disclosed']), lua=cand['lua_errors'])
        else:
            script = read(BATCH / 'validation' / f'{n}.json')
            c_f[n] = dict(start=dict(landing=landing(base, script), t1=board(base['checkpoints'].get('own_turn1_end')), lp1_t3=(base['checkpoints'].get('own_turn3_end') or {}).get('lp1'),
                                     game_over=base['game_over'], medius=medius_analysis(base, Path('/nonexistent'))['medius_activated']),
                          candidate=dict(landing=landing(cand, script), t1=board(cand['checkpoints'].get('own_turn1_end')), lp1_t3=(cand['checkpoints'].get('own_turn3_end') or {}).get('lp1'),
                                         game_over=cand['game_over'], t3_attacks=attacks(cand), visible=len(cand['disclosed']), lua=cand['lua_errors'], failure=cand['failure']),
                          no_retention=dict(landing=landing(prevc, script), lp1_t3=(prevc['checkpoints'].get('own_turn3_end') or {}).get('lp1')),
                          first_divergence=first_own_divergence(cand, base))
    start_ok = {n for n, x in c_rt1.items() if x['start']} | {n for n, x in c_f.items() if (x['start']['landing'] or {}).get('all_present')}
    prev_ok = {n for n, x in c_rt1.items() if x['no_retention']} | {n for n, x in c_f.items() if (x['no_retention']['landing'] or {}).get('all_present')}
    cand_ok = {n for n, x in c_rt1.items() if x['candidate']} | {n for n, x in c_f.items() if (x['candidate']['landing'] or {}).get('all_present')}
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'), manifest_sha256=sha(PREV / 'data/manifest.json'),
                   engine=dict(used=len(starts), cap=28, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e)),
                               by_kind=dict(collections.Counter(e['purpose'].split()[0] for e in starts))),
                   training=training, A=a, B=b, C=dict(rt1=c_rt1, f=c_f, start_successes=sorted(start_ok), candidate_successes=sorted(cand_ok),
                                                         lost=sorted(start_ok - cand_ok), new=sorted(cand_ok - start_ok),
                                                         no_retention_successes=sorted(prev_ok), no_retention_lost=sorted(start_ok - prev_ok), no_retention_new=sorted(prev_ok - start_ok)),
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'])
    print('TRAIN', json.dumps(training['updates'], ensure_ascii=False, default=float))
    for n, c in training['at'][256]['context'].items():
        print('CONTEXT', n, c['preserved_steps'], 'kl', round(c['kl_mean'], 6), round(c['kl_max'], 6), 'changed', [(x['step'], x['start']['name'], x['now']['name'], round(x['p_start_of_start_choice'], 3), round(x['p_now_of_start_choice'], 3)) for x in c['changed']])
    for site, x in a.items():
        m, sm = x['medius'], x['start_model_same_history']['medius']
        print('A', site, 'window', x['window_info']['candidate_choice'], '| start', x['window_info']['start_model_choice_in_old_record'],
              '| chain', m['responses_in_chain'], 'beast', m['beast_arrived'], 'medius', m['medius_on_field'], m.get('medius_status_after'),
              '| t2', x['t2_end']['monsters'], 'opp', x['t2_end']['opponent_monsters'], '| t3', x['t3_attacks'], (x['t3_end'] or {}).get('lp'), x['game_over'],
              '| start t3', (x['start_model_same_history']['t3_end'] or {}).get('lp'), x['start_model_same_history']['game_over'],
              '| noret window', x['no_retention_window']['candidate_choice'], 'noret t3', (x['no_retention']['t3_end'] or {}).get('lp'),
              '| prefix differ', len(x['replayed_prefix']['candidate_would_differ']), 'of', x['replayed_prefix']['own_decisions'])
    for n, x in b.items():
        cnd, st = x['candidate'], x['start_model']
        print('B', n, '| div', cnd['first_divergence_from_start_model'])
        for lab, y in [('cand', cnd), ('noret', x['no_retention']), ('start', st)]:
            m = y['medius']
            print('   ', lab, 't1', y['t1']['monsters'], y['t1']['spell_trap'], 'hand', y['t1']['hand'])
            print('   ', lab, 'draw/standby', [z['action'] for z in y['draw_standby']])
            print('   ', lab, 'window', m['window'] and (m['window']['board']['monsters'], m['window']['candidates'], '->', m['window']['executed']),
                  '| chain', [z['effect'] for z in m['responses_in_chain']], 'beast', m['beast_arrived'], 'medius', m['medius_on_field'], m.get('medius_status_after'))
            print('   ', lab, 't2', y['t2_end']['monsters'], 'hand', y['t2_end']['hand'], 'opp', y['t2_end']['opponent_monsters'], '| t3', y['t3_attacks'], (y['t3_end'] or {}).get('lp'), y['game_over'])
    for n, x in c_rt1.items():
        print('C', n, x)
    for n, x in c_f.items():
        print('C', n, 'start', x['start']['landing'], x['start']['lp1_t3'], '| noret', x['no_retention']['landing'], x['no_retention']['lp1_t3'], '| cand', x['candidate']['landing'], x['candidate']['lp1_t3'], x['candidate']['game_over'],
              '| div', x['first_divergence'] and (x['first_divergence']['step'], x['first_divergence']['turn'], x['first_divergence']['candidate'], x['first_divergence']['start_model']))
    print('C lost', summary['C']['lost'], 'new', summary['C']['new'], '| no-retention lost', summary['C']['no_retention_lost'], 'new', summary['C']['no_retention_new'])
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
