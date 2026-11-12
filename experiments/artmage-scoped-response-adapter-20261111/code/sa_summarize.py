"""Offline summary of the scoped response adapter (no engine, no model): start model / no-retention / retention / this candidate."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111')
PREV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
RET = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
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
    out = dict(adapter=adapter_rows(rec), reason=rec['reason'], failure=rec['failure'], game_over=rec['game_over'],
               t1=board(cp.get('own_turn1_end')), draw_standby=turn_actions(rec, 2, (1, 2)), medius=medius_analysis(rec, d / 'observations.npz'),
               t2_end=board(cp.get('opp_turn2_end')), t3_actions=turn_actions(rec, 3), t3_attacks=attacks(rec), t3_end=board(cp.get('own_turn3_end')),
               visible=len(rec['disclosed']), lua=rec['lua_errors'])
    if old is not None:
        out['first_divergence_from_start_model'] = first_own_divergence(rec, old)
    return out


def adapter_rows(rec):
    """Adapter activations mapped to the record: call k = the k-th own decision of the trace."""
    pol = rec.get('policy')
    if not pol:
        return None
    own = [r for r in rec['trace'] if r['snapshot']['to_play'] == 0]
    out = []
    for x in pol['adapter_active']:
        r = own[x['call']]
        acts = {a['index']: a for a in r['snapshot']['actions']}
        out.append(dict(step=r['step'], chain=[NAMES.get(c['code'], c['code']) for c in r['snapshot']['public_chain']], frozen=label(acts[x['frozen_argmax']]),
                        final=label(acts[x['final_argmax']]), executed=label(r['selected']), changed=x['frozen_argmax'] != x['final_argmax'],
                        max_abs_delta=x['max_abs_delta_legal']))
    return dict(decisions=pol['decisions'], hashes=dict(base=pol['base_sha256'][:12], head=pol['head_sha256'][:12], adapter=pol['adapter_sha256'][:12]),
                type=pol['type'], version=pol['version'], active=out)


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
    upd = [json.loads(l) for l in (ROOT / 'train/updates.jsonl').read_text().splitlines()]
    result = read(ROOT / 'train/result.json')
    prep = read(ROOT / 'cache/prepare.json')
    training = dict(result={k: result[k] for k in ['status', 'updates_done', 'seconds', 'config', 'saved', 'targets', 'clipped_updates', 'first_update_ce_below_ln2']},
                    post_training=dict(result['post_training'], position_actions_on_target=result['post_training']['position_actions_on_target']),
                    updates=dict(loss_first_last=[upd[0]['loss'], upd[-1]['loss']], grad_norm_first_last=[upd[0]['grad_norm'], upd[-1]['grad_norm']],
                                 grad_norm_max=float(max(u['grad_norm'] for u in upd)), update_norm_mean=float(np.mean([u['update_norm'] for u in upd])),
                                 max_abs_delta_last=upd[-1]['max_abs_delta']),
                    prechecks={k: prep[k] for k in ['zero_adapter', 'synthetic_adapter', 'factory_refusals', 'gate_vs_snapshot', 'old40_gated_decisions', 'old40_decisions', 'targets', 'all_prechecks_passed']})
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
                       no_retention_window=read(PREV / 'runs' / f'A-{site}' / 'record.json')['window'],
                       retention=run_summary(read(RET / 'runs' / f'A-{site}' / 'record.json'), RET / 'runs' / f'A-{site}'),
                       retention_window=read(RET / 'runs' / f'A-{site}' / 'record.json')['window'],
                       first_divergence_from_start_model=first_own_divergence(rec, old))
    # B
    b = {}
    for n in B_NAMES:
        d = ROOT / 'runs' / f'B-{n}'
        rec, rep = read(d / 'record.json'), read(d / 'replay.json')
        old = read(FLC / 'runs' / n / 'record.json')
        prev = read(PREV / 'runs' / f'B-{n}' / 'record.json')
        b[n] = dict(candidate=run_summary(rec, d, old), start_model=run_summary(old, FLC / 'runs' / n),
                    no_retention=run_summary(prev, PREV / 'runs' / f'B-{n}', old), no_retention_branch_model_choice=prev['branch']['model_choice'],
                    retention=run_summary(read(RET / 'runs' / f'B-{n}' / 'record.json'), RET / 'runs' / f'B-{n}', old),
                    prefix=f"{rec['prefix_replayed']}/{rec['prefix_length']}", branch_executed=rec['branch']['executed'], branch_model_choice=rec['branch']['model_choice'],
                    replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']})
    # C
    c_rt1, c_f = {}, {}
    for n in C_NAMES:
        cand, base = read(ROOT / 'eval/retain/candidate' / f'{n}.json'), read(HB / 'eval/retain/candidate' / f'{n}.json')
        prevc = read(PREV / 'eval/retain/candidate' / f'{n}.json')
        retc = read(RET / 'eval/retain/candidate' / f'{n}.json')
        side = read(ROOT / 'eval/retain/candidate' / f'{n}.policy.json')['policy']
        identical = len(base['trace']) == len(cand['trace']) and all(x['snapshot'] == y['snapshot'] and (x['selected']['act'], x['selected']['code'], x['selected'].get('effect'), x['selected'].get('spec'))
                                                                     == (y['selected']['act'], y['selected']['code'], y['selected'].get('effect'), y['selected'].get('spec')) for x, y in zip(base['trace'], cand['trace']))
        if n.startswith('Artmage-g'):
            same = len(base['trace']) == len(cand['trace']) and all((x['selected']['act'], x['selected']['code'], x['selected'].get('effect')) ==
                                                                    (y['selected']['act'], y['selected']['code'], y['selected'].get('effect')) for x, y in zip(base['trace'], cand['trace']))
            c_rt1[n] = dict(start=base['passed'], no_retention=prevc['passed'], retention=retc['passed'], candidate=cand['passed'], same_action_sequence=same,
                            identical_to_start=identical, adapter_active=len(side['adapter_active']), decisions=side['decisions'], visible=len(cand['disclosed']), lua=cand['lua_errors'])
        else:
            script = read(BATCH / 'validation' / f'{n}.json')
            c_f[n] = dict(start=dict(landing=landing(base, script), t1=board(base['checkpoints'].get('own_turn1_end')), lp1_t3=(base['checkpoints'].get('own_turn3_end') or {}).get('lp1'),
                                     game_over=base['game_over'], medius=medius_analysis(base, Path('/nonexistent'))['medius_activated']),
                          candidate=dict(landing=landing(cand, script), t1=board(cand['checkpoints'].get('own_turn1_end')), lp1_t3=(cand['checkpoints'].get('own_turn3_end') or {}).get('lp1'),
                                         game_over=cand['game_over'], t3_attacks=attacks(cand), visible=len(cand['disclosed']), lua=cand['lua_errors'], failure=cand['failure']),
                          no_retention=dict(landing=landing(prevc, script), lp1_t3=(prevc['checkpoints'].get('own_turn3_end') or {}).get('lp1')),
                          retention=dict(landing=landing(retc, script), lp1_t3=(retc['checkpoints'].get('own_turn3_end') or {}).get('lp1')),
                          identical_to_start=identical, adapter_active=len(side['adapter_active']), decisions=side['decisions'],
                          first_divergence=first_own_divergence(cand, base))
    start_ok = {n for n, x in c_rt1.items() if x['start']} | {n for n, x in c_f.items() if (x['start']['landing'] or {}).get('all_present')}
    prev_ok = {n for n, x in c_rt1.items() if x['no_retention']} | {n for n, x in c_f.items() if (x['no_retention']['landing'] or {}).get('all_present')}
    ret_ok = {n for n, x in c_rt1.items() if x['retention']} | {n for n, x in c_f.items() if (x['retention']['landing'] or {}).get('all_present')}
    cand_ok = {n for n, x in c_rt1.items() if x['candidate']} | {n for n, x in c_f.items() if (x['candidate']['landing'] or {}).get('all_present')}
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'), manifest_sha256=sha(PREV / 'data/manifest.json'),
                   candidate_manifest_sha256=sha(ROOT / 'train/candidate/manifest.json'),
                   engine=dict(used=len(starts), cap=28, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e)),
                               by_kind=dict(collections.Counter(e['purpose'].split()[0] for e in starts))),
                   training=training, A=a, B=b, C=dict(rt1=c_rt1, f=c_f, start_successes=sorted(start_ok), candidate_successes=sorted(cand_ok),
                                                         lost=sorted(start_ok - cand_ok), new=sorted(cand_ok - start_ok),
                                                         no_retention_successes=sorted(prev_ok), no_retention_lost=sorted(start_ok - prev_ok), no_retention_new=sorted(prev_ok - start_ok),
                                                         retention_lost=sorted(start_ok - ret_ok), retention_new=sorted(ret_ok - start_ok)),
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'])
    print('TRAIN', json.dumps(dict(training['updates'], clipped=training['result']['clipped_updates'], ce_below=training['result']['first_update_ce_below_ln2'],
                                   off=training['post_training']['comparison']['off_target_logit_differences'], pos_off=training['post_training']['position_actions_off_target_differences']), default=float))
    for site, x in a.items():
        m, sm = x['medius'], x['start_model_same_history']['medius']
        print('A', site, 'window', x['window_info']['candidate_choice'], '| start', x['window_info']['start_model_choice_in_old_record'],
              '| chain', m['responses_in_chain'], 'beast', m['beast_arrived'], 'medius', m['medius_on_field'], m.get('medius_status_after'),
              '| t2', x['t2_end']['monsters'], 'opp', x['t2_end']['opponent_monsters'], '| t3', x['t3_attacks'], (x['t3_end'] or {}).get('lp'), x['game_over'],
              '| start t3', (x['start_model_same_history']['t3_end'] or {}).get('lp'), x['start_model_same_history']['game_over'],
              '| noret window', x['no_retention_window']['candidate_choice'], 'noret t3', (x['no_retention']['t3_end'] or {}).get('lp'),
              '| ret t3', (x['retention']['t3_end'] or {}).get('lp'), '| adapter', x['adapter'], '| div', x['first_divergence_from_start_model'],
              '| prefix differ', len(x['replayed_prefix']['candidate_would_differ']), 'of', x['replayed_prefix']['own_decisions'])
    for n, x in b.items():
        cnd, st = x['candidate'], x['start_model']
        print('B', n, '| div', cnd['first_divergence_from_start_model'])
        print('   adapter', cnd['adapter'])
        for lab, y in [('cand', cnd), ('start', st)]:
            m = y['medius']
            print('   ', lab, 't1', y['t1']['monsters'], y['t1']['spell_trap'], 'hand', y['t1']['hand'])
            print('   ', lab, 'draw/standby', [z['action'] for z in y['draw_standby']])
            print('   ', lab, 'window', m['window'] and (m['window']['board']['monsters'], m['window']['candidates'], '->', m['window']['executed']),
                  '| chain', [z['effect'] for z in m['responses_in_chain']], 'beast', m['beast_arrived'], 'medius', m['medius_on_field'], m.get('medius_status_after'))
            print('   ', lab, 't2', y['t2_end']['monsters'], 'hand', y['t2_end']['hand'], 'opp', y['t2_end']['opponent_monsters'], '| t3', y['t3_attacks'], (y['t3_end'] or {}).get('lp'), y['game_over'])
    for n, x in c_rt1.items():
        print('C', n, x)
    for n, x in c_f.items():
        print('C', n, 'identical', x['identical_to_start'], 'active', x['adapter_active'], 'start', (x['start']['landing'] or {}).get('all_present'), x['start']['lp1_t3'], '| noret', (x['no_retention']['landing'] or {}).get('all_present'), '| ret', (x['retention']['landing'] or {}).get('all_present'), '| cand', x['candidate']['landing'], x['candidate']['lp1_t3'], x['candidate']['game_over'],
              '| div', x['first_divergence'] and (x['first_divergence']['step'], x['first_divergence']['turn'], x['first_divergence']['candidate'], x['first_divergence']['start_model']))
    print('C lost', summary['C']['lost'], 'new', summary['C']['new'], '| no-retention lost', summary['C']['no_retention_lost'], 'new', summary['C']['no_retention_new'], '| retention lost', summary['C']['retention_lost'], 'new', summary['C']['retention_new'])
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
