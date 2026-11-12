"""Offline summary of the functional comparison (no engine): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import flc_runs as F  # noqa: E402

_db = sqlite3.connect(P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb')
NAMES = {c: n for c, n in _db.execute('select id,name from texts')}
RACE = {c: r for c, r in _db.execute('select id,race from datas')}
KEY_EFFECTS = {F.DIAKTOROS: 'ディアクトロス', F.IMPASTO: '奪還', F.FINEMELT: 'ファインメルト', F.GRAPH: 'グラフレア', 97434754: 'リテラ', F.MAIDEN: '知恵の乙女',
               F.AWAKENING: '覚醒', F.SHI: '獄神使', F.ASSAULT: '襲撃', F.BEAST: '獄神獣', F.MEDIUS: 'メディウス', 74733322: '学都', 4993187: 'ファンシーボール', 29301450: 'リトルナイト'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(c):
    return NAMES.get(c, str(c))


def names(cs):
    return [name(c) for c in cs] if cs is not None else None


def board(b):
    if not b:
        return None
    races = {RACE.get(c, 0) for c in b['monsters_face_up']}
    return dict(monsters=names(b['monsters']), races_static=len(races), spell_trap_face_up=names(b['spell_trap_face_up']),
                spell_trap_set=names(sorted((collections.Counter(b['spell_trap']) - collections.Counter(b['spell_trap_face_up'])).elements())),
                hand=names(b['hand']), graveyard=names(b['graveyard']), banished=names(b['banished']), extra_face_up=names(b['extra_face_up']),
                lp0=b['lp0'], lp1=b['lp1'], opponent_monsters=names(b['opponent_monsters']), opponent_spell_trap=names(b['opponent_spell_trap']),
                opponent_graveyard=names(b['opponent_graveyard']), opponent_banished=names(b['opponent_banished']))


def chains(trace, turn):
    """Group turn rows into chains: each chain from its first activation (chain_size goes 0 -> 1) to chain_size 0 again."""
    out, cur = [], None
    for i, r in enumerate(trace):
        s = r['snapshot']
        if s['turn_count'] != turn:
            continue
        if cur is not None and s['chain_size'] == 0 and i > cur['start']:
            cur['end'] = i
            out.append(cur)
            cur = None
        if r['selected']['act'] == 'Activate' and s['chain_size'] == 0 and cur is None and s['message'] in ('select_idlecmd', 'select_chain', 'select_effectyn'):
            cur = dict(start=i)
    return out


def chain_end(tr, act_i):
    """Index of the first row after the chain started by the activation at act_i has fully resolved (chain size back to 0)."""
    start = next((k for k in range(act_i + 1, len(tr)) if tr[k]['snapshot']['chain_size'] >= 1), None)
    if start is None:
        return None
    return next((k for k in range(start, len(tr)) if tr[k]['snapshot']['chain_size'] == 0), None)


def own(s, loc):
    return sorted(c['code'] for c in s['cards'] if c['controller'] == 0 and c['location'] == loc)


def opp(s, loc):
    return sorted(c['code'] for c in s['cards'] if c['controller'] == 1 and c['location'] == loc)


# effect labels: the engine's effect description index (10010 + n) is the card text string str(n+1);
# the circled number is taken from the card text that string describes (cards.cdb, checked by hand 2026-11-11)
EFFECT_LABEL = {(74631897, 10010): '乙女①(セット)', (74631897, 10011): '乙女②(相手ターン融合)', (27184601, 10010): 'ディアクトロス①(表示形式)', (27184601, 10011): 'ディアクトロス②(無効・破壊)',
                (34541940, 10010): 'ファインメルト①(特殊召喚・ドロー)', (34541940, 10011): 'ファインメルト③(相手モンスター効果無効・攻撃力半分)',
                (97434754, 10010): 'リテラ①(特殊召喚・回収)', (97434754, 10011): 'リテラ③(手札・墓地から特殊召喚、自身を手札へ)',
                (60946049, 10010): 'グラフレア①(特殊召喚・魔法セット)', (60946049, 10011): 'グラフレア③(魔法罠破壊)', (23829452, 10010): '獄神獣②(融合召喚)', (23829452, 10011): '獄神獣③(魔法罠サーチ)',
                (31822037, 10010): '獄神使①(サーチ)', (31822037, 10011): '獄神使②(L召喚)', (23599634, 10010): '覚醒①(デッキから特殊召喚)', (23599634, 10011): '覚醒②(墓地から除外・戻して破壊)',
                (97556336, 10010): 'メディウス①', (97556336, 10011): 'メディウス②(墓地から特殊召喚)', (53589300, 10010): 'ネルヴァ②(全破壊に変更)',
                (44654994, 10010): '奪還①(無効・破壊)', (23434538, 10010): '増殖するG', (59438930, 0): '幽鬼うさぎ'}


def effect_label(code, effect):
    return EFFECT_LABEL.get((code, effect)) or f"{NAMES.get(code, code)}:e{effect}"


def response_windows(d):
    """Every own chain/yes-no window from the opponent's first scripted action until the opponent leaves its main phase (turn 2)."""
    tr = d['trace']
    start = next((i for i, t in enumerate(tr) if t['snapshot']['turn_count'] == 2 and t['snapshot']['to_play'] == 1 and t['selected']['act'] in ('Summon', 'Activate')), None)
    out = []
    if start is None:
        return out
    for t in tr[start:]:
        s = t['snapshot']
        if s['turn_count'] != 2:
            break
        if s['to_play'] == 0 and s['message'] in ('select_chain', 'select_effectyn', 'select_yesno'):
            sel = t['selected']
            out.append(dict(step=t['step'], chain=[NAMES.get(c['code'], c['code']) for c in s['public_chain']],
                            offered=[effect_label(a['code'], a.get('effect')) + (f"[{a.get('spec')}]" if a.get('spec') else '') for a in s['actions'] if a['act'] == 'Activate'],
                            chosen=effect_label(sel['code'], sel.get('effect')) if sel['act'] == 'Activate' else sel['act']))
        if s['to_play'] == 1 and s['phase'] == 4 and t['selected']['act'] == 'None' and s['message'] == 'select_idlecmd':
            break
    return out


def turn2_analysis(d):
    tr = d['trace']
    rows2 = [r for r in tr if r['snapshot']['turn_count'] == 2]
    offered, used = collections.Counter(), []
    for i, r in enumerate(tr):
        s = r['snapshot']
        if s['turn_count'] != 2 or s['to_play'] != 0:
            continue
        for a in s['actions']:
            if a['act'] == 'Activate':
                offered[(a['code'], a.get('effect'))] += 1
        a = r['selected']
        if a['act'] in ('Activate', 'SpSummon'):
            j = chain_end(tr, i)
            before, after = s, (tr[j]['snapshot'] if j is not None else None)
            used.append(dict(step=r['step'], phase=s['phase'], card=name(a['code']), effect=a.get('effect'),
                             chain_at_activation=[name(c['code']) for c in s['public_chain']],
                             own_monsters_after=names(own(after, 4)) if after else None, opponent_monsters_after=names(opp(after, 4)) if after else None,
                             opponent_graveyard_after=names(opp(after, 16)) if after else None, own_hand_after=names(own(after, 2)) if after else None))
    used_keys = {(r['selected']['code'], r['selected'].get('effect')) for r in rows2 if r['snapshot']['to_play'] == 0 and r['selected']['act'] == 'Activate'}
    unused = [dict(card=name(c), effect=e, windows=n) for (c, e), n in sorted(offered.items()) if (c, e) not in used_keys]
    t1 = d['checkpoints'].get('own_turn1_end') or {}
    on_field = set(t1.get('monsters', [])) | set(t1.get('spell_trap', []))
    offered_cards = {c for c, _ in offered}
    not_offered = [KEY_EFFECTS[c] for c in sorted(on_field) if c in KEY_EFFECTS and c not in offered_cards]
    # opponent action outcome
    ev = d['opponent_events']
    out = dict(script=[e['what'] for e in ev])
    if d['opp'] == 'OM':
        act_i = next((i for i, r in enumerate(tr) if r['snapshot']['to_play'] == 1 and r['selected']['act'] == 'Activate' and r['selected']['code'] == F.MEDIUS
                      and r['snapshot']['turn_count'] == 2), None)
        if act_i is None:
            out['medius_effect'] = 'not activated'
        else:
            end = chain_end(tr, act_i)
            responses = [dict(card=name(tr[k]['selected']['code']), effect=tr[k]['selected'].get('effect')) for k in range(act_i + 1, end or len(tr))
                         if tr[k]['snapshot']['to_play'] == 0 and tr[k]['selected']['act'] == 'Activate']
            after = tr[end]['snapshot'] if end is not None else None
            arrived = after is not None and F.BEAST in opp(after, 4)
            out.update(medius_effect='activated', own_responses_in_chain=responses, beast_arrived=arrived,
                       medius_on_field_after=after is not None and F.MEDIUS in opp(after, 4),
                       opponent_field_after=names(opp(after, 4)) if after else None, opponent_graveyard_after=names(opp(after, 16)) if after else None,
                       opponent_banished_after=names(opp(after, 32)) if after else None)
    else:
        act_i = next((i for i, r in enumerate(tr) if r['snapshot']['to_play'] == 1 and r['selected']['act'] == 'Activate' and r['selected']['code'] == F.INFINITE), None)
        tgt = next((e for e in ev if e['what'] == '無限泡影 target'), None)
        if act_i is None:
            out['infinite'] = 'not activated'
        else:
            end = chain_end(tr, act_i)
            responses = [dict(card=name(tr[k]['selected']['code']), effect=tr[k]['selected'].get('effect')) for k in range(act_i + 1, end or len(tr))
                         if tr[k]['snapshot']['to_play'] == 0 and tr[k]['selected']['act'] == 'Activate']
            after = tr[end]['snapshot'] if end is not None else None
            target = tgt['chosen'] if tgt else None
            # track the targeted card by its zone, not by its name (same-name copies can be on the field)
            pick = next((r for r in tr if r['step'] == tgt['step']), None) if tgt else None
            spec = (pick or {}).get('selected', {}).get('spec', '') if pick else ''
            seq = int(spec[2:]) - 1 if spec.startswith('om') and spec[2:].isdigit() else None
            def at_zone(snap):
                return next((c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 4 and c['sequence'] == seq), None)
            on_field = after is not None and seq is not None and at_zone(after) == target
            left = None
            if seq is not None:
                for k in range(act_i, len(tr)):
                    if tr[k]['snapshot']['turn_count'] != 2:
                        break
                    if at_zone(tr[k]['snapshot']) != target:
                        mover = tr[k - 1]
                        left = dict(step=tr[k]['step'], after_own_action=dict(act=mover['selected']['act'], card=name(mover['selected']['code']) if mover['selected']['code'] else '',
                                                                               effect=mover['selected'].get('effect'), spec=mover['selected'].get('spec'), who=mover['who']),
                                    chain_at_that_point=[name(c['code']) for c in tr[k - 1]['snapshot']['public_chain']],
                                    target_location_after=next((c['location'] for c in tr[k]['snapshot']['cards'] if c['controller'] == 0 and c['code'] == target and c['location'] != 4), None))
                        break
            stop = left['step'] if left else None
            later_offered = sorted({(name(a['code']), a.get('effect')) for r in tr[end or 0:] if r['snapshot']['turn_count'] == 2 and r['snapshot']['to_play'] == 0
                                    and (stop is None or r['step'] < stop)
                                    for a in r['snapshot']['actions'] if a['act'] == 'Activate' and a['code'] == target and a.get('spec') == f'm{seq + 1}'}) if end and seq is not None else []
            same_name_others_after = sum(1 for c in (after or {}).get('cards', []) if c['controller'] == 0 and c['location'] == 4 and c['code'] == target and c['sequence'] != seq)
            out.update(infinite='activated', target=name(target) if target else None, target_zone=f'own monster zone seq {seq}' if seq is not None else None,
                       target_legal=[name(c) for c, _ in (tgt or {}).get('legal', [])],
                       own_responses_in_chain=responses, target_on_field_after_chain=on_field, same_name_copies_elsewhere_after_chain=same_name_others_after,
                       infinite_in_opponent_graveyard_after=after is not None and F.INFINITE in opp(after, 16),
                       target_left_field_in_turn2=left,
                       target_effect_offered_later_in_turn2_same_zone=later_offered,
                       negation_state='not observable in the snapshot (no disabled flag); indirect: effects of the SAME zone card offered later in turn 2 listed')
    return dict(offered_unused=unused, used=used, key_cards_on_field_never_offered=not_offered, opponent_action=out, response_windows=response_windows(d))


def turn_used(trace, turn):
    out = []
    for r in trace:
        s = r['snapshot']
        if s['turn_count'] == turn and s['to_play'] == 0 and (r['selected']['act'] in ('Activate', 'SpSummon', 'Summon', 'Set', 'MSet', 'SSet') or 'Attack' in r['selected']['act']):
            out.append(f"{r['selected']['act']}:{name(r['selected']['code'])}")
    return out


def prefix_identity():
    """Inputs before the branch must be identical across the 4 runs of a hand (2 branches x 2 opponent scripts)."""
    out = {}
    for cost in ['1cost', '2cost']:
        arrays = {}
        for n in [x for x in F.conditions() if f'-{cost}-' in x]:
            d = read(ROOT / 'runs' / n / 'record.json')
            z = np.load(ROOT / 'runs' / n / 'observations.npz')
            k = int(np.searchsorted(z['steps'], d['branch']['step']))  # own decisions before the branch prompt (+ the prompt itself)
            arrays[n] = {key: z[key][:k + 1] for key in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain']}
            arrays[n]['labels'] = z['labels'][:k]
        names_ = list(arrays)
        ref = arrays[names_[0]]
        out[cost] = dict(runs=names_, decisions_compared=len(ref['labels']) + 1,
                         identical=all(all(np.array_equal(ref[key], arrays[n][key]) for key in ref) for n in names_[1:]))
    return out


def main():
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    rows = {}
    for n in F.conditions():
        d, rep = read(ROOT / 'runs' / n / 'record.json'), read(ROOT / 'runs' / n / 'replay.json')
        cp = d['checkpoints']
        hand0 = collections.Counter(d['hand'])
        t1 = cp.get('own_turn1_end')
        rows[n] = dict(cost=d['cost'], branch=d['branch_choice'], opp=d['opp'], model_would_choose=d['branch']['model_choice'],
                       prefix=f"{d['prefix_replayed']}/{d['prefix_length']}", prefix_model_agrees=sum(x['model_agrees'] for x in d['prefix_log']),
                       t1=board(t1), t1_used=turn_used(d['trace'], 1),
                       t1_hand_spent=names(sorted((hand0 - collections.Counter(t1['hand'])).elements())) if t1 else None,
                       t2=turn2_analysis(d), t2_end=board(cp.get('opp_turn2_end')), t3_used=turn_used(d['trace'], 3), t3_end=board(cp.get('own_turn3_end')),
                       game_over=d['game_over'], reason=d['reason'], visible=len(d['disclosed']), lua=d['lua_errors'],
                       replay=dict(steps=f"{rep['reproduced_steps']}/{rep['of']}", mismatch=rep['mismatch'], final_equal=rep['final_equal'],
                                   checkpoints_equal=rep['checkpoints_equal'], game_over_equal=rep['game_over_equal'], observations_match=rep['observations_match']))
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'),
                   engine=dict(used=sum(1 for r in ledger if r['event'] == 'start'), cap=17,
                               statuses=dict(collections.Counter(r['status'] for r in ledger if r['event'] == 'finish'))),
                   prefix_identity=prefix_identity(), conditions=rows, code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print('ENGINE', summary['engine'], 'PREFIX', summary['prefix_identity'])
    for n, r in rows.items():
        t1, t2, t3 = r['t1'] or {}, r['t2'], r['t3_end'] or {}
        print('=' * 8, n, '| model would choose', r['model_would_choose'], '| prefix', r['prefix'], r['prefix_model_agrees'])
        print('  T1 mon', t1.get('monsters'), 'races', t1.get('races_static'), '| ST up', t1.get('spell_trap_face_up'), 'set', t1.get('spell_trap_set'), '| hand', t1.get('hand'), '| spent', r['t1_hand_spent'], '| GY', len(t1.get('graveyard') or []), 'EXup', t1.get('extra_face_up'))
        print('  T2 opp', json.dumps(t2['opponent_action'], ensure_ascii=False))
        print('  T2 used', [(u['card'], u['effect'], u['phase'], u['chain_at_activation']) for u in t2['used']])
        print('  T2 offered-unused', [(u['card'], u['effect'], u['windows']) for u in t2['offered_unused']], '| never offered', t2['key_cards_on_field_never_offered'])
        e2 = r['t2_end'] or {}
        print('  T2 end own', e2.get('monsters'), 'hand', e2.get('hand'), '| opp', e2.get('opponent_monsters'), 'oppGY', e2.get('opponent_graveyard'))
        print('  T3 used', r['t3_used'][:14], '| T3 end own', t3.get('monsters'), 'hand', t3.get('hand'), 'lp1', t3.get('lp1'), '| game over', r['game_over'])
        print('  replay', r['replay'])


if __name__ == '__main__':
    main()
