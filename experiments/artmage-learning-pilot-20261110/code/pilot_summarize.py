"""Offline summary of the Artmage learning pilot (no engine): summary.json + printed tables."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
CDB = P / 'evaluation/opponent-article-basics-20261108/runtime/agent/assets/locale/ja/cards.cdb'
_db = sqlite3.connect(CDB)
NAMES = {code: name for code, name in _db.execute('select id,name from texts')}
OLD_REGRESSION = P / 'experiments/artmage-parallel-20261110/regression'
DEMO_RECORDS = {'C1': P / 'experiments/artmage-parallel-20261110/demos/C1/record.json', 'C2': P / 'experiments/artmage-parallel-20261110/demos/C2/record.json',
                'C1v': ROOT / 'eval/variants/C1v.json', 'C2v': ROOT / 'eval/variants/C2v.json'}
PASS_ACTS = {'Cancel'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def name(code):
    return NAMES.get(code, str(code))


def meaning(a):
    return (a['act'], a['code'], a.get('effect'))


def main_moves(trace, turn, player=0):
    """Player-0 decisions in a turn that are not passes / placements / positions (the route skeleton)."""
    out = []
    for r in trace:
        s = r['snapshot']
        if s['to_play'] != player or s['turn_count'] != turn:
            continue
        a = r['selected']
        if a['act'] in PASS_ACTS or s['message'] in ('select_place', 'select_position', 'select_disfield'):
            continue
        if s['message'] in ('select_chain', 'select_effectyn', 'select_yesno') and a['act'] != 'Activate':
            continue
        if s['message'] == 'select_idlecmd' and a['act'] == 'None':
            out.append(('Phase', 0, a.get('phase')))
            continue
        out.append(meaning(a))
    return out


def divergence(model, ref):
    for i, (a, b) in enumerate(zip(model, ref)):
        if a != b:
            return i
    return None if len(model) == len(ref) else min(len(model), len(ref))


def offered_used(trace, turn):
    offered, used = collections.Counter(), collections.Counter()
    for r in trace:
        s = r['snapshot']
        if s['to_play'] != 0 or s['turn_count'] != turn:
            continue
        for a in s['actions']:
            if a['act'] in ('Activate', 'SpSummon', 'Summon'):
                offered[(a['act'], a['code'], a.get('effect'))] += 1
        a = r['selected']
        if a['act'] in ('Activate', 'SpSummon', 'Summon', 'Set', 'MSet', 'SSet') or 'Attack' in a['act']:
            used[(a['act'], a['code'], a.get('effect'))] += 1
    return offered, used


def fmt(key):
    act, code, eff = key
    return f'{act}:{name(code)}' + (f'[{eff}]' if eff not in (None, -1, 0) else '')


def attacks(trace):
    out = []
    for i, r in enumerate(trace):
        s = r['snapshot']
        if s['to_play'] == 0 and s['turn_count'] == 3 and 'Attack' in r['selected']['act']:
            out.append(f"{r['selected']['act']}:{name(r['selected']['code'])}")
    return out


def nt1_row(which, cond):
    d = read(ROOT / 'eval/nt1' / which / f'{cond}.json')
    ref = read(DEMO_RECORDS[cond])
    model_t1, ref_t1 = main_moves(d['trace'], 1), main_moves(ref['trace'], 1)
    div = divergence(model_t1, ref_t1)
    cp = d['checkpoints']
    off2, used2 = offered_used(d['trace'], 2)
    off3, used3 = offered_used(d['trace'], 3)
    return dict(
        model=which, condition=cond, reason=d['reason'], failure=d['failure'], game_over=d['game_over'], fallbacks=d['opponent_fallbacks'],
        decisions=d['decisions'], model_decisions=d['model_decisions'], lua_errors=d['lua_errors'], visibility_events=d.get('disclosed'),
        legacy_goal=d['legacy_goal_at_own_turn1_end'],
        turn1=dict(moves=[fmt(m) if m[0] != 'Phase' else f'Phase:{m[2]}' for m in model_t1], first_divergence_from_script=div,
                   script_move_at_divergence=(fmt(ref_t1[div]) if div is not None and div < len(ref_t1) else None),
                   model_move_at_divergence=(fmt(model_t1[div]) if div is not None and div < len(model_t1) and model_t1[div][0] != 'Phase' else
                                             (f'Phase:{model_t1[div][2]}' if div is not None and div < len(model_t1) else None)),
                   end=cp.get('own_turn1_end')),
        turn2=dict(own_used=[fmt(k) for k in used2], own_offered=sorted({fmt(k) for k in off2}), end=cp.get('opp_turn2_end')),
        turn3=dict(used=[fmt(k) for k in used3], attacks=attacks(d['trace']), end=cp.get('own_turn3_end')))


def board_names(b):
    if not b:
        return None
    return dict(monsters=[name(c) for c in b['monsters']], spell_trap=[name(c) for c in b['spell_trap']], hand=[name(c) for c in b['hand']],
                graveyard=len(b['graveyard']), banished=len(b['banished']), lp0=b['lp0'], lp1=b['lp1'])


def rt1_rows():
    rows = []
    for cid in ['Artmage-g06-p0', 'Artmage-g07-p0', 'Artmage-g08-p0', 'Artmage-g09-p0', 'Artmage-g10-p0', 'Artmage-g11-p0']:
        cand = read(ROOT / 'eval/rt1/candidate' / f'{cid}.json')
        base_path = ROOT / 'eval/rt1/start' / f'{cid}.json'
        base = read(base_path) if base_path.exists() else read(OLD_REGRESSION / f'{cid}.json')
        same = len(base['trace']) == len(cand['trace']) and all(meaning(a['selected']) == meaning(b['selected']) for a, b in zip(base['trace'], cand['trace']))
        rows.append(dict(case_id=cid, baseline_source='pilot start run' if base_path.exists() else 'artmage-parallel regression (start model)',
                         baseline_passed=base['passed'], candidate_passed=cand['passed'], candidate_reason=cand['reason'], baseline_steps=len(base['trace']),
                         candidate_steps=len(cand['trace']), same_action_sequence=same, visibility_events=cand.get('disclosed')))
    return rows


def exposure():
    eps = [json.loads(l) for l in (ROOT / 'train/attempt-2/episodes.jsonl').read_text().splitlines()]
    reg = read(ROOT / 'registration.json')
    targets = {k: collections.Counter(v) for k, v in reg['evaluation']['evaluation_hands'].items()}
    found = []
    for e in eps:
        for seat, hand in e['hands'].items():
            h = collections.Counter(hand)
            exact = e['hand_first_turn'][seat] == 1
            for name_, t in targets.items():
                if (exact and h == t) or (not exact and not (t - h)):
                    found.append(dict(episode=e['episode'], seat=int(seat), learner=int(seat) == e['learner_seat'], hand=hand, evaluation=name_,
                                      kind='exact' if exact else 'superset (hand seen after a draw)'))
    supersets = sum(1 for e in eps for seat in e['hands'] if e['hand_first_turn'][seat] != 1)
    return dict(episodes=len(eps), hands_recorded=sum(len(e['hands']) for e in eps), hands_seen_after_draw=supersets, overlaps=found)


def main():
    result = read(ROOT / 'train/attempt-2/result.json')
    windows = [json.loads(l) for l in (ROOT / 'train/attempt-2/windows.jsonl').read_text().splitlines()]
    mbs = [json.loads(l) for l in (ROOT / 'train/attempt-2/minibatches.jsonl').read_text().splitlines()]
    eps = [json.loads(l) for l in (ROOT / 'train/attempt-2/episodes.jsonl').read_text().splitlines()]
    per_source = collections.defaultdict(list)
    for m in mbs:
        for i, src in enumerate(['old', 'C1', 'C2']):
            if m['per_source'][i][0] > 0:
                per_source[src].append(m['per_source'][i])
    import numpy as np
    stat = lambda k: dict(mean=float(np.mean([m[k] for m in mbs])), max=float(np.max([m[k] for m in mbs])))
    training = dict(
        attempt_2=dict(total=result['total'], learner=result['learner'], frozen=result['frozen'], episodes=result['episodes'], windows=result['windows'],
                       minibatches=result['minibatches'], rehearsal_counts=result['rehearsal_counts'], loss_steps_ppo=sum(w['loss_steps'] for w in windows),
                       learner_adapter_controlled=result['learner_adapter_controlled'], bootstraps=collections.Counter(w['bootstrap'] for w in windows),
                       learner_outcomes=collections.Counter(str(e.get('learner_outcome')) for e in eps),
                       learner_first_seat_episodes=sum(1 for e in eps if e['learner_seat'] == 0), seconds=result['seconds'],
                       visibility=result['visibility'], saved=result['saved'], adapter=result['adapter'], lua_errors=result['native_lua_errors']),
        attempt_1_failed=dict(decisions=sum(e['decisions'] for e in (json.loads(l) for l in (ROOT / 'train/attempt-1-failed/episodes.jsonl').read_text().splitlines())),
                              windows=sum(1 for _ in (ROOT / 'train/attempt-1-failed/windows.jsonl').read_text().splitlines())),
        metrics={k: stat(k) for k in ['approx_kl', 'clipfrac', 'pg', 'v', 'ent', 'reh_ce', 'ppo_grad', 'reh_grad_weighted', 'combined_grad', 'ppo_reh_cosine', 'base_update', 'head_update']},
        rehearsal_by_source={s: dict(minibatches=len(v), loss_steps=float(sum(x[0] for x in v)), ce_first=v[0][1], ce_last=v[-1][1], ce_mean=float(np.mean([x[1] for x in v])),
                                     weighted_grad_mean=float(np.mean([x[2] for x in v]))) for s, v in per_source.items()})
    training['cumulative_decisions'] = training['attempt_1_failed']['decisions'] + result['total']
    nt1 = {f'{w}/{c}': nt1_row(w, c) for w in ['start', 'candidate'] for c in ['C1', 'C2', 'C1v', 'C2v']}
    variants = {c: dict(script_completed=read(ROOT / f'eval/variants/{c}.json')['script_completed'],
                        own_turn_end=board_names(read(ROOT / f'eval/variants/{c}.json')['checkpoints'].get('own_turn_end')),
                        next_own_turn_start=board_names(read(ROOT / f'eval/variants/{c}.json')['checkpoints'].get('next_own_turn_start')),
                        visibility_events=read(ROOT / f'eval/variants/{c}.json').get('disclosed')) for c in ['C1v', 'C2v']}
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'),
                   evaluation_runs=dict(used=sum(1 for r in ledger if r['event'] == 'start'), cap=18, statuses=[r['status'] for r in ledger if r['event'] == 'finish']),
                   training=training, rt1=rt1_rows(), nt1=nt1, variants=variants, exposure=exposure(),
                   code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))})
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + '\n')
    print(json.dumps(dict(training=training['attempt_2'] | {'cumulative': training['cumulative_decisions']}), ensure_ascii=False, default=str)[:1500])
    print(json.dumps(training['rehearsal_by_source'], ensure_ascii=False))
    for r in summary['rt1']:
        print('RT1', r)
    for k, r in nt1.items():
        print('NT1', k, r['reason'], 'div', r['turn1']['first_divergence_from_script'], 'script:', r['turn1']['script_move_at_divergence'], 'model:', r['turn1']['model_move_at_divergence'])
        print('   t1 moves', r['turn1']['moves'][:14])
        print('   t1 end', board_names(r['turn1']['end']))
        print('   t2 own used', r['turn2']['own_used'], '| t3 used', r['turn3']['used'], 'attacks', r['turn3']['attacks'])
        print('   t3 end', board_names(r['turn3']['end']), 'legacy goal', r['legacy_goal'], 'vis', r['visibility_events'])
    print('EXPOSURE', summary['exposure'])
    print('VARIANTS', json.dumps(variants, ensure_ascii=False)[:800])


if __name__ == '__main__':
    main()
