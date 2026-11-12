"""Pure statistics for the RL-training battles (no engine, no model): the end classification of scripts/evaluate.py
(reviewed version 67db15ff, loaded by file and hash-checked) and the seed-pair summary with the spec's exclusions."""
import collections
import hashlib
import importlib.util
from pathlib import Path

EVALUATE = Path('/home/hatk8/projects/yugioh-ai/scripts/evaluate.py')
EVALUATE_SHA = '67db15ffeb9f94884dda433352f52170ac0b7d60de46056c88f0317177d9d7d0'
NATIVE_TIMEOUT = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_module():
    if sha(EVALUATE) != EVALUATE_SHA:
        raise RuntimeError('scripts/evaluate.py changed')
    spec = importlib.util.spec_from_file_location('reviewed_evaluate_for_rlt', EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# ------------------------------------------------------------------ end classification (scripts/evaluate.py rules)
def classify_game(reward, terminated, truncated, termination_reason, win_reason, actor, candidate_seat, max_step_wall, timeout=NATIVE_TIMEOUT):
    tr = None if termination_reason is None else int(termination_reason)
    engine_timeout = tr == 2
    timeout_suspected = tr is None and max_step_wall >= timeout
    step_limit = (not engine_timeout) and (tr == 1 or bool(truncated))
    is_term = bool(terminated) and not step_limit and not engine_timeout
    if engine_timeout:
        cand_reward, outcome = 0.0, 'draw'
    else:
        cand_reward = float(reward) if actor == candidate_seat else -float(reward)
        outcome = 'win' if cand_reward > 0 else 'loss' if cand_reward < 0 else 'draw'
    abnormal = step_limit or outcome == 'draw' or engine_timeout or timeout_suspected or not is_term
    label = ('engine_timeout' if engine_timeout else 'timeout_suspected' if timeout_suspected else 'truncated' if step_limit else
             'draw_zero_reward' if outcome == 'draw' else 'normal_lp0' if win_reason == 1 else 'normal_deckout_or_special' if win_reason == -1
             else f'other_reason_{win_reason}')
    if step_limit or engine_timeout:
        outcome, cand_reward = 'draw', 0.0
    return dict(outcome=outcome, win=int(outcome == 'win'), loss=int(outcome == 'loss'), draw=int(outcome == 'draw'), candidate_reward=cand_reward,
                terminated=is_term, truncated=step_limit, engine_timeout=engine_timeout, timeout_suspected=timeout_suspected, abnormal=abnormal,
                reason_label=label, termination_reason=tr, win_reason=win_reason)


def pair_summary(games, ev):
    """Verified = seed pairs without timeout / suspected timeout / step-limit truncation / error (spec 5); raw kept."""
    by = collections.defaultdict(dict)
    for g in games:
        by[g['pair']][g['candidate_seat']] = g
    raw_scores, verified_scores, verified, excluded, kinds = [], [], [], [], collections.Counter()
    for pid in sorted(by):
        seat = by[pid]
        if 0 not in seat or 1 not in seat:
            excluded.append(dict(pair=pid, why='incomplete pair'))
            continue
        e0, e1 = seat[0], seat[1]
        raw_scores.append(0.5 * (e0['result']['win'] + e1['result']['win']))
        bad = [e for e in (e0, e1) if e.get('error') or e['result']['engine_timeout'] or e['result']['timeout_suspected'] or e['result']['truncated']]
        if bad:
            excluded.append(dict(pair=pid, why=[(e['candidate_seat'], e.get('error') or e['result']['reason_label']) for e in bad]))
            continue
        verified_scores.append(0.5 * (e0['result']['win'] + e1['result']['win']))
        verified.extend([e0, e1])
        w = (e0['result']['outcome'], e1['result']['outcome'])
        kinds['2-0' if w == ('win', 'win') else '0-2' if w == ('loss', 'loss') else 'first-seat-both' if w == ('win', 'loss') else
              'second-seat-both' if w == ('loss', 'win') else 'with-draw'] += 1
    n = len(verified)
    wins = sum(e['result']['win'] for e in verified)
    draws = sum(e['result']['draw'] for e in verified)
    losses = sum(e['result']['loss'] for e in verified)
    low, high, se, k = ev.paired_wilson_ci95(verified_scores, wins, n)
    rlow, rhigh, _, _ = ev.paired_wilson_ci95(raw_scores, 0, 0)
    seats = {}
    for s, label in [(0, 'first'), (1, 'second')]:
        es = [e for e in verified if e['candidate_seat'] == s]
        w = sum(e['result']['win'] for e in es)
        lo, hi = ev.wilson_ci95(w, len(es))
        seats[label] = dict(games=len(es), wins=w, losses=sum(e['result']['loss'] for e in es), draws=sum(e['result']['draw'] for e in es),
                            win_rate=w / len(es) if es else None, wilson_ci95_reference=[lo, hi])
    return dict(verified_pairs=len(verified_scores), verified_games=n, wins=wins, losses=losses, draws=draws,
                win_rate=wins / n if n else None, draw_rate=draws / n if n else None, ci95_pairs=[low, high], pair_se=se, n_eff=k,
                score_rate_draw_half=(wins + 0.5 * draws) / n if n else None, pair_kinds=dict(kinds), seats=seats, excluded_pairs=excluded,
                raw=dict(pairs=len(raw_scores), games=len(games), wins=sum(g['result']['win'] for g in games),
                         losses=sum(g['result']['loss'] for g in games), draws=sum(g['result']['draw'] for g in games), ci95_pairs=[rlow, rhigh]),
                interval='scripts/evaluate.py paired_wilson_ci95 (Clopper-Pearson, floor/ceil of the sum of pair scores wins/2; unit = seed pair)')
