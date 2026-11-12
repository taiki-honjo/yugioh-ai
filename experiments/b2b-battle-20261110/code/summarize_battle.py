"""Offline summary of the B2B-candidate battle evaluation; each stage reported separately (never pooled)."""
import collections
import json
from pathlib import Path
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-battle-20261110'
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402


def stage(path, requested):
    d = C.read_json(path)
    r = d['results']
    pairs = collections.defaultdict(list)
    for e in d['episodes']:
        pairs[e['pair_id']].append(e)
    kinds = collections.Counter()
    for eps in pairs.values():
        wins = sum(e['win'] for e in eps if e.get('included_in_verified_score', True))
        kinds['both_won' if wins == 2 else 'split' if wins == 1 else 'both_lost'] += 1
    reasons = collections.Counter(e['reason_label'] for e in d['episodes'])
    return dict(file=str(path), sha256=C.sha(path), requested=requested, total=r['total_episodes'], verified=r['verified_episodes'],
                wins=r['wins'], losses=r['losses'], draws=r['draws'], win_rate=r['win_rate'], paired_ci95=r['win_rate_ci95'],
                se_paired=r['win_rate_se_paired'], first_player=r['first_player'], second_player=r['second_player'],
                pairs=dict(kinds), terminations=d['terminations'], reason_labels=dict(reasons),
                requires_rerun_or_exclusion=r['requires_rerun_or_exclusion'], clean=C.evaluation_is_clean(d, requested),
                promotion_rule=C.promotion_ok(d, requested), candidate=d['candidate'], opponent=d['opponent'],
                wave_seeds=sorted({e['wave_seed'] for e in d['episodes']}))


def main():
    plan = C.read_json(OUT / 'plan.json')
    decision = C.read_json(OUT / 'decision.json')
    directory = OUT / 'root/cycles/000001-b2b-battle'
    out = dict(plan_sha256=C.sha(OUT / 'plan.json'), decision=dict(promoted_by_current_rule=decision['promoted'], clean=decision['clean'], adopted=False))
    out['scout'] = stage(Path(decision['evidence']), 128)
    out['reference'] = stage(Path(decision['reference_evidence']), 64)
    if decision.get('confirmation_evidence'):
        out['confirmation'] = stage(Path(decision['confirmation_evidence']), 256)
    for key in ['scout', 'reference', 'confirmation']:
        if key in out:
            planned = plan['games'][key]['wave_seeds']
            out[key]['wave_seeds_as_registered'] = sorted(planned) == out[key]['wave_seeds']
    out['stage1_pass'] = out['scout']['promotion_rule'] and out['reference']['clean'] and out['reference']['win_rate'] >= 0.5
    out['episode_ledger'] = [json.loads(l) for l in (OUT / 'episode-ledger.jsonl').read_text().splitlines()]
    out['episodes_used'] = sum(r['episodes'] for r in out['episode_ledger'])
    C.atomic_json(OUT / 'summary.json', out)
    for key in ['scout', 'reference', 'confirmation']:
        if key in out:
            s = out[key]
            print(key, json.dumps({k: s[k] for k in ['total', 'verified', 'wins', 'losses', 'draws', 'win_rate', 'paired_ci95', 'first_player', 'second_player',
                                                     'pairs', 'reason_labels', 'clean', 'promotion_rule', 'wave_seeds_as_registered']}, ensure_ascii=False))
    print('stage1_pass', out['stage1_pass'], '| episodes used', out['episodes_used'], '| promoted_by_current_rule', decision['promoted'])


if __name__ == '__main__':
    main()
