"""Read-only tables for the 2x2 run. Uses summary.json, which contains only runs whose recorded
conditions and evidence matched the plan."""
import json
from pathlib import Path
E = Path(__file__).resolve().parent
s = json.loads((E / 'summary.json').read_text())
assert s['conditions_verified'] and s['complete'], (s.get('missing_runs'))
runs = {(r['arm'], r['seed']): r for r in s['runs']}
arms, seeds = list(s['plan']['arms']), s['plan']['seeds']
short = dict(zip(arms, 'ABCD'))
base = json.loads((E / arms[0] / 'opening-baseline.json').read_text())
baseline_passed = set(base['passed_case_ids'])
out = []
w = out.append
w('## per run')
w('| cond | seed | openings | retention | lost | new | train CE | val CE | battle W-L-D (verified) | abnormal | truncated | engine timeout | suspected | eval error |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in arms:
    for sd in seeds:
        r = runs[(a, sd)]; b = r['battle']; res = b.get('results'); t = b.get('terminations') or {}
        bt = f"{res['wins']}-{res['losses']}-{res['draws']} ({res['verified_episodes']})" if res else '—'
        w(f"| {short[a]} | {sd} | {r['opening_passed']}/{r['opening_total']} | {'pass' if r['regression_passed'] else 'fail'} | "
          f"{len(r['lost_case_ids'])} | {len(r['new_case_ids'])} | {r['teacher_forced']['train']['ce']:.4f} | "
          f"{r['teacher_forced']['validation']['ce']:.4f} | {bt} | {t.get('abnormal_total', '—')} | {t.get('truncated', '—')} | "
          f"{t.get('engine_timeout', '—')} | {t.get('timeout_suspected', '—')} | {b.get('exception') or 0} |")
w('\n## lost cases per run')
for a in arms:
    for sd in seeds:
        r = runs[(a, sd)]
        w(f"- {short[a]}-{sd}: lost {r['lost_case_ids']} new {r['new_case_ids']}")
w('\n## paired comparisons (same seed): cases failing in one condition but not the other')
for x, y in [('A', 'C'), ('A', 'B'), ('C', 'D'), ('B', 'D')]:
    ax, ay = arms['ABCD'.index(x)], arms['ABCD'.index(y)]
    for sd in seeds:
        lx, ly = set(runs[(ax, sd)]['lost_case_ids']), set(runs[(ay, sd)]['lost_case_ids'])
        w(f"- {x} vs {y} seed {sd}: lost {len(lx)} vs {len(ly)}; only {x}: {sorted(lx - ly)}; only {y}: {sorted(ly - lx)}")
    tx = sum(len(runs[(ax, sd)]['lost_case_ids']) for sd in seeds); ty = sum(len(runs[(ay, sd)]['lost_case_ids']) for sd in seeds)
    w(f"  total lost (counted per seed): {x}={tx} {y}={ty}")
w('\n## update size and gradients')
w('| cond | seed | lr | base rel L2 | base max abs | head rel L2 | head changed | PPO grad norm | rehearsal grad norm | cosine(PPO,rehearsal) | rehearsal CE | rehearsal decisions | train s | RSS MiB | GPU peak MiB |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in arms:
    for sd in seeds:
        r = runs[(a, sd)]; wc = r['weight_change']; rh = r['rehearsal']; res = r['resources'].get('train', {})
        g = (f"{rh['mean']['ppo_grad_norm']:.1f} | {rh['mean']['rehearsal_grad_norm']:.2f} | {rh['mean']['ppo_rehearsal_cosine']:+.3f} | "
             f"{rh['mean']['ce']:.3f} | {rh['loss_decisions']}") if rh else '— | — | — | — | 0'
        w(f"| {short[a]} | {sd} | {r['learning_rate']:.1e} | {wc['base']['relative_l2']:.2e} | {wc['base']['max_abs']:.2e} | "
          f"{wc['chain-head']['relative_l2']:.2e} | {wc['chain-head']['changed']} | {g} | {res.get('wall_seconds')} | {res.get('max_rss_mib')} | "
          f"{res.get('gpu_memory_used_peak_mib')} |")
w('\n## per minibatch cosine (rehearsal arms)')
for a in arms:
    for sd in seeds:
        path = E / a / 'cycles' / f"{s['plan']['steps']}-seed{sd}" / 'rehearsal-metrics.jsonl'
        if path.exists():
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            w(f"- {short[a]}-{sd}: " + ', '.join(f"{m['ppo_rehearsal_cosine']:+.3f}" for r in rows for m in r['minibatches']))
w('\n## per-case table (cases changed anywhere)')
ids = [c['case_id'] for c in base['cases']]
w('| case | base | ' + ' | '.join(f'{short[a]}{str(sd)[-3:]}' for a in arms for sd in seeds) + ' |')
w('|' + '---|' * (2 + len(arms) * len(seeds)))
for cid in ids:
    vals = [runs[(a, sd)]['cases'][cid] for a in arms for sd in seeds]
    if all(vals) and cid in baseline_passed:
        continue
    w(f"| {cid} | {'○' if cid in baseline_passed else '×'} | " + ' | '.join('○' if v else '×' for v in vals) + ' |')
(E / 'analysis.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
