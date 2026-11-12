"""Run the pre-registered battle plan: 128-game scout; 256-game confirmation ONLY if the existing
promotion criterion holds; at most 384 games. No adoption, no training, no resident operation."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal_ab as AB

plan_path = OUT / 'battle-plan.json'
plan = C.read_json(plan_path)
assert C.sha(plan_path) == 'aa27b26311fec6c1d135f0c5d9a03d6de2f921b2370030e7433a99111af91eed', 'battle plan changed after registration'
cfg = C.load_config(plan['config'])
state = C.read_json(Path(cfg['root']) / 'state.json')
for side in ['candidate', 'opponent']:
    for name, key in [('base.flax_model', 'base_sha256'), ('chain-head.flax_model', 'head_sha256')]:
        assert C.sha(Path(plan[side]['path']) / name) == plan[side][key], (side, name)
production = Path(C.read_json(AB.PRODUCTION_CONFIG)['root'])
for name in ['base.flax_model', 'chain-head.flax_model']:
    assert C.sha(production / 'models/initial' / name) == C.sha(Path(plan['opponent']['path']) / name)
assert plan['scout']['episodes'] + plan['confirmation']['episodes'] <= plan['max_battles']
before = AB.production_fingerprint()
if AB.resident_learners():
    raise RuntimeError('A resident learner is running')
AB.raise_on_termination_signal()
B = AB.backend()  # run_job with resource accounting

def battle(label, episodes, seed):
    result = OUT / f'{label}-{episodes}.json'
    if not result.exists():
        with AB.arm_active(cfg):  # lifts only this experiment arm's own STOP marker; restored on every exit path
            B.control.run_job(cfg, state, [cfg['python'], '-u', str(P / 'scripts/continuous_learning.py'), 'job-eval',
                                           '--config', cfg['_file'], '--candidate', plan['candidate']['path'],
                                           '--opponent', plan['opponent']['path'], '--episodes', str(episodes),
                                           '--seed', str(seed), '--result', str(result)], OUT, label)
    summary = C.read_json(result)
    return summary, dict(clean=C.evaluation_is_clean(summary, episodes), promotion_ok=C.promotion_ok(summary, episodes))

record = dict(plan=str(plan_path), plan_sha256=C.sha(plan_path), games_played=0, adopted=False)
scout, verdict = battle('scout', plan['scout']['episodes'], plan['scout']['seed'])
record.update(games_played=plan['scout']['episodes'], scout=dict(results=scout['results'], terminations=scout['terminations'], **verdict))
print('SCOUT', json.dumps(dict(verdict, **{k: scout['results'][k] for k in ['wins', 'losses', 'draws', 'win_rate', 'win_rate_ci95']})), flush=True)
if verdict['promotion_ok']:
    confirm, verdict2 = battle('confirmation', plan['confirmation']['episodes'], plan['confirmation']['seed'])
    record['games_played'] += plan['confirmation']['episodes']
    record.update(confirmation=dict(results=confirm['results'], terminations=confirm['terminations'], **verdict2),
                  existing_promotion_rule_satisfied=verdict2['promotion_ok'])
    print('CONFIRMATION', json.dumps(dict(verdict2, **{k: confirm['results'][k] for k in ['wins', 'losses', 'draws', 'win_rate', 'win_rate_ci95']})), flush=True)
else:
    record.update(confirmation=None, confirmation_skipped_reason='scout did not satisfy promotion_ok (existing criterion)',
                  existing_promotion_rule_satisfied=False)
assert record['games_played'] <= plan['max_battles']
after = AB.production_fingerprint()
record.update(production_fingerprint_unchanged=before == after,
              resources={p.stem: C.read_json(p) for p in OUT.glob('*-resources.json')})
C.atomic_json(OUT / 'battle-result.json', record)
if before != after:
    raise RuntimeError('Production fingerprint changed')
print('DONE', json.dumps(dict(games=record['games_played'], promotion_rule=record['existing_promotion_rule_satisfied'], adopted=False)), flush=True)
