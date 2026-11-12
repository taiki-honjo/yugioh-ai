"""Pre-registered 128-game evaluation of the FINAL cumulative-16384 candidate vs the adopted model.
Runs only if the retention rule held on both tests. No confirmation run, no adoption, no training."""
import json, os, stat, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent; E = OUT.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal_ab as AB

plan_path = OUT / 'battle-plan.json'
plan = C.read_json(plan_path)
assert C.sha(plan_path) == 'df87dd41ecfe4e84cce8940a03292435dfe3a074c524384f5cf0c05d440fb691', 'battle plan changed after registration'
comparison = C.read_json(E / 'fixed-test-comparison.json')['comparisons'][0]
record = dict(plan=str(plan_path), plan_sha256=C.sha(plan_path), games_played=0, adopted=False,
              retention_on_both=comparison['retention_on_both'])
if not comparison['retention_on_both']:
    record['skipped_reason'] = 'retention rule did not hold; no games were played'
    C.atomic_json(OUT / 'battle-result.json', record)
    print('SKIPPED', json.dumps(record), flush=True)
    raise SystemExit(0)

cfg = C.load_config(plan['config'])
state = C.read_json(Path(cfg['root']) / 'state.json')
run = C.read_json(Path(cfg['root']) / 'cycles/12288-seed10016384/run-result.json')
candidate = Path(run['candidate'])
assert (C.sha(candidate / 'base.flax_model'), C.sha(candidate / 'chain-head.flax_model')) == (run['base_sha256'], run['head_sha256'])
assert (comparison['base_sha256'], comparison['head_sha256']) == (run['base_sha256'], run['head_sha256'])
for name in ['base.flax_model', 'chain-head.flax_model']:  # fix the candidate before any game
    os.chmod(candidate / name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
opponent = Path(plan['opponent']['path'])
assert (C.sha(opponent / 'base.flax_model'), C.sha(opponent / 'chain-head.flax_model')) == (plan['opponent']['base_sha256'], plan['opponent']['head_sha256'])
production = Path(C.read_json(AB.PRODUCTION_CONFIG)['root'])
assert all(C.sha(production / 'models/initial' / n) == C.sha(opponent / n) for n in ['base.flax_model', 'chain-head.flax_model'])
before = AB.production_fingerprint()
if AB.resident_learners():
    raise RuntimeError('A resident learner is running')
AB.raise_on_termination_signal()
B = AB.backend()
result = OUT / f"games-{plan['episodes']}.json"
if not result.exists():
    with AB.arm_active(cfg):
        B.control.run_job(cfg, state, [cfg['python'], '-u', str(P / 'scripts/continuous_learning.py'), 'job-eval',
                                       '--config', cfg['_file'], '--candidate', str(candidate), '--opponent', str(opponent),
                                       '--episodes', str(plan['episodes']), '--seed', str(plan['seed']), '--result', str(result)],
                          OUT, 'games')
summary = C.read_json(result)
record.update(games_played=plan['episodes'], candidate=str(candidate), base_sha256=run['base_sha256'], head_sha256=run['head_sha256'],
              results=summary['results'], terminations=summary['terminations'],
              clean=C.evaluation_is_clean(summary, plan['episodes']), promotion_ok_for_information=C.promotion_ok(summary, plan['episodes']))
assert record['games_played'] <= plan['max_battles']
after = AB.production_fingerprint()
record.update(production_fingerprint_unchanged=before == after, resources=C.read_json(OUT / 'games-resources.json'))
C.atomic_json(OUT / 'battle-result.json', record)
if before != after:
    raise RuntimeError('Production fingerprint changed')
r = summary['results']
print('GAMES', json.dumps(dict(wins=r['wins'], losses=r['losses'], draws=r['draws'], win_rate=r['win_rate'], ci95=r['win_rate_ci95'],
                               clean=record['clean'], promotion_ok=record['promotion_ok_for_information'])), flush=True)
