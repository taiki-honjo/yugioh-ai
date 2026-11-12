"""User-requested 128-game comparison of the cumulative-16384 candidate vs the adopted model.
Keeps the original plan and its retention-failure skip record unchanged. No confirmation run,
no adoption, no training. The experiment arm's STOP is restored on every exit path."""
import json, os, stat, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal_ab as AB

plan_path = OUT / 'battle-plan-requested-20261109.json'
plan = C.read_json(plan_path)
assert C.sha(plan_path) == 'f70dc1eb3437e84ca6124d5771298bd09875151125033d88572e6c5bff5428fc', 'requested plan changed'
for key in ['original_plan', 'original_skip_record']:
    assert C.sha(OUT / plan[key]['file']) == plan[key]['sha256'], f'{key} must stay unchanged'
cfg = C.load_config(plan['config'])
state = C.read_json(Path(cfg['root']) / 'state.json')
candidate, opponent = Path(plan['candidate']['path']), Path(plan['opponent']['path'])
for side, path in [('candidate', candidate), ('opponent', opponent)]:
    assert (C.sha(path / 'base.flax_model'), C.sha(path / 'chain-head.flax_model')) == (plan[side]['base_sha256'], plan[side]['head_sha256']), side
for name in ['base.flax_model', 'chain-head.flax_model']:  # fix the candidate before any game
    os.chmod(candidate / name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
production = Path(C.read_json(AB.PRODUCTION_CONFIG)['root'])
assert all(C.sha(production / 'models/initial' / n) == C.sha(opponent / n) for n in ['base.flax_model', 'chain-head.flax_model'])
arm_hashes, production_hashes = C.read_json(Path(cfg['root']) / 'snapshot/hashes.json'), C.read_json(production / 'snapshot/hashes.json')
for name in ['policy-code/center_placement.py', 'policy-code/battle_position.py', 'evaluator.py']:
    assert arm_hashes[name] == production_hashes[name], name
before = AB.production_fingerprint()
if AB.resident_learners():
    raise RuntimeError('A resident learner is running')
AB.raise_on_termination_signal()
B = AB.backend()
result = OUT / 'requested-games-128.json'
if not result.exists():
    with AB.arm_active(cfg):  # lifts only this experiment arm's STOP; restored on every exit path
        B.control.run_job(cfg, state, [cfg['python'], '-u', str(P / 'scripts/continuous_learning.py'), 'job-eval',
                                       '--config', cfg['_file'], '--candidate', str(candidate), '--opponent', str(opponent),
                                       '--episodes', str(plan['episodes']), '--seed', str(plan['seed']), '--result', str(result)],
                          OUT, 'requested-games')
summary = C.read_json(result)
after = AB.production_fingerprint()
record = dict(plan=str(plan_path), plan_sha256=C.sha(plan_path), games_played=plan['episodes'], adopted=False,
              retention_failure_record_unchanged=all(C.sha(OUT / plan[k]['file']) == plan[k]['sha256'] for k in ['original_plan', 'original_skip_record']),
              candidate=str(candidate), base_sha256=plan['candidate']['base_sha256'], head_sha256=plan['candidate']['head_sha256'],
              results=summary['results'], terminations=summary['terminations'],
              clean=C.evaluation_is_clean(summary, plan['episodes']), promotion_ok_for_information=C.promotion_ok(summary, plan['episodes']),
              arm_stop_restored=(Path(cfg['root']) / 'STOP').read_text() == 'now',
              production_fingerprint_unchanged=before == after, resources=C.read_json(OUT / 'requested-games-resources.json'))
assert record['games_played'] <= plan['max_battles']
C.atomic_json(OUT / 'requested-battle-result.json', record)
if before != after:
    raise RuntimeError('Production fingerprint changed')
r = summary['results']
print('GAMES', json.dumps(dict(wins=r['wins'], losses=r['losses'], draws=r['draws'], win_rate=r['win_rate'], ci95=r['win_rate_ci95'],
                               clean=record['clean'], promotion_ok=record['promotion_ok_for_information'],
                               stop_restored=record['arm_stop_restored'])), flush=True)
