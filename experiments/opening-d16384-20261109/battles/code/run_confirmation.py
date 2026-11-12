"""Pre-registered 256-game confirmation of the fixed cumulative-16384 candidate vs the adopted model.
Judged alone by the existing criterion. Earlier records are kept unchanged. No adoption, no training,
no further games. The experiment arm's STOP is restored on every exit path."""
import json, sys
from pathlib import Path
P = Path('/home/hatk8/projects/yugioh-ai'); OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C
import opening_rehearsal_ab as AB

plan_path = OUT / 'battle-plan-confirmation-20261109.json'
plan = C.read_json(plan_path)
assert C.sha(plan_path) == '0dc703026eff52e6adb3d4157cbadb21cf56994cee87893e928368ebbe12990c', 'confirmation plan changed'
def kept():
    return {name: C.sha(OUT / name) == digest for name, digest in plan['kept_unchanged'].items()}
assert all(kept().values()), kept()
cfg = C.load_config(plan['config'])
state = C.read_json(Path(cfg['root']) / 'state.json')
candidate, opponent = Path(plan['candidate']['path']), Path(plan['opponent']['path'])
for side, path in [('candidate', candidate), ('opponent', opponent)]:
    assert (C.sha(path / 'base.flax_model'), C.sha(path / 'chain-head.flax_model')) == (plan[side]['base_sha256'], plan[side]['head_sha256']), side
assert not any((candidate / n).stat().st_mode & 0o222 for n in ['base.flax_model', 'chain-head.flax_model']), 'candidate must stay read-only'
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
result = OUT / 'confirmation-games-256.json'
if not result.exists():
    with AB.arm_active(cfg):
        B.control.run_job(cfg, state, [cfg['python'], '-u', str(P / 'scripts/continuous_learning.py'), 'job-eval',
                                       '--config', cfg['_file'], '--candidate', str(candidate), '--opponent', str(opponent),
                                       '--episodes', str(plan['episodes']), '--seed', str(plan['seed']), '--result', str(result)],
                          OUT, 'confirmation-games')
summary = C.read_json(result)
after = AB.production_fingerprint()
arm_state = C.read_json(Path(cfg['root']) / 'state.json')
record = dict(plan=str(plan_path), plan_sha256=C.sha(plan_path), games_played=plan['episodes'], seed=plan['seed'], adopted=False,
              earlier_records_unchanged=kept(), candidate=str(candidate),
              base_sha256=plan['candidate']['base_sha256'], head_sha256=plan['candidate']['head_sha256'],
              results=summary['results'], terminations=summary['terminations'],
              clean=C.evaluation_is_clean(summary, plan['episodes']),
              existing_battle_criterion_met_by_these_256_games=C.promotion_ok(summary, plan['episodes']),
              arm_stop=(Path(cfg['root']) / 'STOP').read_text(), arm_status=arm_state['status'], arm_child_pid=arm_state.get('child_pid'),
              leftover_jobs=AB.leftover_jobs(cfg['root']),
              production_fingerprint_unchanged=before == after, resources=C.read_json(OUT / 'confirmation-games-resources.json'))
assert record['games_played'] <= plan['max_battles'] and all(record['earlier_records_unchanged'].values())
C.atomic_json(OUT / 'confirmation-battle-result.json', record)
if before != after:
    raise RuntimeError('Production fingerprint changed')
r = summary['results']
print('CONFIRMATION', json.dumps(dict(wins=r['wins'], losses=r['losses'], draws=r['draws'], win_rate=r['win_rate'], ci95=r['win_rate_ci95'],
                                      clean=record['clean'], criterion=record['existing_battle_criterion_met_by_these_256_games'],
                                      stop=record['arm_stop'], child_pid=record['arm_child_pid'], leftover=record['leftover_jobs'])), flush=True)
