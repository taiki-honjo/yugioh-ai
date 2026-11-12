"""Battle evaluation of the single B2B-trial candidate against the adopted model (isolated root; nothing adopted).

The current decision path (continuous_learning.evaluate_candidate) is used unchanged: scout 128 games vs best, reference
64 games vs the root's initial model (same base seed, as in the current code), and - only if the scout passes
promotion_ok, the reference is clean and its win rate >= 0.5 - a 256-game confirmation with base seed + 1.
Two deliberate substitutions, both recorded:
  * its first step, the 66-case opening regression, returns the EXISTING retention record of this exact candidate
    (experiments/b2b-train-20261109/retention, 65/66, identical passing set) instead of re-running it;
  * every battle job is counted in an episode ledger that refuses to exceed 448 games.
The evaluation root's STOP is lifted only while evaluate_candidate runs. Production is not touched.

  battle.py prepare | run | summarize
"""
import json
import os
from pathlib import Path
import random
import shutil
import stat
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/b2b-battle-20261110'
ROOT = OUT / 'root'
CONFIG_OUT = OUT / 'config.json'
CANDIDATE_SRC = P / 'experiments/b2b-train-20261109/root/cycles/000001-b2b-trial/candidate'
ADOPTED_SRC = P / 'experiments/adoption-d16384-20261109/model'
RETENTION = P / 'experiments/b2b-train-20261109/retention'
EPISODE_CAP = 448
SCOUT_SEED, REJECTED_SEED = 80021112, 80021110
PAST_BASE_SEEDS = [3026, 6037, 9107, 11037, 41037, 811037, 80001109, 80002991, 80004985, 80016384, 80016385, 90000555]
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402


def wave(base, w):  # scripts/evaluate.py make_wave_seed
    return random.Random(base * 1000003 + w * 9973 + 100000).randint(1, 100_000_000)


def copy_model(src, dst, expected):
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
    for name, digest in expected.items():
        assert C.sha(dst / name) == digest, (dst, name)
    for f in dst.iterdir():
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    dst.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def prepare():
    assert not OUT.joinpath('plan.json').exists(), 'already prepared'
    cand = C.read_json(P / 'experiments/b2b-train-20261109/candidate.json')
    adopted = dict(base='86b4358dcfec696cb4b3cc4c19781c34292b2509bf037b9e4f44d9ed8ec59d1c', head='6ca3221f1828eecc5cea817003c8df5e8802b75fabef0f4aa33105ad0f979fce')
    copy_model(CANDIDATE_SRC, OUT / 'models/candidate', {'base.flax_model': cand['base_sha256'], 'chain-head.flax_model': cand['head_sha256']})
    copy_model(ADOPTED_SRC, OUT / 'models/adopted', {'base.flax_model': adopted['base'], 'chain-head.flax_model': adopted['head']})
    prod = json.loads((P / 'config/continuous-elfnotes.json').read_text())
    cfg = dict(prod, root=str(ROOT), initial_checkpoint=str(ADOPTED_SRC),
               experiment_note='isolated battle-evaluation root; never run the loop here (STOP=now); nothing is adopted')
    for key in ['rehearsal', 'previous_production_root', 'previous_production_config_sha256', 'keep_all_checkpoints', 'save_interval_updates']:
        cfg.pop(key, None)
    C.atomic_json(CONFIG_OUT, cfg)
    ROOT.mkdir(parents=True)
    (ROOT / 'STOP').write_text('now')
    cfg = C.load_config(CONFIG_OUT)
    C.backend().initialize(cfg)
    C.runtime_check(cfg)
    for name, digest in [('base.flax_model', adopted['base']), ('chain-head.flax_model', adopted['head'])]:
        assert C.sha(ROOT / 'models/initial' / name) == digest  # best == initial == adopted model
    ours = C.read_json(ROOT / 'snapshot/hashes.json')
    theirs = C.read_json(Path(prod['root']) / 'snapshot/hashes.json')
    differing = sorted(n for n in set(ours) | set(theirs) if ours.get(n) != theirs.get(n))
    assert set(differing) <= {'policy-code/verify_opponent_article_basics.py', 'train.py', 'opening_rehearsal.py', 'rehearsal/train.npz',
                              'rehearsal/train-witnesses.json', 'rehearsal/settings.json'}, differing
    # evaluation code, policy code, adapters, runtime and manifest identical to production (only the embedded path and
    # the rehearsal/trainer files, which battles do not use, differ)
    past = {wave(b, w): (b, w) for b in PAST_BASE_SEEDS for w in range(64)}
    seeds = dict(scout=(SCOUT_SEED, 16), reference=(SCOUT_SEED, 8), confirmation=(SCOUT_SEED + 1, 32))
    waves = {k: [wave(b, w) for w in range(n)] for k, (b, n) in seeds.items()}
    assert not set(waves['scout']) & set(past) and not set(waves['confirmation']) & set(past)
    assert not set(waves['scout']) & set(waves['confirmation']) and set(waves['reference']) <= set(waves['scout'])
    retention = C.read_json(RETENTION / 'retention-summary.json')
    assert retention['candidate_base_sha256'] == cand['base_sha256'] and retention['all_baseline_cases_retained']
    plan = dict(
        registered_jst=C.now(), spec='docs/b2b-battle-decision-2026-11-10.md',
        candidate=dict(path=str(OUT / 'models/candidate'), source=str(CANDIDATE_SRC), base_sha256=cand['base_sha256'], head_sha256=cand['head_sha256'],
                       training_seed=cand['training_seed'], decisions=cand['decisions']),
        opponent=dict(path=str(ROOT / 'models/initial'), fixed_copy=str(OUT / 'models/adopted'), base_sha256=adopted['base'], head_sha256=adopted['head'],
                      note='adopted d16384 pair; in this root best == initial == this pair, so scout and reference face the same weights'),
        setting='Elfnotes + Kewl Tune mirror; same deck, native, vocabulary, evaluator, placement v2 and position adapters for both; '
                'seat-swapped same-seed pairs; both models decide freely from turn 1 (no scripted landing, no demonstration data at inference)',
        versions=dict(config_sha256=C.sha(CONFIG_OUT), snapshot_hashes_sha256=C.sha(ROOT / 'snapshot/hashes.json'),
                      snapshot_files_differing_from_production=differing, continuous_learning_sha256=C.sha(P / 'scripts/continuous_learning.py'),
                      backend_sha256=C.sha(P / 'scripts/continuous_opening_backend.py'), battle_driver_sha256=C.sha(Path(__file__))),
        games=dict(scout=dict(episodes=128, pairs=64, base_seed=SCOUT_SEED, wave_seeds=waves['scout']),
                   reference=dict(episodes=64, pairs=32, base_seed=SCOUT_SEED, wave_seeds=waves['reference'],
                                  note='same base seed as the scout (current code): its 8 waves are the scout\'s first 8; reported separately, not independent evidence'),
                   confirmation=dict(episodes=256, pairs=128, base_seed=SCOUT_SEED + 1, wave_seeds=waves['confirmation'], only_if='stage-1 conditions all hold'),
                   cap=EPISODE_CAP),
        seed_check=dict(rejected=dict(base_seed=REJECTED_SEED, reason=f'wave 11 equals a past wave seed (base 811037, wave 59)'),
                        past_base_seeds=PAST_BASE_SEEDS, past_waves_checked=64, no_overlap_with_past=True, scout_confirmation_disjoint=True),
        stage1_pass='scout 128 and reference 64 each fully verified; abnormal 0; no rerun/exclusion requests; scout paired CI95 lower bound > 0.5; '
                    'reference win rate >= 0.5 (current promotion_ok / evaluation_is_clean / reference rule)',
        stage2_pass='256 games alone: fully verified, abnormal 0, no rerun/exclusion, paired CI95 lower bound > 0.5; never pooled with 128 or 64',
        retention=dict(reused_record=str(RETENTION / 'retention-summary.json'), sha256=C.sha(RETENTION / 'retention-summary.json'),
                       candidate_passed=retention['candidate_passed'], identical_passing_set=retention['all_baseline_cases_retained'],
                       note='not re-run inside this battle budget'),
        on_failure='an incomplete evaluation is recorded as incomplete; no extra games, no seed substitution, no relaxed conditions',
        outcome_use='no adoption and no gate change even if every stage passes; production, cycle-5 candidate, STOP and pending unchanged')
    C.atomic_json(OUT / 'plan.json', plan)
    print('PREPARED', json.dumps(dict(plan_sha256=C.sha(OUT / 'plan.json'), differing=differing, scout_seed=SCOUT_SEED, confirmation_seed=SCOUT_SEED + 1)))


def run():
    plan = C.read_json(OUT / 'plan.json')
    directory = ROOT / 'cycles/000001-b2b-battle'
    assert not directory.exists(), 'evaluation already run'
    cfg = C.load_config(CONFIG_OUT)
    C.runtime_check(cfg)
    candidate = Path(plan['candidate']['path'])
    assert C.sha(candidate / 'base.flax_model') == plan['candidate']['base_sha256']
    retention = C.read_json(RETENTION / 'retention-summary.json')
    B = C.backend()

    def reuse_retention(cfg_, state_, pending_, candidate_):
        return dict(passed=True, baseline_passed=65, candidate_passed=retention['candidate_passed'], total=retention['total'],
                    evidence=str(RETENTION / 'opening-retention.json'), lost_case_ids=retention['lost_case_ids'],
                    reused_existing_record=True, record_sha256=C.sha(RETENTION / 'retention-summary.json'))
    B.regression_gate = reuse_retention
    original_run_job = C.run_job
    ledger = OUT / 'episode-ledger.jsonl'

    def counted_run_job(cfg_, state_, command, directory_, name):
        command = [str(x) for x in command]
        episodes = int(command[command.index('--episodes') + 1])
        used = sum(json.loads(l)['episodes'] for l in ledger.read_text().splitlines()) if ledger.exists() else 0
        if used + episodes > EPISODE_CAP:
            raise RuntimeError(f'episode cap {EPISODE_CAP} would be exceeded ({used}+{episodes})')
        row = dict(job=name, episodes=episodes, seed=int(command[command.index('--seed') + 1]),
                   opponent=command[command.index('--opponent') + 1], time_jst=C.now())
        with ledger.open('a') as h:
            h.write(json.dumps(row) + '\n')
        return original_run_job(cfg_, state_, command, directory_, name)
    C.run_job = counted_run_job
    state = C.read_json(ROOT / 'state.json')
    pending = dict(cycle=1, directory=str(directory), eval_seed=plan['games']['scout']['base_seed'], phase='evaluating', candidate=str(candidate))
    directory.mkdir(parents=True)
    assert (ROOT / 'STOP').read_text() == 'now'
    (ROOT / 'STOP').unlink()
    try:
        decision = C.evaluate_candidate(cfg, state, pending, candidate)
    finally:
        (ROOT / 'STOP').write_text('now')
        C.save_state(cfg, C.read_json(ROOT / 'state.json'), child_pid=None, phase='idle', status='stopped')
        C.run_job = original_run_job
    C.atomic_json(OUT / 'decision.json', dict(decision, adopted=False, note='evaluation only; nothing adopted'))
    print('DECISION', json.dumps(dict(promoted_by_current_rule=decision['promoted'], clean=decision['clean'],
                                      scout=decision.get('scout', {}).get('win_rate_ci95'), reference=decision.get('reference', {}).get('win_rate'),
                                      confirmation=decision.get('confirmation', {}).get('win_rate_ci95'))))


if __name__ == '__main__':
    dict(prepare=prepare, run=run)[sys.argv[1]]()
