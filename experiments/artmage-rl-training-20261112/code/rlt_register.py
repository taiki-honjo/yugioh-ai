"""Register the RL training before any engine run: seeds, code, pins, model copies, data, budgets, evaluation design.

  rlt_register.py register    writes data/seeds.json and registration.json once (refuses if either exists)
No engine duel and no model evaluation here (the isolated entry hash-checks its pins on import).
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import rlt_train as RT  # noqa: E402

SPEC = ROOT / 'spec/artmage-rl-training-plan-2026-11-12.md'
SPEC_SHA = 'c044e72a83e9ef0ab61cd8fabd92d15f48819a843f3e35231b9c6111b41e5039'
HB = P / 'experiments/artmage-hand-branch-20261110'
PILOT = P / 'experiments/artmage-learning-pilot-20261110'
JST = timezone(timedelta(hours=9))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_retention_reuse():
    """Start-model retention records of the hand-branch study: model, case, code, adapter/engine pins and conditions."""
    import hb_runs as HR  # this root's frozen copy (same file hash as the hand-branch one)
    amend = json.loads((HB / 'amendment-1.json').read_text())
    ledger = [json.loads(l) for l in (HB / 'eval-runs.jsonl').read_text().splitlines()]
    rows = {}
    for name in ['Artmage-g06-p0', 'Artmage-g07-p0', 'Artmage-g08-p0', 'Artmage-g09-p0', 'Artmage-g10-p0', 'Artmage-g11-p0'] + [f'F{i}-V' for i in range(7)]:
        path = HB / 'eval/retain/candidate' / f'{name}.json'
        rec = json.loads(path.read_text())
        if name.startswith('Artmage-g'):
            split = 'validation' if name in ['Artmage-g06-p0', 'Artmage-g07-p0'] else 'holdout'
            case = next(c for c in json.loads(HR.OLD_MANIFEST.read_text())['decks']['Artmage']['cases'][split] if c['case_id'] == name)
            same_case = (rec['seed'], rec['hand']) == (case['seed'], case['hand'])
        else:
            spec = HR.hand_spec(name)
            same_case = (rec['seed'], rec['hand']) == (spec['seed'], spec['hand'])
        start = next(r for r in ledger if r['event'] == 'start' and r['purpose'] == f'retain {name}')
        finish = next(r for r in ledger if r['event'] == 'finish' and r['run'] == start['run'])
        rows[name] = dict(path=str(path), sha256=sha(path), model_sha256=rec['model_sha256'], model_ok=tuple(rec['model_sha256']) == RT.START_SHA,
                          case_ok=same_case, run=start['run'], run_status=finish['status'], run_time=start['time_jst'])
    code_ok = all(amend['code'][n] == sha(ROOT / 'code' / n) for n in ['hb_runs.py', 'artmage_entry.py', 'artmage_pins.py', 'artmage_teacher.py',
                                                                         'batch_plans.py', 'hb_plans.py'])
    return dict(records=rows, all_models_ok=all(r['model_ok'] for r in rows.values()), all_cases_ok=all(r['case_ok'] for r in rows.values()),
                code_equal_to_hand_branch_amendment_1=code_ok, hand_branch_amendment_1_registered=amend['registered_jst'],
                conditions=dict(RT1_CAP=HR.RT1_CAP, FREE_CAP=HR.FREE_CAP, same_file='hb_runs.py 88e69412 (constants in the file)'),
                runs_after_amendment=all(r['run_time'] > amend['registered_jst'] for r in rows.values()))


def register():
    if (ROOT / 'registration.json').exists() or (ROOT / 'data/seeds.json').exists():
        raise SystemExit('registered already')
    if sha(SPEC) != SPEC_SHA:
        raise SystemExit('spec copy differs')
    import artmage_entry as E
    pins = E.verify_pins()
    import rlt_model as M
    import rlt_eval as EV
    training = RT.training_seeds()
    evaluation = RT.evaluation_seeds(training)
    seeds = dict(training=training, evaluation=evaluation,
                 rule=dict(training=f'rlt_train.episode_seeds({RT.TRAIN_SEED}, {RT.EPISODE_CAP}) (pilot generator: random.Random(seed*1000003+77).randint(1, 2**31-2)); episode e uses training[e]',
                           evaluation=f'walk episode_seeds({RT.EVAL_SEED}, 100000) in order, skip any of the 256 training seeds and repeats, keep the first {RT.EVAL_PAIRS}; pair i plays evaluation[i] twice (candidate seat 0, then 1)'),
                 overlap=len(set(training) & set(evaluation)), training_distinct=len(set(training)))
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'data/seeds.json').write_text(json.dumps(seeds, indent=1) + '\n')
    hb_manifest = json.loads((HB / 'data/manifest.json').read_text())
    reg = dict(
        registered_jst=datetime.now(JST).isoformat(timespec='seconds'),
        spec=dict(path=str(SPEC), sha256=SPEC_SHA, source='C:/Users/hatk8/Documents/Codex/2026-10-27/new-chat/outputs/artmage-rl-training-plan-2026-11-12.md'),
        purpose='Artmage mirror, natural dealing and draws: learn base + chain-head from win/loss (PPO) against the frozen start model, with rehearsal of the 40 old sequences; then 128 battles + 13 retention cases.',
        start=dict(base=str(RT.START[0]), head=str(RT.START[1]), sha256=list(RT.START_SHA),
                   origin='experiments/artmage-hand-branch-20261110/train/candidate (research candidate: basic openings + hand branch); copied read-only',
                   opponent='the same two files, never updated (separate in-memory copy of the start params)',
                   not_used='3-file response adapters (c110f964 etc.) are kept unchanged and not mixed in'),
        code={p.name: sha(p) for p in sorted((ROOT / 'code').glob('*.py'))},
        copied_from=dict(
            unchanged={n: str(HB / 'code' / n) for n in ['artmage_entry.py', 'artmage_pins.py', 'artmage_teacher.py', 'pilot_data.py', 'batch_data.py',
                                                         'batch_plans.py', 'hb_data.py', 'hb_plans.py', 'hb_runs.py']},
            modified={'rlt_model.py': dict(source=str(PILOT / 'code/pilot_model.py'), source_sha256=sha(PILOT / 'code/pilot_model.py')),
                      'rlt_train.py': dict(source=str(PILOT / 'code/pilot_train.py'), source_sha256=sha(PILOT / 'code/pilot_train.py'),
                                           note='pilot version after its amendment-1 (visibility audit by origin)')}),
        changes_from_pilot=[
            'value loss 0.5*(V-target)^2 without the min(., vloss_clip=1.0) cap (zero value gradient for |error| > sqrt(2)); PPO ratio clip unchanged',
            'start/opponent = the hand-branch candidate b9c7751f/77191d5d (pilot: the registered initial pair)',
            'caps: 32,768 environment decisions (learner + frozen) or 256 episodes; per game 4,000 decisions (native max_steps) and native timeout 30 s (pilot 2,048 / 10 s)',
            'rehearsal: the 40 old sequences via hb_data.BlockSampler (2 existing-group batches + 2 pair batches per block of 4, 2 sequences each, padded to 115) instead of the pilot 14/1/1 old/C1/C2 schedule; per-source logging for 9 sources',
            'episode ends classified by termination_reason: natural (+1/-1/0 from the actor view), step_limit (0, separate count), timeout (open-window steps of that episode dropped, no outcome)',
            'decision cap reached while the frozen side moves: the unfinished episode steps of the last window are dropped from the update (pilot: neutral bootstrap = last value); learner next: exact bootstrap',
            'the last allowed episode flushes a short window; graceful wall-time stop at window boundaries',
            'progress.json / jsonl appends are fsync-ed; inference time and per-role decisions logged',
            'adapter-owned learner decisions excluded from the loss (unchanged rule) and tested'],
        pins=pins,
        data=dict(manifest=str(HB / 'data/manifest.json'), manifest_sha256=sha(HB / 'data/manifest.json'),
                  existing_manifest=hb_manifest['existing']['manifest'], existing_manifest_sha256=hb_manifest['existing']['sha256'],
                  sequences=40, existing=24, pairs=16, sources=list(RT.SOURCES), rehearsal_length=115,
                  masks='unchanged: loss = own-turn non-position decisions + scripted opponent-turn non-pass non-position; context = every player-0 decision',
                  excluded='validation hands, opponent future scripts, response sequences: not read'),
        seeds=dict(path=str(ROOT / 'data/seeds.json'), sha256=sha(ROOT / 'data/seeds.json'), training_seed=RT.TRAIN_SEED, rehearsal_seed=RT.REHEARSAL_SEED,
                   evaluation_seed=RT.EVAL_SEED, overlap=seeds['overlap']),
        training=dict(seed=RT.TRAIN_SEED, rehearsal_seed=RT.REHEARSAL_SEED, first_learner_seat=0, config=dict(M.CONFIG),
                      window_min_decisions=RT.WINDOW, cap_decisions=RT.CAP, cap_episodes=RT.EPISODE_CAP, max_game_decisions=RT.MAX_GAME_DECISIONS,
                      native_timeout_seconds=RT.NATIVE_TIMEOUT, sampling='both sides sample from their own final logits (one PRNG key from the training seed)',
                      device='GPU (one job; refuses if another compute process is on the GPU). If the GPU is unavailable before start: CPU with the same settings, decided before start, never switched mid-run',
                      wall=dict(graceful_stop_after_seconds=RT.WALL_LIMIT, job_timeout_seconds=3600, note='spec: 60 min on GPU (120 min on CPU)'),
                      candidate='final state after the last window; no checkpoint selection',
                      counts_logged=['environment decisions (learner/frozen)', 'PPO loss steps', 'rehearsal sequences and loss/context steps', 'updates', 'windows',
                                     'episode ends by kind', 'dropped steps (timeout / cut)', 'adapter-controlled learner steps', 'visibility events'],
                      on_error='no restart, no second candidate: the state is saved as train/stopped-state and the trial is reported as not established',
                      not_established_if=['clear exposure', 'environment/engine error', 'non-finite weights', 'saved files do not reproduce the in-memory logits/argmax/adapter actions']),
        evaluation=dict(
            battles=dict(games=128, pairs=RT.EVAL_PAIRS, seeds='data/seeds.json evaluation (pair i: candidate seat 0 then 1)', opponent='start model b9c7751f/77191d5d',
                         policies='artmage_entry.create_policy with explicit base/head hashes, positions=True (common adapter) for both; argmax; own RNN per seat, reset every game',
                         env='rlt_train.make_env: Artmage vs Artmage, fixture empty (normal dealing/draws), max_steps 4,000, timeout 30 s',
                         classification='scripts/evaluate.py rules (67db15ff): engine_timeout = termination_reason 2; step limit = termination_reason 1 or truncated; natural otherwise (win/loss by the actor-view reward, win_reason 1 LP / -1 deck-out or special)',
                         verified='seed pairs containing a timeout, a suspected timeout, a step-limit truncation or an error are excluded from the verified win rate and interval (raw kept)',
                         win_rate='wins / verified games; draws reported separately; interval: evaluate.paired_wilson_ci95 over verified pair scores (wins/2 per pair); seat Wilson intervals are references only; score rate with draws = 0.5 reported without an interval',
                         no_more='no confirmation games, no other opponents, no retries; a retention loss does not stop the battles',
                         wall='battle job stops starting games after 6,600 s (job timeout 7,000 s); retention job timeout 1,800 s; evaluation total <= 120 min',
                         records='per game: result, end kind, decisions/inference time per side, every decision (actor, message, action meaning, public chain), public board at every turn start (LP, hand/deck/extra counts, face-up field/graveyard/face-up banished codes; face-down cards without identity), visibility audit; the native snapshot is used for logs only'),
            retention=dict(cases=EV.RETAIN, runner='hb_runs.py 88e69412 (frozen copy) rt1 / free_run, output root rebound to this study',
                           baseline=start_retention_reuse(), start_successes_expected=12,
                           landing='RT1: goal check of the old manifest; F*-V: every validation-script turn-1 monster present at the end of own turn 1 (same rule as the response study)',
                           fixture='fixture duels: no normal draws (draw count 0); effect draws occur'),
            engine_cap=dict(total=141, battles=128, retention=13, retries=0)),
        limits=dict(memory_gib=5, swap=0, cpu_percent=300, nice=10, jobs=1, gpu_jobs=1),
        known_limits=['native ygopro.h: a natural draw with no winner returns reward -1 to the acting player (reward = winner == player ? 1 : -1); the API exposes no winner, so such a draw cannot be told from a loss (both in training and in scripts/evaluate.py)',
                      'no claim that information leakage is zero; visibility events are recorded with origin (selection_prompt / unverified)'],
        not_done=['adoption', 'production restart', 'normal operation', 'B2B', 'confirmation games', 'other candidates/seeds', 'LLM/API calls'],
    )
    (ROOT / 'registration.json').write_text(json.dumps(reg, ensure_ascii=False, indent=1) + '\n')
    print('REGISTERED', sha(ROOT / 'registration.json'), 'seeds', sha(ROOT / 'data/seeds.json'))
    print(json.dumps(reg['evaluation']['retention']['baseline'], indent=1)[:3000])


if __name__ == '__main__':
    if sys.argv[1:] != ['register']:
        raise SystemExit(__doc__)
    register()
