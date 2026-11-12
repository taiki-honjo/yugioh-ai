"""Engine side of the opening-capability-eval-v1 shadow evaluation: records, replay check, O1 probe and the
conditional model-use test, for two TRAIN-split cases only.

Shadow evaluation: own output root, production files are only read (pinned snapshot code, adopted model),
no training, no battles, no change to the gate, rewards, rehearsal data or the production root.
Every new real-engine run (one duel from the opening hand) is counted in engine-runs.jsonl; the limit is 12.

  probe.py launch <record|replay|model-use> <P1|P2>   # runs the job under the same environment as gate jobs
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

P = Path('/home/hatk8/projects/yugioh-ai')
OUT = P / 'experiments/capability-probe-20261109'
CONFIG = P / 'config/continuous-elfnotes.json'
ADOPTED = P / 'experiments/adoption-d16384-20261109/model'
CASES = {'P1': 'cue-reco-jj-zaren-g00-p0', 'P2': 'elf-rukina-regina-g00-p0'}
KEY_CARDS = [84815190, 50954680, 24092792, 88170262, 14558127, 13597785, 4891376, 14442329, 78058681]
OPP_STEP_CAP = 250
sys.path.insert(0, str(OUT / 'code'))
sys.path.insert(0, str(P / 'scripts'))
import analysis as A  # noqa: E402
import continuous_learning as C  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_json(path, json.loads(json.dumps(value, default=str)))


def setup():
    cfg = C.load_config(CONFIG)
    root = Path(cfg['root'])
    C.runtime_check(cfg)  # pinned runtime + snapshot hashes, read only
    sys.path.insert(0, str(root / 'snapshot/policy-code'))
    import all_openings_common as common  # same import as the gate's openings_job (installs teacher preferences)
    import all_opening_cases as cases
    for name in ['all_openings_common', 'all_opening_witness', 'all_opening_cases', 'rukina_recovery_witness']:
        assert Path(sys.modules[name].__file__).resolve().is_relative_to((root / 'snapshot/policy-code').resolve()), name
    return cfg, root, common, cases


def train_case(root, label):
    manifest = json.loads((root / 'snapshot/opening-manifest.json').read_text())['decks']['ElfnotesKewlTune']['cases']
    case = next(c for c in manifest['train'] if c['case_id'] == CASES[label])
    key = tuple(sorted(case['hand']))
    leaked = [s for s in ['validation', 'holdout'] if any(tuple(sorted(c['hand'])) == key for c in manifest[s])]
    assert case.get('split', 'train') == 'train' and not leaked, (case['case_id'], leaked)
    return case


def versions(cfg, root):
    hashes = json.loads((root / 'snapshot/hashes.json').read_text())
    runtime = json.loads((root / 'snapshot/runtime.json').read_text())
    scripts = Path(cfg['runtime']) / 'agent/scripts/script'
    return dict(snapshot_hashes_sha256=C.sha(root / 'snapshot/hashes.json'),
                policy_code={k: v for k, v in hashes.items() if k.startswith('policy-code/') and k.endswith('.py')},
                native_sha256=C.sha(runtime['native_binary']), deck_sha256=C.sha(runtime['deck']),
                card_scripts={code: C.sha(scripts / f'c{code}.lua') for code in KEY_CARDS},
                card_script_dir=str(scripts.resolve()),
                analysis_sha256=C.sha(OUT / 'code/analysis.py'), probe_sha256=C.sha(Path(__file__)))


def name_actions(row):
    """Add the card name to the executed action (the snapshot's action list already carries names)."""
    selected = dict(row['selected'])
    named = row['snapshot']['actions'][selected['index']] if selected.get('index') is not None and selected['index'] < len(row['snapshot']['actions']) else {}
    selected['name'] = named.get('name', '')
    return dict(row, selected_meaning=A.meaning(selected))


class Engine:
    def __init__(self, common):
        self.w = common.v                 # all_opening_witness
        self.v = common.v.v               # rukina_recovery_witness
        self.native = common.v.native
        self.common = common

    def snapshot(self):
        return self.v.compact(self.native.benchmark_snapshot())


def o1_opponent(engine, s, state):
    """The existing O1 opponent of all_opening_witness.run(): Rukina (1) to the centre, then Rukina (2),
    search target Spirit when offered; everything else passive."""
    v = engine.v
    if s['turn_player'] == 1 and s['phase'] == 4:
        target = state['steps'][0] if state['steps'] else None
        found = next((a for a in s['actions'] if target and all(a[k] == value for k, value in target.items())), None)
        if found:
            state['steps'].pop(0)
            if found['act'] == 'Activate':
                state['search_started'] = True
            return found['index']
        if s['message'] == 'select_place':
            return next(a['index'] for a in s['actions'] if a['place'] == 3)
        if s['message'] == 'select_position':
            return 0
        if s['message'] in ['select_card', 'select_unselect_card']:
            return next((a['index'] for a in s['actions'] if a['code'] == v.SPIRIT), 0)
    return v.passive(s)


def forced_observe(common, policy, obs, info, idx):
    """Teacher-forced history for the model: the RNN state is advanced on the same observations the executed
    route produced (the state update does not depend on the chosen action), and the placement helper's memory
    observes the EXECUTED action. Not replayed: placement 'events' for forced msg-11 choices."""
    import numpy as np
    placement = policy.base
    core = placement.base
    core.state, _ = core.infer(core.base, core.head, core.state, obs, common.features(info))
    actions = np.asarray(obs['actions_'])[0]
    cards = np.asarray(obs['cards_'])[0]
    row = actions[idx]
    code = placement.id_codes.get(int(row[1]) * 256 + int(row[2]), 0)
    spec = int(row[0])
    hand = bool(spec and spec <= len(cards) and int(cards[spec - 1, 2]) == 2 and int(cards[spec - 1, 4]) == 0)
    placement.memory.observe(int(row[3]), int(row[4]), code, hand, int(np.asarray(info['public_chain_length'])[0]))


def drive(engine, case, record, mode, policy=None):
    """Replay the recorded index sequence while checking the observable snapshot and the action meaning at every
    step. mode 'replay': the whole record. mode 'o1': own turn from the record, then the O1 opponent with our side
    passing (candidates listed). mode 'model': own turn from the record (teacher-forced model history), then the O1
    opponent with our side decided by the model."""
    import numpy as np
    env, obs, info = engine.v.create(case)
    audit = engine.w.VisibilityAudit()
    rows, mismatch, failure = [], None, None
    trace = record['trace']
    prefix = trace if mode == 'replay' else [r for r in trace if r['snapshot']['turn_player'] == 0]
    if mode != 'replay':
        assert len(prefix) == next((i for i, r in enumerate(trace) if r['snapshot']['turn_player'] == 1), len(trace))
    try:
        for row in prefix:
            raw = engine.native.benchmark_snapshot()
            s = engine.v.compact(raw)
            idx = row['selected']['index']
            if idx >= len(raw['actions']):
                mismatch = dict(step=row['step'], keys=['action_index_out_of_range']); break
            keys = A.compare_step(row, s, raw['actions'][idx])
            if keys:
                mismatch = dict(step=row['step'], keys=keys); break
            if raw['to_play'] == 0:
                audit.check(obs, info, raw)
                if mode == 'model':
                    forced_observe(engine.common, policy, obs, info, idx)
            rows.append(dict(step=row['step'], snapshot=s, selected=raw['actions'][idx], source='record'))
            obs, _, term, trunc, info = env.step(np.array([idx], np.int32))
            if term[0] or trunc[0]:
                failure = 'terminated_or_truncated'; break
        replay_end = engine.snapshot()
        live = dict(steps=[engine.v.a('SpSummon', engine.v.RUK), engine.v.a('Activate', engine.v.RUK)], search_started=False)
        if mode != 'replay' and mismatch is None and failure is None:
            step = rows[-1]['step'] + 1 if rows else 0
            for _ in range(OPP_STEP_CAP):
                raw = engine.native.benchmark_snapshot()
                if live['search_started'] and raw['chain_size'] == 0:
                    break
                if raw['turn_count'] > 2:
                    failure = 'opponent_turn_passed'; break
                if raw['to_play'] == 0:
                    audit.check(obs, info, raw)
                    idx = policy.act(obs, info) if mode == 'model' else engine.v.passive(raw)
                    source = 'model' if mode == 'model' else 'probe-pass'
                else:
                    idx = o1_opponent(engine, raw, live); source = 'o1-opponent'
                rows.append(dict(step=step, snapshot=engine.v.compact(raw), selected=raw['actions'][idx], source=source))
                obs, _, term, trunc, info = env.step(np.array([idx], np.int32)); step += 1
                if term[0] or trunc[0]:
                    failure = 'terminated_or_truncated'; break
            else:
                failure = 'opponent_step_cap'
    except Exception as exc:  # recorded, never counted as a capability result
        failure = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    final = engine.snapshot()
    lua = engine.native.get_lua_error_count()
    env.close(); del env
    import gc; gc.collect()
    assert engine.native.get_env_resource_stats()['live_duels'] == 0
    return dict(rows=rows, final=final, replay_end=replay_end, mismatch=mismatch, failure=failure, lua_errors=lua,
                visibility_audit=dict(checks=audit.checks, legal_hand_peek_rows=audit.hand_peeks, legal_deck_peek_rows=audit.deck_peeks))


def analyse(rows, final):
    return dict(o1=A.o1_stages(rows, final), windows=A.windows(rows), baronne=A.baronne_checks(rows))


def main(command, label):
    cfg, root, common, cases = setup()
    raw_case = train_case(root, label)
    case = cases.normalize(raw_case)
    if command == 'check':  # imports, pins, case isolation and versions only; no duel is created
        print('CHECK', json.dumps(dict(case_id=raw_case['case_id'], kind=raw_case['kind'], hand=raw_case['hand'], seed=raw_case['seed'],
                                       versions=versions(cfg, root), runs=len(A.Ledger(OUT / 'engine-runs.jsonl').started()))), flush=True)
        return
    engine = Engine(common)
    ledger = A.Ledger(OUT / 'engine-runs.jsonl')
    meta = dict(label=label, case_id=raw_case['case_id'], split='train', family=raw_case['eval_family'], kind=raw_case['kind'],
                hand=raw_case['hand'], seed=raw_case['seed'], arm=raw_case.get('arm'), versions=versions(cfg, root),
                teacher_route=True, model_received_goals_or_plans=False)
    record_path = OUT / 'records' / f'{label}-{raw_case["case_id"]}.json'
    if command == 'record':
        run = ledger.start('record (teacher route through run())', raw_case['case_id'], C.now())
        try:
            result, _ = engine.w.run(case, policy=None)
            trace = [name_actions(r) for r in result['trace']]
            legacy = {k: result.get(k) for k in ['passed', 'board', 'chosen_variant', 'failure', 'jj_release_step', 'ash_used_as_synchro_material',
                                                 'recovered_ash_activated', 'opponent_search_negated', 'opponent_search_resolved',
                                                 'crystal_first_preference_met', 'boss_field_arrival_order', 'normal_summons_diagnostic', 'lua_errors']}
            out = dict(meta, engine_run=run, purpose='record', legacy=legacy, visibility_audit=result['visibility_audit'],
                       own_end=result['own_end'], final=result['final'], trace=trace, analysis=analyse(result['trace'], result['final']),
                       note='teacher-built state for spec development; not the model reaching it by itself')
            write(record_path, out)
            ledger.finish(run, 'ok', C.now(), legacy_passed=legacy['passed'], failure=legacy['failure'])
            print('RECORD', json.dumps(dict(run=run, legacy=legacy, o1=out['analysis']['o1']['outcome'])), flush=True)
        except BaseException as exc:
            ledger.finish(run, 'error', C.now(), error=f'{type(exc).__name__}: {exc}'); raise
        return
    record = json.loads(record_path.read_text())
    policy = None
    if command == 'model-use':
        backend = C.backend()
        policy = backend.create_policy(cfg, ADOPTED)
        policy.reset()
        import jax
        meta.update(model=dict(path=str(ADOPTED), base_sha256=C.sha(ADOPTED / 'base.flax_model'),
                               head_sha256=C.sha(ADOPTED / 'chain-head.flax_model'), device=str(jax.devices()[0]),
                               adapters='placement v2 + position (production create_policy)'),
                    rnn_history='teacher-forced over the recorded own turn; model decides from the opponent turn on',
                    teacher_route=True)
    mode = dict(replay='replay' if label == 'P1' else 'o1', **{'model-use': 'model'})[command]
    purpose = dict(replay='replay check (whole record)', o1='replay check (own turn) + O1 probe, our side passes',
                   model='conditional model-use: teacher-forced own turn, model in the opponent turn')[mode]
    run = ledger.start(purpose, raw_case['case_id'], C.now())
    try:
        got = drive(engine, case, record, mode, policy)
        rows = [name_actions(r) for r in got['rows']]
        reproduced_prefix = sum(1 for r in got['rows'] if r['source'] == 'record')
        out = dict(meta, engine_run=run, purpose=purpose, mode=mode, mismatch=got['mismatch'], failure=got['failure'],
                   lua_errors=got['lua_errors'], visibility_audit=got['visibility_audit'], reproduced_steps=reproduced_prefix,
                   final=got['final'], trace=rows,
                   analysis=analyse(got['rows'], got['final']) if mode != 'replay' else None)
        if mode == 'replay':
            out['final_matches_record'] = A.normalized(got['replay_end']) == A.normalized(record['final'])
            out['analysis'] = analyse(got['rows'], got['replay_end'])
        else:
            out['own_end_matches_record'] = A.normalized(got['replay_end']) == A.normalized(record['own_end'])
        write(OUT / {'replay': 'replays', 'o1': 'probes', 'model': 'model-use'}[mode] / f'{label}-{raw_case["case_id"]}.json', out)
        ledger.finish(run, 'ok' if got['mismatch'] is None and got['failure'] is None else 'completed-with-findings', C.now(),
                      mismatch=got['mismatch'], failure=got['failure'])
        print('RESULT', json.dumps(dict(run=run, mode=mode, mismatch=got['mismatch'], failure=got['failure'], reproduced=reproduced_prefix,
                                        final_matches=out.get('final_matches_record'), own_end_matches=out.get('own_end_matches_record'),
                                        o1=(out['analysis'] or {}).get('o1', {}).get('outcome'))), flush=True)
    except BaseException as exc:
        ledger.finish(run, 'error', C.now(), error=f'{type(exc).__name__}: {exc}'); raise


def launch(command, label):
    cfg = C.load_config(CONFIG)
    env = C.job_env(cfg)
    log = OUT / 'logs' / f'{command}-{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f"MemoryMax={cfg['job_memory_gib']}G", '-p', 'MemorySwapMax=0',
           '-p', f"CPUQuota={cfg['cpu_percent']}%", 'nice', '-n', '10', cfg['python'], '-B', '-u', str(Path(__file__).resolve()),
           'job', command, label]
    with log.open('w') as handle:
        done = subprocess.run(cmd, cwd=Path(cfg['runtime']) / 'agent/scripts', env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(''.join(l for l in log.read_text().splitlines(True) if l.startswith(('CHECK', 'RECORD', 'RESULT', 'Traceback', 'AssertionError', 'RuntimeError', 'KeyError', 'TypeError'))), end='')
    print(f'exit={done.returncode} log={log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'job'])
    parser.add_argument('command', choices=['check', 'record', 'replay', 'model-use'])
    parser.add_argument('label', choices=sorted(CASES))
    args = parser.parse_args()
    (launch if args.mode == 'launch' else main)(args.command, args.label)
