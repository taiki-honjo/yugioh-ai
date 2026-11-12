"""Offline reuse check (no engine, no model): which probe conditions are the SAME decision sequence as an old record.

For each site, the old comparison record (BV1-1cost-VARNISH-OM / BV1-2cost-ASSAULT-OM) is hash-pinned and its replay
result read. The registered intervention rules (otp_runs.Intervener) are applied to every own post-branch decision of
the old trace with the recorded action as the model's choice. If no rule ever executes a different action, the probe
condition would feed the engine the identical action sequence from the identical fixture/seed (the old replay showed
the engine reproduces a recorded sequence exactly), so the old record stands for it and no new run is made.
Writes reuse.json once.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import otp_runs as O  # noqa: E402

OLD = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_divergence(trace, d1, r1):
    iv, probe = O.Intervener(d1, r1), O.Probe()
    branched, divergence = False, None
    for r in trace:
        s = dict(r['snapshot'], _step=r['step'])
        probe(s)
        if r['who'] == 'branch-control':
            branched = True
            continue
        if s['to_play'] != 0 or not branched or divergence:
            continue
        forced = iv(s, r['step'], r['selected']['index'])
        if forced is not None and forced != r['selected']['index']:
            x = iv.log[-1]
            divergence = dict(step=r['step'], rule=x['rule'], recorded=O.meaning(r['selected']),
                              rule_executes=next(O.meaning(a) for a in s['actions'] if a['index'] == forced))
    return divergence, iv.log, probe.cp


def main():
    out_path = ROOT / 'reuse.json'
    if out_path.exists():
        raise SystemExit('reuse.json exists; not rewritten')
    amendment = json.loads((OLD / 'amendment-1.json').read_text())
    ledger = [json.loads(l) for l in (OLD / 'eval-runs.jsonl').read_text().splitlines()]
    sites = {}
    for site, old in O.SITES.items():
        d = OLD / 'runs' / old
        rec = json.loads((d / 'record.json').read_text())
        rep = json.loads((d / 'replay.json').read_text())
        runs = {e['run']: e for e in ledger if e.get('event') == 'start'}
        status = {}
        for i, e in enumerate(ledger):
            if e.get('event') == 'start' and e['case_id'] == old:
                status[e['purpose'].split()[0]] = dict(run=e['run'], status=ledger[i + 1].get('status'))
        replay_ok = rep['mismatch'] is None and rep['failure'] is None and rep['checkpoints_equal'] and rep['game_over_equal'] and rep['observations_match'] \
            and rep['final_equal'] in (True, None) and rep['reproduced_steps'] == rep['of']
        conds = {}
        for cond in ['D0R0', 'D0R1', 'D1R0', 'D1R1']:
            d1, r1 = cond[1] == '1', cond[3] == '1'
            div, log, cp = first_divergence(rec['trace'], d1, r1)
            conds[cond] = dict(first_divergence=div, rule_log=[{k: x[k] for k in x if k != 'legal'} for x in log],
                               reuse_old_record=div is None and replay_ok and rec['model_sha256'] == list(O.MODEL_SHA),
                               medius_window_step=(cp.get('medius_response_window') or {}).get('step'),
                               after_medius_chain_step=(cp.get('after_medius_chain') or {}).get('step'))
        sites[site] = dict(old_condition=old, files={p: sha(d / p) for p in ['record.json', 'observations.npz', 'replay.json']},
                           old_engine_runs=status, replay=dict({k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal',
                                                                                    'observations_match', 'reproduced_steps', 'of']}, ok=replay_ok),
                           model_sha256=rec['model_sha256'], model_matches=rec['model_sha256'] == list(O.MODEL_SHA),
                           old_record_who_labels='prefix-replay / branch-control / model / opponent-*; a reused R1 decision is labelled "model" there',
                           conditions=conds)
    out = dict(method='offline: hash-pinned old records; registered rules applied to the recorded post-branch own decisions; no engine, no model',
               old_root=str(OLD), old_registration_sha256=sha(OLD / 'registration.json'), old_amendment_sha256=sha(OLD / 'amendment-1.json'),
               old_runner_executed_sha256=amendment['code']['flc_runs.py'], frozen_runner_copy_sha256=sha(ROOT / 'code/flc_runs.py'),
               new_runner_sha256=sha(ROOT / 'code/otp_runs.py'), sites=sites,
               note='The new runner differs in code (hash above); equality is established for the executed decision sequence, not claimed for the code.')
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    for site, x in sites.items():
        print(site, x['old_condition'], 'replay_ok', x['replay']['ok'], 'model', x['model_matches'])
        for cond, y in x['conditions'].items():
            print('  ', cond, 'reuse' if y['reuse_old_record'] else 'NEW RUN', json.dumps(y['first_divergence'], ensure_ascii=False),
                  [(z['rule'], z['step'], z['executed'], z.get('model_agrees')) for z in y['rule_log']][:3], 'window', y['medius_window_step'], 'after', y['after_medius_chain_step'])
    print('REUSE', sha(out_path))


if __name__ == '__main__':
    main()
