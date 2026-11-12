"""Offline full-action check (no engine): the new candidate is fed every own observation of each saved record from the
start (RNN reset per record, each observation once, through the 3-file entry with the position adapter); every chosen
action index is compared with the recorded executed index. A condition is reused only if ALL own actions match, the record
replayed exactly and its hashes are pinned. Writes check.json once, then amendment-2.json with the runs still needed.

  rs_check.py check
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import sa_model as SM  # noqa: E402
import pilot_data as PD  # noqa: E402
import numpy as np  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
CV = P / 'experiments/artmage-response-coverage-20261111'
UP = P / 'experiments/artmage-unseen-response-probe-20261111'
FLC8 = ['BV1-1cost-VARNISH-OM', 'BV1-1cost-VARNISH-OI', 'BV1-1cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OI',
        'BV1-2cost-VARNISH-OM', 'BV1-2cost-VARNISH-OI', 'BV1-2cost-ASSAULT-OM', 'BV1-2cost-ASSAULT-OI']
PROBE4 = ['H1-ASH', 'H1-GAMMA', 'H2-ASH', 'H2-GAMMA']
JST = timezone(timedelta(hours=9))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def obs_info(z, t):
    obs = {k: z[k][t:t + 1] for k in PD.OBS_KEYS}
    obs['mask_'] = None
    ch = z['chain'][t]
    info = dict(num_options=np.array([int(np.sum(z['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
    return obs, info


def check_record(d, policy):
    rec = json.loads((d / 'record.json').read_text())
    rep = json.loads((d / 'replay.json').read_text())
    replay_ok = (rep['mismatch'] is None and rep['failure'] is None and rep['observations_match'] and rep['checkpoints_equal'] and rep['game_over_equal']
                 and rep['final_equal'] in (True, None) and rep['reproduced_steps'] == rep['of'])
    z = {k: v for k, v in np.load(d / 'observations.npz').items()}
    rows = {r['step']: r for r in rec['trace']}
    policy.reset()
    first = None
    gated = []
    for t, st in enumerate(z['steps'].tolist()):
        obs, info = obs_info(z, t)
        a = int(policy.act(obs, info))
        r = rows[st]
        assert r['selected']['index'] == int(z['labels'][t])
        if SM.gate(z['actions_'][t], z['chain'][t]):
            acts = {x['index']: x for x in r['snapshot']['actions']}
            gated.append(dict(step=st, new=(acts[a]['act'], acts[a]['code'], acts[a].get('effect')),
                              recorded=(r['selected']['act'], r['selected']['code'], r['selected'].get('effect'))))
        if a != int(z['labels'][t]) and first is None:
            acts = {x['index']: x for x in r['snapshot']['actions']}
            first = dict(step=st, npz_index=t, message=r['snapshot']['message'], new=dict(index=a, act=acts[a]['act'], code=acts[a]['code'], effect=acts[a].get('effect')),
                         recorded=dict(index=r['selected']['index'], act=r['selected']['act'], code=r['selected']['code'], effect=r['selected'].get('effect')))
    return dict(record=str(d), record_sha256=sha(d / 'record.json'), observations_sha256=sha(d / 'observations.npz'), replay_sha256=sha(d / 'replay.json'),
                replay_ok=replay_ok, own_decisions=len(z['steps']), all_actions_equal=first is None, first_difference=first, gated=gated,
                reuse=first is None and replay_ok)


def main():
    out_path = ROOT / 'check.json'
    if out_path.exists() or (ROOT / 'amendment-2.json').exists():
        raise SystemExit('check exists')
    result = json.loads((ROOT / 'train/result.json').read_text())
    assert result['status'] == 'complete'
    policy = SM.create_adapter_policy(ROOT / 'train/candidate', positions=True)
    checks = {}
    for n in FLC8:
        checks[f'flc:{n}'] = check_record(CV / 'runs' / f'B-{n}', policy)
    for n in PROBE4:
        checks[f'probe:{n}'] = check_record(UP / 'runs' / f'{n}-CAND', policy)
    out = dict(method='offline: the new 3-file candidate through every own observation of each saved record (position adapter on), action indices compared',
               candidate=policy.policy_metadata, checks=checks)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + '\n')
    flc_to_run = [n for n in FLC8 if not checks[f'flc:{n}']['reuse']]
    probe_to_run = ['H1-USAGI-BASE', 'H1-USAGI-CAND'] + [f'{n}-CAND' for n in PROBE4 if not checks[f'probe:{n}']['reuse']]
    reg = json.loads((ROOT / 'registration.json').read_text())
    am = dict(amendment=2, registered_jst=datetime.now(JST).isoformat(timespec='seconds'), registration_sha256=sha(ROOT / 'registration.json'),
              amendment_1_sha256=sha(ROOT / 'amendment-1.json'), change='runs still needed after the offline full-action check (no engine); conditions unchanged',
              check=dict(file=str(out_path), sha256=sha(out_path)), reused={k: v['record'] for k, v in checks.items() if v['reuse']},
              flc_to_run=flc_to_run, probe_to_run=probe_to_run, code=reg['code'])
    (ROOT / 'amendment-2.json').write_text(json.dumps(am, ensure_ascii=False, indent=1) + '\n')
    for k, v in checks.items():
        print('CHECK', k, 'reuse' if v['reuse'] else 'RUN', v['own_decisions'], json.dumps(v['first_difference'], ensure_ascii=False), [(g['step'], g['new'], g['recorded']) for g in v['gated']])
    print('AMENDMENT-2', sha(ROOT / 'amendment-2.json'), 'flc_to_run', flc_to_run, 'probe_to_run', probe_to_run)


if __name__ == '__main__':
    {'check': main}[sys.argv[1]]()
