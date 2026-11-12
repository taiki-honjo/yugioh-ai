"""Offline summary of the relative-score retention study (no engine). Scores at gated screens are recomputed by teacher-forcing
the frozen Policy along the saved NPZ (CPU) and applying the NEW adapter (c110f964) and, for reference only, the coverage adapter
(00977941) to the same inputs."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import rs_probe_runs as PR  # noqa: E402
import sa_model as SM  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from opponent_article_bc_common import Policy, load  # noqa: E402

P = Path('/home/hatk8/projects/yugioh-ai')
CV = P / 'experiments/artmage-response-coverage-20261111'
UP = P / 'experiments/artmage-unseen-response-probe-20261111'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
NAME = {(23434538, 10010): 'G', (0, -1): 'pass', (27184601, 10011): 'ディアクトロス②', (34541940, 10011): 'ファインメルト③', (23829452, 10010): '獄神獣②',
        (59438930, 0): 'うさぎ', (44654994, 10010): '奪還①', (31822037, 10011): '獄神使②', (97434754, 10011): 'リテラ③', (23599634, 10011): '覚醒②'}
OBS_KEYS = ('cards_', 'global_', 'actions_', 'h_actions_')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def lab(a):
    return NAME.get((a['code'], a.get('effect')), f"{a.get('name') or a['code']}:e{a.get('effect')}")


def obs_info(z, t):
    obs = {k: z[k][t:t + 1] for k in OBS_KEYS}
    obs['mask_'] = None
    ch = z['chain'][t]
    return obs, dict(num_options=np.array([int(np.sum(z['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                     public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                     public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))


def gated_screens(d, base, adapters):
    rec = read(d / 'record.json')
    z = {k: v for k, v in np.load(d / 'observations.npz').items()}
    rows = {r['step']: r for r in rec['trace']}
    base.reset()
    out = []
    for t, st in enumerate(z['steps'].tolist()):
        obs, info = obs_info(z, t)
        base.state, logits = base.infer(base.base, base.head, base.state, obs, SM.features(info))
        if not SM.gate(z['actions_'][t], z['chain'][t]):
            continue
        f = np.asarray(logits)[0]
        acts = rows[st]['snapshot']['actions']
        legal = [a['index'] for a in acts]
        pi = next((a['index'] for a in acts if a['act'] == 'Cancel'), None)
        res = dict(step=st, latest_chain_card=rows[st]['snapshot']['public_chain'][-1]['code'], executed=lab(rows[st]['selected']), who=rows[st]['who'], adapters={})
        for name, ad in adapters.items():
            d_ = np.asarray(SM.adapter_delta(ad, jnp.asarray(z['chain'][t][None]), jnp.asarray(z['actions_'][t][None])))[0]
            fin = f + d_
            v0 = np.array([f[i] for i in legal]); p0 = np.exp(v0 - v0.max()); p0 /= p0.sum()
            v1 = np.array([fin[i] for i in legal]); p1 = np.exp(v1 - v1.max()); p1 /= p1.sum()
            opts = []
            for j, a in enumerate(acts):
                i = a['index']
                opts.append(dict(option=lab(a), spec=a.get('spec'), frozen_logit=float(f[i]), delta=float(d_[i]),
                                 relative_delta_vs_pass=float(d_[i] - d_[pi]) if pi is not None else None,
                                 old_margin_vs_pass=float(f[i] - f[pi]) if pi is not None else None,
                                 new_margin_vs_pass=float(fin[i] - fin[pi]) if pi is not None else None, p_frozen=float(p0[j]), p_final=float(p1[j])))
            res['adapters'][name] = dict(choice=lab(acts[int(np.argmax(v1))]), frozen_choice=lab(acts[int(np.argmax(v0))]), options=opts)
        out.append(res)
    return out


def identical(a, b):
    if len(a['trace']) != len(b['trace']):
        return False
    return all(x['snapshot'] == y['snapshot'] and x['selected']['index'] == y['selected']['index'] for x, y in zip(a['trace'], b['trace']))


def outcome_probe(rec):
    cp = rec['checkpoints']
    a, b = cp.get('opponent_activation') or {}, cp.get('target_chain_resolved') or {}
    return dict(hand_before=a.get('hand'), hand_after=b.get('hand'), monsters_after=b.get('monsters'), graveyard_after=b.get('graveyard'),
                opponent_monsters_after=b.get('opponent_monsters'), opponent_gy_after=b.get('opponent_graveyard'),
                g_kept=PR.G in (b.get('hand') or []), medius_on_field=PR.MEDIUS in (b.get('monsters') or []), medius_in_gy=PR.MEDIUS in (b.get('graveyard') or []))


def main():
    base = Policy(str(PR.BASE[0]), str(PR.BASE[1]))
    new = load(ROOT / 'train/candidate/response-adapter.flax_model')
    cov = load(CV / 'train/candidate/response-adapter.flax_model')
    ads = {'new': new, 'coverage': cov}
    res = read(ROOT / 'train/result.json')
    upd = [json.loads(l) for l in (ROOT / 'train/updates.jsonl').read_text().splitlines()]
    check = read(ROOT / 'check.json')
    am2 = read(ROOT / 'amendment-2.json')
    # training screens: new vs coverage relative deltas from the cache
    z = np.load(ROOT / 'cache/screens.npz')
    train_screens = {}
    for k in ['S1', 'S2', 'M', 'P']:
        prot, pi = z[f'{k}__protect'], int(z[f'{k}__pass'])
        row = {}
        for name, ad in ads.items():
            d_ = np.asarray(SM.adapter_delta(ad, jnp.asarray(z[f'{k}__chain'][None]), jnp.asarray(z[f'{k}__actions'][None])))[0]
            rel = d_ - d_[pi]
            row[name] = dict(max_abs_rel_protected=float(np.abs(rel[prot]).max()), mean_abs_rel_protected=float(np.abs(rel[prot]).mean()),
                             rel={int(i): float(rel[i]) for i in np.flatnonzero(z[f'{k}__legal'])}, pass_delta=float(d_[pi]))
        train_screens[k] = row
    # existing 12 + new runs
    flc = {}
    for n in [x.split(':', 1)[1] for x in check['checks'] if x.startswith('flc:')]:
        reused = check['checks'][f'flc:{n}']['reuse']
        d = (CV / 'runs' / f'B-{n}') if reused else (ROOT / 'runs' / f'B-{n}')
        rec = read(d / 'record.json')
        flc[n] = dict(source='reused coverage record (all own actions equal)' if reused else 'new record + replay (this study)', record=str(d),
                      identical_to_coverage_record=identical(rec, read(CV / 'runs' / f'B-{n}' / 'record.json')),
                      identical_to_start_model=identical(rec, read(FLC / 'runs' / n / 'record.json')),
                      replay=None if reused else read(d / 'replay.json'), screens=gated_screens(d, base, ads),
                      t3_lp=(rec['checkpoints'].get('own_turn3_end') or {}).get('lp1') if 'own_turn3_end' in rec['checkpoints'] else None, game_over=rec['game_over'])
    probe = {}
    for n in ['H1-ASH', 'H1-GAMMA', 'H2-ASH', 'H2-GAMMA']:
        d = UP / 'runs' / f'{n}-CAND'
        probe[n] = dict(source='reused unseen-probe record (all own actions equal)', screens=gated_screens(d, base, ads), outcome=outcome_probe(read(d / 'record.json')))
    for pol in ['BASE', 'CAND']:
        d = ROOT / 'runs' / f'H1-USAGI-{pol}'
        rec = read(d / 'record.json')
        probe[f'H1-USAGI-{pol}'] = dict(source='new record + replay', policy=rec['policy'] and {k: rec['policy'].get(k) for k in ['type', 'adapter_sha256', 'base_sha256', 'head_sha256']},
                                         adapter_active=(rec['policy'] or {}).get('adapter_active'), target=rec['target'], reason=rec['reason'],
                                         opponent=[e['what'] for e in rec['opponent_events']], screens=gated_screens(d, base, ads), outcome=outcome_probe(rec),
                                         replay=read(d / 'replay.json'))
    b, c = read(ROOT / 'runs/H1-USAGI-BASE/record.json'), read(ROOT / 'runs/H1-USAGI-CAND/record.json')
    probe['H1-USAGI-BASE-vs-CAND-identical'] = identical(b, c)
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_1_sha256=sha(ROOT / 'amendment-1.json'), amendment_2_sha256=sha(ROOT / 'amendment-2.json'),
                   engine=dict(used=len(starts), cap=30, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e))),
                   training=dict(screens=res['screens'], clipped=res['clipped_updates'], post=res['post_training']['comparison'],
                                 position_off=res['post_training']['position_actions_off_target_differences'], reload=res['saved']['reload'],
                                 adapter=res['saved']['adapter_sha256'], manifest=res['saved']['manifest_sha256'],
                                 terms=dict(ce=[upd[0]['ce'], upd[-1]['ce']], kl=[upd[0]['kl'], upd[-1]['kl']], rel=[upd[0]['rel'], upd[-1]['rel']], rel_max=max(u['rel'] for u in upd)),
                                 grad_norms={k: dict(mean=float(np.mean([u[k] for u in upd])), max=float(max(u[k] for u in upd))) for k in ['ce_grad_norm', 'kl_grad_norm', 'rel_grad_norm', 'grad_norm']},
                                 update_norm_mean=float(np.mean([u['update_norm'] for u in upd]))),
                   train_screens_relative=train_screens, check=dict(reused=am2['reused'], flc_to_run=am2['flc_to_run'], probe_to_run=am2['probe_to_run'],
                                                                    first_differences={k: v['first_difference'] for k, v in check['checks'].items() if not v['reuse']}),
                   flc=flc, probe=probe)
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + '\n')
    print('ENGINE', summary['engine'])
    for k, v in train_screens.items():
        print('TRAIN', k, {n: (round(x['max_abs_rel_protected'], 4), round(x['pass_delta'], 2)) for n, x in v.items()})
    for n, v in flc.items():
        sc = v['screens'][0] if v['screens'] else None
        print('FLC', n, v['source'][:28], 'id_cov', v['identical_to_coverage_record'], 'id_start', v['identical_to_start_model'], '| exec', sc and sc['executed'],
              '| new', sc and sc['adapters']['new']['choice'], 'cov', sc and sc['adapters']['coverage']['choice'],
              '| rel new', sc and {o['option']: round(o['relative_delta_vs_pass'], 2) for o in sc['adapters']['new']['options']},
              '| rel cov', sc and {o['option']: round(o['relative_delta_vs_pass'], 2) for o in sc['adapters']['coverage']['options']})
    for n, v in probe.items():
        if not isinstance(v, dict):
            print('PROBE', n, v); continue
        for sc in v['screens']:
            print('PROBE', n, 'latest', sc['latest_chain_card'], 'exec', sc['executed'], {a: [(o['option'], round(o['frozen_logit'], 2), round(o['delta'], 2), round(o['relative_delta_vs_pass'], 2),
                                                                                                round(o['old_margin_vs_pass'], 2), round(o['new_margin_vs_pass'], 2), round(o['p_final'], 5)) for o in x['options']]
                                                                                           for a, x in sc['adapters'].items()})
        print('   outcome', json.dumps(v.get('outcome'), ensure_ascii=False))
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
