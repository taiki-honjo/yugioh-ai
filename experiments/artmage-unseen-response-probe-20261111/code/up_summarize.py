"""Offline summary of the unseen-response probe (no engine). The frozen logits at each gated screen are recomputed by
teacher-forcing the frozen Policy along the saved NPZ (CPU, the evaluation backend); the adapter delta is the coverage
candidate's (00977941) on the same inputs (actual for CAND runs, counterfactual for BASE / control runs)."""
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-unseen-response-probe-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import up_runs as U  # noqa: E402
import sa_model as SM  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from opponent_article_bc_common import Policy, load  # noqa: E402

LABEL = {(23434538, 10010): 'G', (0, -1): 'pass'}
OBS_KEYS = ('cards_', 'global_', 'actions_', 'h_actions_')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def lab(a):
    return LABEL.get((a['code'], a.get('effect'))) or f"{a.get('name') or a['code']}:e{a.get('effect')}"


def gated(s):
    ch = s['public_chain']
    return bool(s['message'] == 'select_chain' and ch and ch[-1]['player'] != s['to_play'])


def obs_info(z, t):
    obs = {k: z[k][t:t + 1] for k in OBS_KEYS}
    obs['mask_'] = None
    ch = z['chain'][t]
    info = dict(num_options=np.array([int(np.sum(z['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
    return obs, info


def screens(name, base, adapter):
    """Every gated own screen of the record: legal options with frozen logit, delta, relative delta vs pass, margins, probabilities."""
    d = ROOT / 'runs' / name
    rec = read(d / 'record.json')
    z = {k: v for k, v in np.load(d / 'observations.npz').items()}
    rows = {r['step']: r for r in rec['trace']}
    out = []
    base.reset()
    for t, st in enumerate(z['steps'].tolist()):
        obs, info = obs_info(z, t)
        base.state, logits = base.infer(base.base, base.head, base.state, obs, SM.features(info))
        snap = rows[st]['snapshot']
        if not SM.gate(z['actions_'][t], z['chain'][t]):
            continue
        assert gated(snap)
        f = np.asarray(logits)[0]
        delta = np.asarray(SM.adapter_delta(adapter, jnp.asarray(z['chain'][t][None]), jnp.asarray(z['actions_'][t][None])))[0]
        acts = snap['actions']
        legal = [a['index'] for a in acts]
        pass_i = next((a['index'] for a in acts if a['act'] == 'Cancel'), None)
        fin = f + delta

        def probs(x):
            v = np.array([x[i] for i in legal]); v = np.exp(v - v.max()); v /= v.sum()
            return {i: float(p) for i, p in zip(legal, v)}
        p0, p1 = probs(f), probs(fin)
        opts = []
        for a in acts:
            i = a['index']
            rel = float(delta[i] - delta[pass_i]) if pass_i is not None else None
            old_m = float(f[i] - f[pass_i]) if pass_i is not None else None
            opts.append(dict(index=i, option=lab(a), controller=0, spec=a.get('spec'), frozen_logit=float(f[i]), delta=float(delta[i]), final_logit=float(fin[i]),
                             relative_delta_vs_pass=rel, old_margin_vs_pass=old_m, new_margin_vs_pass=(old_m + rel) if rel is not None else None,
                             p_frozen=p0[i], p_final=p1[i]))
        chosen = rows[st]['selected']
        out.append(dict(step=st, npz_index=t, chain=[(c['player'], c['code']) for c in snap['public_chain']], latest=snap['public_chain'][-1]['code'],
                        hand=sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 2),
                        field=sorted(c['code'] for c in snap['cards'] if c['controller'] == 0 and c['location'] == 4),
                        options=opts, frozen_choice=lab(next(a for a in acts if a['index'] == max(legal, key=lambda i: f[i]))),
                        final_choice=lab(next(a for a in acts if a['index'] == max(legal, key=lambda i: fin[i]))), executed=lab(chosen), who=rows[st]['who'],
                        p_pass_frozen=p0.get(pass_i), p_pass_final=p1.get(pass_i)))
    return out


def outcome(rec):
    cp = rec['checkpoints']
    a, b = cp.get('opponent_activation') or {}, cp.get('target_chain_resolved') or {}
    hand_a, hand_b = collections.Counter(a.get('hand', [])), collections.Counter(b.get('hand', []))
    used_g = U.G in hand_a and U.G not in hand_b and U.G in b.get('graveyard', [])
    new_cards = sorted((hand_b - hand_a).elements())
    return dict(own_hand_before=a.get('hand'), own_hand_after=b.get('hand'), g_used=used_g, cards_added_to_hand=new_cards,
                own_monsters_before=a.get('monsters'), own_monsters_after=b.get('monsters'), medius_on_own_field_after=U.MEDIUS in b.get('monsters', []),
                medius_in_own_gy_after=U.MEDIUS in b.get('graveyard', []), opponent_monsters_after=b.get('opponent_monsters'),
                opponent_gy_after=b.get('opponent_graveyard'), gamma_on_field=U.GAMMA in b.get('opponent_monsters', []), driver_on_field=U.DRIVER in b.get('opponent_monsters', []))


def same_until(a, b, step):
    return all(x['snapshot'] == y['snapshot'] and x['selected']['index'] == y['selected']['index'] for x, y in zip(a['trace'][:step], b['trace'][:step]))


def main():
    base = Policy(str(U.BASE[0]), str(U.BASE[1]))
    adapter = load(U.CAND / 'response-adapter.flax_model')
    am = read(ROOT / 'amendment-1.json')
    names = U.COMPARISON + am['controls_to_run']
    recs = {n: read(ROOT / 'runs' / n / 'record.json') for n in names}
    out = {}
    for n in names:
        rec = recs[n]
        act_step = rec['checkpoints'].get('opponent_activation', {}).get('step')
        rep = read(ROOT / 'runs' / n / 'replay.json')
        out[n] = dict(target=rec['target'], reason=rec['reason'], opponent=[e['what'] for e in rec['opponent_events']], opponent_activation_step=act_step,
                      policy=rec['policy'] and {k: rec['policy'].get(k) for k in ['type', 'base_sha256', 'head_sha256', 'adapter_sha256', 'manifest_sha256', 'decisions']},
                      adapter_active=(rec['policy'] or {}).get('adapter_active'), control_log=rec['control_log'], screens=screens(n, base, adapter), outcome=outcome(rec),
                      replay={k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'game_over_equal', 'observations_match']},
                      visible=len(rec['disclosed']), lua=rec['lua_errors'])
    pairs = {}
    for h in ['H1', 'H2']:
        for o in ['ASH', 'GAMMA']:
            b, c = recs[f'{h}-{o}-BASE'], recs[f'{h}-{o}-CAND']
            zb, zc = np.load(ROOT / 'runs' / f'{h}-{o}-BASE' / 'observations.npz'), np.load(ROOT / 'runs' / f'{h}-{o}-CAND' / 'observations.npz')
            same_obs = all(np.array_equal(zb[k], zc[k]) for k in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain', 'labels'])
            pairs[f'{h}-{o}'] = dict(base_vs_cand_identical_trace=len(b['trace']) == len(c['trace']) and same_until(b, c, len(b['trace'])), base_vs_cand_identical_inputs=bool(same_obs),
                                     cand_adapter_activations=len(c['policy']['adapter_active']), cand_activations_before_opponent=sum(1 for x in c['policy']['adapter_active']
                                                                                                                                     if x['call'] < sum(1 for r in c['trace'][:b['checkpoints']['opponent_activation']['step']] if r['snapshot']['to_play'] == 0)))
        a, g = recs[f'{h}-ASH-BASE'], recs[f'{h}-GAMMA-BASE']
        s = a['checkpoints']['opponent_activation']['step']
        za, zg = np.load(ROOT / 'runs' / f'{h}-ASH-BASE' / 'observations.npz'), np.load(ROOT / 'runs' / f'{h}-GAMMA-BASE' / 'observations.npz')
        k = int(np.searchsorted(za['steps'], s))
        pairs[f'{h}-ASH-vs-GAMMA-before-activation'] = dict(activation_step=[s, g['checkpoints']['opponent_activation']['step']], trace_equal=same_until(a, g, s),
                                                            inputs_equal=bool(all(np.array_equal(za[x][:k], zg[x][:k]) for x in ['cards_', 'global_', 'actions_', 'h_actions_', 'chain'])))
    ledger = [json.loads(l) for l in (ROOT / 'eval-runs.jsonl').read_text().splitlines()]
    starts = [e for e in ledger if e.get('event') == 'start']
    summary = dict(registration_sha256=sha(ROOT / 'registration.json'), amendment_sha256=sha(ROOT / 'amendment-1.json'),
                   engine=dict(used=len(starts), cap=26, statuses=dict(collections.Counter(e.get('status') for e in ledger if 'status' in e))),
                   reused_controls={n: d['reused_record'] for n, d in am['controls'].items()}, runs=out, pairs=pairs)
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + '\n')
    print('ENGINE', summary['engine'], 'reused', summary['reused_controls'])
    for k, v in pairs.items():
        print('PAIR', k, v)
    for n, r in out.items():
        print('=====', n, r['target'], r['reason'], '| adapter', r['adapter_active'] and len(r['adapter_active']), '| outcome', json.dumps({k: r['outcome'][k] for k in ['g_used', 'cards_added_to_hand', 'medius_on_own_field_after', 'gamma_on_field', 'driver_on_field', 'own_hand_after']}, ensure_ascii=False))
        for sc in r['screens']:
            print('   screen', sc['step'], 'latest', sc['latest'], 'frozen', sc['frozen_choice'], 'final', sc['final_choice'], 'executed', sc['executed'], sc['who'], 'p_pass', round(sc['p_pass_frozen'], 5), '->', round(sc['p_pass_final'], 5))
            for o in sc['options']:
                print('      ', o['option'], 'logit', round(o['frozen_logit'], 2), 'delta', round(o['delta'], 2), 'rel', o['relative_delta_vs_pass'] and round(o['relative_delta_vs_pass'], 2),
                      'margin', o['old_margin_vs_pass'] is not None and round(o['old_margin_vs_pass'], 2), '->', o['new_margin_vs_pass'] is not None and round(o['new_margin_vs_pass'], 2),
                      'p', round(o['p_frozen'], 5), '->', round(o['p_final'], 5))
    print('SUMMARY', sha(ROOT / 'summary.json'))


if __name__ == '__main__':
    main()
