"""Relative-score retention jobs (CPU; no engine duel). Derived from the coverage study's cv_train.py: same structure
(sa_model.py unchanged), same 4 screens, same CE / KL; ADDED: relative_retention = mean over screens of mean over the
protected options of (delta(option) - delta(pass))^2, coefficient 1.0. Protected = legal options except pass and (on S1 / S2)
every option with the same meaning as the taught one.

  rs_train.py prepare   the 4-screen cache (+ pass index, protected masks), equality with the coverage cache, zero-adapter
                        equivalence, gate check -> cache/screens.npz, cache/prepare.json (written once)
  rs_train.py train     512 updates of a new adapter (seed 11111501, zero output), save the 3-file candidate, post checks
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-relative-score-retention-20261111')
COVERAGE = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import sa_model as SM  # noqa: E402
import artmage_entry as E  # noqa: E402
import pilot_data as PD  # noqa: E402
import rl_data as RD  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from opponent_article_bc_common import Policy  # noqa: E402

for _m in (SM, E, PD, RD):
    assert Path(_m.__file__).resolve().parent == ROOT / 'code', _m.__file__
FROZEN = (ROOT / 'models/frozen/base.flax_model', ROOT / 'models/frozen/chain-head.flax_model')
SEED = 11111501
COEF = 1.0
COEF_REL = 1.0
CONFIG = dict(updates=512, learning_rate=1e-3, adam_eps=1e-5, max_grad_norm=1.0, seed=SEED, kl_coefficient=COEF, relative_coefficient=COEF_REL,
              batch='all 4 screens every update')
CACHE = ROOT / 'cache/screens.npz'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
PROBE = P / 'experiments/artmage-opponent-turn-probe-20261111'
SCREENS = {
    'S1': dict(dir=PROBE / 'runs/S1-D0R1', step=149, kind='ce', want=('Activate', 27184601, 10011), note='improve: ディアクトロス② (R1-executed in the probe; the start model passed)'),
    'S2': dict(dir=FLC / 'runs/BV1-2cost-ASSAULT-OM', step=169, kind='ce', want=('Activate', 34541940, 10011), note='improve: ファインメルト③ (the start model\'s own choice)'),
    'M': dict(dir=FLC / 'runs/BV1-1cost-ASSAULT-OM', step=139, kind='kl', want=('Activate', 23829452, 10010), note='keep: the start model\'s distribution (its first choice 獄神獣②)'),
    'P': dict(dir=FLC / 'runs/BV1-2cost-VARNISH-OI', step=102, kind='kl', want=('Cancel', 0, -1), note='keep: the start model\'s distribution (its first choice pass)'),
}
ORDER = ['S1', 'S2', 'M', 'P']
FLC8 = ['BV1-1cost-VARNISH-OM', 'BV1-1cost-VARNISH-OI', 'BV1-1cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OI',
        'BV1-2cost-VARNISH-OM', 'BV1-2cost-VARNISH-OI', 'BV1-2cost-ASSAULT-OM', 'BV1-2cost-ASSAULT-OI']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered(need_amendment=False):
    reg = json.loads((ROOT / 'registration.json').read_text())
    amendments = sorted(ROOT.glob('amendment-*.json'))
    latest = json.loads(amendments[-1].read_text()) if amendments else reg
    for name, digest in latest['code'].items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    if (sha(FROZEN[0]), sha(FROZEN[1])) != SM.FROZEN_SHA:
        raise RuntimeError('frozen copies changed')
    if need_amendment:
        fixed = latest.get('cache')
        if not fixed or sha(CACHE) != fixed['sha256']:
            raise RuntimeError('cache not fixed by an amendment, or changed')
    return reg, latest


def npz_sequence(d):
    with np.load(d / 'observations.npz') as z:
        s = {k: z[k] for k in PD.OBS_KEYS}
        s.update(chain=z['chain'].astype(np.int32), labels=z['labels'].astype(np.int32), steps=z['steps'])
    return s


def all_sequences():
    """The 42 training-study sequences + the 8 functional-comparison runs (saved NPZ)."""
    existing, names, pairs, responses, metas, stats = RD.load()
    items, labels = [], []
    for g in sorted(existing):
        for s, n in zip(existing[g], names[g]):
            items.append(s); labels.append(f'{g}:{n}')
    for g in sorted(pairs):
        for p in pairs[g]:
            for n, variant, seq in p['hands']:
                items.append(seq); labels.append(f'pair:{n}')
    for n in RD.NAMES:
        items.append(responses[n]); labels.append(n)
    for n in FLC8:
        items.append(npz_sequence(FLC / 'runs' / n)); labels.append(f'flc:{n}')
    assert len(items) == 50
    return items, labels


def obs_info(s, t):
    obs = {k: s[k][t:t + 1] for k in PD.OBS_KEYS}
    obs['mask_'] = None
    ch = s['chain'][t]
    info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
    return obs, info


def walk(items, adapter_policy=None):
    base = Policy(str(FROZEN[0]), str(FROZEN[1]))
    out = []
    for s in items:
        base.reset()
        if adapter_policy is not None:
            adapter_policy.reset()
        rows = []
        for t in range(len(s['labels'])):
            obs, info = obs_info(s, t)
            base.state, l0 = base.infer(base.base, base.head, base.state, obs, SM.features(info))
            row = dict(frozen=np.asarray(l0)[0], gate=SM.gate(s['actions_'][t], s['chain'][t]))
            if adapter_policy is not None:
                l1, gated = adapter_policy.step(obs, info)
                row.update(final=np.asarray(l1)[0], gated=gated,
                           same_state=all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(base.state), jax.tree.leaves(adapter_policy.inner.state))))
            rows.append(row)
        out.append(rows)
    return out


def position_actions(items, make):
    acts = []
    for s in items:
        pol = make()
        pol.reset()
        acts.append([int(pol.act(*obs_info(s, t))) for t in range(len(s['labels']))])
    return acts


def compare(walked, labels):
    off_diff, state_diff, on = 0, 0, []
    for rows, name in zip(walked, labels):
        for t, r in enumerate(rows):
            if r['gate'] != r['gated']:
                raise RuntimeError(f'gate disagreement in {name} at {t}')
            state_diff += int(not r['same_state'])
            if not r['gated']:
                off_diff += int(not np.array_equal(r['frozen'], r['final']))
            else:
                on.append(dict(sequence=name, index=t, frozen_argmax=int(r['frozen'].argmax()), final_argmax=int(r['final'].argmax()),
                               max_abs_change=float(np.abs(r['final'] - r['frozen']).max())))
    return dict(off_target_logit_differences=off_diff, rnn_state_differences=state_diff, on_target=on,
                decisions=sum(len(r) for r in walked), gated=sum(1 for rows in walked for r in rows if r['gate']))


def three_file_folder(adapter, note):
    d = Path(tempfile.mkdtemp(prefix='cv-', dir=str(ROOT / 'cache')))
    for f in FROZEN:
        shutil.copyfile(f, d / f.name)
    SM.save_adapter(adapter, d / 'response-adapter.flax_model')
    SM.write_manifest(d, d / 'base.flax_model', d / 'chain-head.flax_model', d / 'response-adapter.flax_model', dict(note=note))
    return d


def screen_rows():
    """Per screen: history (own observations to the screen, inclusive), the record row, checks."""
    out = {}
    for k in ORDER:
        sc = SCREENS[k]
        d = sc['dir']
        rec = json.loads((d / 'record.json').read_text())
        rep = json.loads((d / 'replay.json').read_text())
        if not (rep['mismatch'] is None and rep['failure'] is None and rep['observations_match'] and rep['checkpoints_equal']):
            raise RuntimeError(f'{k}: source replay did not match')
        if list(rec['model_sha256']) != list(SM.FROZEN_SHA):
            raise RuntimeError(f'{k}: source record is not the start model')
        s = npz_sequence(d)
        steps = s['steps'].tolist()
        i = steps.index(sc['step'])  # NPZ index from the steps array, never the step number itself
        row = next(r for r in rec['trace'] if r['step'] == sc['step'])
        snap = row['snapshot']
        g_snap = bool(snap['message'] == 'select_chain' and snap['public_chain'] and snap['public_chain'][-1]['player'] != snap['to_play'])
        if not (g_snap and SM.gate(s['actions_'][i], s['chain'][i])):
            raise RuntimeError(f'{k}: screen not in the gate')
        sel = row['selected']
        if (sel['act'], sel['code'], sel.get('effect')) != sc['want']:
            raise RuntimeError(f"{k}: recorded action {(sel['act'], sel['code'], sel.get('effect'))} != {sc['want']}")
        if int(s['labels'][i]) != sel['index']:
            raise RuntimeError(f'{k}: NPZ label differs from the record')
        hist = {key: v[:i + 1] for key, v in s.items() if key != 'steps'}
        out[k] = dict(history=hist, index=i, row=row, who=row['who'],
                      files={f: sha(d / f) for f in ['record.json', 'observations.npz', 'replay.json']})
    return out


MASKED = -1e9  # stored log-probability of illegal options (finite)


def legal_logsoftmax(logits, legal):
    x = np.where(legal, logits, -np.inf)
    m = x.max()
    lse = m + np.log(np.exp(x - m).sum())
    return np.where(legal, logits - lse, MASKED)


def run_prepare():
    registered()
    if CACHE.exists() or (ROOT / 'cache/prepare.json').exists():
        raise SystemExit('cache exists; prepared once')
    print('DEVICES', jax.devices(), flush=True)
    assert jax.devices()[0].platform == 'cpu'
    t0 = time.time()
    screens = screen_rows()
    walked_screens = walk([screens[k]['history'] for k in ORDER])
    cache, info = {}, {}
    for j, k in enumerate(ORDER):
        sc, h, i = SCREENS[k], screens[k]['history'], screens[k]['index']
        frozen = walked_screens[j][i]['frozen'].astype(np.float32)
        legal = h['actions_'][i, :, 3] != 0
        acts = {a['index']: a for a in screens[k]['row']['snapshot']['actions']}
        fa = int(np.where(legal, frozen, -np.inf).argmax())
        executed = screens[k]['row']['selected']['index']
        if sc['kind'] == 'kl' and fa != executed:
            raise RuntimeError(f'{k}: frozen first choice {fa} differs from the recorded start-model action {executed}')
        ref = legal_logsoftmax(frozen.astype(np.float64), legal).astype(np.float32)
        cache[f'{k}__frozen_logits'] = frozen
        cache[f'{k}__chain'] = h['chain'][i].astype(np.int32)
        cache[f'{k}__actions'] = h['actions_'][i]
        cache[f'{k}__legal'] = legal
        cache[f'{k}__label'] = np.int32(executed if sc['kind'] == 'ce' else -1)
        cache[f'{k}__ref_logp'] = ref
        # pass = the snapshot's Cancel option (checked against the NPZ legal row); protected = legal minus pass minus the taught meaning
        passes = [a['index'] for a in acts.values() if a['act'] == 'Cancel']
        if len(passes) != 1 or not legal[passes[0]] or sorted(acts) != [int(x) for x in np.flatnonzero(legal)]:
            raise RuntimeError(f'{k}: pass / legal options do not match the snapshot')
        taught = set()
        if sc['kind'] == 'ce':
            want = (acts[executed]['act'], acts[executed]['code'], acts[executed].get('effect'))
            taught = {ix for ix, a in acts.items() if (a['act'], a['code'], a.get('effect')) == want}
        protect = legal.copy()
        protect[passes[0]] = False
        for ix in taught:
            protect[ix] = False
        cache[f'{k}__pass'] = np.int32(passes[0])
        cache[f'{k}__protect'] = protect
        p = np.exp(np.where(legal, ref, -np.inf))
        info[k] = dict(kind=sc['kind'], note=sc['note'], step=sc['step'], npz_index=i, who=screens[k]['who'], files=screens[k]['files'],
                       chain=h['chain'][i].tolist(), frozen_first=dict(index=fa, act=acts[fa]['act'], code=acts[fa]['code'], effect=acts[fa].get('effect')),
                       recorded=dict(index=executed, act=acts[executed]['act'], code=acts[executed]['code'], effect=acts[executed].get('effect')),
                       frozen_probs={f"{acts[ix]['act']}:{acts[ix]['code']}:{acts[ix].get('effect')}": float(p[ix]) for ix in acts},
                       gated_steps_in_history=[t for t in range(len(h['labels'])) if SM.gate(h['actions_'][t], h['chain'][t])],
                       pass_index=passes[0], taught_indices=sorted(taught), protected=[dict(index=int(ix), act=acts[int(ix)]['act'], code=acts[int(ix)]['code'],
                                                                                            effect=acts[int(ix)].get('effect')) for ix in np.flatnonzero(protect)])
    np.savez_compressed(CACHE, **cache)
    with np.load(COVERAGE / 'cache/screens.npz') as zc:
        coverage_equal = {key: bool(np.array_equal(zc[key], cache[key])) for key in zc.files}
    # zero-adapter equivalence on 42 + 8 sequences, and position-adapter actions
    items, labels = all_sequences()
    zero = three_file_folder(SM.init_adapter(SEED), 'zero-initialised adapter (precheck only)')
    try:
        zero_cmp = compare(walk(items, adapter_policy=SM.create_adapter_policy(zero, positions=False)), labels)
        acts_orig = position_actions(items, lambda: E.create_policy(FROZEN[0], FROZEN[1], positions=True, expected=SM.FROZEN_SHA))
        acts_zero = position_actions(items, lambda: SM.create_adapter_policy(zero, positions=True))
    finally:
        shutil.rmtree(zero)
    # gate from features vs snapshot on the 8 comparison runs + S1 source
    gv, total = [], 0
    for d in [FLC / 'runs' / n for n in FLC8] + [PROBE / 'runs/S1-D0R1']:
        rec = json.loads((d / 'record.json').read_text())
        s = npz_sequence(d)
        rows = {r['step']: r for r in rec['trace']}
        for i, st in enumerate(s['steps'].tolist()):
            snap = rows[st]['snapshot']
            g1 = bool(snap['message'] == 'select_chain' and snap['public_chain'] and snap['public_chain'][-1]['player'] != snap['to_play'])
            if g1 != SM.gate(s['actions_'][i], s['chain'][i]):
                gv.append(dict(run=d.name, step=st))
            total += 1
    gated_flc = {labels[i]: [t for t in range(len(items[i]['labels'])) if SM.gate(items[i]['actions_'][t], items[i]['chain'][t])] for i in range(len(items)) if labels[i].startswith('flc:')}
    result = dict(backend=str(jax.devices()[0]), seconds=time.time() - t0, frozen_sha256=list(SM.FROZEN_SHA), sa_model_sha256=sha(ROOT / 'code/sa_model.py'),
                  screens=info, cache=dict(file=str(CACHE), sha256=sha(CACHE), source='frozen Policy.infer on the CPU along the saved histories (the evaluation backend)'),
                  zero_adapter=dict(comparison=zero_cmp, on_target_equal=all(e['max_abs_change'] == 0.0 for e in zero_cmp['on_target']), position_actions_equal=acts_orig == acts_zero,
                                    sequences=len(items)),
                  gate_vs_snapshot=dict(decisions=total, mismatches=gv), gated_npz_indices_in_the_8_runs=gated_flc,
                  coverage_cache=dict(file=str(COVERAGE / 'cache/screens.npz'), sha256=sha(COVERAGE / 'cache/screens.npz'), arrays_equal=coverage_equal))
    ok = (all(coverage_equal.values()) and zero_cmp['off_target_logit_differences'] == 0 and zero_cmp['rnn_state_differences'] == 0 and result['zero_adapter']['on_target_equal']
          and acts_orig == acts_zero and not gv)
    result['all_prechecks_passed'] = ok
    (ROOT / 'cache/prepare.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('PREPARE', json.dumps(dict(ok=ok, zero=dict(off=zero_cmp['off_target_logit_differences'], states=zero_cmp['rnn_state_differences'], decisions=zero_cmp['decisions'],
                                                      gated=zero_cmp['gated'], position_equal=acts_orig == acts_zero),
                                     gate=dict(decisions=total, mismatches=len(gv)), gated_flc=gated_flc,
                                     screens={k: dict(index=v['npz_index'], frozen_first=v['frozen_first'], recorded=v['recorded'], who=v['who']) for k, v in info.items()},
                                     protected={k: [(x['code'], x['effect']) for x in v['protected']] for k, v in info.items()},
                                     coverage_cache_equal=all(coverage_equal.values()), cache=result['cache']['sha256']), ensure_ascii=False), flush=True)


def load_batch():
    with np.load(CACHE) as z:
        c = {k: z[k] for k in z.files}
    return dict(frozen=jnp.asarray(np.stack([c[f'{k}__frozen_logits'] for k in ORDER])), chain=jnp.asarray(np.stack([c[f'{k}__chain'] for k in ORDER])),
                actions=jnp.asarray(np.stack([c[f'{k}__actions'] for k in ORDER])), legal=jnp.asarray(np.stack([c[f'{k}__legal'] for k in ORDER])),
                label=jnp.asarray(np.stack([max(int(c[f'{k}__label']), 0) for k in ORDER]).astype(np.int32)),
                ref_logp=jnp.asarray(np.stack([c[f'{k}__ref_logp'] for k in ORDER])),
                is_ce=jnp.asarray(np.array([SCREENS[k]['kind'] == 'ce' for k in ORDER])),
                pass_index=jnp.asarray(np.stack([c[f'{k}__pass'] for k in ORDER]).astype(np.int32)),
                protect=jnp.asarray(np.stack([c[f'{k}__protect'] for k in ORDER])))


def log_q(adapter, batch):
    delta = SM.ADAPTER.apply(adapter, batch['chain'], batch['actions'])
    return jax.nn.log_softmax(jnp.where(batch['legal'], batch['frozen'] + delta, -1e9), -1), delta


def ce_term(adapter, batch):
    lq, delta = log_q(adapter, batch)
    ce = -jnp.take_along_axis(lq, batch['label'][:, None], -1)[:, 0]
    w = batch['is_ce'].astype(jnp.float32)
    return jnp.sum(ce * w) / jnp.sum(w), dict(ce=ce, delta=delta)


def kl_term(adapter, batch):
    lq, _ = log_q(adapter, batch)
    ref = jax.lax.stop_gradient(batch['ref_logp'])
    p = jnp.where(batch['legal'], jnp.exp(ref), 0.0)
    kl = jnp.sum(jnp.where(batch['legal'], p * (ref - lq), 0.0), -1)
    w = 1.0 - batch['is_ce'].astype(jnp.float32)
    return jnp.sum(kl * w) / jnp.sum(w), dict(kl=kl)


def rel_term(adapter, batch):
    """mean over screens (with protected options) of mean over protected options of (delta(o) - delta(pass))^2. Invariant to a common shift."""
    delta = SM.ADAPTER.apply(adapter, batch['chain'], batch['actions'])
    rel = delta - jnp.take_along_axis(delta, batch['pass_index'][:, None], -1)
    m = batch['protect'].astype(jnp.float32)
    n = jnp.sum(m, -1)
    per = jnp.sum(rel ** 2 * m, -1) / jnp.maximum(n, 1.0)
    valid = (n > 0).astype(jnp.float32)
    return jnp.sum(per * valid) / jnp.maximum(jnp.sum(valid), 1.0), dict(per_screen_rel=per, max_abs_rel=jnp.max(jnp.abs(rel) * m, -1))


def make_step(optimizer, batch, coef=COEF, coef_rel=COEF_REL):
    @jax.jit
    def step(p, o):
        (ce, aux_ce), g_ce = jax.value_and_grad(ce_term, has_aux=True)(p, batch)
        (kl, aux_kl), g_kl = jax.value_and_grad(kl_term, has_aux=True)(p, batch)
        (rl, aux_rl), g_rl = jax.value_and_grad(rel_term, has_aux=True)(p, batch)
        g = jax.tree.map(lambda a, b, c: a + coef * b + coef_rel * c, g_ce, g_kl, g_rl)
        gn = optax.global_norm(g)
        upd, o = optimizer.update(g, o, p)
        new = optax.apply_updates(p, upd)
        return new, o, dict(loss=ce + coef * kl + coef_rel * rl, ce=ce, kl=kl, rel=rl, per_screen_ce=aux_ce['ce'], per_screen_kl=aux_kl['kl'],
                            per_screen_rel=aux_rl['per_screen_rel'], max_abs_rel=aux_rl['max_abs_rel'], ce_grad_norm=optax.global_norm(g_ce),
                            kl_grad_norm=optax.global_norm(g_kl), rel_grad_norm=optax.global_norm(g_rl), grad_norm=gn, clipped=gn > CONFIG['max_grad_norm'],
                            update_norm=optax.global_norm(upd), max_abs_delta=jnp.max(jnp.abs(jnp.where(batch['legal'], aux_ce['delta'], 0.0))))
    return step


def screen_detail(adapter, batch, prep):
    lq, delta = log_q(adapter, batch)
    lq, delta = np.asarray(lq), np.asarray(delta)
    out = {}
    for j, k in enumerate(ORDER):
        legal = np.asarray(batch['legal'][j])
        p = np.exp(np.where(legal, lq[j], -np.inf))
        ref = np.exp(np.where(legal, np.asarray(batch['ref_logp'][j]), -np.inf))
        idx = np.flatnonzero(legal)
        info = prep['screens'][k]
        target = info['recorded']['index'] if SCREENS[k]['kind'] == 'ce' else info['frozen_first']['index']
        out[k] = dict(kind=SCREENS[k]['kind'], argmax=int(p.argmax()), target_index=target, argmax_is_target=int(p.argmax()) == target,
                      p_target=float(p[target]), p_frozen=dict((int(i), float(ref[i])) for i in idx), p_now=dict((int(i), float(p[i])) for i in idx),
                      delta=dict((int(i), float(delta[j, i])) for i in idx),
                      pass_index=int(batch['pass_index'][j]),
                      relative_delta={int(i): float(delta[j, i] - delta[j, int(batch['pass_index'][j])]) for i in idx},
                      old_margin_vs_pass={int(i): float(batch['frozen'][j, i] - batch['frozen'][j, int(batch['pass_index'][j])]) for i in idx},
                      max_abs_rel_protected=float(max([abs(delta[j, i] - delta[j, int(batch['pass_index'][j])]) for i in np.flatnonzero(np.asarray(batch['protect'][j]))] or [0.0])),
                      ce=float(-lq[j, target]) if SCREENS[k]['kind'] == 'ce' else None,
                      kl=float(np.sum(np.where(legal, ref * (np.asarray(batch['ref_logp'][j]) - lq[j]), 0.0))) if SCREENS[k]['kind'] == 'kl' else None)
    return out


def run_train():
    reg, latest = registered(need_amendment=True)
    out = ROOT / 'train'
    if (out / 'result.json').exists() or (out / 'updates.jsonl').exists():
        raise SystemExit('training output exists; one candidate only')
    out.mkdir(exist_ok=True)
    print('DEVICES', jax.devices(), flush=True)
    prep = json.loads((ROOT / 'cache/prepare.json').read_text())
    if not prep['all_prechecks_passed']:
        raise SystemExit('prechecks did not pass')
    batch = load_batch()
    params = SM.init_adapter(SEED)
    optimizer = optax.chain(optax.clip_by_global_norm(CONFIG['max_grad_norm']), optax.adam(CONFIG['learning_rate'], eps=CONFIG['adam_eps']))
    opt_state = optimizer.init(params)
    assert sorted(x.shape for x in jax.tree.leaves(opt_state) if hasattr(x, 'shape') and x.ndim > 0) == sorted([x.shape for x in jax.tree.leaves(params)] * 2)
    step = make_step(optimizer, batch)
    diag = {0: screen_detail(params, batch, prep)}
    print('SCREENS 0', json.dumps({k: dict(argmax_is_target=v['argmax_is_target'], p_target=round(v['p_target'], 5), ce=v['ce'], kl=v['kl'], max_rel=v['max_abs_rel_protected']) for k, v in diag[0].items()}), flush=True)
    t0 = time.time()
    with (out / 'updates.jsonl').open('w') as h:
        for u in range(CONFIG['updates']):
            params, opt_state, m = step(params, opt_state)
            row = {k: np.asarray(v).tolist() for k, v in m.items()}
            row['update'] = u + 1
            h.write(json.dumps(row) + '\n')
            if not np.isfinite(row['loss']):
                raise SystemExit(f'non-finite loss at update {u + 1}')
    diag[CONFIG['updates']] = screen_detail(params, batch, prep)
    print('SCREENS 512', json.dumps({k: dict(argmax_is_target=v['argmax_is_target'], p_target=round(v['p_target'], 5), ce=v['ce'], kl=v['kl'], max_rel=v['max_abs_rel_protected']) for k, v in diag[512].items()}), flush=True)
    cand = out / 'candidate'
    cand.mkdir()
    for f in FROZEN:
        shutil.copyfile(f, cand / f.name)
    SM.save_adapter(params, cand / 'response-adapter.flax_model')
    manifest = SM.write_manifest(cand, cand / 'base.flax_model', cand / 'chain-head.flax_model', cand / 'response-adapter.flax_model',
                                 dict(seed=SEED, updates=CONFIG['updates'], screens=ORDER, kl_coefficient=COEF, relative_coefficient=COEF_REL, registration_sha256=sha(ROOT / 'registration.json')))
    for f in cand.iterdir():
        os.chmod(f, 0o444)
    assert (sha(cand / 'base.flax_model'), sha(cand / 'chain-head.flax_model')) == SM.FROZEN_SHA
    items, labels = all_sequences()
    walked = walk(items, adapter_policy=SM.create_adapter_policy(cand, positions=False))
    post = compare(walked, labels)
    acts_orig = position_actions(items, lambda: E.create_policy(FROZEN[0], FROZEN[1], positions=True, expected=SM.FROZEN_SHA))
    acts_new = position_actions(items, lambda: SM.create_adapter_policy(cand, positions=True))
    gated_idx = {(labels[i], t) for i, rows in enumerate(walked) for t, r in enumerate(rows) if r['gate']}
    pos_off_diff = sum(1 for i in range(len(items)) for t in range(len(items[i]['labels'])) if (labels[i], t) not in gated_idx and acts_orig[i][t] != acts_new[i][t])
    # reload: the 3-file entry on the 4 screens vs the in-memory adapter
    screens = screen_rows()
    reloaded = walk([screens[k]['history'] for k in ORDER], adapter_policy=SM.create_adapter_policy(cand, positions=False))
    lq_mem, _ = log_q(params, batch)
    reload_diff = {}
    for j, k in enumerate(ORDER):
        r = reloaded[j][screens[k]['index']]
        mem = np.asarray(batch['frozen'][j]) + np.asarray(SM.ADAPTER.apply(params, batch['chain'][j:j + 1], batch['actions'][j:j + 1]))[0]
        reload_diff[k] = dict(gated=r['gated'], max_abs_logit_diff=float(np.abs(r['final'] - mem).max()))
    import flax.serialization as ser
    finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ser.msgpack_restore((cand / 'response-adapter.flax_model').read_bytes())))
    upd = [json.loads(l) for l in (out / 'updates.jsonl').read_text().splitlines()]
    result = dict(status='complete', updates_done=CONFIG['updates'], seconds=time.time() - t0, config=CONFIG, manifest=manifest,
                  saved=dict(candidate_base_sha256=SM.FROZEN_SHA[0], candidate_head_sha256=SM.FROZEN_SHA[1], adapter_sha256=sha(cand / 'response-adapter.flax_model'),
                             manifest_sha256=sha(cand / 'manifest.json'), frozen_pair_unchanged=True, adapter_weights_finite=finite, reload=reload_diff),
                  screens=diag, clipped_updates=sum(1 for r in upd if r['clipped']),
                  post_training=dict(comparison=post, position_actions_off_target_differences=pos_off_diff,
                                     position_actions_on_target=[dict(sequence=labels[i], index=t, original=acts_orig[i][t], new=acts_new[i][t])
                                                                 for i in range(len(items)) for t in range(len(items[i]['labels'])) if (labels[i], t) in gated_idx]))
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('RESULT', json.dumps(dict(clipped=result['clipped_updates'], adapter=result['saved']['adapter_sha256'], off=post['off_target_logit_differences'],
                                    states=post['rnn_state_differences'], position_off=pos_off_diff, on_target=post['on_target'], reload=reload_diff, finite=finite)), flush=True)


if __name__ == '__main__':
    try:
        {'prepare': run_prepare, 'train': run_train}[sys.argv[1]]()
    except BaseException:
        traceback.print_exc()
        raise
