"""Scoped response adapter jobs (CPU only; no engine duel; the engine library is initialised by the isolated entry).

  sa_train.py prepare   frozen inference along the saved histories: target cache (S1/S2 window: frozen logits, chain
                        feature, action features, legal mask, label), zero-adapter equivalence on all 42 sequences,
                        gate-from-features vs gate-from-snapshot on saved records, synthetic non-zero adapter test,
                        factory refusal tests -> cache/targets.npz, cache/prepare.json (written once)
  sa_train.py train     512 updates of the NEW adapter only (Adam 1e-3, eps 1e-5, clip 1.0 on the adapter, seed 11111501),
                        save the 3-file candidate + manifest, post-training checks (off-target identity, reload)
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

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-scoped-response-adapter-20261111')
P = Path('/home/hatk8/projects/yugioh-ai')
sys.path.insert(0, str(ROOT / 'code'))
import sa_model as SM  # noqa: E402  (imports the isolated entry first)
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
CONFIG = dict(updates=512, learning_rate=1e-3, adam_eps=1e-5, max_grad_norm=1.0, seed=SEED, batch='S1 + S2 every update')
CACHE = ROOT / 'cache/targets.npz'
GATE_RECORDS = ([P / 'experiments/artmage-functional-landing-compare-20261110/runs' / n for n in
                 ['BV1-1cost-VARNISH-OM', 'BV1-1cost-VARNISH-OI', 'BV1-1cost-ASSAULT-OM', 'BV1-1cost-ASSAULT-OI',
                  'BV1-2cost-VARNISH-OM', 'BV1-2cost-VARNISH-OI', 'BV1-2cost-ASSAULT-OM', 'BV1-2cost-ASSAULT-OI']]
                + [P / 'experiments/artmage-opponent-turn-probe-20261111/runs' / n for n in ['S1-D1R0', 'S1-D0R1', 'S1-D1R1', 'S2-D1R0', 'S2-D1R1']]
                + [P / 'experiments/artmage-response-learning-20261111/runs' / n for n in
                   ['A-S1', 'A-S2', 'B-BV1-1cost-VARNISH-OM', 'B-BV1-2cost-ASSAULT-OM', 'B-BV1-1cost-ASSAULT-OM', 'B-BV1-2cost-VARNISH-OM']])


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


def all_sequences():
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
    assert len(items) == 42
    return items, labels, responses, metas


def obs_info(s, t):
    obs = {k: s[k][t:t + 1] for k in PD.OBS_KEYS}
    obs['mask_'] = None
    ch = s['chain'][t]
    info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
    return obs, info


def frozen_policy():
    return Policy(str(FROZEN[0]), str(FROZEN[1]))


def walk(items, adapter_folder=None, adapter_policy=None):
    """Teacher-forced along the saved histories. Returns per sequence: frozen logits, final logits, states-equal flags, gates.
    With an adapter policy, the frozen Policy and the adapter policy are stepped side by side on the same inputs."""
    base = frozen_policy()
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
                same_state = all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(base.state), jax.tree.leaves(adapter_policy.inner.state)))
                row.update(final=np.asarray(l1)[0], gated=gated, same_state=same_state)
            rows.append(row)
        out.append(rows)
    return out


def position_actions(items, make):
    """Post-position-adapter actions along the saved histories for a policy factory (as in saved_checks)."""
    acts = []
    for s in items:
        pol = make()
        pol.reset()
        a = []
        for t in range(len(s['labels'])):
            obs, info = obs_info(s, t)
            a.append(int(pol.act(obs, info)))
        acts.append(a)
    return acts


def compare(walked, labels):
    """Off-target: exact logit equality and identical states; on-target: list the changes."""
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


def gate_vs_snapshots():
    out, total, mismatches, gated = [], 0, [], 0
    for d in GATE_RECORDS:
        rec = json.loads((d / 'record.json').read_text())
        with np.load(d / 'observations.npz') as z:
            steps, acts, chain = z['steps'].tolist(), z['actions_'], z['chain']
        rows = {r['step']: r for r in rec['trace']}
        n_g = 0
        for i, st in enumerate(steps):
            snap = rows[st]['snapshot']
            pub = snap['public_chain']
            g_snap = bool(snap['message'] == 'select_chain' and pub and pub[-1]['player'] != snap['to_play'])
            g_feat = SM.gate(acts[i], chain[i])
            if g_snap != g_feat:
                mismatches.append(dict(run=str(d), step=st))
            n_g += int(g_feat)
        total += len(steps)
        gated += n_g
        out.append(dict(run=str(d.relative_to(P)), own_decisions=len(steps), gated=n_g, record_sha256=sha(d / 'record.json'), observations_sha256=sha(d / 'observations.npz')))
    return dict(runs=out, own_decisions=total, gated=gated, mismatches=mismatches)


def run_prepare():
    registered()
    if CACHE.exists() or (ROOT / 'cache/prepare.json').exists():
        raise SystemExit('cache exists; prepared once')
    print('DEVICES', jax.devices(), flush=True)
    assert jax.devices()[0].platform == 'cpu'
    items, labels, responses, metas = all_sequences()
    t0 = time.time()
    # zero adapter (the initial state of the registered seed) as a 3-file folder
    init = SM.init_adapter(SEED)
    zero_out = float(sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(init['params']['output'])))
    tmp_zero = Path(tempfile.mkdtemp(prefix='sa-zero-', dir=str(ROOT / 'cache')))
    for f in FROZEN:
        shutil.copyfile(f, tmp_zero / f.name)
    SM.save_adapter(init, tmp_zero / 'response-adapter.flax_model')
    SM.write_manifest(tmp_zero, tmp_zero / 'base.flax_model', tmp_zero / 'chain-head.flax_model', tmp_zero / 'response-adapter.flax_model', dict(note='zero-initialised adapter (precheck only)'))
    zero_pol = SM.create_adapter_policy(tmp_zero, positions=False)
    walked = walk(items, adapter_policy=zero_pol.adapter_policy if hasattr(zero_pol, 'adapter_policy') else zero_pol)
    zero_cmp = compare(walked, labels)
    zero_on_equal = all(e['max_abs_change'] == 0.0 for e in zero_cmp['on_target'])
    acts_orig = position_actions(items, lambda: E.create_policy(FROZEN[0], FROZEN[1], positions=True, expected=SM.FROZEN_SHA))
    acts_zero = position_actions(items, lambda: SM.create_adapter_policy(tmp_zero, positions=True))
    # target cache from the frozen logits of the walk
    cache, targets = {}, {}
    for n in RD.NAMES:
        si, m = labels.index(n), metas[n]
        t = m['target_index']
        s = responses[n]
        if not SM.gate(s['actions_'][t], s['chain'][t]):
            raise RuntimeError(f'{n}: target is not inside the gate')
        legal = s['actions_'][t, :, 3] != 0
        cache[f'{n}__frozen_logits'] = walked[si][t]['frozen'].astype(np.float32)
        cache[f'{n}__chain'] = s['chain'][t].astype(np.int32)
        cache[f'{n}__actions'] = s['actions_'][t]
        cache[f'{n}__legal'] = legal
        cache[f'{n}__label'] = np.int32(s['labels'][t])
        fl = np.where(legal, walked[si][t]['frozen'], -np.inf)
        p = np.exp(fl - fl.max()); p /= p.sum()
        targets[n] = dict(index=t, step=m['target_step'], label=int(s['labels'][t]), label_name=m['label']['name'], cancel_index=m['cancel_index'],
                          p_label_frozen=float(p[s['labels'][t]]), p_pass_frozen=float(p[m['cancel_index']]), chain=s['chain'][t].tolist(), gated_steps_in_history=[
                              i for i in range(len(s['labels'])) if SM.gate(s['actions_'][i], s['chain'][i])])
    np.savez_compressed(CACHE, **cache)
    # synthetic non-zero adapter: off-target ignored, on-target used
    import flax
    rnd = flax.core.unfreeze(SM.init_adapter(SEED + 7))
    rnd['params']['output']['kernel'] = jax.random.normal(jax.random.PRNGKey(5), rnd['params']['output']['kernel'].shape)
    tmp_rnd = Path(tempfile.mkdtemp(prefix='sa-rnd-', dir=str(ROOT / 'cache')))
    for f in FROZEN:
        shutil.copyfile(f, tmp_rnd / f.name)
    SM.save_adapter(rnd, tmp_rnd / 'response-adapter.flax_model')
    SM.write_manifest(tmp_rnd, tmp_rnd / 'base.flax_model', tmp_rnd / 'chain-head.flax_model', tmp_rnd / 'response-adapter.flax_model', dict(note='synthetic'))
    sub = [labels.index(n) for n in RD.NAMES] + [0, 30]
    rnd_cmp = compare(walk([items[i] for i in sub], adapter_policy=SM.create_adapter_policy(tmp_rnd, positions=False)), [labels[i] for i in sub])
    # factory refusals
    refusals = {}
    bad = Path(tempfile.mkdtemp(prefix='sa-bad-', dir=str(ROOT / 'cache')))
    for f in FROZEN:
        shutil.copyfile(f, bad / f.name)
    shutil.copyfile(tmp_rnd / 'response-adapter.flax_model', bad / 'response-adapter.flax_model')
    m = json.loads((tmp_rnd / 'manifest.json').read_text())
    for case, mutate in [('adapter hash mismatch', lambda x: x['adapter'].update(sha256='0' * 64)), ('wrong version', lambda x: x.update(version=99)),
                         ('missing adapter', lambda x: x['adapter'].update(file='absent.flax_model'))]:
        mm = json.loads(json.dumps(m)); mutate(mm)
        (bad / 'manifest.json').write_text(json.dumps(mm))
        try:
            SM.create_adapter_policy(bad, positions=False)
            refusals[case] = 'NOT refused'
        except RuntimeError as exc:
            refusals[case] = f'refused: {exc}'
    for d in [tmp_zero, tmp_rnd, bad]:
        shutil.rmtree(d)
    gv = gate_vs_snapshots()
    old40 = [i for i, l in enumerate(labels) if l not in RD.NAMES]
    result = dict(backend=str(jax.devices()[0]), seconds=time.time() - t0, frozen_sha256=list(SM.FROZEN_SHA),
                  zero_adapter=dict(output_layer_abs_sum=zero_out, comparison=zero_cmp, on_target_equal=zero_on_equal,
                                    position_actions_equal=acts_orig == acts_zero, sequences=len(items)),
                  synthetic_adapter=dict(comparison={k: v for k, v in rnd_cmp.items()}, sequences=[labels[i] for i in sub]),
                  factory_refusals=refusals, gate_vs_snapshot=gv,
                  old40_gated_decisions=sum(1 for i in old40 for r in walked[i] if r['gate']), old40_decisions=sum(len(walked[i]) for i in old40),
                  targets=targets, cache=dict(file=str(CACHE), sha256=sha(CACHE), source='frozen Policy.infer on the CPU along the saved histories (default precision, the evaluation backend)'))
    ok = (zero_out == 0.0 and zero_cmp['off_target_logit_differences'] == 0 and zero_cmp['rnn_state_differences'] == 0 and zero_on_equal
          and acts_orig == acts_zero and not gv['mismatches'] and rnd_cmp['off_target_logit_differences'] == 0 and rnd_cmp['rnn_state_differences'] == 0
          and any(e['max_abs_change'] > 0 for e in rnd_cmp['on_target']) and all(v.startswith('refused') for v in refusals.values()))
    result['all_prechecks_passed'] = ok
    (ROOT / 'cache/prepare.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('PREPARE', json.dumps(dict(ok=ok, zero=dict(off=zero_cmp['off_target_logit_differences'], states=zero_cmp['rnn_state_differences'], gated=zero_cmp['gated'],
                                                      decisions=zero_cmp['decisions'], on_equal=zero_on_equal, position_equal=acts_orig == acts_zero),
                                     synthetic=dict(off=rnd_cmp['off_target_logit_differences'], on=[(e['sequence'], e['index'], e['max_abs_change']) for e in rnd_cmp['on_target']]),
                                     gate=dict(decisions=gv['own_decisions'], gated=gv['gated'], mismatches=len(gv['mismatches'])), refusals=refusals,
                                     old40_gated=result['old40_gated_decisions'], targets={n: {k: t[k] for k in ['index', 'p_label_frozen', 'p_pass_frozen', 'gated_steps_in_history']} for n, t in targets.items()},
                                     cache=result['cache']['sha256'])), flush=True)


def legal_ce(adapter, batch):
    delta = SM.ADAPTER.apply(adapter, batch['chain'], batch['actions'])
    logits = jnp.where(batch['legal'], batch['frozen'] + delta, -1e9)
    logp = jax.nn.log_softmax(logits, -1)
    ce = -jnp.take_along_axis(logp, batch['label'][:, None], -1)[:, 0]
    return jnp.mean(ce), dict(per_seq_ce=ce, delta=delta)


def target_detail(adapter, batch, names, cancel):
    _, aux = legal_ce(adapter, batch)
    logits = np.where(np.asarray(batch['legal']), np.asarray(batch['frozen']) + np.asarray(aux['delta']), -np.inf)
    out = {}
    for b, n in enumerate(names):
        p = np.exp(logits[b] - logits[b].max()); p /= p.sum()
        legal = np.asarray(batch['legal'][b])
        d = np.asarray(aux['delta'][b])
        out[n] = dict(p_label=float(p[int(batch['label'][b])]), p_pass=float(p[cancel[n]]), argmax_is_label=int(p.argmax()) == int(batch['label'][b]),
                      ce=float(aux['per_seq_ce'][b]), delta_legal={int(i): float(d[i]) for i in np.flatnonzero(legal)})
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
    with np.load(CACHE) as z:
        c = {k: z[k] for k in z.files}
    names = RD.NAMES
    batch = dict(frozen=jnp.asarray(np.stack([c[f'{n}__frozen_logits'] for n in names])), chain=jnp.asarray(np.stack([c[f'{n}__chain'] for n in names])),
                 actions=jnp.asarray(np.stack([c[f'{n}__actions'] for n in names])), legal=jnp.asarray(np.stack([c[f'{n}__legal'] for n in names])),
                 label=jnp.asarray(np.stack([c[f'{n}__label'] for n in names]).astype(np.int32)))
    cancel = {n: prep['targets'][n]['cancel_index'] for n in names}
    params = SM.init_adapter(SEED)
    optimizer = optax.chain(optax.clip_by_global_norm(CONFIG['max_grad_norm']), optax.adam(CONFIG['learning_rate'], eps=CONFIG['adam_eps']))
    opt_state = optimizer.init(params)
    opt_leaves = sorted(x.shape for x in jax.tree.leaves(opt_state) if hasattr(x, 'shape') and x.ndim > 0)
    par_leaves = sorted(x.shape for x in jax.tree.leaves(params))
    assert opt_leaves == sorted(par_leaves * 2), 'optimizer state must cover the adapter parameters only'

    @jax.jit
    def step(p, o):
        (loss, aux), g = jax.value_and_grad(legal_ce, has_aux=True)(p, batch)
        gn = optax.global_norm(g)
        upd, o = optimizer.update(g, o, p)
        new = optax.apply_updates(p, upd)
        return new, o, dict(loss=loss, per_seq_ce=aux['per_seq_ce'], grad_norm=gn, clipped=gn > CONFIG['max_grad_norm'], update_norm=optax.global_norm(upd),
                            max_abs_delta=jnp.max(jnp.abs(jnp.where(batch['legal'], aux['delta'], 0.0))))
    diag = {0: target_detail(params, batch, names, cancel)}
    print('TARGET 0', json.dumps({n: {k: round(v, 5) if isinstance(v, float) else v for k, v in d.items() if k != 'delta_legal'} for n, d in diag[0].items()}), flush=True)
    t0 = time.time()
    with (out / 'updates.jsonl').open('w') as h:
        for u in range(CONFIG['updates']):
            params, opt_state, m = step(params, opt_state)
            row = {k: np.asarray(v).tolist() for k, v in m.items()}
            row['update'] = u + 1
            h.write(json.dumps(row) + '\n')
            if not np.isfinite(row['loss']):
                raise SystemExit(f'non-finite loss at update {u + 1}')
    diag[CONFIG['updates']] = target_detail(params, batch, names, cancel)
    print('TARGET 512', json.dumps({n: {k: round(v, 5) if isinstance(v, float) else v for k, v in d.items() if k != 'delta_legal'} for n, d in diag[512].items()}), flush=True)
    cand = out / 'candidate'
    cand.mkdir()
    for f in FROZEN:
        shutil.copyfile(f, cand / f.name)
    SM.save_adapter(params, cand / 'response-adapter.flax_model')
    manifest = SM.write_manifest(cand, cand / 'base.flax_model', cand / 'chain-head.flax_model', cand / 'response-adapter.flax_model',
                                 dict(seed=SEED, updates=CONFIG['updates'], registration_sha256=sha(ROOT / 'registration.json')))
    for f in cand.iterdir():
        os.chmod(f, 0o444)
    assert (sha(cand / 'base.flax_model'), sha(cand / 'chain-head.flax_model')) == SM.FROZEN_SHA
    # post-training checks (CPU, same backend as the cache and the engine evaluation)
    items, labels, responses, metas = all_sequences()
    loaded = SM.create_adapter_policy(cand, positions=False)
    walked = walk(items, adapter_policy=loaded)
    post = compare(walked, labels)
    acts_orig = position_actions(items, lambda: E.create_policy(FROZEN[0], FROZEN[1], positions=True, expected=SM.FROZEN_SHA))
    acts_new = position_actions(items, lambda: SM.create_adapter_policy(cand, positions=True))
    gated_idx = {(labels[i], t) for i, rows in enumerate(walked) for t, r in enumerate(rows) if r['gate']}
    pos_off_diff = sum(1 for i in range(len(items)) for t in range(len(items[i]['labels'])) if (labels[i], t) not in gated_idx and acts_orig[i][t] != acts_new[i][t])
    # reload equality: in-memory adapter vs the saved file on the targets
    mem = {n: np.asarray(c[f'{n}__frozen_logits']) + np.asarray(SM.adapter_delta(params, jnp.asarray(c[f'{n}__chain'][None]), jnp.asarray(c[f'{n}__actions'][None])))[0] for n in names}
    file_logits = {n: walked[labels.index(n)][prep['targets'][n]['index']]['final'] for n in names}
    reload_equal = {n: float(np.abs(mem[n] - file_logits[n]).max()) for n in names}
    import flax.serialization as ser
    finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ser.msgpack_restore((cand / 'response-adapter.flax_model').read_bytes())))
    upd = [json.loads(l) for l in (out / 'updates.jsonl').read_text().splitlines()]
    result = dict(status='complete', updates_done=CONFIG['updates'], seconds=time.time() - t0, config=CONFIG, manifest=manifest,
                  saved=dict(candidate_base_sha256=SM.FROZEN_SHA[0], candidate_head_sha256=SM.FROZEN_SHA[1], adapter_sha256=sha(cand / 'response-adapter.flax_model'),
                             manifest_sha256=sha(cand / 'manifest.json'), frozen_pair_unchanged=True, adapter_weights_finite=finite,
                             reload_max_abs_logit_diff_on_targets=reload_equal),
                  targets=diag, clipped_updates=sum(1 for r in upd if r['clipped']), first_update_ce_below_ln2={
                      n: next((r['update'] for r in upd if r['per_seq_ce'][b] < np.log(2)), None) for b, n in enumerate(names)},
                  post_training=dict(comparison=post, position_actions_off_target_differences=pos_off_diff,
                                     position_actions_on_target=[dict(sequence=labels[i], index=t, original=acts_orig[i][t], new=acts_new[i][t])
                                                                 for i in range(len(items)) for t in range(len(items[i]['labels'])) if (labels[i], t) in gated_idx]))
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('RESULT', json.dumps(dict(clipped=result['clipped_updates'], ce_below_ln2=result['first_update_ce_below_ln2'], adapter=result['saved']['adapter_sha256'],
                                    off_target_logit_diff=post['off_target_logit_differences'], state_diff=post['rnn_state_differences'], position_off_diff=pos_off_diff,
                                    on_target=post['on_target'], reload=reload_equal, finite=finite)), flush=True)


if __name__ == '__main__':
    try:
        {'prepare': run_prepare, 'train': run_train}[sys.argv[1]]()
    except BaseException:
        traceback.print_exc()
        raise
