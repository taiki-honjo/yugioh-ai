"""Context-retention job (derived from the response study's rl_train.py; the ONLY training change is the retention term
on response minibatches: CE_response + 1.0 * KL_context; old minibatches keep CE only). Same start model, data,
sampler (seed 11111301, blocks 3/3/2, identical draw order), 256 updates, optimizer and limits.

  cr_train.py reference   GPU: the start model's legal-normalised log-probabilities along the two recorded response
                          histories (the response minibatch layout) -> reference/start-gpu.npz (written once)
  cr_train.py refcheck    CPU: the same computation on the CPU, compared with the GPU reference (argmax / max |diff|)
  cr_train.py train       GPU: training, diagnostics at 0 / 256, save, reload checks
No engine duel is created here (the engine library is initialised by the isolated entry, nothing more).
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
PREV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402
import batch_model as BM  # noqa: E402
import batch_plans as BP  # noqa: E402
import cr_data as CD  # noqa: E402
import cr_model as CM  # noqa: E402
import hb_plans as HP  # noqa: E402
import pilot_data as PD  # noqa: E402
import pilot_model as PM  # noqa: E402
import rl_data as RD  # noqa: E402
from opponent_article_bc_common import features  # noqa: E402

START = (ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
START_SHA = tuple(RD.MODEL_SHA)
SEED = 11111301
VARNISH, ASSAULT = 74011784, 1122030
for _m in (E, BM, BP, CD, CM, HP, PD, PM, RD):
    assert Path(_m.__file__).resolve().parent == ROOT / 'code', _m.__file__


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered(need_reference=False):
    reg = json.loads((ROOT / 'registration.json').read_text())
    amendments = sorted(ROOT.glob('amendment-*.json'))
    latest = json.loads(amendments[-1].read_text()) if amendments else reg
    for name, digest in latest['code'].items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    if need_reference:
        fixed = latest.get('reference')
        if not fixed or sha(CD.REF) != fixed['sha256']:
            raise RuntimeError('reference not fixed by an amendment, or changed')
    return reg, latest


def load_all():
    existing, names, pairs, responses, metas, stats = RD.load()
    items, labels, groups = [], [], []
    for g in sorted(existing):
        for s, n in zip(existing[g], names[g]):
            items.append(s); labels.append(n); groups.append(g)
    for g in sorted(pairs):
        for p in pairs[g]:
            for n, variant, seq in p['hands']:
                items.append(seq); labels.append(n); groups.append(variant)
    for n in RD.NAMES:
        items.append(responses[n]); labels.append(n); groups.append(n)
    assert len(items) == 42
    length = max(len(s['labels']) for s in items)
    return existing, names, pairs, responses, metas, stats, items, labels, groups, length


# ------------------------------------------------------------------ reference
def compute_pair_logp(params, bstats, responses, length):
    batch = CD.response_pair_batch(responses, length)
    fn = jax.jit(CM.legal_logp)
    return np.asarray(fn(params, bstats, {k: jnp.asarray(v) for k, v in batch.items()})), batch


def run_reference():
    registered()
    if CD.REF.exists():
        raise SystemExit('reference exists; computed once')
    print('DEVICES', jax.devices(), flush=True)
    assert jax.devices()[0].platform == 'gpu', 'the reference must be computed on the training backend (GPU)'
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, _ = PM.load_pair(*START)
    existing, names, pairs, responses, metas, stats, items, labels, groups, length = load_all()
    lp, batch = compute_pair_logp(params, bstats, responses, length)
    arrays, info = {}, {}
    for b, n in enumerate(RD.NAMES):
        k = len(responses[n]['labels'])
        lg = CD.legal(responses[n])
        keep = CD.preserve_mask(responses[n], metas[n])
        arrays[f'{n}__logp'] = np.where(lg, lp[:k, b], CD.MASKED).astype(np.float32)
        arrays[f'{n}__legal'] = lg
        arrays[f'{n}__preserve'] = keep
        executed = responses[n]['labels'][:k]
        argmax = np.where(lg, lp[:k, b], -np.inf).argmax(-1)
        info[n] = dict(steps=k, preserved=int(keep.sum()), excluded=dict(window=1, position=int(((responses[n]['message'] == PD.POSITION_MESSAGE) & (np.arange(k) < metas[n]['target_index'])).sum()),
                                                                          single_legal=int(((lg.sum(1) < 2) & (np.arange(k) < metas[n]['target_index']) & (responses[n]['message'] != PD.POSITION_MESSAGE)).sum())),
                       preserved_steps_where_start_argmax_differs_from_executed=[int(t) for t in np.flatnonzero(keep) if argmax[t] != executed[t]])
    CD.REF.parent.mkdir(exist_ok=True)
    np.savez_compressed(CD.REF, **arrays)
    meta = dict(file=str(CD.REF), sha256=sha(CD.REF), backend=str(jax.devices()[0]), matmul_precision='highest', model=list(START_SHA),
                layout='response minibatch (S1, S2) padded to the 42-sequence length, as in training', length=length, sequences=info,
                note='executed (recorded) actions only advance the history; the stored distribution is the start model\'s own')
    (CD.REF.with_suffix('.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=1) + '\n')
    print('REFERENCE', json.dumps(dict(sha256=meta['sha256'], sequences=info)), flush=True)


def run_refcheck():
    registered()
    out = ROOT / 'reference/cpu-check.json'
    if out.exists():
        raise SystemExit('cpu check exists')
    print('DEVICES', jax.devices(), flush=True)
    assert jax.devices()[0].platform == 'cpu'
    params, bstats, _ = PM.load_pair(*START)
    existing, names, pairs, responses, metas, stats, items, labels, groups, length = load_all()
    ref = CD.load_reference(responses, metas)
    lp, _ = compute_pair_logp(params, bstats, responses, length)
    res = {}
    for b, n in enumerate(RD.NAMES):
        k = len(responses[n]['labels'])
        lg, keep = ref[n]['legal'], ref[n]['preserve'] > 0
        cpu = np.where(lg, lp[:k, b], -np.inf)
        gpu = np.where(lg, ref[n]['logp'], -np.inf)
        diff = np.abs(np.where(lg, np.exp(lp[:k, b]) - np.exp(ref[n]['logp']), 0.0)).max(-1)
        disagree = [int(t) for t in np.flatnonzero(keep) if cpu[t].argmax() != gpu[t].argmax()]
        res[n] = dict(preserved=int(keep.sum()), argmax_disagreements=disagree, max_abs_prob_diff_preserved=float(diff[keep].max()) if keep.any() else 0.0)
    out.write_text(json.dumps(dict(backend=str(jax.devices()[0]), reference_sha256=sha(CD.REF), sequences=res), indent=1) + '\n')
    print('REFCHECK', json.dumps(res), flush=True)


# ------------------------------------------------------------------ diagnostics (unchanged from rl_train unless noted)
def study_branches(items, labels):
    out = []
    for si, s in enumerate(items):
        for b in HP.branch_steps(s, E._IDS):
            out.append(dict(sequence=si, name=labels[si], step=b['step'], branch=b['branch'], label_card=b['label']))
    return out


def card_of(seq, step, index):
    row = seq['actions_'][step, index]
    return E._IDS.get(int(row[1]) * 256 + int(row[2]), 0)


def branch_detail(logps, items, branches):
    rows = []
    for b in branches:
        s, lp = items[b['sequence']], logps[b['sequence']][b['step']]
        valid = np.where(s['actions_'][b['step'], :, 3] != 0)[0]
        probs = {}
        for i in valid:
            c = card_of(s, b['step'], i)
            if c in (BP.NERVED, BP.BEAST, BP.VARNISH, BP.ASSAULT):
                probs[c] = probs.get(c, 0.0) + float(np.exp(lp[i]))
        rows.append(dict(name=b['name'], step=b['step'], branch=b['branch'], label_card=b['label_card'],
                         p_label=round(probs.get(b['label_card'], 0.0), 4), options={str(k): round(v, 4) for k, v in probs.items()}))
    return rows


def response_detail(logps, labels, metas):
    out = {}
    for n, m in metas.items():
        lp = logps[labels.index(n)][m['target_index']]
        opts = [dict(o, p=float(np.exp(lp[o['index']]))) for o in m['options']]
        out[n] = dict(p_label=float(np.exp(lp[m['label']['index']])), p_pass=float(np.exp(lp[m['cancel_index']])) if m['cancel_index'] is not None else None,
                      argmax=int(lp.argmax()), argmax_is_label=int(lp.argmax()) == m['label']['index'], options=opts)
    return out


def diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches):
    logps = BM.evaluate_sequences(params, bstats, items, length, logp_fn)
    by_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        sub_br = [dict(b, sequence=idx.index(b['sequence'])) for b in branches if b['sequence'] in idx]
        by_group[g] = BM.summarize([logps[i] for i in idx], [items[i] for i in idx], sub_br)
    old = [i for i, g in enumerate(groups) if g not in RD.NAMES]
    by_group['old-40'] = BM.summarize([logps[i] for i in old], [items[i] for i in old])
    return by_group, logps


def meaning(m, index):
    a = m['actions'].get(int(index)) or m['actions'].get(str(int(index))) or {}
    return dict(index=int(index), act=a.get('act'), code=a.get('code'), name=a.get('name'), effect=a.get('effect'), spec=a.get('spec'))


def context_diag(params, bstats, responses, ref, length, meanings):
    """NEW: on the preserved steps of the two response histories (same pair layout as the reference): KL(start || now),
    argmax changes with card / effect / zone, the start model's top-1 minus top-2 probability, and the watched steps
    (the Graphrea set prompt and the opponent-turn draw/standby selections)."""
    lp_now, _ = compute_pair_logp(params, bstats, responses, length)
    out = {}
    for b, n in enumerate(RD.NAMES):
        k = len(responses[n]['labels'])
        lg, keep, ref_lp = ref[n]['legal'], ref[n]['preserve'] > 0, ref[n]['logp']
        now = np.where(lg, lp_now[:k, b], CD.MASKED)
        p = np.where(lg, np.exp(ref_lp), 0.0)
        kl = np.sum(np.where(lg, p * (ref_lp - now), 0.0), -1)
        steps = np.flatnonzero(keep)
        a_start, a_now = np.where(lg, ref_lp, -np.inf).argmax(-1), np.where(lg, now, -np.inf).argmax(-1)
        srt = -np.sort(-np.where(lg, np.exp(ref_lp), -1.0), -1)
        margin = srt[:, 0] - np.maximum(srt[:, 1], 0.0)
        ms = meanings[n]
        def row(t):
            return dict(index=int(t), step=ms[t]['step'], turn=ms[t]['turn'], phase=ms[t]['phase'], message=ms[t]['message'],
                        start=meaning(ms[t], a_start[t]), now=meaning(ms[t], a_now[t]), executed=meaning(ms[t], ms[t]['executed']),
                        p_start_of_start_choice=float(np.exp(ref_lp[t, a_start[t]])), p_now_of_start_choice=float(np.exp(now[t, a_start[t]])),
                        start_margin=float(margin[t]), kl=float(kl[t]))
        changed = [row(t) for t in steps if a_start[t] != a_now[t]]
        watched = []
        for t in steps:
            codes = {a['code'] for a in ms[t]['actions'].values()}
            graph = ms[t]['turn'] == 1 and ms[t]['message'] == 'select_card' and {VARNISH, ASSAULT} <= codes
            opp_sel = ms[t]['turn'] == 2 and ms[t]['phase'] in (1, 2) and ms[t]['message'] in ('select_card', 'select_unselect_card', 'select_chain', 'select_effectyn')
            if graph or opp_sel:
                watched.append(dict(row(t), kind='Graphrea set' if graph else 'opponent turn draw/standby'))
        out[n] = dict(preserved_steps=int(len(steps)), kl_mean=float(kl[steps].mean()), kl_max=float(kl[steps].max()), kl_max_step=int(ms[int(steps[kl[steps].argmax()])]['step']),
                      argmax_changed=len(changed), changed=changed, watched=watched,
                      start_margin=dict(min=float(margin[steps].min()), below_0_1=int((margin[steps] < 0.1).sum())))
    return out


class MemoryPolicy:
    def __init__(self, params, bstats):
        self.params, self.bstats = params, bstats
        self.reset()

    def reset(self):
        self.state = PM.init_state(1)

    def act(self, obs, info):
        o = {k: jnp.asarray(obs[k]) for k in PM.OBS_KEYS}
        self.state, logits, _ = PM.infer(self.params, self.bstats, self.state, o, jnp.asarray(features(info)))
        return int(np.asarray(logits)[0].argmax())


def saved_checks(params, items, labels, out):
    """Unchanged from rl_train.saved_checks."""
    cand = (out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    want = (sha(cand[0]), sha(cand[1]))
    reloaded = E.create_policy(cand[0], cand[1], positions=False, expected=want)
    _, bstats, _ = PM.load_pair(*cand)
    max_diff, argmax_equal = 0.0, True
    for s in items:
        st_mem, st_file = PM.init_state(1), reloaded.state
        for t in range(len(s['labels'])):
            o = {k: jnp.asarray(s[k][t:t + 1]) for k in PM.OBS_KEYS}
            ch = jnp.asarray(s['chain'][t:t + 1])
            st_mem, l_mem, _ = PM.infer(params, bstats, st_mem, o, ch)
            st_file, l_file = reloaded.infer(reloaded.base, reloaded.head, st_file, dict(o, mask_=None), ch)
            max_diff = max(max_diff, float(jnp.max(jnp.abs(l_mem - l_file))))
            argmax_equal &= int(jnp.argmax(l_mem[0])) == int(jnp.argmax(l_file[0]))
    module = E._adapter()
    adapter_equal = True
    for s in items:
        finals = []
        for pol in [E.create_policy(cand[0], cand[1], positions=True, expected=want), module.PositionPolicy(MemoryPolicy(params, bstats), E._IDS)]:
            pol.reset()
            acts = []
            for t in range(len(s['labels'])):
                obs = {k: s[k][t:t + 1] for k in PM.OBS_KEYS}
                obs['mask_'] = None
                ch = s['chain'][t]
                info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                            public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                            public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
                acts.append(int(pol.act(obs, info)))
            finals.append(acts)
        adapter_equal &= finals[0] == finals[1]
    finite = {}
    import flax.serialization as ser
    for name, path in zip(['base', 'head'], cand):
        tree = ser.msgpack_restore(path.read_bytes())
        finite[name] = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))
    return dict(candidate_base_sha256=want[0], candidate_head_sha256=want[1], reload_through_isolated_entry=True,
                max_abs_logit_diff_memory_vs_saved=max_diff, argmax_equal=argmax_equal, adapter_final_actions_equal=adapter_equal,
                saved_weights_finite=finite, sequences_checked=len(items))


def previous_draws():
    return [(u['kind'], u['sequences']) for u in (json.loads(l) for l in (PREV / 'train/updates.jsonl').read_text().splitlines())]


def run_train():
    reg, latest = registered(need_reference=True)
    out = ROOT / 'train'
    out.mkdir(exist_ok=True)
    if (out / 'result.json').exists() or (out / 'updates.jsonl').exists():
        raise SystemExit('training output exists; one candidate only')
    assert reg['training']['sampling_seed'] == SEED and BM.CONFIG['updates'] == 256 and CM.COEF == reg['training']['retention_coefficient'] == 1.0
    print('DEVICES', jax.devices(), flush=True)
    assert jax.devices()[0].platform == 'gpu'
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, original = PM.load_pair(*START)
    start_params = jax.tree.map(lambda x: x.copy(), params)
    existing, names, pairs, responses, metas, stats, items, labels, groups, length = load_all()
    ref = CD.load_reference(responses, metas)
    meanings = {n: CD.step_meanings(n) for n in RD.NAMES}
    # the draw order must be the response study's order (same seed, same sampler)
    probe = RD.BlockSampler8(existing, pairs, SEED)
    order = []
    for _ in range(BM.CONFIG['updates']):
        kind, v = probe.next()
        _, used = CD.make_batch(existing, pairs, responses, ref, (kind, v), length) if kind != 'response' else (None, list(v))
        order.append((kind, used))
    if order != previous_draws():
        raise SystemExit('draw order differs from the response study')
    sampler = RD.BlockSampler8(existing, pairs, SEED)
    optimizer = BM.make_optimizer()
    opt_state = optimizer.init(params)
    step_fn = jax.jit(lambda p, o, b: BM.update(p, o, bstats, b, optimizer))
    resp_fn = jax.jit(lambda p, o, b: CM.update_response(p, o, bstats, b, optimizer))
    logp_fn = jax.jit(BM.step_logp)
    branches = study_branches(items, labels)
    t0 = time.time()
    aborted = None
    for u in range(BM.CONFIG['updates'] + 1):
        if u in BM.CONFIG['diagnostics_at']:
            by_group, logps = diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches)
            resp = response_detail(logps, labels, metas)
            ctx = context_diag(params, bstats, responses, ref, length, meanings)
            with (out / 'diagnostics.jsonl').open('a') as h:
                h.write(json.dumps(dict(update=u, groups=by_group, responses=resp, context=ctx, branches=branch_detail(logps, items, branches)), default=float) + '\n')
            print('DIAG', u, json.dumps({g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in by_group.items()}), f'{time.time() - t0:.0f}s', flush=True)
            for n, r in resp.items():
                print('RESPONSE', u, n, 'p_label', round(r['p_label'], 4), 'p_pass', round(r['p_pass'], 4), 'argmax_is_label', r['argmax_is_label'], flush=True)
            for n, c in ctx.items():
                print('CONTEXT', u, n, 'kl_mean', round(c['kl_mean'], 6), 'kl_max', round(c['kl_max'], 6), 'argmax_changed', c['argmax_changed'],
                      [(x['step'], x['start']['name'], x['now']['name']) for x in c['changed']][:6], flush=True)
        if u == BM.CONFIG['updates']:
            break
        picked = sampler.next()
        batch, used = CD.make_batch(existing, pairs, responses, ref, picked, length)
        fn = resp_fn if picked[0] == 'response' else step_fn
        params, opt_state, m = fn(params, opt_state, {k: jnp.asarray(v) for k, v in batch.items()})
        row = {k: np.asarray(v).tolist() for k, v in m.items()}
        row.update(update=u + 1, kind=picked[0], sequences=used)
        with (out / 'updates.jsonl').open('a') as h:
            h.write(json.dumps(row, default=float) + '\n')
        if not row['finite']:
            aborted = f'non-finite loss/gradient/weights at update {u + 1}'
            break
        if (ROOT / 'STOP').read_text().strip() == 'now':
            aborted = 'STOP=now requested during the job'
            break
    stats_out = dict(updates_done=u if aborted else BM.CONFIG['updates'], sampled=sampler.counts(), data=stats, length=length,
                     seconds=time.time() - t0, aborted=aborted)
    if aborted:
        (out / 'result.json').write_text(json.dumps(dict(status='aborted', **stats_out), ensure_ascii=False, indent=1, default=float) + '\n')
        print('ABORTED', aborted, flush=True)
        return
    (out / 'candidate').mkdir(exist_ok=True)
    PM.save_pair(params, original, out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    for name in ['base.flax_model', 'chain-head.flax_model']:
        os.chmod(out / 'candidate' / name, 0o444)
    change = {part: float(PM.tree_norm(jax.tree.map(lambda a, b: a - b, params[part], start_params[part]))) for part in ['base', 'head']}
    upd = [json.loads(l) for l in (out / 'updates.jsonl').read_text().splitlines()]
    result = dict(status='complete', **stats_out, parameter_change_norm=change, both_parts_changed=all(v > 0 for v in change.values()),
                  clipped_updates=sum(1 for r in upd if r['clipped']), saved=saved_checks(params, items, labels, out))
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('RESULT', json.dumps({k: result[k] for k in ['status', 'updates_done', 'sampled', 'parameter_change_norm', 'clipped_updates']}), flush=True)
    print('SAVED', json.dumps(result['saved']), flush=True)


if __name__ == '__main__':
    try:
        {'reference': run_reference, 'refcheck': run_refcheck, 'train': run_train}[sys.argv[1]]()
    except BaseException:
        traceback.print_exc()
        raise
