"""Response study supervised job (derived from hb_train.py of the hand-branch study; changes: data = old 40 + 2 response
sequences, sampler = blocks of 8 (3 old group / 3 old pair / 2 response), 256 updates, diagnostics at 0 and 256).
Start = the unadopted hand-branch research candidate (b9c7751f / 77191d5d), fresh Adam state, one candidate.

  rl_train.py diag    offline diagnostics before training (start model; conflicts; teacher-forced CE/top-1; response probabilities)
  rl_train.py train   training (GPU), diagnostics at 0 / 256, save, reload checks
No engine duel is created here (the engine library is initialised by the isolated entry, nothing more).
"""
import collections
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

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402
import batch_model as BM  # noqa: E402
import batch_plans as BP  # noqa: E402
import hb_plans as HP  # noqa: E402
import pilot_data as PD  # noqa: E402
import pilot_model as PM  # noqa: E402
import rl_data as RD  # noqa: E402
from opponent_article_bc_common import features  # noqa: E402

START = (ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
START_SHA = tuple(RD.MODEL_SHA)
SEED = 11111301
for _m in (E, BM, BP, HP, PD, PM, RD):
    assert Path(_m.__file__).resolve().parent == ROOT / 'code', _m.__file__


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered():
    reg = json.loads((ROOT / 'registration.json').read_text())
    amendments = sorted(ROOT.glob('amendment-*.json'))
    code = json.loads(amendments[-1].read_text())['code'] if amendments else reg['code']
    for name, digest in code.items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    return reg


def all_items(existing, names, pairs, responses):
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
    return items, labels, groups


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
    """P(label) and P(the other studied option) at each Graphrea/Medius branch step (as in the hand-branch study)."""
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


def response_detail(logps, items, labels, metas):
    """At each response window: P(registered effect), P(pass = Cancel) and every option's probability; argmax."""
    out = {}
    for n, m in metas.items():
        lp = logps[labels.index(n)][m['target_index']]
        opts = [dict(o, p=float(np.exp(lp[o['index']]))) for o in m['options']]
        out[n] = dict(p_label=float(np.exp(lp[m['label']['index']])), p_pass=float(np.exp(lp[m['cancel_index']])) if m['cancel_index'] is not None else None,
                      argmax=int(lp.argmax()), argmax_is_label=int(lp.argmax()) == m['label']['index'], options=opts)
    return out


def conflicts(items, labels):
    """Same raw input AND same full history (all earlier steps of the sequence) with different labels, on loss steps.
    Current-input-only matches are counted separately and are not conflicts (hand-branch definition, unchanged)."""
    full, current = collections.defaultdict(set), collections.defaultdict(set)
    where_current = collections.defaultdict(list)
    for si, s in enumerate(items):
        h = hashlib.sha256()
        for t in range(len(s['labels'])):
            step = hashlib.sha256(b''.join(np.ascontiguousarray(s[k][t]).tobytes() for k in PD.OBS_KEYS + ('chain',))).hexdigest()
            prefix = h.hexdigest()
            if s['loss'][t] > 0:
                full[(prefix, step)].add(int(s['labels'][t]))
                current[step].add(int(s['labels'][t]))
                where_current[step].append((labels[si], t))
            h.update(step.encode())
            h.update(str(int(s['labels'][t])).encode())
    response_current = {}
    for n in RD.NAMES:
        si = labels.index(n)
        s = items[si]
        t = int(np.flatnonzero(s['loss'] > 0)[0])
        step = hashlib.sha256(b''.join(np.ascontiguousarray(s[k][t]).tobytes() for k in PD.OBS_KEYS + ('chain',))).hexdigest()
        response_current[n] = dict(same_current_input_elsewhere=[x for x in where_current[step] if x[0] != n], labels=sorted(current[step]))
    return dict(full_history_conflicts=sum(1 for v in full.values() if len(v) > 1),
                current_input_only_label_differences=sum(1 for v in current.values() if len(v) > 1),
                response_windows=response_current)


def diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches):
    logps = BM.evaluate_sequences(params, bstats, items, length, logp_fn)
    by_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        sub_items = [items[i] for i in idx]
        sub_lp = [logps[i] for i in idx]
        sub_br = [dict(b, sequence=idx.index(b['sequence'])) for b in branches if b['sequence'] in idx]
        by_group[g] = BM.summarize(sub_lp, sub_items, sub_br)
    old = [i for i, g in enumerate(groups) if g not in RD.NAMES]
    by_group['old-40'] = BM.summarize([logps[i] for i in old], [items[i] for i in old])
    return by_group, logps


def load_all():
    existing, names, pairs, responses, metas, stats = RD.load()
    items, labels, groups = all_items(existing, names, pairs, responses)
    assert len(items) == 42, len(items)
    return existing, names, pairs, responses, metas, stats, items, labels, groups


def run_diag():
    registered()
    existing, names, pairs, responses, metas, stats, items, labels, groups = load_all()
    length = max(len(s['labels']) for s in items)
    logp_fn = jax.jit(BM.step_logp)
    branches = study_branches(items, labels)
    for s in items:
        n = len(s['labels'])
        assert (s['actions_'][np.arange(n), s['labels'], 3] != 0)[s['loss'] > 0].all()
    out = dict(data=stats, length=length, sequences=len(items), labels_legal=True, conflicts=conflicts(items, labels),
               masks='old 40: unchanged (own-turn non-position + scripted opponent-turn non-position); responses: only the window action',
               rnn_reset='every sequence unrolls from a zero state (one duel, one player)')
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, _ = PM.load_pair(*START)
    by_group, logps = diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches)
    out['start'] = by_group
    out['responses_start'] = response_detail(logps, items, labels, metas)
    out['branches_start'] = branch_detail(logps, items, branches)
    (ROOT / 'diag').mkdir(exist_ok=True)
    path = ROOT / 'diag/pre-training.json'
    if path.exists():
        raise SystemExit('pre-training diagnostics exist; not rewritten')
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n')
    print('DIAG', json.dumps(dict(conflicts={k: out['conflicts'][k] for k in ['full_history_conflicts', 'current_input_only_label_differences']},
                                  start={g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in by_group.items()})), flush=True)
    for n, r in out['responses_start'].items():
        print('RESPONSE', n, 'p_label', round(r['p_label'], 4), 'p_pass', round(r['p_pass'], 4), 'argmax_is_label', r['argmax_is_label'], flush=True)
    print('CURRENT', json.dumps(out['conflicts']['response_windows'], ensure_ascii=False), flush=True)


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
    """Unchanged from hb_train.saved_checks: reload through the isolated entry; logits and post-adapter actions vs memory."""
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


def run_train():
    reg = registered()
    out = ROOT / 'train'
    out.mkdir(exist_ok=True)
    if (out / 'result.json').exists() or (out / 'updates.jsonl').exists():
        raise SystemExit('training output exists; one candidate only')
    assert reg['training']['sampling_seed'] == SEED and BM.CONFIG['updates'] == 256
    pre = json.loads((ROOT / 'diag/pre-training.json').read_text())
    if pre['conflicts']['full_history_conflicts'] != 0:
        raise SystemExit('full-history label conflicts in the data; not training (report first)')
    print('DEVICES', jax.devices(), flush=True)
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, original = PM.load_pair(*START)
    start_params = jax.tree.map(lambda x: x.copy(), params)
    existing, names, pairs, responses, metas, stats, items, labels, groups = load_all()
    length = max(len(s['labels']) for s in items)
    sampler = RD.BlockSampler8(existing, pairs, SEED)
    optimizer = BM.make_optimizer()
    opt_state = optimizer.init(params)
    step_fn = jax.jit(lambda p, o, b: BM.update(p, o, bstats, b, optimizer))
    logp_fn = jax.jit(BM.step_logp)
    branches = study_branches(items, labels)
    t0 = time.time()
    aborted = None
    for u in range(BM.CONFIG['updates'] + 1):
        if u in BM.CONFIG['diagnostics_at']:
            by_group, logps = diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches)
            resp = response_detail(logps, items, labels, metas)
            with (out / 'diagnostics.jsonl').open('a') as h:
                h.write(json.dumps(dict(update=u, groups=by_group, responses=resp, branches=branch_detail(logps, items, branches)), default=float) + '\n')
            print('DIAG', u, json.dumps({g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in by_group.items()}), f'{time.time() - t0:.0f}s', flush=True)
            for n, r in resp.items():
                print('RESPONSE', u, n, 'p_label', round(r['p_label'], 4), 'p_pass', round(r['p_pass'], 4), 'argmax_is_label', r['argmax_is_label'], flush=True)
        if u == BM.CONFIG['updates']:
            break
        picked = sampler.next()
        batch, used = RD.make_batch(existing, pairs, responses, picked, length)
        params, opt_state, m = step_fn(params, opt_state, {k: jnp.asarray(v) for k, v in batch.items()})
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
        {'diag': run_diag, 'train': run_train}[sys.argv[1]]()
    except BaseException:
        traceback.print_exc()
        raise
