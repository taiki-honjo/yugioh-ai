"""Hand-branch supervised job: start = the unadopted basic-opening research candidate (f47cd2b6 / 92b452da);
1,024 optimizer updates, batch 2 (blocks: 2 existing minibatches + 2 paired minibatches); one candidate.

  hb_train.py diag    offline diagnostics before training (start model)
  hb_train.py train   training (GPU), diagnostics at 0/256/512/768/1,024 updates, save, reload checks
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

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402
import batch_data as BD  # noqa: E402
import batch_model as BM  # noqa: E402
import batch_plans as BP  # noqa: E402
import hb_data as HD  # noqa: E402
import hb_plans as HP  # noqa: E402
import pilot_data as PD  # noqa: E402
import pilot_model as PM  # noqa: E402
from opponent_article_bc_common import features  # noqa: E402

START = (ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
START_SHA = ('f47cd2b664b6153bec85484a037ecd888244baafd1b503e01655681a6f929186', '92b452daf1a8e9b30d657963fa48db8ddc46d8dfb552e9802d878540e7a823bd')
PILOT = (Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110/train/attempt-2/candidate/base.flax_model'),
         Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110/train/attempt-2/candidate/chain-head.flax_model'))
PILOT_SHA = ('58a39a61a12f5ee3a142fbbba2bdc71c9e27ae6026b092037b55843feca9742a', 'fabf2b857bceecf690f59044cd42b6d8ef6237be33bef11216bf289fe10a518a')


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


def flat(seqs, names):
    items, labels, groups = [], [], []
    for g in sorted(seqs):
        for s, n in zip(seqs[g], names[g]):
            items.append(s); labels.append(n); groups.append(g)
    return items, labels, groups


def all_items(existing, names, pairs):
    items, labels, groups = flat(existing, names)
    for g in sorted(pairs):
        for p in pairs[g]:
            for n, variant, seq in p['hands']:
                items.append(seq); labels.append(n); groups.append(variant)
    return items, labels, groups


def study_branches(items, labels):
    out = []
    for si, s in enumerate(items):
        for b in HP.branch_steps(s, E._IDS):
            out.append(dict(sequence=si, name=labels[si], step=b['step'], branch=b['branch'], label_card=b['label']))
    return out


def branch_detail(logps, items, branches):
    """P(label) and P(the other studied option) at each branch step."""
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


def card_of(seq, step, index):
    row = seq['actions_'][step, index]
    return E._IDS.get(int(row[1]) * 256 + int(row[2]), 0)


def named_branches(items, labels):
    """C1-type: the Medius① choice (label ネルヴェド while 獄神獣 is offered). C2-type: the 改変 search (label リテラ while 覚醒 is offered)."""
    out = []
    for si, s in enumerate(items):
        n = int(np.sum(s['actions_'][0, :, 3] != 0))
        for t in range(len(s['labels'])):
            if s['loss'][t] <= 0:
                continue
            valid = np.where(s['actions_'][t, :, 3] != 0)[0]
            offered = {card_of(s, t, i) for i in valid}
            chosen = card_of(s, t, int(s['labels'][t]))
            if chosen == BP.NERVED and BP.BEAST in offered and s['message'][t] != 11:
                out.append(dict(sequence=si, name=labels[si], step=t, branch='Medius①: ネルヴェド (獄神獣 offered)'))
            if chosen == BP.LITERA and BP.AWAKENING in offered and s['message'][t] != 11:
                out.append(dict(sequence=si, name=labels[si], step=t, branch='改変/学都: リテラ (覚醒 offered)'))
        del n
    return out


def first_divergence(logps, items, labels):
    out = []
    for si, (lp, s) in enumerate(zip(logps, items)):
        idx = next((t for t in range(len(s['labels'])) if s['loss'][t] > 0 and int(lp[t].argmax()) != int(s['labels'][t])), None)
        if idx is not None:
            out.append(dict(sequence=si, name=labels[si], step=int(idx), branch='first start-model divergence (teacher forcing)'))
    return out


def conflicts(items, labels):
    """Same raw input AND same full history (all earlier steps of the sequence) with different labels, on loss steps.
    Current-input-only matches are counted separately and are not conflicts."""
    full, current = collections.defaultdict(set), collections.defaultdict(set)
    for si, s in enumerate(items):
        h = hashlib.sha256()
        for t in range(len(s['labels'])):
            step = hashlib.sha256(b''.join(np.ascontiguousarray(s[k][t]).tobytes() for k in PD.OBS_KEYS + ('chain',))).hexdigest()
            prefix = h.hexdigest()
            if s['loss'][t] > 0:
                full[(prefix, step)].add(int(s['labels'][t]))
                current[step].add(int(s['labels'][t]))
            h.update(step.encode())
            h.update(str(int(s['labels'][t])).encode())
    return dict(full_history_conflicts=sum(1 for v in full.values() if len(v) > 1),
                current_input_only_label_differences=sum(1 for v in current.values() if len(v) > 1))


def diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches):
    logps = BM.evaluate_sequences(params, bstats, items, length, logp_fn)
    by_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        sub_items = [items[i] for i in idx]
        sub_lp = [logps[i] for i in idx]
        sub_br = [dict(b, sequence=idx.index(b['sequence'])) for b in branches if b['sequence'] in idx]
        by_group[g] = BM.summarize(sub_lp, sub_items, sub_br)
    return by_group, logps


def run_diag():
    registered()
    existing, names, pairs, stats = HD.load()
    items, labels, groups = all_items(existing, names, pairs)
    length = max(len(s['labels']) for s in items)
    logp_fn = jax.jit(BM.step_logp)
    branches = study_branches(items, labels)
    out = dict(data=stats, length=length, sequences=len(items), labels_legal=True, conflicts=conflicts(items, labels),
               masks='context all player-0 decisions; loss own-turn non-position + scripted opponent-turn non-position',
               rnn_reset='every sequence unrolls from a zero state (one duel, one player)')
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, _ = PM.load_pair(*START)
    by_group, logps = diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches)
    out['start'] = by_group
    out['branches_start'] = branch_detail(logps, items, branches)
    out['first_divergence_start'] = first_divergence(logps, items, labels)
    (ROOT / 'diag').mkdir(exist_ok=True)
    (ROOT / 'diag/pre-training.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + '\n')
    print('DIAG', json.dumps(dict(conflicts=out['conflicts'], start={g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in by_group.items()})), flush=True)
    for r in out['branches_start']:
        print('BRANCH', r['name'], r['step'], r['branch'], 'label', r['label_card'], r['options'], flush=True)


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
    if (out / 'result.json').exists():
        raise SystemExit('training result exists; one candidate only')
    print('DEVICES', jax.devices(), flush=True)
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, original = PM.load_pair(*START)
    start_params = jax.tree.map(lambda x: x.copy(), params)
    existing, names, pairs, stats = HD.load()
    items, labels, groups = all_items(existing, names, pairs)
    length = max(len(s['labels']) for s in items)
    sampler = HD.BlockSampler(existing, pairs, reg['training']['sampling_seed'])
    optimizer = BM.make_optimizer()
    opt_state = optimizer.init(params)
    step_fn = jax.jit(lambda p, o, b: BM.update(p, o, bstats, b, optimizer))
    logp_fn = jax.jit(BM.step_logp)
    pre = json.loads((ROOT / 'diag/pre-training.json').read_text())
    branches = study_branches(items, labels)
    t0 = time.time()
    aborted = None
    for u in range(BM.CONFIG['updates'] + 1):
        if u in BM.CONFIG['diagnostics_at']:
            by_group, logps = diagnostics(params, bstats, items, labels, groups, length, logp_fn, branches)
            with (out / 'diagnostics.jsonl').open('a') as h:
                h.write(json.dumps(dict(update=u, groups=by_group, branches=branch_detail(logps, items, branches)), default=float) + '\n')
            print('DIAG', u, json.dumps({g: (round(v['ce_pooled'], 3), round(v['top1'], 3)) for g, v in by_group.items()}), f'{time.time() - t0:.0f}s', flush=True)
        if u == BM.CONFIG['updates']:
            break
        picked = sampler.next()
        batch, used = HD.make_batch(existing, pairs, picked, length)
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
