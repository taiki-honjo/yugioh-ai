"""Opening rehearsal: an auxiliary acceptable-action loss on the TRAIN split only.

Local-only. No LLM/API/network client is used or added.

Inspired by experience replay for continual learning (Rolnick et al., NeurIPS
2019), but this is NOT a CLEAR implementation: there is no V-trace replay of old
experience, no value cloning and no policy KL here. PPO keeps optimizing fresh
self-play rollouts exactly as before; for every PPO minibatch this module adds

    lambda_bc * CE_acceptable(train demonstrations)

to the gradient, where

    CE_acceptable(s) = -log sum_{a in acceptable(s) & legal(s)} p_current(a | own history)

computed on the final base + chain-head logits. Every train sequence is unrolled
from a zero RNN state at its first own decision (one duel, one player), the same
way the evaluated policy resets. Steps whose message is select_position (5) or
select_place (11) stay in the unroll as history but are excluded from the loss,
because the external placement/position adapters own those choices. Padding is
excluded through the dataset weights. validation/holdout are refused.

B2B extension (experiments/b2b-train-20261109; pinned in that root's snapshot as opening_rehearsal.py):
with settings['b2b'] absent, every function below behaves exactly like scripts/opening_rehearsal.py.
With settings['b2b'] present, verified train-only B2B demonstrations are added as SEPARATE data with their
own manifest and hash. Their steps carry two masks: the valid-step mask (weights; the whole duel history the
policy actually saw, unrolled through the RNN) and an independent loss mask (loss_weights; only the labelled
B2B / attack decisions). Old sequences keep their original masks. Sampling follows a fixed schedule: every 16
rehearsal sequences (8 minibatches of 2) hold 14 old sequences (uniform from the 416) and one of each demo, at
positions shuffled by a fixed seed. CE and gradient norms are recorded separately for old and B2B sources.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import numpy as np

DECK = 'ElfnotesKewlTune'
SPLIT = 'train'
OBS_KEYS = ('cards_', 'global_', 'actions_', 'h_actions_')
STEP_KEYS = OBS_KEYS + ('chain',)
ADAPTER_MESSAGES = (5, 11)
SNAPSHOT_DIR = 'rehearsal'
METRIC_NAMES = ('ce', 'accuracy', 'loss_steps', 'ppo_grad_norm', 'rehearsal_grad_norm',
                'combined_grad_norm', 'rehearsal_base_grad_norm', 'rehearsal_head_grad_norm',
                'ppo_head_grad_norm', 'ppo_rehearsal_cosine')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_settings(settings):
    """Reject anything except the first, BC-only comparison on the train split."""
    if not isinstance(settings, dict) or not settings.get('enabled'):
        raise ValueError('Rehearsal settings must be an enabled mapping')
    if settings.get('split') != SPLIT:
        raise ValueError('Opening rehearsal accepts the train split only; validation/holdout are refused')
    if float(settings.get('lambda_kl', 0.0)) != 0.0:
        raise ValueError('lambda_kl is not implemented in this stage; it must be 0')
    lam = float(settings['lambda_bc'])
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('lambda_bc must be a positive finite number')
    per = int(settings['sequences_per_minibatch'])
    if per < 1:
        raise ValueError('sequences_per_minibatch must be positive')
    if len(str(settings.get('sha256', ''))) != 64:
        raise ValueError('Pin the train.npz sha256 in the rehearsal settings')
    base = dict(enabled=True, split=SPLIT, lambda_bc=lam, lambda_kl=0.0, sequences_per_minibatch=per,
                seed=int(settings['seed']), sha256=str(settings['sha256']))
    if settings.get('b2b') is None:
        return base  # identical to scripts/opening_rehearsal.py
    return dict(base, b2b=validate_b2b(settings['b2b'], per))


B2B_FILES = ('b2b-demos.npz', 'b2b-demos-manifest.json')
SOURCE_OLD = 0


def validate_b2b(b2b, per_minibatch):
    if not isinstance(b2b, dict) or not b2b.get('enabled'):
        raise ValueError('b2b settings must be an enabled mapping')
    for key in ['sha256', 'manifest_sha256']:
        if len(str(b2b.get(key, ''))) != 64:
            raise ValueError(f'Pin b2b {key}')
    block, old = int(b2b['block_sequences']), int(b2b['old_per_block'])
    demos = {str(k): int(v) for k, v in b2b['demos_per_block'].items()}
    if block <= 0 or old < 0 or any(v < 0 for v in demos.values()) or old + sum(demos.values()) != block:
        raise ValueError('b2b block composition must add up to block_sequences')
    if block % per_minibatch:
        raise ValueError('b2b block must be a whole number of minibatches')
    return dict(enabled=True, sha256=str(b2b['sha256']), manifest_sha256=str(b2b['manifest_sha256']),
                block_sequences=block, old_per_block=old, demos_per_block=demos, schedule_seed=int(b2b['schedule_seed']))


def dataset_directory(manifest_path, deck=DECK):
    """The dataset is always the one built next to the given manifest; no free path."""
    return Path(manifest_path).parent / 'dataset' / deck


def hand_key(case):
    return tuple(sorted(int(x) for x in case['hand']))


def check_split_isolation(manifest, deck=DECK):
    cases = manifest['decks'][deck]['cases']
    hands = {split: {hand_key(c) for c in cases[split]} for split in ['train', 'validation', 'holdout']}
    for other in ['validation', 'holdout']:
        overlap = hands[SPLIT] & hands[other]
        if overlap:
            raise ValueError(f'train shares {len(overlap)} complete starting hands with {other}')
    ids = {split: {c['case_id'] for c in cases[split]} for split in cases}
    return dict(train_cases=len(cases[SPLIT]), train_hands=len(hands[SPLIT]),
                validation_hands=len(hands['validation']), holdout_hands=len(hands['holdout']),
                hand_overlap_with_validation=0, hand_overlap_with_holdout=0), ids


def prepare(arrays):
    """Validate a padded split and return trimmed per-step arrays plus offsets.

    Pure numpy; raises on inconsistent data instead of silently training on it.
    """
    lengths = np.asarray(arrays['lengths'], np.int64)
    weights = np.asarray(arrays['weights'])
    n, T = weights.shape
    if len(lengths) != n or lengths.min() < 1 or lengths.max() > T:
        raise ValueError('Inconsistent sequence lengths')
    expected = (np.arange(T)[None, :] < lengths[:, None])
    if not np.array_equal(weights > 0, expected):
        raise ValueError('weights do not match lengths (padding would leak into the loss)')
    actions = np.asarray(arrays['actions_'])
    legal = actions[..., 3] != 0
    message = actions[:, :, 0, 3].astype(np.int64)
    adapter = np.isin(message, ADAPTER_MESSAGES)
    raw_targets = np.asarray(arrays['targets']) > 0
    targets = raw_targets & legal
    loss_mask = expected & ~adapter
    count = targets.sum(-1)
    empty = loss_mask & (count == 0)
    if empty.any():
        raise ValueError(f'{int(empty.sum())} loss steps have no legal acceptable action')
    chain = np.asarray(arrays['chain'])
    if chain.shape != (n, T, 4) or chain.min() < 0 or chain.max() > np.iinfo(np.int32).max:
        raise ValueError('Unexpected chain feature layout')
    labels = np.asarray(arrays['labels'])
    label_ok = np.take_along_axis(raw_targets, labels[..., None].astype(np.int64), -1)[..., 0]
    if not label_ok[expected].all():
        raise ValueError('A demonstrated label is missing from its acceptable-action set')
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = {k: np.concatenate([np.asarray(arrays[k])[i, :lengths[i]] for i in range(n)]) for k in OBS_KEYS}
    flat['chain'] = np.concatenate([chain[i, :lengths[i]] for i in range(n)]).astype(np.int32)
    flat['targets'] = np.concatenate([targets[i, :lengths[i]] for i in range(n)]).astype(np.uint8)
    flat['loss_mask'] = np.concatenate([loss_mask[i, :lengths[i]] for i in range(n)]).astype(np.float32)
    stats = dict(sequences=int(n), decisions=int(lengths.sum()), max_length=int(lengths.max()),
                 padded_length=int(T), loss_steps=int(loss_mask.sum()),
                 adapter_steps_excluded=int((expected & adapter).sum()),
                 padding_steps_excluded=int((~expected).sum()),
                 multi_target_loss_steps=int((loss_mask & (count > 1)).sum()),
                 illegal_targets_dropped=int((raw_targets & ~legal)[expected].sum()),
                 messages_in_loss={str(m): int(((message == m) & loss_mask).sum()) for m in np.unique(message[loss_mask])})
    return dict(flat=flat, offsets=offsets, lengths=lengths.astype(np.int64), stats=stats)


def load_train_split(manifest_path, data_dir, split=SPLIT, deck=DECK, expected_sha256=None):
    """Load ONLY train.npz, after hash, witness and hand-isolation checks."""
    if split != SPLIT:
        raise ValueError('Opening rehearsal accepts the train split only; validation/holdout are refused')
    data_dir = Path(data_dir)
    path = data_dir / 'train.npz'
    manifest = read(manifest_path)
    if manifest.get('status') != 'complete' or manifest.get('normal_summon_usage_scored') is not False:
        raise ValueError('Manifest is not the completed normal-summon-agnostic policy')
    if manifest.get('goals_and_plans_in_model_inputs') is not False:
        raise ValueError('Manifest does not certify goal-free model inputs')
    record = manifest['decks'][deck]
    actual = sha(path)
    if actual != record[SPLIT]['sha256']:
        raise ValueError('train.npz does not match the manifest checksum')
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('train.npz does not match the pinned rehearsal checksum')
    isolation, ids = check_split_isolation(manifest, deck)
    witnesses = read(data_dir / 'train-witnesses.json')
    if [w['case_id'] for w in witnesses] != [c['case_id'] for c in manifest['decks'][deck]['cases'][SPLIT]]:
        raise ValueError('train witnesses do not follow the train case order')
    if any(w['case_id'] in ids['validation'] | ids['holdout'] or not w['passed'] for w in witnesses):
        raise ValueError('train witnesses include a non-train or failed case')
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if not np.array_equal(arrays['lengths'], [w['decisions'] for w in witnesses]):
        raise ValueError('train.npz lengths do not match the train witnesses')
    prepared = prepare(arrays)
    del arrays
    prepared['stats'].update(isolation, sha256=actual, split=SPLIT, record_decisions=int(record[SPLIT]['decisions']))
    if prepared['stats']['decisions'] != record[SPLIT]['decisions'] or prepared['stats']['sequences'] != record[SPLIT]['trajectories']:
        raise ValueError('train.npz size differs from the manifest record')
    prepared['families'] = [w['family'] for w in witnesses]
    prepared['case_ids'] = [w['case_id'] for w in witnesses]
    return prepared


def prepare_demos(arrays):
    """Demonstrations with SEPARATE masks: weights = valid steps (all unrolled), loss_weights = labelled steps.

    Context steps (valid, not labelled) need no acceptable action; labelled steps must have a legal acceptable
    set containing the demonstrated label and may not be adapter-controlled or padding."""
    lengths = np.asarray(arrays['lengths'], np.int64)
    weights = np.asarray(arrays['weights'])
    n, T = weights.shape
    if len(lengths) != n or lengths.min() < 1 or lengths.max() > T:
        raise ValueError('Inconsistent sequence lengths')
    valid = (np.arange(T)[None, :] < lengths[:, None])
    if not np.array_equal(weights > 0, valid):
        raise ValueError('weights do not match lengths (padding would leak into the history)')
    labelled = np.asarray(arrays['loss_weights']) > 0
    if (labelled & ~valid).any():
        raise ValueError('a labelled step lies in the padding')
    actions = np.asarray(arrays['actions_'])
    legal = actions[..., 3] != 0
    message = actions[:, :, 0, 3].astype(np.int64)
    adapter = np.isin(message, ADAPTER_MESSAGES)
    if (labelled & adapter).any():
        raise ValueError('a labelled step is owned by the placement/position adapters')
    raw_targets = np.asarray(arrays['targets']) > 0
    targets = raw_targets & legal
    loss_mask = valid & labelled
    count = targets.sum(-1)
    if (loss_mask & (count == 0)).any():
        raise ValueError('a labelled step has no legal acceptable action')
    if (raw_targets & ~legal)[loss_mask].any():
        raise ValueError('an acceptable action of a labelled step is not legal')
    labels = np.asarray(arrays['labels'])
    label_ok = np.take_along_axis(targets, labels[..., None].astype(np.int64), -1)[..., 0]
    if not label_ok[loss_mask].all():
        raise ValueError('A demonstrated label is missing from its acceptable-action set')
    if raw_targets[valid & ~loss_mask].any():
        raise ValueError('context steps must not carry acceptable actions')
    chain = np.asarray(arrays['chain'])
    if chain.shape != (n, T, 4) or chain.min() < 0 or chain.max() > np.iinfo(np.int32).max:
        raise ValueError('Unexpected chain feature layout')
    sources = np.asarray(arrays['source'], np.int64)
    if sources.shape != (n,) or (sources <= SOURCE_OLD).any():
        raise ValueError('demo sources must be positive ids')
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = {k: np.concatenate([np.asarray(arrays[k])[i, :lengths[i]] for i in range(n)]) for k in OBS_KEYS}
    flat['chain'] = np.concatenate([chain[i, :lengths[i]] for i in range(n)]).astype(np.int32)
    flat['targets'] = np.concatenate([targets[i, :lengths[i]] for i in range(n)]).astype(np.uint8)
    flat['loss_mask'] = np.concatenate([loss_mask[i, :lengths[i]] for i in range(n)]).astype(np.float32)
    flat['source'] = np.concatenate([np.full(lengths[i], sources[i]) for i in range(n)]).astype(np.int32)
    stats = dict(sequences=int(n), decisions=int(lengths.sum()), max_length=int(lengths.max()), padded_length=int(T),
                 loss_steps=int(loss_mask.sum()), context_steps=int((valid & ~loss_mask).sum()),
                 adapter_steps_in_context=int((valid & adapter).sum()), padding_steps_excluded=int((~valid).sum()),
                 multi_target_loss_steps=int((loss_mask & (count > 1)).sum()),
                 loss_steps_per_sequence=[int(x) for x in loss_mask.sum(1)],
                 messages_in_loss={str(m): int(((message == m) & loss_mask).sum()) for m in np.unique(message[loss_mask])})
    return dict(flat=flat, offsets=offsets, lengths=lengths.astype(np.int64), stats=stats, sources=sources)


def load_demo_split(manifest_path, data_dir, b2b, deck=DECK):
    """Load the B2B demonstrations after hash, manifest, split and hand-isolation checks."""
    data_dir = Path(data_dir)
    path, demo_manifest_path = data_dir / B2B_FILES[0], data_dir / B2B_FILES[1]
    if sha(path) != b2b['sha256']:
        raise ValueError('b2b-demos.npz does not match the pinned checksum')
    if sha(demo_manifest_path) != b2b['manifest_sha256']:
        raise ValueError('b2b-demos-manifest.json does not match the pinned checksum')
    demo_manifest = read(demo_manifest_path)
    if demo_manifest.get('split') != SPLIT or demo_manifest.get('npz_sha256') != b2b['sha256']:
        raise ValueError('demo manifest is not the train-only manifest of this file')
    if demo_manifest.get('validation_or_holdout_used') is not False or demo_manifest.get('legacy_gate_flag_used') is not False:
        raise ValueError('demo manifest must certify no validation/holdout and no legacy-gate flag')
    manifest = read(manifest_path)
    cases = manifest['decks'][deck]['cases']
    train_ids = {c['case_id'] for c in cases[SPLIT]}
    other_hands = {hand_key(c) for split in ['validation', 'holdout'] for c in cases[split]}
    for demo in demo_manifest['demos']:
        if demo['case_id'] not in train_ids:
            raise ValueError(f"demo {demo['name']} is not built from a train case")
        if tuple(sorted(int(x) for x in demo['hand'])) in other_hands:
            raise ValueError(f"demo {demo['name']} shares its starting hand with validation/holdout")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if [int(x) for x in arrays['lengths']] != [d['decisions'] for d in demo_manifest['demos']] or \
            [int(x) for x in arrays['source']] != [d['source_id'] for d in demo_manifest['demos']]:
        raise ValueError('b2b-demos.npz does not follow its manifest')
    prepared = prepare_demos(arrays)
    if set(prepared['stats'].keys()) and prepared['stats']['loss_steps'] != sum(d['loss_steps'] for d in demo_manifest['demos']):
        raise ValueError('labelled steps differ from the manifest')
    prepared['names'] = {d['source_id']: d['name'] for d in demo_manifest['demos']}
    prepared['stats'].update(sha256=b2b['sha256'], manifest_sha256=b2b['manifest_sha256'])
    return prepared


def merge(old, demos):
    """One flat store: old sequences first (source 0), then the demonstrations."""
    n_old, n_demo = len(old['lengths']), len(demos['lengths'])
    flat = {}
    for k in STEP_KEYS + ('targets', 'loss_mask'):
        flat[k] = np.concatenate([old['flat'][k], demos['flat'][k]]).astype(old['flat'][k].dtype)
    flat['source'] = np.concatenate([np.zeros(len(old['flat']['loss_mask']), np.int32), demos['flat']['source']])
    lengths = np.concatenate([old['lengths'], demos['lengths']])
    sources = np.concatenate([np.zeros(n_old, np.int64), demos['sources']])
    stats = dict(old=old['stats'], b2b=demos['stats'], sequences=int(n_old + n_demo),
                 max_length=int(max(old['stats']['max_length'], demos['stats']['max_length'])))
    return dict(flat=flat, offsets=np.concatenate([[0], np.cumsum(lengths)]), lengths=lengths, stats=stats, sources=sources,
                old_count=n_old, demo_indices={int(s): n_old + i for i, s in enumerate(demos['sources'])},
                names=demos.get('names', {}))


def batch(prepared, indices, length):
    """Time-major batch [length, B, ...]; padding repeats the last real step (no new inputs)."""
    flat, offsets, lengths = prepared['flat'], prepared['offsets'], prepared['lengths']
    indices = np.asarray(indices, np.int64)
    if lengths[indices].max() > length:
        raise ValueError('Batch length is shorter than a sequence')
    t = np.arange(length)[:, None]
    rows = offsets[indices][None, :] + np.minimum(t, lengths[indices][None, :] - 1)
    out = {k: flat[k][rows] for k in STEP_KEYS + ('targets',)}
    out['loss_mask'] = flat['loss_mask'][rows] * (t < lengths[indices][None, :])
    if 'source' in flat:  # only for the merged B2B store; the old path keeps its exact keys
        out['source'] = flat['source'][rows]
    return out


class Sampler:
    """Uniform with-replacement sampling of whole train sequences; independent numpy RNG."""

    def __init__(self, prepared, minibatches, per_minibatch, seed, length=None):
        self.prepared = prepared
        self.minibatches, self.per_minibatch = int(minibatches), int(per_minibatch)
        self.length = int(length or prepared['stats']['max_length'])
        self.rng = np.random.default_rng(seed)

    def next(self):
        idx = self.rng.integers(0, len(self.prepared['lengths']), size=(self.minibatches, self.per_minibatch))
        parts = [batch(self.prepared, row, self.length) for row in idx]
        return {k: np.stack([p[k] for p in parts]) for k in parts[0]}, idx


class MixedSampler:
    """Fixed composition: every block of `block_sequences` slots holds `old_per_block` old sequences (uniform over
    the old store) and `demos_per_block[source]` slots of each demonstration; slot positions are shuffled per block.
    Blocks are consumed in order, minibatches x per_minibatch slots per call."""

    def __init__(self, prepared, minibatches, per_minibatch, seed, b2b, length=None):
        self.prepared = prepared
        self.minibatches, self.per_minibatch = int(minibatches), int(per_minibatch)
        self.length = int(length or prepared['stats']['max_length'])
        seq = np.random.SeedSequence(seed)
        slot_seed, old_seed = seq.spawn(2)
        self.slot_rng, self.old_rng = np.random.default_rng(slot_seed), np.random.default_rng(old_seed)
        self.b2b, self.queue, self.drawn = b2b, [], []
        if (self.minibatches * self.per_minibatch) % b2b['block_sequences'] and b2b['block_sequences'] % (self.minibatches * self.per_minibatch):
            raise ValueError('blocks and PPO updates must align')

    def _block(self):
        kinds = [SOURCE_OLD] * self.b2b['old_per_block']
        for source, count in sorted(self.b2b['demos_per_block'].items()):
            kinds += [int(source)] * count
        kinds = list(self.slot_rng.permutation(kinds))
        out = []
        for kind in kinds:
            if kind == SOURCE_OLD:
                out.append((SOURCE_OLD, int(self.old_rng.integers(0, self.prepared['old_count']))))
            else:
                out.append((int(kind), int(self.prepared['demo_indices'][int(kind)])))
        return out

    def next(self):
        need = self.minibatches * self.per_minibatch
        while len(self.queue) < need:
            self.queue += self._block()
        take, self.queue = self.queue[:need], self.queue[need:]
        self.drawn += take
        idx = np.array([i for _, i in take], np.int64).reshape(self.minibatches, self.per_minibatch)
        parts = [batch(self.prepared, row, self.length) for row in idx]
        return {k: np.stack([p[k] for p in parts]) for k in parts[0]}, idx

    def counts(self):
        out = {}
        for source, _ in self.drawn:
            out[str(source)] = out.get(str(source), 0) + 1
        return out


# ---------------------------------------------------------------- JAX functions

def unroll_logits(agent, variables, steps):
    """Final (base + chain-head) logits for time-major steps [T, B, ...].

    The RNN state starts at zero for every sequence (sequence boundary = new duel)
    and advances through every step, including steps excluded from the loss.
    """
    import jax
    import jax.numpy as jnp
    width = steps['global_'].shape[1]
    initial = jax.tree.map(jnp.asarray, agent.init_rnn_state(width))

    def step(state, x):
        obs = dict(cards_=x['cards_'], global_=x['global_'], actions_=x['actions_'],
                   h_actions_=x['h_actions_'], mask_=None, chain_=x['chain'])
        state, logits = agent.apply(variables, obs, state)[:2]
        return state, logits
    return jax.lax.scan(jax.checkpoint(step), initial, {k: steps[k] for k in STEP_KEYS})[1]


def acceptable_action_ce(logits, steps):
    """Mean over loss steps of -log P(acceptable & legal set); also top-1 acceptable rate."""
    import jax
    import jax.numpy as jnp
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    targets = steps['targets'] > 0
    ce = -jax.nn.logsumexp(jnp.where(targets, logp, -1e9), axis=-1)
    mask = steps['loss_mask'].astype(jnp.float32)
    count = jnp.sum(mask)
    denom = jnp.maximum(count, 1.0)
    hit = jnp.take_along_axis(targets, logits.argmax(-1)[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(mask > 0, ce, 0.0)) / denom, dict(accuracy=jnp.sum(hit * mask) / denom, loss_steps=count, per_step=ce)


def rehearsal_gradient(agent, params, batch_stats, steps):
    import jax

    def loss(p):
        value, metrics = acceptable_action_ce(unroll_logits(agent, {'params': p, 'batch_stats': batch_stats}, steps), steps)
        return value, metrics
    (value, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params)
    return value, metrics, grads


def combine(agent, params, batch_stats, ppo_grads, steps, lambda_bc):
    """PPO gradient + lambda_bc * rehearsal gradient, and diagnostic norms."""
    import jax
    import jax.numpy as jnp
    import optax
    value, metrics, grads = rehearsal_gradient(agent, params, batch_stats, steps)
    total = jax.tree.map(lambda a, b: a + lambda_bc * b, ppo_grads, grads)
    dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(ppo_grads), jax.tree.leaves(grads)))
    cosine = dot / jnp.maximum(optax.global_norm(ppo_grads) * optax.global_norm(grads), 1e-12)
    stats = jnp.stack([value, metrics['accuracy'], metrics['loss_steps'], optax.global_norm(ppo_grads),
                       optax.global_norm(grads), optax.global_norm(total), optax.global_norm(grads['base']),
                       optax.global_norm(grads['head']), optax.global_norm(ppo_grads['head']), cosine])
    return total, stats


METRIC_NAMES_B2B = METRIC_NAMES + ('ce_old', 'ce_b2b', 'loss_steps_old', 'loss_steps_b2b',
                                   'weighted_grad_norm_old', 'weighted_grad_norm_b2b', 'sequences_old', 'sequences_b2b')


def combine_mixed(agent, params, batch_stats, ppo_grads, steps, lambda_bc):
    """Same objective as combine() (mean CE over all loss steps of the minibatch), split by source.

    One forward unroll; the rehearsal gradient is the sum of the two source parts (each = gradient of its share of
    the minibatch mean), so it equals combine()'s gradient. Weighted norms are lambda_bc * ||part||."""
    import jax
    import jax.numpy as jnp
    import optax
    logits, pullback = jax.vjp(lambda p: unroll_logits(agent, {'params': p, 'batch_stats': batch_stats}, steps), params)
    targets = steps['targets'] > 0
    mask = steps['loss_mask'].astype(jnp.float32)
    b2b = (steps['source'] > SOURCE_OLD).astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask), 1.0)

    def ce_of(lg):
        logp = jax.nn.log_softmax(lg.astype(jnp.float32), axis=-1)
        return -jax.nn.logsumexp(jnp.where(targets, logp, -1e9), axis=-1)

    def share(m):
        return lambda lg: jnp.sum(jnp.where(m > 0, ce_of(lg), 0.0)) / denom
    m_old, m_b2b = mask * (1.0 - b2b), mask * b2b
    g_old = pullback(jax.grad(share(m_old))(logits))[0]
    g_b2b = pullback(jax.grad(share(m_b2b))(logits))[0]
    grads = jax.tree.map(lambda a, b: a + b, g_old, g_b2b)
    ce = ce_of(logits)
    value = jnp.sum(jnp.where(mask > 0, ce, 0.0)) / denom
    hit = jnp.take_along_axis(targets, logits.argmax(-1)[..., None], axis=-1)[..., 0]
    accuracy = jnp.sum(hit * mask) / denom
    total = jax.tree.map(lambda a, b: a + lambda_bc * b, ppo_grads, grads)
    dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(ppo_grads), jax.tree.leaves(grads)))
    cosine = dot / jnp.maximum(optax.global_norm(ppo_grads) * optax.global_norm(grads), 1e-12)
    n_old, n_b2b = jnp.sum(m_old), jnp.sum(m_b2b)
    seq_b2b = jnp.sum(b2b[0])
    stats = jnp.stack([value, accuracy, jnp.sum(mask), optax.global_norm(ppo_grads), optax.global_norm(grads),
                       optax.global_norm(total), optax.global_norm(grads['base']), optax.global_norm(grads['head']),
                       optax.global_norm(ppo_grads['head']), cosine,
                       jnp.sum(jnp.where(m_old > 0, ce, 0.0)) / jnp.maximum(n_old, 1.0),
                       jnp.sum(jnp.where(m_b2b > 0, ce, 0.0)) / jnp.maximum(n_b2b, 1.0), n_old, n_b2b,
                       lambda_bc * optax.global_norm(g_old), lambda_bc * optax.global_norm(g_b2b),
                       b2b.shape[1] - seq_b2b, seq_b2b])
    return total, stats


# ---------------------------------------------------------------- snapshot / trainer integration

TRAINER_PATCHES = [
    ('        key: jax.random.PRNGKey,\n    ):\n',
     '        key: jax.random.PRNGKey,\n        rehearsal_batches,\n    ):\n'),
    ('            if args.segment_length is None:\n                def update_minibatch(agent_state, minibatch):\n'
     '                    (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)), grads = \\\n'
     '                        loss_grad_fn(agent_state.params, agent_state.batch_stats, *minibatch)\n'
     '                    grads = jax.lax.pmean(grads, axis_name="local_devices")\n'
     '                    agent_state = agent_state.apply_gradients(grads=grads)\n'
     "                    agent_state = agent_state.replace(batch_stats=state_updates['batch_stats'])\n"
     '                    return agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl)\n',
     "            assert args.segment_length is None, 'opening rehearsal supports the default minibatch path only'\n"
     '            if args.segment_length is None:\n                def update_minibatch(agent_state, minibatch):\n'
     '                    *minibatch, rehearsal_batch = minibatch\n'
     '                    (loss, (state_updates, pg_loss, v_loss, ent_loss, approx_kl)), grads = \\\n'
     '                        loss_grad_fn(agent_state.params, agent_state.batch_stats, *minibatch)\n'
     '                    grads, rehearsal_metrics = OPENING_REHEARSAL.combine(\n'
     '                        agent, agent_state.params, agent_state.batch_stats, grads, rehearsal_batch)\n'
     '                    grads = jax.lax.pmean(grads, axis_name="local_devices")\n'
     '                    agent_state = agent_state.apply_gradients(grads=grads)\n'
     "                    agent_state = agent_state.replace(batch_stats=state_updates['batch_stats'])\n"
     '                    return agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl, rehearsal_metrics)\n'),
    ('            agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(\n                update_minibatch,\n',
     '            agent_state, (loss, pg_loss, v_loss, ent_loss, approx_kl, rehearsal_metrics) = jax.lax.scan(\n                update_minibatch,\n'),
    ('                    b_next_data,\n                ),\n            )\n'
     '            return (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl)\n',
     '                    b_next_data,\n                    rehearsal_batches,\n                ),\n            )\n'
     '            return (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl, rehearsal_metrics)\n'),
    ('        (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl) = jax.lax.scan(\n',
     '        (agent_state, key), (loss, pg_loss, v_loss, ent_loss, approx_kl, rehearsal_metrics) = jax.lax.scan(\n'),
    ('        return agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, key\n',
     '        return agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, key, rehearsal_metrics\n'),
    ('        (agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, learner_keys) = multi_device_update(\n'
     '            agent_state,\n            *list(zip(*sharded_data_list)),\n            learner_keys,\n        )\n',
     '        (agent_state, loss, pg_loss, v_loss, ent_loss, approx_kl, learner_keys, rehearsal_metrics) = multi_device_update(\n'
     '            agent_state,\n            *list(zip(*sharded_data_list)),\n            learner_keys,\n'
     '            OPENING_REHEARSAL.next_device_batches(args, learner_devices),\n        )\n'
     '        OPENING_REHEARSAL.record(args, learner_policy_version, rehearsal_metrics, time.time() - training_time_start)\n'),
    ('    install_training_backend(globals())\n    main()\n',
     '    install_training_backend(globals())\n    from opening_rehearsal import install_rehearsal\n'
     '    install_rehearsal(globals())\n    main()\n    OPENING_REHEARSAL.finish()\n'),
]


def patch_trainer(text):
    """Apply the rehearsal hooks on top of continuous_opening_backend.patched_trainer()."""
    for old, new in TRAINER_PATCHES:
        if text.count(old) != 1:
            raise RuntimeError(f'Unexpected trainer source for rehearsal patch: {old[:80]!r}')
        text = text.replace(old, new)
    compile(text, 'rehearsal-trainer.py', 'exec')
    return text


def prepare_snapshot(cfg, snapshot, trainer_text, copy):
    """Pin module + train data in a NEW snapshot; return the rehearsal trainer text."""
    settings = validate_settings(cfg['rehearsal'])
    snapshot = Path(snapshot)
    source = dataset_directory(cfg['opening_manifest'])
    target = snapshot / SNAPSHOT_DIR
    for name in ['train.npz', 'train-witnesses.json']:
        copy(source / name, target / name)
    copy(Path(__file__), snapshot / 'opening_rehearsal.py')
    prepared = load_train_split(snapshot / 'opening-manifest.json', target, SPLIT, expected_sha256=settings['sha256'])
    (target / 'settings.json').write_text(json.dumps(dict(
        settings=settings, data=prepared['stats'], source_dataset=str(source / 'train.npz'),
        validation_or_holdout_loaded=False, objective='ppo_loss + lambda_bc * acceptable_action_ce(train)',
        clear_full_implementation=False, network_or_llm_calls=False), ensure_ascii=False, indent=2) + '\n')
    return patch_trainer(trainer_text)


class TrainerRehearsal:
    def __init__(self, cfg, prepared, settings):
        self.cfg, self.prepared, self.settings = cfg, prepared, settings
        self.sampler = None
        self.log = None
        self.updates = 0

    def combine(self, agent, params, batch_stats, ppo_grads, steps):
        if self.settings.get('b2b'):
            return combine_mixed(agent, params, batch_stats, ppo_grads, steps, self.settings['lambda_bc'])
        return combine(agent, params, batch_stats, ppo_grads, steps, self.settings['lambda_bc'])

    def next_device_batches(self, args, devices):
        import jax
        if self.sampler is None:
            if args.update_epochs != 1:
                raise RuntimeError('Rehearsal batches are drawn once per PPO update (update_epochs=1)')
            if self.settings.get('b2b'):
                seed = [self.settings['b2b']['schedule_seed'], int(args.seed)]
                self.sampler = MixedSampler(self.prepared, args.num_minibatches, self.settings['sequences_per_minibatch'],
                                            seed, self.settings['b2b'])
            else:
                seed = np.random.SeedSequence([self.settings['seed'], int(args.seed)])
                self.sampler = Sampler(self.prepared, args.num_minibatches, self.settings['sequences_per_minibatch'], seed)
            self.log = Path(args.ckpt_dir).parent / 'rehearsal-metrics.jsonl'
            self.indices = []
        shards = []
        for _ in devices:
            steps, idx = self.sampler.next()
            self.indices.append(idx.tolist())
            shards.append(steps)
        return jax.device_put_sharded(shards, devices)

    def record(self, args, version, metrics, seconds):
        values = np.asarray(metrics)  # [devices, epochs, minibatches, metric]
        if not np.all(np.isfinite(values)):
            raise RuntimeError('Non-finite rehearsal metrics')
        self.updates += 1
        rows = values.reshape(-1, values.shape[-1])
        names = METRIC_NAMES_B2B if self.settings.get('b2b') else METRIC_NAMES
        entry = dict(update=int(version), learner_seconds=round(float(seconds), 4),
                     sampled_sequences=self.indices[-len(values):],
                     minibatches=[dict(zip(names, map(float, r))) for r in rows],
                     mean={k: float(rows[:, i].mean()) for i, k in enumerate(names)})
        if self.settings.get('b2b'):
            entry['sampled_sources'] = [[int(self.prepared['sources'][i]) for i in np.asarray(idx).ravel()] for idx in self.indices[-len(values):]]
            entry['source_counts_so_far'] = self.sampler.counts()
        with self.log.open('a') as f:
            f.write(json.dumps(entry) + '\n')
        print('OPENING_REHEARSAL', json.dumps(dict(update=entry['update'], **entry['mean'])), flush=True)

    def finish(self):
        print('OPENING_REHEARSAL_UPDATES', self.updates, flush=True)
        if self.settings.get('b2b') and self.sampler is not None:
            print('OPENING_REHEARSAL_SOURCES', json.dumps(dict(counts=self.sampler.counts(), order=self.sampler.drawn)), flush=True)


def install_rehearsal(namespace):
    cfg = read(os.environ['YGO_CONTINUOUS_CONFIG'])
    settings = validate_settings(cfg['rehearsal'])
    snapshot = Path(cfg['root']) / 'snapshot'
    hashes = read(snapshot / 'hashes.json')
    for name in ['opening_rehearsal.py', f'{SNAPSHOT_DIR}/train.npz', f'{SNAPSHOT_DIR}/train-witnesses.json', 'opening-manifest.json']:
        if sha(snapshot / name) != hashes[name]:
            raise RuntimeError(f'Pinned rehearsal input changed: {name}')
    if Path(__file__).resolve() != (snapshot / 'opening_rehearsal.py').resolve():
        raise RuntimeError('Trainer imported an unpinned rehearsal module')
    started = time.perf_counter()
    prepared = load_train_split(snapshot / 'opening-manifest.json', snapshot / SNAPSHOT_DIR, SPLIT, expected_sha256=settings['sha256'])
    if settings.get('b2b'):
        for name in B2B_FILES:
            if sha(snapshot / SNAPSHOT_DIR / name) != hashes[f'{SNAPSHOT_DIR}/{name}']:
                raise RuntimeError(f'Pinned rehearsal input changed: {SNAPSHOT_DIR}/{name}')
        demos = load_demo_split(snapshot / 'opening-manifest.json', snapshot / SNAPSHOT_DIR, settings['b2b'])
        prepared = merge(prepared, demos)
    print('OPENING_REHEARSAL_DATA', json.dumps(dict(prepared['stats'], load_seconds=round(time.perf_counter() - started, 2)), default=str), flush=True)
    namespace['OPENING_REHEARSAL'] = TrainerRehearsal(cfg, prepared, settings)
