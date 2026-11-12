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
                'ppo_head_grad_norm')


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
    return dict(enabled=True, split=SPLIT, lambda_bc=lam, lambda_kl=0.0, sequences_per_minibatch=per,
                seed=int(settings['seed']), sha256=str(settings['sha256']))


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
    stats = jnp.stack([value, metrics['accuracy'], metrics['loss_steps'], optax.global_norm(ppo_grads),
                       optax.global_norm(grads), optax.global_norm(total), optax.global_norm(grads['base']),
                       optax.global_norm(grads['head']), optax.global_norm(ppo_grads['head'])])
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
        return combine(agent, params, batch_stats, ppo_grads, steps, self.settings['lambda_bc'])

    def next_device_batches(self, args, devices):
        import jax
        if self.sampler is None:
            if args.update_epochs != 1:
                raise RuntimeError('Rehearsal batches are drawn once per PPO update (update_epochs=1)')
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
        entry = dict(update=int(version), learner_seconds=round(float(seconds), 4),
                     sampled_sequences=self.indices[-len(values):],
                     minibatches=[dict(zip(METRIC_NAMES, map(float, r))) for r in rows],
                     mean={k: float(rows[:, i].mean()) for i, k in enumerate(METRIC_NAMES)})
        with self.log.open('a') as f:
            f.write(json.dumps(entry) + '\n')
        print('OPENING_REHEARSAL', json.dumps(dict(update=entry['update'], **entry['mean'])), flush=True)

    def finish(self):
        print('OPENING_REHEARSAL_UPDATES', self.updates, flush=True)


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
    print('OPENING_REHEARSAL_DATA', json.dumps(dict(prepared['stats'], load_seconds=round(time.perf_counter() - started, 2))), flush=True)
    namespace['OPENING_REHEARSAL'] = TrainerRehearsal(cfg, prepared, settings)
