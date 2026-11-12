"""JAX model functions for the Artmage learning pilot: the pinned base (RNNAgent) + chain head, as used by
opponent_article_bc_common.Policy, with both parts trainable under one optimizer.

Params tree: {'base': base['params'], 'head': head['params']}; base batch_stats are carried unchanged.
No engine is touched here.
"""
from pathlib import Path

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from opponent_article_bc_common import AGENT, HEAD

OBS_KEYS = ('cards_', 'global_', 'actions_', 'h_actions_')
CONFIG = dict(learning_rate=1.5e-5, adam_eps=1e-5, max_grad_norm=1.0, clip_coef=0.25, dual_clip_coef=3.0, ent_coef=0.01,
              vf_coef=1.0, vloss_clip=1.0, gamma=1.0, gae_lambda=0.95, rehearsal_lambda=5.0, rehearsal_kl=0.0,
              norm_adv=False, upgo=False, update_epochs=1, minibatches_per_window=4, chunk_length=32)


# ------------------------------------------------------------------ files
def load_pair(base_path, head_path):
    base = serialization.msgpack_restore(Path(base_path).read_bytes())
    head = serialization.msgpack_restore(Path(head_path).read_bytes())
    params = {'base': base['params'], 'head': head['params']}
    return jax.device_put(params), jax.device_put(base.get('batch_stats', {})), base


def save_pair(params, original_base, base_path, head_path):
    """Write base (original key order: batch_stats, params) and chain head exactly as the original files are laid out."""
    params = jax.device_get(params)
    base = {k: (params['base'] if k == 'params' else v) for k, v in original_base.items()}
    Path(base_path).write_bytes(serialization.msgpack_serialize(base))
    Path(head_path).write_bytes(serialization.msgpack_serialize({'params': params['head']}))


# ------------------------------------------------------------------ forward
def apply_step(params, batch_stats, state, obs, chain):
    """One decision for a batch: returns (state, final logits, value). Same arithmetic as Policy.infer."""
    x = {k: obs[k] for k in OBS_KEYS}
    x['mask_'] = None
    state, logits, value, _ = AGENT.apply({'params': params['base'], 'batch_stats': batch_stats}, x, state)
    logits = logits + HEAD.apply({'params': params['head']}, chain, obs['actions_'])
    return state, logits, value[..., 0]


infer = jax.jit(apply_step)


def init_state(batch):
    return jax.tree.map(jnp.asarray, AGENT.init_rnn_state(batch))


def unroll(params, batch_stats, state, steps, first=None):
    """Time-major steps [T, B, ...]; optional first[T, B] resets the state to zero before that step."""
    first = jnp.zeros(steps['actions_'].shape[:2], bool) if first is None else first

    def step(carry, x):
        carry = jax.tree.map(lambda s: jnp.where(x['first'][:, None], 0.0, s), carry)
        carry, logits, value = apply_step(params, batch_stats, carry, x, x['chain'])
        return carry, (logits, value)
    xs = {k: steps[k] for k in OBS_KEYS + ('chain',)}
    xs['first'] = first
    _, (logits, values) = jax.lax.scan(jax.checkpoint(step), state, xs)
    return logits, values


# ------------------------------------------------------------------ losses
def ppo_loss(params, batch_stats, chunk):
    """chunk: time-major [T, 1] learner decisions with init_state, first, actions, old_logits, advantages, targets, mask."""
    logits, values = unroll(params, batch_stats, chunk['init_state'], chunk, chunk['first'])
    logp = jax.nn.log_softmax(logits, -1)
    old_logp = jax.nn.log_softmax(chunk['old_logits'], -1)
    a = chunk['actions'][..., None]
    ratio = jnp.exp(jnp.take_along_axis(logp, a, -1)[..., 0] - jnp.take_along_axis(old_logp, a, -1)[..., 0])
    adv = jax.lax.stop_gradient(chunk['advantages'])
    clipped = jnp.fmin(ratio * adv, jnp.clip(ratio, 1 - CONFIG['clip_coef'], 1 + CONFIG['clip_coef']) * adv)
    clipped = jnp.where(adv >= 0, clipped, jnp.fmax(clipped, CONFIG['dual_clip_coef'] * adv))
    pg = -clipped
    v = jnp.minimum(0.5 * (values - chunk['targets']) ** 2, CONFIG['vloss_clip'])
    p = jnp.exp(logp)
    ent = -jnp.sum(jnp.where(p > 0, p * logp, 0.0), -1)
    approx_kl = (ratio - 1) - jnp.log(ratio)
    mask = chunk['mask']
    n = jnp.maximum(jnp.sum(mask), 1.0)
    pg, v, ent, kl = [jnp.sum(x * mask) / n for x in (pg, v, ent, approx_kl)]
    loss = pg - CONFIG['ent_coef'] * ent + CONFIG['vf_coef'] * v
    return loss, dict(pg=pg, v=v, ent=ent, approx_kl=kl, steps=jnp.sum(mask),
                      clipfrac=jnp.sum((jnp.abs(ratio - 1) > CONFIG['clip_coef']) * mask) / n)


def rehearsal_step_ce(params, batch_stats, batch):
    """Per-step CE of the demonstrated label (single acceptable action), time-major [T, B]."""
    logits, _ = unroll(params, batch_stats, init_state(batch['labels'].shape[1]), batch)
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    ce = -jnp.take_along_axis(logp, batch['labels'][..., None], -1)[..., 0]
    hit = (logits.argmax(-1) == batch['labels']).astype(jnp.float32)
    return ce, hit


def tree_norm(tree):
    return optax.global_norm(tree)


def make_optimizer():
    return optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(CONFIG['max_grad_norm']),
                                             optax.adam(CONFIG['learning_rate'], eps=CONFIG['adam_eps'])), max_consecutive_errors=10)


def update_step(params, opt_state, batch_stats, chunk, reh, optimizer):
    """PPO gradient on learner decisions + lambda * rehearsal gradient; one Adam step after global-norm clipping.

    Rehearsal loss = sum over loss steps of CE / number of loss steps in the minibatch (2 sequences). Its gradient is
    taken per sequence (vector-Jacobian product with that sequence's mask) so each source's share can be logged;
    the per-sequence gradients add up to the rehearsal gradient exactly."""
    (loss, aux), g_ppo = jax.value_and_grad(ppo_loss, has_aux=True)(params, batch_stats, chunk)
    ce, vjp, hit = jax.vjp(lambda p: rehearsal_step_ce(p, batch_stats, reh), params, has_aux=True)
    mask = reh['loss']
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    seq_grads = [vjp(jnp.zeros_like(mask).at[:, j].set(mask[:, j] / denom))[0] for j in range(mask.shape[1])]
    g_reh = jax.tree.map(lambda *g: sum(g), *seq_grads)
    total = jax.tree.map(lambda a, b: a + CONFIG['rehearsal_lambda'] * b, g_ppo, g_reh)
    src = reh['source']
    per_source = []
    for k in range(3):
        g_k = jax.tree.map(lambda *g: sum(jnp.where(src[j] == k, g[j], 0.0) for j in range(len(g))), *seq_grads)
        steps_k = jnp.sum(jnp.where(src[None, :] == k, mask, 0.0))
        ce_k = jnp.sum(jnp.where(src[None, :] == k, ce * mask, 0.0)) / jnp.maximum(steps_k, 1.0)
        per_source.append(jnp.stack([steps_k, ce_k, CONFIG['rehearsal_lambda'] * tree_norm(g_k)]))
    dot = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(g_ppo), jax.tree.leaves(g_reh)))
    cos = dot / jnp.maximum(tree_norm(g_ppo) * tree_norm(g_reh), 1e-12)
    updates, opt_state = optimizer.update(total, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics = dict(loss=loss, **aux, reh_ce=jnp.sum(ce * mask) / denom, reh_acc=jnp.sum(hit * mask) / denom, reh_steps=jnp.sum(mask),
                   ppo_grad=tree_norm(g_ppo), reh_grad_weighted=CONFIG['rehearsal_lambda'] * tree_norm(g_reh), combined_grad=tree_norm(total),
                   combined_grad_after_clip=jnp.minimum(tree_norm(total), CONFIG['max_grad_norm']), ppo_reh_cosine=cos,
                   base_update=tree_norm(updates['base']), head_update=tree_norm(updates['head']),
                   per_source=jnp.stack(per_source))
    return new_params, opt_state, metrics


# ------------------------------------------------------------------ advantages (numpy)
def gae(values, rewards, dones, bootstrap, gamma=CONFIG['gamma'], lam=CONFIG['gae_lambda']):
    """One learner segment (time order). dones[i] = episode ended after decision i. bootstrap = V after the last step."""
    n = len(values)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for i in reversed(range(n)):
        next_value = bootstrap if i == n - 1 else values[i + 1]
        nonterminal = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + np.asarray(values, np.float32)
