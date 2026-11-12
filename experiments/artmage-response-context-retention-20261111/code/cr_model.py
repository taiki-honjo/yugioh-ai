"""Response loss with context retention (JAX). Old minibatches keep batch_model.update (CE only) unchanged.

Response minibatch: loss = CE_response + COEF * KL_context
  CE_response  = batch_model.sequence_loss (the window action only; per-sequence mean, then mean over the 2 sequences)
  KL_context   = mean over sequences of (mean over preserved steps of KL(p_start_legal || p_current_legal)), temperature 1
p_start is the fixed reference array (constant input; stop_gradient applied), p_current the candidate's own logits along the
same observation history; both are normalised over the legal options only (illegal options never enter the sum).
The two terms' gradients are taken separately and summed; the optimizer (global-norm clip 1.0, Adam) sees the sum.
"""
import jax
import jax.numpy as jnp
import optax

import batch_model as BM

COEF = 1.0


def kl_context(params, batch_stats, batch):
    _, logits = BM.step_logp(params, batch_stats, batch)
    lg = batch['legal']
    logq = jax.nn.log_softmax(jnp.where(lg, logits.astype(jnp.float32), -1e9), -1)
    logp = jax.lax.stop_gradient(batch['ref_logp'])
    p = jnp.where(lg, jnp.exp(logp), 0.0)
    kl_t = jnp.sum(jnp.where(lg, p * (logp - logq), 0.0), -1)  # [T, B]
    m = batch['kl_mask']
    steps = jnp.sum(m, 0)
    per_seq = jnp.sum(kl_t * m, 0) / jnp.maximum(steps, 1.0)
    valid = (steps > 0).astype(jnp.float32)
    kl = jnp.sum(per_seq * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    return kl, dict(kl_per_seq=per_seq, kl_max_step=jnp.max(kl_t * m, 0), kl_steps=steps)


def update_response(params, opt_state, batch_stats, batch, optimizer, coef=COEF):
    (ce, aux_ce), g_ce = jax.value_and_grad(BM.sequence_loss, has_aux=True)(params, batch_stats, batch)
    (kl, aux_kl), g_kl = jax.value_and_grad(kl_context, has_aux=True)(params, batch_stats, batch)
    grads = jax.tree.map(lambda a, b: a + coef * b, g_ce, g_kl)
    gnorm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new = optax.apply_updates(params, updates)
    loss = ce + coef * kl
    finite = jnp.isfinite(loss) & jnp.isfinite(gnorm) & jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(new)]))
    return new, opt_state, dict(loss=loss, ce=ce, kl=kl, ce_grad_norm=optax.global_norm(g_ce), kl_grad_norm=optax.global_norm(g_kl),
                                grad_norm=gnorm, clipped=gnorm > BM.CONFIG['max_grad_norm'], finite=finite,
                                base_update=optax.global_norm(updates['base']), head_update=optax.global_norm(updates['head']), **aux_ce, **aux_kl)


def legal_logp(params, batch_stats, batch):
    """Legal-normalised log-probabilities [T, B, A] (illegal = -1e9-dominated), same arithmetic as kl_context."""
    _, logits = BM.step_logp(params, batch_stats, batch)
    return jax.nn.log_softmax(jnp.where(batch['actions_'][..., 3] != 0, logits.astype(jnp.float32), -1e9), -1)
