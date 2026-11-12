"""Supervised learning functions for the batch (base + chain head, one optimizer). No engine, no PPO.

Loss = mean over the batch's sequences of (mean CE of the demonstrated label over that sequence's loss steps).
This differs from the pilot rehearsal (mean over all loss steps of the minibatch): a long sequence does not get
more weight because it has more decisions.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import artmage_entry  # noqa: F401  (pinned files checked; sets the model-code path; no engine duel)
import pilot_model as PM

CONFIG = dict(learning_rate=1.5e-5, adam_eps=1e-5, max_grad_norm=1.0, updates=256, batch_sequences=2, kl=0.0, ppo=False,
              value_loss=False, entropy_loss=False, diagnostics_at=[0, 256])  # response study: 256 updates, start + final only


def make_optimizer():
    return optax.chain(optax.clip_by_global_norm(CONFIG['max_grad_norm']), optax.adam(CONFIG['learning_rate'], eps=CONFIG['adam_eps']))


def step_logp(params, batch_stats, batch):
    """Teacher-forced log-probabilities, time-major [T, B, A]; every sequence starts from a zero RNN state."""
    logits, _ = PM.unroll(params, batch_stats, PM.init_state(batch['labels'].shape[1]), batch)
    return jax.nn.log_softmax(logits.astype(jnp.float32), -1), logits


def sequence_loss(params, batch_stats, batch):
    logp, logits = step_logp(params, batch_stats, batch)
    ce = -jnp.take_along_axis(logp, batch['labels'][..., None], -1)[..., 0]
    mask = batch['loss']
    per_seq_steps = jnp.sum(mask, 0)
    per_seq = jnp.sum(ce * mask, 0) / jnp.maximum(per_seq_steps, 1.0)
    valid = (per_seq_steps > 0).astype(jnp.float32)
    loss = jnp.sum(per_seq * valid) / jnp.maximum(jnp.sum(valid), 1.0)
    hit = jnp.sum((logits.argmax(-1) == batch['labels']) * mask, 0) / jnp.maximum(per_seq_steps, 1.0)
    return loss, dict(per_seq_ce=per_seq, per_seq_acc=hit, per_seq_steps=per_seq_steps)


def update(params, opt_state, batch_stats, batch, optimizer):
    (loss, aux), grads = jax.value_and_grad(sequence_loss, has_aux=True)(params, batch_stats, batch)
    gnorm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new = optax.apply_updates(params, updates)
    finite = jnp.isfinite(loss) & jnp.isfinite(gnorm) & jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(new)]))
    return new, opt_state, dict(loss=loss, grad_norm=gnorm, clipped=gnorm > CONFIG['max_grad_norm'], finite=finite,
                                base_update=optax.global_norm(updates['base']), head_update=optax.global_norm(updates['head']), **aux)


def evaluate_sequences(params, batch_stats, seqs, length, logp_fn):
    """Per sequence: per-step log-probabilities of every option (teacher forcing). Returns list of numpy [n, A]."""
    out = []
    import pilot_data as PD
    for s in seqs:
        b = PD.batch([s], length=length)
        logp = np.asarray(logp_fn(params, batch_stats, {k: jnp.asarray(v) for k, v in b.items()})[0])[:len(s['labels']), 0]
        out.append(logp)
    return out


def summarize(logps, seqs, branches=None):
    """CE / top-1 over loss steps (pooled and per-sequence mean) and P(label) at the given branch steps."""
    ce_steps, hits, per_seq = [], [], []
    for lp, s in zip(logps, seqs):
        m = s['loss'] > 0
        ce = -lp[np.arange(len(s['labels'])), s['labels']]
        top = lp.argmax(-1) == s['labels']
        ce_steps.extend(ce[m].tolist())
        hits.extend(top[m].tolist())
        per_seq.append(float(ce[m].mean()) if m.any() else None)
    out = dict(loss_steps=len(ce_steps), ce_pooled=float(np.mean(ce_steps)) if ce_steps else None,
               ce_per_sequence_mean=float(np.mean([x for x in per_seq if x is not None])) if per_seq else None,
               top1=float(np.mean(hits)) if hits else None)
    if branches is not None:
        out['branches'] = [dict(b, p_label=float(np.exp(logps[b['sequence']][b['step'], seqs[b['sequence']]['labels'][b['step']]])))
                           for b in branches]
    return out
