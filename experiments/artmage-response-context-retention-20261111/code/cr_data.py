"""Context-retention data (pure numpy; no engine, no model).

Same 42 sequences as the response study (rl_data.load(): the response-learning manifest 2281d191..., hash-checked).
The new part is the REFERENCE: for each of the two response sequences, the start model's own legal-normalised
log-probabilities at every own step, computed with teacher forcing along the RECORDED history (the executed actions
move the duel forward; they are NOT used as labels of the retention term). The reference is computed once on the GPU
(the training backend) by cr_train.py reference, stored in reference/start-gpu.npz and fixed by hash before training.

Preserved steps (KL mask): own steps strictly before the response window, message != select_position (adapter-owned),
at least 2 legal options. The window itself is excluded (it is re-taught by the CE), padding is excluded.
"""
import json
from pathlib import Path

import numpy as np

import hb_data as HD
import pilot_data as PD
import rl_data as RD

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-context-retention-20261111')
REF = ROOT / 'reference/start-gpu.npz'
NAMES = RD.NAMES
MASKED = -1e9  # log-probability stored for illegal options (finite: no NaN through where/softmax)


def legal(seq):
    return seq['actions_'][:, :, 3] != 0


def preserve_mask(seq, meta):
    n = len(seq['labels'])
    count = legal(seq).sum(1)
    keep = (np.arange(n) < meta['target_index']) & (seq['message'] != PD.POSITION_MESSAGE) & (count >= 2)
    return keep.astype(np.float32)


def step_meanings(name):
    """Own steps of the source record up to the window: (step, turn, phase, message, actions[index -> meaning])."""
    src = RD.SOURCES[name]
    rec = json.loads((src['dir'] / 'record.json').read_text())
    rows = [r for r in rec['trace'] if r['snapshot']['to_play'] == 0 and r['step'] <= src['step']]
    return [dict(step=r['step'], turn=r['snapshot']['turn_count'], phase=r['snapshot']['phase'], message=r['snapshot']['message'], who=r['who'],
                 executed=r['selected']['index'],
                 actions={a['index']: dict(act=a['act'], code=a['code'], name=a.get('name', ''), effect=a.get('effect'), spec=a.get('spec'))
                          for a in r['snapshot']['actions']}) for r in rows]


def response_pair_batch(responses, length):
    """The response minibatch exactly as the response study built it (rl_data.make_batch): S1, S2 padded to `length`."""
    return PD.batch([responses[n] for n in NAMES], length=length)


def load_reference(responses, metas):
    with np.load(REF) as z:
        ref = {k: z[k] for k in z.files}
    out = {}
    for n in NAMES:
        seq, meta = responses[n], metas[n]
        k = len(seq['labels'])
        logp, lg, keep = ref[f'{n}__logp'], ref[f'{n}__legal'], ref[f'{n}__preserve']
        if logp.shape[0] != k or not np.array_equal(lg, legal(seq)) or not np.array_equal(keep, preserve_mask(seq, meta)):
            raise ValueError(f'{n}: reference does not match the sequence')
        out[n] = dict(logp=logp.astype(np.float32), legal=lg, preserve=keep)
    return out


def add_reference(batch, responses, ref, length):
    """Time-major ref_logp [L, 2, A], legal [L, 2, A], kl_mask [L, 2]; padding repeats the last row with mask 0."""
    t = np.arange(length)
    cols_lp, cols_lg, cols_m = [], [], []
    for n in NAMES:
        k = len(responses[n]['labels'])
        idx = np.minimum(t, k - 1)
        cols_lp.append(ref[n]['logp'][idx])
        cols_lg.append(ref[n]['legal'][idx])
        cols_m.append(ref[n]['preserve'][idx] * (t < k))
    out = dict(batch)
    out.update(ref_logp=np.stack(cols_lp, 1).astype(np.float32), legal=np.stack(cols_lg, 1), kl_mask=np.stack(cols_m, 1).astype(np.float32))
    if not np.array_equal(out['legal'], out['actions_'][..., 3] != 0):
        raise ValueError('reference legal mask differs from the batch observations')
    return out


def make_batch(existing, pairs, responses, ref, item, length):
    kind, v = item
    if kind == 'response':
        assert tuple(v) == tuple(NAMES)
        return add_reference(response_pair_batch(responses, length), responses, ref, length), list(v)
    return HD.make_batch(existing, pairs, item, length)
