"""Rehearsal data for the Artmage learning pilot (pure numpy; no engine, no model).

Sources (TRAIN only; validation/holdout are never read here):
  old  experiments/opponent-article-training-20261108/.../dataset/Artmage/train.npz  (12 sequences, 1,077 decisions)
  C1   experiments/artmage-parallel-20261110/demos/C1/observations.npz + record.json (襲撃＋罠コスト1枚)
  C2   experiments/artmage-parallel-20261110/demos/C2/observations.npz + record.json (学都＋魔法罠コスト1枚)

Context mask = every real player-0 decision (it goes through the RNN). Loss mask:
  old: every real decision except select_position (message type 5, owned by the position adapter)
  C1/C2: own-turn (turn 1) decisions except select_position, plus the scripted opponent-turn steps
         (C1: 獄神使② → W:P Fancy Ball); unscripted opponent-turn passes stay context only.
select_place (11) stays in the loss: no adapter owns it for Artmage (the isolated entry's neural policy decides it).
Single demonstrated target per step, so the acceptable-action CE is the CE of the executed label.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path('/home/hatk8/projects/yugioh-ai')
OLD = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/dataset/Artmage/train.npz'
OLD_SHA = '61bc7cb52209c4c15b97e952694dd4a748752f59ea1734f99f4dd5ed197f3a14'
DEMO_DIR = P / 'experiments/artmage-parallel-20261110/demos'
DEMO_SHA = {'C1': dict(observations='8425bd1cf43608f417c0fca9bdf2f00fa7fe3cd95463e31460ab622307cd935d',
                       record='41fe4c4fc1c0e956ac983202676afeb23b5cdef4413c54c0d699d41b63776d84'),
            'C2': dict(observations='41be0b84d1d45803b3afe66d1c279c6f5764fed6feb6a96c0837ddc9e2dceff4',
                       record='7b6b5e633fef3662e1077390e6340eeb6cb97405de6e65307101158e87bdee0d')}
OBS_KEYS = ('cards_', 'global_', 'actions_', 'h_actions_')
STEP_KEYS = OBS_KEYS + ('chain',)
POSITION_MESSAGE = 5
SOURCES = ('old', 'C1', 'C2')
LENGTH = 113  # longest sequence (C1); every rehearsal batch uses this fixed length


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def message_types(actions):
    """Message type of each step = feature 3 of the first action row (same as the production rehearsal)."""
    return np.asarray(actions)[..., 0, 3].astype(np.int64)


def load_old():
    if sha(OLD) != OLD_SHA:
        raise ValueError('old train.npz checksum differs')
    with np.load(OLD) as z:
        arrays = {k: z[k] for k in z.files}
    lengths = arrays['lengths'].astype(np.int64)
    weights = arrays['weights']
    n, T = weights.shape
    real = np.arange(T)[None, :] < lengths[:, None]
    if not np.array_equal(weights > 0, real):
        raise ValueError('old weights do not match lengths')
    seqs = []
    for i in range(n):
        L = int(lengths[i])
        msg = message_types(arrays['actions_'][i, :L])
        seq = {k: arrays[k][i, :L] for k in OBS_KEYS}
        seq.update(chain=arrays['chain'][i, :L].astype(np.int32), labels=arrays['labels'][i, :L].astype(np.int32),
                   loss=(msg != POSITION_MESSAGE).astype(np.float32), message=msg)
        seqs.append(seq)
    return seqs


def load_demo(cond):
    obs_path, rec_path = DEMO_DIR / cond / 'observations.npz', DEMO_DIR / cond / 'record.json'
    if sha(obs_path) != DEMO_SHA[cond]['observations']:
        raise ValueError(f'{cond} observations checksum differs')
    if sha(rec_path) != DEMO_SHA[cond]['record']:
        raise ValueError(f'{cond} record checksum differs')
    record = json.loads(rec_path.read_text())
    if not record['script_completed'] or record['failure']:
        raise ValueError(f'{cond} demo is not a completed script')
    with np.load(obs_path) as z:
        arrays = {k: z[k] for k in z.files}
    rows = {r['step']: r for r in record['trace'] if r['snapshot']['to_play'] == 0}
    steps = arrays['steps'].tolist()
    if sorted(rows) != steps:
        raise ValueError(f'{cond} observation steps differ from the record')
    msg = message_types(arrays['actions_'])
    loss = np.zeros(len(steps), np.float32)
    for i, step in enumerate(steps):
        row = rows[step]
        if (msg[i] == POSITION_MESSAGE) != (row['snapshot']['message'] == 'select_position'):
            raise ValueError(f'{cond} message type mismatch at step {step}')
        if row['selected']['index'] != int(arrays['labels'][i]):
            raise ValueError(f'{cond} label mismatch at step {step}')
        own_turn = row['snapshot']['turn_count'] == 1
        scripted_opponent_turn = row['snapshot']['turn_count'] == 2 and row['why']['kind'] != 'default-pass'
        if (own_turn and msg[i] != POSITION_MESSAGE) or (scripted_opponent_turn and msg[i] != POSITION_MESSAGE):
            loss[i] = 1.0
    seq = {k: arrays[k] for k in OBS_KEYS}
    seq.update(chain=arrays['chain'].astype(np.int32), labels=arrays['labels'].astype(np.int32), loss=loss, message=msg)
    return seq, record


def build():
    old = load_old()
    c1, r1 = load_demo('C1')
    c2, r2 = load_demo('C2')
    seqs = {'old': old, 'C1': [c1], 'C2': [c2]}
    for name, items in seqs.items():
        for s in items:
            n = len(s['labels'])
            if n > LENGTH:
                raise ValueError('sequence longer than the fixed rehearsal length')
            legal = s['actions_'][np.arange(n), s['labels'], 3] != 0
            if not legal[s['loss'] > 0].all():
                raise ValueError(f'{name}: a loss label is not a legal action')
    stats = {}
    for name, items in seqs.items():
        msgs = np.concatenate([s['message'] for s in items])
        loss = np.concatenate([s['loss'] for s in items]) > 0
        stats[name] = dict(sequences=len(items), decisions=int(sum(len(s['labels']) for s in items)), loss_steps=int(loss.sum()),
                           context_only_steps=int((~loss).sum()), position_steps_excluded=int((msgs == POSITION_MESSAGE).sum()),
                           messages_in_loss={str(m): int(((msgs == m) & loss).sum()) for m in np.unique(msgs[loss])})
    return seqs, stats


def batch(items, length=LENGTH):
    """Time-major [length, B, ...]; padding repeats the last real step and carries zero loss/context weight."""
    out = {}
    n = [len(s['labels']) for s in items]
    t = np.arange(length)
    for k in STEP_KEYS + ('labels', 'loss'):
        cols = []
        for s, L in zip(items, n):
            idx = np.minimum(t, L - 1)
            col = s[k][idx]
            if k == 'loss':
                col = col * (t < L)
            cols.append(col)
        out[k] = np.stack(cols, axis=1)
    out['real'] = np.stack([(t < L).astype(np.float32) for L in n], axis=1)
    return out


class Schedule:
    """Blocks of 16 sequence slots: 14 old / 1 C1 / 1 C2, slot order shuffled by a fixed seed; old slots draw a
    train sequence uniformly with replacement. Two consecutive slots form one minibatch."""

    def __init__(self, seed, per_block=(14, 1, 1), per_minibatch=2):
        self.rng = np.random.default_rng(seed)
        self.per_block, self.per_minibatch = per_block, per_minibatch
        self.queue, self.drawn = [], []

    def _block(self):
        slots = ['old'] * self.per_block[0] + ['C1'] * self.per_block[1] + ['C2'] * self.per_block[2]
        order = self.rng.permutation(len(slots))
        for i in order:
            src = slots[i]
            self.queue.append((src, int(self.rng.integers(0, 12)) if src == 'old' else 0))

    def next(self):
        while len(self.queue) < self.per_minibatch:
            self._block()
        picked, self.queue = self.queue[:self.per_minibatch], self.queue[self.per_minibatch:]
        self.drawn.extend(picked)
        return picked

    def counts(self):
        out = {s: 0 for s in SOURCES}
        for src, _ in self.drawn:
            out[src] += 1
        return out


def rehearsal_batch(seqs, picked):
    items = [seqs[src][i] for src, i in picked]
    b = batch(items)
    b['source'] = np.array([SOURCES.index(src) for src, _ in picked], np.int32)
    return b


if __name__ == '__main__':
    _, stats = build()
    print(json.dumps(stats, ensure_ascii=False, indent=1))
