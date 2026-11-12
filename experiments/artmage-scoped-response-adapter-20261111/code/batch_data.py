"""Training data for the basic-opening batch (pure numpy; no engine, no model).

Groups F0-F6. F0 = old train (12 sequences, unchanged file); F1 = C1 + new F1 demos; F3 = C2 + new F3 demos; other
groups = new demos only. A new demo enters only if its record completed its script (opponent-turn entries may stay
unexecuted) and its replay matched every step, the end state and the observations. Validation hands live in
validation/ as JSON only; the loader reads nothing but the files listed in data/manifest.json.

Masks (same rule as the pilot): context = every player-0 decision; loss = own-turn decisions except select_position
(adapter-owned), plus scripted opponent-turn steps (kind != default-pass) except select_position.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import batch_plans as BP
import pilot_data as PD

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
MANIFEST = ROOT / 'data/manifest.json'
GROUPS = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_status(name):
    d = ROOT / 'demos' / name
    if not (d / 'record.json').exists():
        return False, 'not recorded'
    rec = json.loads((d / 'record.json').read_text())
    if not rec.get('script_completed') or rec.get('failure'):
        return False, f"script not completed: {rec.get('failure') or rec.get('remaining_required')}"
    if not (d / 'replay.json').exists():
        return False, 'not replayed'
    rep = json.loads((d / 'replay.json').read_text())
    ok = (not rep['mismatch'] and not rep['failure'] and rep['reproduced_steps'] == rep['of'] and rep['final_equal'] is not False
          and rep['checkpoints_equal'] and rep['observations_match'])
    return (True, 'ok') if ok else (False, f"replay differs: {json.dumps({k: rep[k] for k in ['mismatch', 'failure', 'final_equal', 'checkpoints_equal', 'observations_match']})}")


def load_new(name):
    d = ROOT / 'demos' / name
    record = json.loads((d / 'record.json').read_text())
    with np.load(d / 'observations.npz') as z:
        arrays = {k: z[k] for k in z.files}
    rows = {r['step']: r for r in record['trace'] if r['snapshot']['to_play'] == 0}
    steps = arrays['steps'].tolist()
    if sorted(rows) != steps:
        raise ValueError(f'{name}: observation steps differ from the record')
    msg = PD.message_types(arrays['actions_'])
    loss = np.zeros(len(steps), np.float32)
    for i, step in enumerate(steps):
        row = rows[step]
        if (msg[i] == PD.POSITION_MESSAGE) != (row['snapshot']['message'] == 'select_position'):
            raise ValueError(f'{name}: message type mismatch at {step}')
        if row['selected']['index'] != int(arrays['labels'][i]):
            raise ValueError(f'{name}: label mismatch at {step}')
        own = row['snapshot']['turn_count'] == 1
        scripted_opp = row['snapshot']['turn_count'] == 2 and row['why']['kind'] not in ('default-pass', 'opponent-passive')
        if (own or scripted_opp) and msg[i] != PD.POSITION_MESSAGE:
            loss[i] = 1.0
    seq = {k: arrays[k] for k in PD.OBS_KEYS}
    seq.update(chain=arrays['chain'].astype(np.int32), labels=arrays['labels'].astype(np.int32), loss=loss, message=msg)
    return seq


def build_manifest():
    """Fix the training data before any model result of this batch is seen. Writes data/manifest.json once."""
    if MANIFEST.exists():
        raise SystemExit('manifest exists; fixed once')
    groups = {g: [] for g in GROUPS}
    excluded = []
    groups['F0'].append(dict(name='old-train', kind='old', path=str(PD.OLD), sha256=PD.OLD_SHA, sequences=12))
    groups['F1'].append(dict(name='C1', kind='demo-pilot', path=str(PD.DEMO_DIR / 'C1/observations.npz'), sha256=PD.DEMO_SHA['C1']['observations'], sequences=1))
    groups['F3'].append(dict(name='C2', kind='demo-pilot', path=str(PD.DEMO_DIR / 'C2/observations.npz'), sha256=PD.DEMO_SHA['C2']['observations'], sequences=1))
    for name, spec in BP.TRAIN.items():
        ok, why = demo_status(name)
        if ok:
            path = ROOT / 'demos' / name / 'observations.npz'
            groups[spec['condition']].append(dict(name=name, kind='demo-batch', path=str(path), sha256=sha(path),
                                                  record_sha256=sha(ROOT / 'demos' / name / 'record.json'), sequences=1,
                                                  hand=spec['hand'], seed=spec['seed']))
        else:
            excluded.append(dict(name=name, condition=spec['condition'], reason=why))
    for entries in groups.values():
        for e in entries:
            assert '/validation/' not in e['path']
    out = dict(status='fixed_before_model_results', groups=groups, excluded=excluded,
               available_groups=[g for g in GROUPS if groups[g]], missing_groups=[g for g in GROUPS if not groups[g]],
               note='Validation hands are not in this manifest and have no observation arrays.')
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    return out


def load():
    """Sequences per group from the fixed manifest (hash-checked). Returns ({group: [seq]}, {group: [name]}, stats)."""
    manifest = json.loads(MANIFEST.read_text())
    seqs, names = {}, {}
    for g, entries in manifest['groups'].items():
        for e in entries:
            if sha(e['path']) != e['sha256']:
                raise ValueError(f"{e['name']}: checksum differs")
            if e['kind'] == 'old':
                items = PD.load_old()
                labels = [f'old-{i}' for i in range(len(items))]
            elif e['kind'] == 'demo-pilot':
                items, labels = [PD.load_demo(e['name'])[0]], [e['name']]
            else:
                items, labels = [load_new(e['name'])], [e['name']]
            seqs.setdefault(g, []).extend(items)
            names.setdefault(g, []).extend(labels)
    stats = {}
    for g, items in seqs.items():
        for s in items:
            n = len(s['labels'])
            legal = s['actions_'][np.arange(n), s['labels'], 3] != 0
            if not legal[s['loss'] > 0].all():
                raise ValueError(f'{g}: a loss label is not a legal action')
        msgs = np.concatenate([s['message'] for s in items])
        loss = np.concatenate([s['loss'] for s in items]) > 0
        stats[g] = dict(sequences=len(items), decisions=int(sum(len(s['labels']) for s in items)), loss_steps=int(loss.sum()),
                        context_only_steps=int((~loss).sum()), position_steps=int((msgs == PD.POSITION_MESSAGE).sum()),
                        max_length=int(max(len(s['labels']) for s in items)))
    return seqs, names, stats


class GroupSampler:
    """Each sequence slot: a group uniformly among the available groups, then a sequence uniformly within it."""

    def __init__(self, seqs, seed, per_batch=2):
        self.groups = sorted(seqs)
        self.sizes = {g: len(seqs[g]) for g in self.groups}
        self.rng = np.random.default_rng(seed)
        self.per_batch = per_batch
        self.drawn = []

    def next(self):
        picked = []
        for _ in range(self.per_batch):
            g = self.groups[int(self.rng.integers(0, len(self.groups)))]
            picked.append((g, int(self.rng.integers(0, self.sizes[g]))))
        self.drawn.extend(picked)
        return picked

    def counts(self):
        out = {g: 0 for g in self.groups}
        for g, _ in self.drawn:
            out[g] += 1
        return out


def make_batch(seqs, picked, length):
    b = PD.batch([seqs[g][i] for g, i in picked], length=length)
    return b
