"""Training data for the hand-branch study (pure numpy; no engine, no model).

Existing: the 24 sequences of the basic-opening batch (its fixed manifest, files unchanged).
New: paired train demos of this study; a pair enters only if BOTH hands completed their script and their replays
matched every step, the end state and the observations. Validation pairs are JSON only and never listed.
Sampling: blocks of 4 minibatches = 2 existing (each slot: one of the 7 groups uniformly, then a sequence uniformly)
+ 2 new (group A/B uniformly, then a complete pair uniformly; both hands of the pair form the minibatch); the slot
order inside a block is shuffled by the registered seed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import batch_data as BD
import hb_plans as HP
import pilot_data as PD

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
MANIFEST = ROOT / 'data/manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_ok(name):
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
    return (True, 'ok') if ok else (False, 'replay differs')


def load_new(name):
    old = BD.ROOT
    BD.ROOT = ROOT
    try:
        return BD.load_new(name)
    finally:
        BD.ROOT = old


def build_manifest():
    if MANIFEST.exists():
        raise SystemExit('manifest exists; fixed once')
    pairs, excluded = [], []
    for p in HP.PAIRS:
        if p['split'] != 'train':
            continue
        names = [f"{p['pair']}-{v}" for v in p['hands']]
        status = {n: demo_ok(n) for n in names}
        if all(ok for ok, _ in status.values()):
            pairs.append(dict(pair=p['pair'], group=p['group'], seed=p['seed'],
                              hands=[dict(name=n, variant=HP.HANDS[n]['variant'], hand=HP.HANDS[n]['hand'],
                                          path=str(ROOT / 'demos' / n / 'observations.npz'), sha256=sha(ROOT / 'demos' / n / 'observations.npz'),
                                          record_sha256=sha(ROOT / 'demos' / n / 'record.json')) for n in names]))
        else:
            excluded.append(dict(pair=p['pair'], status={n: why for n, (ok, why) in status.items()}))
    out = dict(status='fixed_before_model_results', existing=dict(manifest=str(BD.MANIFEST), sha256=sha(BD.MANIFEST)),
               pairs=pairs, excluded=excluded, groups_with_pairs=sorted({p['group'] for p in pairs}),
               note='Validation pairs are not listed and have no observation arrays.')
    MANIFEST.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    return out


def load():
    manifest = json.loads(MANIFEST.read_text())
    if sha(manifest['existing']['manifest']) != manifest['existing']['sha256']:
        raise ValueError('existing manifest changed')
    existing, names, stats = BD.load()
    pairs = {}
    for p in manifest['pairs']:
        seqs = []
        for h in p['hands']:
            if sha(h['path']) != h['sha256']:
                raise ValueError(f"{h['name']}: checksum differs")
            s = load_new(h['name'])
            n = len(s['labels'])
            legal = s['actions_'][np.arange(n), s['labels'], 3] != 0
            if not legal[s['loss'] > 0].all():
                raise ValueError(f"{h['name']}: a loss label is not legal")
            seqs.append((h['name'], h['variant'], s))
        pairs.setdefault(p['group'], []).append(dict(pair=p['pair'], hands=seqs))
    for g, items in pairs.items():
        all_seqs = [s for p in items for _, _, s in p['hands']]
        stats[f'pairs-{g}'] = dict(pairs=len(items), sequences=len(all_seqs), decisions=int(sum(len(s['labels']) for s in all_seqs)),
                                    loss_steps=int(sum(s['loss'].sum() for s in all_seqs)), max_length=int(max(len(s['labels']) for s in all_seqs)))
    return existing, names, pairs, stats


class BlockSampler:
    def __init__(self, existing, pairs, seed):
        self.existing = BD.GroupSampler(existing, seed + 1)
        self.groups = sorted(pairs)
        self.sizes = {g: len(pairs[g]) for g in self.groups}
        self.rng = np.random.default_rng(seed)
        self.queue, self.drawn = [], []

    def _block(self):
        slots = ['existing', 'existing', 'pair', 'pair']
        for i in self.rng.permutation(4):
            if slots[i] == 'existing':
                self.queue.append(('existing', self.existing.next()))
            else:
                g = self.groups[int(self.rng.integers(0, len(self.groups)))]
                self.queue.append(('pair', (g, int(self.rng.integers(0, self.sizes[g])))))

    def next(self):
        if not self.queue:
            self._block()
        item = self.queue.pop(0)
        self.drawn.append(item)
        return item

    def counts(self):
        out = dict(existing_minibatches=0, pair_minibatches=0, existing_groups={}, pairs={})
        for kind, v in self.drawn:
            if kind == 'existing':
                out['existing_minibatches'] += 1
                for g, _ in v:
                    out['existing_groups'][g] = out['existing_groups'].get(g, 0) + 1
            else:
                out['pair_minibatches'] += 1
                key = f'{v[0]}:{v[1]}'
                out['pairs'][key] = out['pairs'].get(key, 0) + 1
        return out


def make_batch(existing, pairs, item, length):
    kind, v = item
    if kind == 'existing':
        items = [existing[g][i] for g, i in v]
        labels = [f'{g}:{i}' for g, i in v]
    else:
        p = pairs[v[0]][v[1]]
        items = [s for _, _, s in p['hands']]
        labels = [n for n, _, _ in p['hands']]
    return PD.batch(items, length=length), labels
