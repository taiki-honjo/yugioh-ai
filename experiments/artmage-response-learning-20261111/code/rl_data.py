"""Training data for the response study (pure numpy; no engine, no model).

Old: the hand-branch manifest (24 basic-batch sequences + 16 paired sequences), files unchanged and hash-checked
through hb_data.load(); its masks and internal sampling rules are unchanged.
New: 2 response sequences. Each one is the recorded own observations from the start of the duel up to and including
the first own response window to the opponent's Medius① (context), with the loss mask 1 ONLY on that window's
recorded action and 0 elsewhere; nothing after the window enters the data. The RNN unrolls every sequence from zero.
  response-S1: probe S1-D0R1 (BV1-1cost-VARNISH-OM path; ディアクトロス② executed by the registered R1 rule), step 149
  response-S2: comparison BV1-2cost-ASSAULT-OM (the model's own ファインメルト③), step 169
These are demonstrations verified in two positions, not labels saying other legal responses are wrong. The two
BV1 hands were validation hands before; from this study on they are training / known diagnostic cases.
Sampling: blocks of 8 minibatches = 3 old GroupSampler + 3 old pair + 2 response (S1 and S2 together); the slot
order inside a block is shuffled by the registered seed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import batch_data as BD
import hb_data as HD
import pilot_data as PD

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-learning-20261111')
MANIFEST = ROOT / 'data/manifest.json'
P = Path('/home/hatk8/projects/yugioh-ai')
VOCAB = P / 'experiments/meta-opponents-20261107/run-20261107-224631/code_list.txt'
PROBE = P / 'experiments/artmage-opponent-turn-probe-20261111'
FLC = P / 'experiments/artmage-functional-landing-compare-20261110'
MEDIUS, FINEMELT, DIAKTOROS = 97556336, 34541940, 27184601
MODEL_SHA = ['b9c7751f5c1f7179faa4911e1a52e8603e332fd9700a0f301dd4e759f8208f66', '77191d5d9a9aaa65d63902c93e47c0e3274009e7a91ee1e8e8abb48fcd73c6a2']
SOURCES = {
    'response-S1': dict(dir=PROBE / 'runs/S1-D0R1', step=149, code=DIAKTOROS, effect=10011, effect_label='ディアクトロス②', site='S1',
                        origin='opponent-turn probe S1-D0R1: BV1-1cost-VARNISH-OM path, the registered R1 rule executed ディアクトロス② (the model chose pass)'),
    'response-S2': dict(dir=FLC / 'runs/BV1-2cost-ASSAULT-OM', step=169, code=FINEMELT, effect=10011, effect_label='ファインメルト③', site='S2',
                        origin='functional comparison BV1-2cost-ASSAULT-OM: the model itself chose ファインメルト③'),
}
NAMES = sorted(SOURCES)
SLOTS = ['existing'] * 3 + ['pair'] * 3 + ['response'] * 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_medius_window(s):
    """Own select_chain in the opponent's main phase 1 of turn 2 with exactly player 1's Medius① on the chain (otp_runs copy)."""
    ch = s['public_chain']
    return (s['to_play'] == 0 and s['turn_count'] == 2 and s['turn_player'] == 1 and s['phase'] == 4 and s['message'] == 'select_chain'
            and len(ch) == 1 and ch[0]['player'] == 1 and ch[0]['code'] == MEDIUS and ch[0]['description'] == MEDIUS * 16)


def build_response(name):
    """Returns (sequence, meta). Every check failing raises."""
    src = SOURCES[name]
    d = src['dir']
    rec = json.loads((d / 'record.json').read_text())
    rep = json.loads((d / 'replay.json').read_text())
    if not (rep['mismatch'] is None and rep['failure'] is None and rep['checkpoints_equal'] and rep['game_over_equal'] and rep['observations_match']
            and rep['final_equal'] in (True, None) and rep['reproduced_steps'] == rep['of']):
        raise ValueError(f'{name}: source replay did not match')
    if list(rec['model_sha256']) != MODEL_SHA:
        raise ValueError(f'{name}: source record was not made by the start model')
    with np.load(d / 'observations.npz') as z:
        arrays = {k: z[k] for k in z.files}
    steps = arrays['steps'].tolist()
    own = sorted(r['step'] for r in rec['trace'] if r['snapshot']['to_play'] == 0)
    if steps != own:
        raise ValueError(f'{name}: observation steps differ from the own decisions of the record')
    windows = [r['step'] for r in rec['trace'] if is_medius_window(r['snapshot'])]
    if not windows or windows[0] != src['step']:
        raise ValueError(f"{name}: first Medius① window is {windows[:1]}, registered {src['step']}")
    i = steps.index(src['step'])
    row = next(r for r in rec['trace'] if r['step'] == src['step'])
    sel = row['selected']
    if (sel['act'], sel['code'], sel.get('effect')) != ('Activate', src['code'], src['effect']):
        raise ValueError(f'{name}: recorded action is not the registered response')
    if int(arrays['labels'][i]) != sel['index']:
        raise ValueError(f'{name}: label index differs from the record')
    if arrays['actions_'][i, sel['index'], 3] == 0:
        raise ValueError(f'{name}: label is not a legal option in the observation')
    codes = [int(line.split()[0]) for line in VOCAB.read_text().splitlines()]
    chain = arrays['chain'][i]  # [length, owner (0 none / 1 own / 2 opponent), card vocabulary id, effect]
    if not (int(chain[0]) == 1 and int(chain[1]) == 2 and codes[int(chain[2]) - 1] == MEDIUS):
        raise ValueError(f'{name}: observed chain is not one opponent Medius link: {chain.tolist()}')
    n = i + 1
    seq = {k: arrays[k][:n].copy() for k in PD.OBS_KEYS}
    loss = np.zeros(n, np.float32)
    loss[i] = 1.0
    seq.update(chain=arrays['chain'][:n].astype(np.int32), labels=arrays['labels'][:n].astype(np.int32), loss=loss,
               message=PD.message_types(arrays['actions_'][:n]))
    meta = dict(name=name, site=src['site'], target_index=i, target_step=src['step'], context_steps=n,
                label=dict(index=sel['index'], code=src['code'], effect=src['effect'], name=src['effect_label']),
                options=[dict(index=a['index'], act=a['act'], code=a['code'], effect=a.get('effect'), spec=a.get('spec')) for a in row['snapshot']['actions']],
                cancel_index=next((a['index'] for a in row['snapshot']['actions'] if a['act'] == 'Cancel'), None),
                observed_chain=chain.tolist())
    return seq, meta


def build_manifest():
    if MANIFEST.exists():
        raise SystemExit('manifest exists; fixed once')
    entries = []
    for name in NAMES:
        _, meta = build_response(name)
        d = SOURCES[name]['dir']
        entries.append(dict(meta, origin=SOURCES[name]['origin'], record=str(d / 'record.json'), record_sha256=sha(d / 'record.json'),
                            observations=str(d / 'observations.npz'), observations_sha256=sha(d / 'observations.npz'),
                            replay=str(d / 'replay.json'), replay_sha256=sha(d / 'replay.json')))
    out = dict(status='fixed_before_model_results', old=dict(manifest=str(HD.MANIFEST), sha256=sha(HD.MANIFEST),
                                                             basic_manifest=str(BD.MANIFEST), basic_sha256=sha(BD.MANIFEST)),
               responses=entries,
               mask='context = every own observation from the duel start to the window (inclusive); loss = 1 only on the window action',
               promotion='The BV1-B-1cost / BV1-B-2cost hands were validation hands of the hand-branch study. From this study on, the two '
                         'response histories are TRAINING data and the two hands are known diagnostic cases; success on them is not '
                         'generalisation to unseen hands. Earlier splits and registrations are not rewritten.',
               not_a_rule='No rule "always negate" and no Finemelt-over-Diaktoros order is taught; each label is the action verified in its own position.')
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    return out


def load():
    """(existing, names, pairs, responses, metas, stats); every file hash-checked against the fixed manifest."""
    manifest = json.loads(MANIFEST.read_text())
    if sha(manifest['old']['manifest']) != manifest['old']['sha256'] or sha(manifest['old']['basic_manifest']) != manifest['old']['basic_sha256']:
        raise ValueError('old manifest changed')
    existing, names, pairs, stats = HD.load()
    responses, metas = {}, {}
    for e in manifest['responses']:
        for k in ['record', 'observations', 'replay']:
            if sha(e[k]) != e[f'{k}_sha256']:
                raise ValueError(f"{e['name']}: {k} checksum differs")
        seq, meta = build_response(e['name'])
        responses[e['name']], metas[e['name']] = seq, meta
    stats['responses'] = {n: dict(context_steps=m['context_steps'], loss_steps=int(responses[n]['loss'].sum()), target_index=m['target_index'])
                          for n, m in metas.items()}
    return existing, names, pairs, responses, metas, stats


class BlockSampler8:
    """Blocks of 8 minibatches: 3 old GroupSampler (its own seed + 1 and rules) + 3 old pair (group A/B uniformly, then a
    pair uniformly; both hands of the pair) + 2 response (S1 and S2 together); slot order shuffled per block."""

    def __init__(self, existing, pairs, seed):
        self.existing = BD.GroupSampler(existing, seed + 1)
        self.groups = sorted(pairs)
        self.sizes = {g: len(pairs[g]) for g in self.groups}
        self.rng = np.random.default_rng(seed)
        self.queue, self.drawn = [], []

    def _block(self):
        for i in self.rng.permutation(len(SLOTS)):
            if SLOTS[i] == 'existing':
                self.queue.append(('existing', self.existing.next()))
            elif SLOTS[i] == 'pair':
                g = self.groups[int(self.rng.integers(0, len(self.groups)))]
                self.queue.append(('pair', (g, int(self.rng.integers(0, self.sizes[g])))))
            else:
                self.queue.append(('response', tuple(NAMES)))

    def next(self):
        if not self.queue:
            self._block()
        item = self.queue.pop(0)
        self.drawn.append(item)
        return item

    def counts(self):
        out = dict(existing_minibatches=0, pair_minibatches=0, response_minibatches=0, existing_groups={}, pairs={}, responses={},
                   sequence_presentations=dict(old=0, new=0))
        for kind, v in self.drawn:
            if kind == 'existing':
                out['existing_minibatches'] += 1
                out['sequence_presentations']['old'] += len(v)
                for g, _ in v:
                    out['existing_groups'][g] = out['existing_groups'].get(g, 0) + 1
            elif kind == 'pair':
                out['pair_minibatches'] += 1
                out['sequence_presentations']['old'] += 2
                key = f'{v[0]}:{v[1]}'
                out['pairs'][key] = out['pairs'].get(key, 0) + 1
            else:
                out['response_minibatches'] += 1
                out['sequence_presentations']['new'] += len(v)
                for n in v:
                    out['responses'][n] = out['responses'].get(n, 0) + 1
        return out


def make_batch(existing, pairs, responses, item, length):
    kind, v = item
    if kind == 'response':
        return PD.batch([responses[n] for n in v], length=length), list(v)
    return HD.make_batch(existing, pairs, item, length)
