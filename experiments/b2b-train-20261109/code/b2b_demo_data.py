"""Build the train-only B2B demonstration data (separate file + manifest; the old 416-sequence data is untouched).

Sources: experiments/b2b-demo-20261109 runs DM-A (engine run 1, replay 4) and DM-B (engine run 3, replay 5); the
crashed run 2 is not used. Each demonstration is one sequence of player-0 decisions (own turn 1 from the landing
record, then turns 2-3). Two masks: every recorded decision is a valid step (RNN history); only the labelled B2B /
attack decisions of turns 2-3 carry loss. Passes, phase changes and adapter-owned prompts stay context.
Acceptable sets: the demonstrated action only, except the B2B (3) target prompt where Cue and Zaren are both
accepted IF the two demonstrations reach that prompt with identical observations and identical history."""
import json
from pathlib import Path
import sys

import numpy as np

P = Path('/home/hatk8/projects/yugioh-ai')
DEMO = P / 'experiments/b2b-demo-20261109'
OUT = P / 'experiments/b2b-train-20261109/data'
sys.path.insert(0, str(P / 'scripts'))
import continuous_learning as C  # noqa: E402

DEMOS = [('DM-A', 1), ('DM-B', 2)]
OBS = ('cards_', 'global_', 'actions_', 'h_actions_', 'chain')
LABELLED = {'b2b', 'b2b-pick', 'b2b-option-special-summon', 'b2b-synchro-yes', 'b2b-synchro-pick', 'track-search',
            'track-search-pick', 'attack-monster', 'attack-direct', 'attack-pick'}
CUE, ZAREN = 16387555, 4891376


def load(name):
    record = json.loads((DEMO / 'conditions' / f'{name}.json').read_text())
    replay = json.loads((DEMO / 'replays' / f'{name}.json').read_text())
    assert record['failure'] is None and record['mismatch'] is None and record['lua_errors'] == 0, name
    assert replay['replayed_engine_run'] == record['engine_run'] and replay['observations_match'] and replay['end_matches'] \
        and replay['mismatch'] is None and replay['reproduced_steps'] == replay['of'], name
    obs_path = DEMO / 'observations' / f'{name}.npz'
    assert C.sha(obs_path) == record['observations']['sha256'], name
    with np.load(obs_path) as z:
        obs = {k: z[k] for k in z.files}
    rows = {r['step']: r for r in record['trace']}
    ours = [r for r in record['trace'] if r['snapshot']['to_play'] == 0]
    assert [r['step'] for r in ours] == obs['step'].tolist(), name
    assert [r['selected']['index'] for r in ours] == obs['label'].tolist(), name
    return record, replay, obs, rows, obs_path


def main():
    loaded = {name: load(name) for name, _ in DEMOS}
    # the B2B (3) target prompt: same step in both, identical observation and identical history?
    target_step = {name: next(r['step'] for r in loaded[name][0]['trace'] if r['source'] == 'b2b-pick') for name, _ in DEMOS}
    assert len(set(target_step.values())) == 1
    step = target_step['DM-A']
    pos = {name: int(np.where(loaded[name][2]['step'] == step)[0][0]) for name, _ in DEMOS}
    shared = pos['DM-A'] == pos['DM-B'] and all(
        np.array_equal(loaded['DM-A'][2][k][:pos['DM-A'] + 1], loaded['DM-B'][2][k][:pos['DM-B'] + 1]) for k in OBS)
    snapshot = loaded['DM-A'][3][step]['snapshot']
    target_options = {a['code']: a['index'] for a in snapshot['actions'] if a['code'] in (CUE, ZAREN)}
    assert shared and set(target_options) == {CUE, ZAREN}, 'target prompt not identical; cannot share an acceptable set'

    T = max(len(loaded[name][2]['step']) for name, _ in DEMOS)
    n = len(DEMOS)
    arrays = {k: np.zeros((n, T) + loaded['DM-A'][2][k].shape[1:], loaded['DM-A'][2][k].dtype) for k in OBS}
    arrays['chain'] = arrays['chain'].astype(np.int64)
    arrays.update(targets=np.zeros((n, T, 128), np.uint8), labels=np.zeros((n, T), np.int32), weights=np.zeros((n, T), np.float32),
                  loss_weights=np.zeros((n, T), np.float32), lengths=np.zeros(n, np.int32), source=np.zeros(n, np.int32))
    manifest_demos = []
    for i, (name, source) in enumerate(DEMOS):
        record, replay, obs, rows, obs_path = loaded[name]
        L = len(obs['step'])
        for k in OBS:
            arrays[k][i, :L] = obs[k]
        arrays['labels'][i, :L] = obs['label']
        arrays['weights'][i, :L] = 1.0
        arrays['lengths'][i], arrays['source'][i] = L, source
        labelled = []
        for t, s in enumerate(obs['step'].tolist()):
            row = rows[s]
            msg = int(obs['actions_'][t, 0, 3])
            if row['source'] not in LABELLED or row['snapshot']['turn_count'] < 2 or msg in (5, 11):
                continue
            label = int(obs['label'][t])
            accept = [label]
            reason = 'demonstrated action only'
            if s == step:
                accept = sorted(target_options.values())
                reason = 'B2B (3) target: Cue and Zaren both verified from this identical prompt (DM-A / DM-B share observations and history)'
            arrays['loss_weights'][i, t] = 1.0
            arrays['targets'][i, t, accept] = 1
            names = {a['index']: a.get('name') for a in row['snapshot']['actions']}
            labelled.append(dict(position=t, step=s, turn=row['snapshot']['turn_count'], message=row['snapshot']['message'],
                                 kind=row['source'], demonstrated=dict(index=label, act=row['selected']['act'], code=row['selected']['code'],
                                                                      name=names.get(label), effect=row['selected'].get('effect')),
                                 acceptable=[dict(index=a, code=row['snapshot']['actions'][a]['code'], name=names.get(a)) for a in accept],
                                 reason=reason))
        own_turn = sum(1 for s in obs['step'].tolist() if rows[s]['snapshot']['turn_count'] == 1)
        manifest_demos.append(dict(name=name, source_id=source, case_id=record['case_id'], hand=json.loads((P / 'experiments/b2b-zaren-ownturn-check-20261109/records/via-zaren.json').read_text())['hand'],
                                   seed=record['seed'], decisions=L, own_turn_context_steps=own_turn, loss_steps=len(labelled),
                                   context_steps=L - len(labelled), labelled=labelled,
                                   record=dict(path=str(DEMO / 'conditions' / f'{name}.json'), engine_run=record['engine_run'],
                                               sha256=C.sha(DEMO / 'conditions' / f'{name}.json')),
                                   replay=dict(path=str(DEMO / 'replays' / f'{name}.json'), engine_run=replay['engine_run'],
                                               sha256=C.sha(DEMO / 'replays' / f'{name}.json'), observations_match=True),
                                   observations=dict(path=str(obs_path), sha256=C.sha(obs_path)),
                                   game_over=record.get('game_over')))
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / 'b2b-demos.npz'
    np.savez_compressed(npz, **arrays)
    manifest = dict(created_jst=C.now(), split='train', npz_sha256=C.sha(npz), demos=manifest_demos,
                    excluded=['experiments/b2b-demo-20261109 engine run 2 (crashed; not a demonstration)'],
                    masks='weights = every recorded player-0 decision (history unrolled through the RNN); loss_weights = labelled B2B/attack decisions of turns 2-3 only',
                    not_labelled='own turn 1 (the same train hand has the P1 route in the old data), passes, phase changes (enter/leave battle), placement/position prompts',
                    acceptable_sets='demonstrated action only, except the shared B2B (3) target prompt {Cue, Zaren}; identical card copies were NOT added',
                    trackmaker_search='Kewl Tune Synchro only (verified); other search targets are not added',
                    level3_tuner_note='the demonstrations use the Ash Blossom version of the own-turn route; the Reco version is not verified',
                    validation_or_holdout_used=False, legacy_gate_flag_used=False, old_train_npz_modified=False,
                    builder_sha256=C.sha(Path(__file__)))
    C.atomic_json(OUT / 'b2b-demos-manifest.json', manifest)
    print(json.dumps(dict(npz_sha256=manifest['npz_sha256'], manifest_sha256=C.sha(OUT / 'b2b-demos-manifest.json'),
                          demos=[dict(name=d['name'], decisions=d['decisions'], loss_steps=d['loss_steps'], context=d['context_steps'],
                                      kinds=[x['kind'] for x in d['labelled']]) for d in manifest_demos]), ensure_ascii=False))


if __name__ == '__main__':
    main()
