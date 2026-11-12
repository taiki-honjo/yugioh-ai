"""Offline summary of the Artmage first stage (no engine): summary.json and data/train-manifest-draft.json."""
import collections
import hashlib
import json
from pathlib import Path
import sys

P = Path('/home/hatk8/projects/yugioh-ai')
W = P / 'experiments/artmage-parallel-20261110'
sys.path.insert(0, str(W / 'code'))
import artmage_teacher as T  # noqa: E402

OLD = P / 'experiments/opponent-article-training-20261108/run-20261108-145203/dataset/Artmage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def subset(expected, got):
    missing = collections.Counter(expected) - collections.Counter(got)
    return dict(expected=expected, present=not missing, missing=sorted(missing.elements()))


def demo(cond):
    rec = read(W / 'demos' / cond / 'record.json')
    rep = read(W / 'demos' / cond / 'replay.json')
    ref = T.CONDITIONS[cond]['reference_end']
    end, nxt = rec['checkpoints']['own_turn_end'], rec['checkpoints'].get('next_own_turn_start')
    rows = [r for r in rec['trace'] if r['snapshot']['to_play'] == 0]
    own = [r for r in rows if r['snapshot']['turn_count'] == 1]
    opp = [r for r in rows if r['snapshot']['turn_count'] == 2]
    scripted_opp = [r['step'] for r in opp if r['why']['kind'] != 'default-pass']
    offered = collections.Counter()
    for o in rec['opponent_turn_offers']:
        for x in o['offered']:
            if x['act'] == 'Activate':
                offered[(x['code'], x['name'], x['effect_index'])] += 1
    loss = [i for i, r in enumerate(rows) if (r['snapshot']['turn_count'] == 1 and r['snapshot']['message'] != 'select_position') or r['step'] in scripted_opp]
    return dict(
        condition=cond, label=rec['label'], reference=rec['reference'], hand=rec['hand'], seed=rec['seed'], engine_run=rec['engine_run'],
        script_completed=rec['script_completed'], failure=rec['failure'], reason=rec['reason'], lua_errors=rec['lua_errors'],
        own_turn_end=end, next_own_turn_start=nxt,
        reference_check=dict(monsters=subset(ref['own_turn_monsters'], end['monsters_face_up']),
                             field=ref['field'] in end['spell_trap_face_up'],
                             set_spell_trap=subset(ref['set_spell_trap'], [c for c in end['spell_trap'] if c not in end['spell_trap_face_up']]),
                             face_up_spell_trap=subset(ref.get('face_up_spell_trap', []), end['spell_trap_face_up']),
                             opponent_turn_reference=ref['opponent_turn']),
        fainmelt_draw=sorted((collections.Counter(end['hand']) + collections.Counter()) - collections.Counter({int(k): v for k, v in rec['marks']['before_draw'].items()})) if cond == 'C2' else None,
        medius_cost=[r['selected']['name'] for r in own if r['why']['kind'] == 'rule'],
        opponent_turn=dict(scripted_steps=scripted_opp, offered_activations=[dict(code=k[0], name=k[1], effect_index=k[2], windows=n) for k, n in sorted(offered.items())]),
        quirks=dict(skipped_optional=[(x['act'], x['code']) for x in rec['skipped_optional']],
                    optional_entries_matched_position_prompt=[r['step'] for r in own if r['snapshot']['message'] == 'select_position' and r['why']['kind'] == 'plan'],
                    adapter_changed_positions=[(r['step'], r['selected']['name'], r['selected']['position']) for r in rows if r['why'].get('adapter_changed')]),
        decisions=dict(total=len(rows), own_turn=len(own), opponent_turn=len(opp), own_turn_position_prompts=sum(1 for r in own if r['snapshot']['message'] == 'select_position'),
                       kinds_own_turn=dict(collections.Counter(r['why']['kind'] for r in own)), kinds_opponent_turn=dict(collections.Counter(r['why']['kind'] for r in opp)),
                       proposed_loss_indices=loss, proposed_loss_count=len(loss)),
        observations=rec['observations'],
        replay=dict(engine_run=rep['engine_run'], reproduced=f"{rep['reproduced_steps']}/{rep['of']}", mismatch=rep['mismatch'], final_equal=rep['final_equal'],
                    checkpoints_equal=rep['checkpoints_equal'], observations_match=rep['observations_match'], lua_errors=rep['lua_errors']),
        files={str(p.relative_to(W)): sha(p) for p in sorted((W / 'demos' / cond).glob('*'))})


def main():
    reg = read(W / 'registration.json')
    ledger = [json.loads(l) for l in (W / 'engine-runs.jsonl').read_text().splitlines()]
    regression = read(W / 'regression/summary.json')
    demos = {c: demo(c) for c in T.CONDITIONS}
    for c in demos:
        if c == 'C1':
            end = demos[c]['own_turn_end']
            demos[c]['fainmelt_draw'] = sorted((collections.Counter(end['hand']) + collections.Counter([T.WARASHI]))
                                               - collections.Counter({int(k): v for k, v in read(W / 'demos/C1/record.json')['marks']['before_draw'].items()}))
    summary = dict(
        registration_sha256=sha(W / 'registration.json'), engine_runs=dict(used=sum(1 for r in ledger if r['event'] == 'start'), cap=reg['engine_run_cap'],
                                                                           statuses=[r['status'] for r in ledger if r['event'] == 'finish']),
        regression=regression, demos=demos, split_check=read(W / 'split-check.json')['passed'],
        md_check={k: v for k, v in read(W / 'md-check.json').items() if k not in ('rows',)},
        code={p.name: sha(p) for p in sorted((W / 'code').glob('*.py'))})
    (W / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')

    manifest = dict(
        status='draft_not_converted_not_used_for_training', date='2026-11-10',
        original=dict(note='Original NPZ files are referenced, never rewritten.',
                      train=dict(path=str(OLD / 'train.npz'), sha256=sha(OLD / 'train.npz'), sequences=12, decisions=1077, use='train'),
                      validation=dict(path=str(OLD / 'validation.npz'), sha256=sha(OLD / 'validation.npz'), use='not for training'),
                      holdout=dict(path=str(OLD / 'holdout.npz'), sha256=sha(OLD / 'holdout.npz'), use='not for training; known regression set (results seen), not an untouched final test')),
        new_train_demos={c: dict(observations=demos[c]['observations']['path'], sha256=demos[c]['observations']['sha256'], record=str(W / 'demos' / c / 'record.json'),
                                 sequence_length=demos[c]['decisions']['total'], own_turn=demos[c]['decisions']['own_turn'],
                                 opponent_turn=demos[c]['decisions']['opponent_turn'], proposed_loss_decisions=demos[c]['decisions']['proposed_loss_count'],
                                 proposed_context_decisions=demos[c]['decisions']['total'], hand=demos[c]['hand'], split='train',
                                 split_check='no shared complete-hand group with old train/validation/holdout')
                         for c in demos},
        conversion_needed=['one padded sequence per demo (length > old max 100 for C1: 113)', 'separate context mask (all player-0 decisions) and loss mask (proposal below)',
                           'chain features already saved per decision (public chain length/role/card/effect)', 'labels = executed action index'],
        proposed_loss_mask='own-turn decisions except select_position (position adapter owns those), plus the scripted opponent-turn steps of C1 (獄神使② → W:P Fancy Ball); unscripted opponent-turn passes stay context only',
        caveat='Old train decisions all have weight 1 (includes position prompts chosen by the old teacher without the adapter); decide before training whether to mask them the same way.')
    (W / 'data').mkdir(exist_ok=True)
    (W / 'data/train-manifest-draft.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(dict(engine_runs=summary['engine_runs'], regression_identical=regression['all_identical'], regression_passed=regression['passed'],
                          demos={c: dict(script_completed=d['script_completed'], reference=d['reference_check'], replay=d['replay']['reproduced'],
                                         obs=d['replay']['observations_match'], loss=d['decisions']['proposed_loss_count'], total=d['decisions']['total'],
                                         draw=d['fainmelt_draw'], medius_cost=d['medius_cost']) for c, d in demos.items()}), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
