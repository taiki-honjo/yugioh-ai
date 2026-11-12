"""Offline tests for the isolated Artmage demo teacher, plans and split check (no engine, no model)."""
import collections
import json
from pathlib import Path
import sys

import unittest

W = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-parallel-20261110')
sys.path.insert(0, str(W / 'code'))
import artmage_teacher as T  # noqa: E402
import split_check as S  # noqa: E402


def act(index, act, code=0, **extra):
    base = dict(index=index, act=act, code=code, effect=-1, spec='', place=0, position=0, phase='', finish=False)
    base.update(extra)
    return base


def snap(message, actions, cards=(), turn=1, turn_player=0, phase=4, to_play=0):
    return dict(message=message, actions=list(actions), cards=list(cards), turn_count=turn, turn_player=turn_player, phase=phase,
                to_play=to_play, public_chain=[])


def card(code, location, sequence=0, controller=0, position=1):
    return dict(code=code, location=location, sequence=sequence, controller=controller, position=position)


def test_matches_uses_message_key():
    step = T.a('Activate', T.BEAST, message='select_idlecmd')
    assert T.matches(step, act(0, 'Activate', T.BEAST), 'select_idlecmd')
    assert not T.matches(step, act(0, 'Activate', T.BEAST), 'select_chain')


def test_optional_entry_is_skipped_only_when_a_later_entry_matches():
    t = T.Teacher([T.opt(T.a('None', T.MEDIUS)), T.a('Summon', T.MEDIUS)])
    s = snap('select_idlecmd', [act(0, 'Summon', T.MEDIUS), act(1, 'None', phase='End')])
    assert t(s) == 0 and t.steps == []
    t = T.Teacher([T.opt(T.a('None', T.MEDIUS)), T.a('Summon', T.MEDIUS)])
    try:
        t(snap('select_idlecmd', [act(0, 'Activate', T.FIELD), act(1, 'None', phase='End')]))
        raise AssertionError('expected Unresolved')
    except T.Unresolved:
        pass


def test_group_any_order_and_lifo_picks():
    plan = [T.group((T.a('Activate', T.MAIDEN), [T.a('None', T.AWAKENING)]), (T.a('Activate', T.BEAST), [T.a('None', T.ASSAULT)]))]
    t = T.Teacher(plan)
    first = snap('select_chain', [act(0, 'Activate', T.BEAST, effect=10011), act(1, 'Activate', T.MAIDEN, effect=10010), act(2, 'Cancel')])
    assert t(first) == 1  # listed order: Maiden first (chain link 1)
    second = snap('select_chain', [act(0, 'Activate', T.BEAST, effect=10011), act(1, 'Cancel')])
    assert t(second) == 0
    # chain resolves last-in-first-out: the Beast pick comes first
    assert [s['code'] for s in t.pending] == [T.ASSAULT, T.AWAKENING] and t.steps == []
    third = snap('select_card', [act(0, 'None', T.AWAKENING, spec='1'), act(1, 'None', T.ASSAULT, spec='1')])
    assert t(third) == 1 and t(dict(third)) == 0 and t.pending == []


def test_group_reversed_engine_order_still_schedules_lifo():
    plan = [T.group((T.a('Activate', T.MAIDEN), [T.a('None', T.AWAKENING)]), (T.a('Activate', T.BEAST), [T.a('None', T.ASSAULT)]))]
    t = T.Teacher(plan)
    assert t(snap('select_effectyn', [act(0, 'Activate', T.BEAST), act(1, 'Cancel')])) == 0
    assert t(snap('select_effectyn', [act(0, 'Activate', T.MAIDEN), act(1, 'Cancel')])) == 0
    assert [s['code'] for s in t.pending] == [T.AWAKENING, T.ASSAULT]


def test_group_member_resolving_before_the_next_trigger_uses_its_own_pick():
    plan = [T.group((T.a('Activate', T.NERVED), [T.opt(T.a('None', T.BEAST))]), (T.a('Activate', T.MAIDEN), [T.opt(T.a('None', T.FIELD))])),
            T.a('Activate', T.BEAST, message='select_idlecmd')]
    t = T.Teacher(plan)
    assert t(snap('select_effectyn', [act(0, 'Activate', T.NERVED), act(1, 'Cancel')])) == 0
    assert t(snap('select_card', [act(0, 'None', T.LITERA, spec='1'), act(1, 'None', T.BEAST, spec='1')])) == 1
    assert t(snap('select_effectyn', [act(0, 'Activate', T.MAIDEN), act(1, 'Cancel')])) == 0
    # Maiden's pick auto-resolved (not shown); stale optional pick must not leak into a later prompt
    assert t(snap('select_idlecmd', [act(0, 'Activate', T.BEAST, effect=10010), act(1, 'None', phase='End')])) == 0
    assert t.pending == [] and t.skipped_optional and t.steps == []


def test_missing_required_group_pick_stops_at_the_main_phase():
    t = T.Teacher([T.group((T.a('Activate', T.MAIDEN), [T.a('None', T.AWAKENING)]),), T.a('None', 0, phase='End')])
    t(snap('select_effectyn', [act(0, 'Activate', T.MAIDEN), act(1, 'Cancel')]))
    try:
        t(snap('select_idlecmd', [act(0, 'None', phase='End')]))
        raise AssertionError('expected Unresolved')
    except T.Unresolved:
        pass


def test_unmatched_chain_prompt_passes_and_is_logged():
    t = T.Teacher([T.a('Activate', T.FIELD)])
    s = snap('select_chain', [act(0, 'Activate', T.ASH), act(1, 'Cancel')])
    assert t(s) == 1
    assert t.unmatched_chain_heads and t.steps


def test_medius_cost_rule_prefers_drawn_non_engine_monster_then_filler():
    hand = [card(T.ASH, 2), card(T.G, 2), card(T.WARASHI, 2)]
    t = T.Teacher([T.mark(T.a('Activate', T.FINEMELT), 'before_draw'), T.a('Activate', T.MEDIUS), T.rule('medius_cost', optional=True), T.a('Set', T.AWAKENING)])
    assert t(snap('select_yesno', [act(0, 'Activate', T.FINEMELT, effect=10012), act(1, 'Cancel')], cards=hand)) == 0
    assert t.marks['before_draw'] == {T.ASH: 1, T.G: 1, T.WARASHI: 1}
    assert t(snap('select_idlecmd', [act(0, 'Activate', T.MEDIUS, effect=10011, spec='g1'), act(1, 'None', phase='End')], cards=hand)) == 0
    drawn = hand + [card(T.PSY_DRIVER, 2)]
    choices = [act(0, 'None', T.ASH, spec='h1'), act(1, 'None', T.PSY_DRIVER, spec='h4'), act(2, 'None', T.WARASHI, spec='h3'),
               act(3, 'None', T.LITERA, spec='m1')]
    assert t(snap('select_card', choices, cards=drawn)) == 1
    t2 = T.Teacher([T.mark(T.a('Activate', T.FINEMELT), 'before_draw'), T.rule('medius_cost', optional=True), T.a('Set', T.AWAKENING)])
    t2(snap('select_yesno', [act(0, 'Activate', T.FINEMELT), act(1, 'Cancel')], cards=hand))
    engine_draw = hand + [card(T.BEAST, 2)]
    choices = [act(0, 'None', T.ASH, spec='h1'), act(1, 'None', T.BEAST, spec='h4'), act(2, 'None', T.WARASHI, spec='h3')]
    assert t2(snap('select_card', choices, cards=engine_draw)) == 2


def test_optional_rule_does_not_block_later_steps():
    t = T.Teacher([T.rule('medius_cost', optional=True), T.a('Set', T.AWAKENING)])
    assert t(snap('select_place', [act(0, 'None', place=1), act(1, 'None', place=2)])) == 0
    assert t(snap('select_idlecmd', [act(0, 'None', phase='End'), act(1, 'Set', T.AWAKENING)])) == 1
    assert t.steps == []


def test_shi_goes_to_extra_zone_whose_bottom_zone_is_free():
    t = T.Teacher([T.a('SpSummon', T.SHI)])
    t(snap('select_idlecmd', [act(0, 'SpSummon', T.SHI), act(1, 'None', phase='End')]))
    places = [act(0, 'None', place=6), act(1, 'None', place=7)]
    occupied = [card(T.MAIDEN, 4, 1)]
    assert t(snap('select_place', places, cards=occupied)) == 1
    t.summon_context = T.SHI
    assert t(snap('select_place', places, cards=[card(T.MAIDEN, 4, 0)])) == 0


def test_opponent_turn_steps_wait_for_opponent_turn():
    t = T.Teacher([T.a('None', 0, phase='End'), T.opp(T.a('Activate', T.SHI))])
    own = snap('select_idlecmd', [act(0, 'Activate', T.SHI), act(1, 'None', phase='End')])
    assert t(own) == 1
    # still own turn (e.g. end phase chain prompt): the opponent-turn entry is not used
    assert t(snap('select_chain', [act(0, 'Activate', T.SHI), act(1, 'Cancel')], phase=512)) == 1
    opp_turn = snap('select_chain', [act(0, 'Activate', T.SHI), act(1, 'Cancel')], turn=2, turn_player=1)
    assert t(opp_turn) == 0 and t.steps == []
    assert T.Teacher([T.opp(T.a('Activate', T.SHI))]).remaining_required() == []


def test_unresolved_reports_head_at_own_main_phase():
    t = T.Teacher([T.a('SpSummon', T.MAIDEN)])
    try:
        t(snap('select_idlecmd', [act(0, 'Summon', T.MEDIUS), act(1, 'None', phase='End')]))
        raise AssertionError('expected Unresolved')
    except T.Unresolved as err:
        assert 'SpSummon' in str(err)


def _recipe():
    main, extra, section = [], [], None
    for line in (W / 'decks/Artmage.ydk').read_text().splitlines():
        line = line.strip()
        if line in ('#main', '#extra', '!side'):
            section = line
        elif line.isdigit():
            (main if section == '#main' else extra if section == '#extra' else []).append(int(line))
    return collections.Counter(main), collections.Counter(extra)


def test_condition_hands_are_legal_five_card_hands_from_the_recipe():
    main, extra = _recipe()
    assert sum(main.values()) == 40 and sum(extra.values()) == 15
    seeds = set()
    for name, c in T.CONDITIONS.items():
        assert len(c['hand']) == 5
        for code, n in collections.Counter(c['hand']).items():
            assert n <= main[code], (name, code)
        seeds.add(c['seed'])
        codes = {s['code'] for s in c['plan'] if '_group' not in s and '_rule' not in s} | {
            m['step']['code'] for s in c['plan'] if '_group' in s for m in s['_group']}
        assert codes <= set(main) | set(extra) | {0}, (name, codes - set(main) - set(extra))
    assert len(seeds) == len(T.CONDITIONS)
    old = json.loads(S.OLD_MANIFEST.read_text())['decks']['Artmage']['cases']
    assert not seeds & {c['seed'] for cases in old.values() for c in cases}


def test_split_check_groups_permutations_and_flags_eval_overlap():
    old = dict(train=[dict(hand=[1, 2, 3, 4, 5])], validation=[dict(hand=[6, 7, 8, 9, 10])], holdout=[dict(hand=[1, 1, 2, 3, 4])])
    out = S.check(old, dict(a=[5, 4, 3, 2, 1], b=[10, 9, 8, 7, 6], c=[1, 2, 3, 4, 6]))
    assert out['new']['a']['old_splits'] == ['train'] and not out['new']['a']['overlaps_validation_or_holdout']
    assert out['new']['b']['overlaps_validation_or_holdout'] and not out['passed']
    assert out['new']['c']['old_splits'] == []
    mixed = S.check(dict(train=[dict(hand=[1, 2, 3, 4, 5])], holdout=[dict(hand=[2, 1, 3, 4, 5])]), {})
    assert mixed['old_groups_in_more_than_one_split'] and not mixed['passed']


class AllTests(unittest.TestCase):
    pass


for _name in [n for n in list(globals()) if n.startswith('test_')]:
    setattr(AllTests, _name, staticmethod(globals()[_name]))
del _name

if __name__ == '__main__':
    unittest.main()
