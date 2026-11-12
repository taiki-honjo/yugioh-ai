"""Offline tests for the functional landing comparison (no engine duel)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-functional-landing-compare-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402
import flc_runs as F  # noqa: E402


def act(index, a, code=0, **extra):
    return dict(dict(index=index, act=a, code=code, effect=-1, spec='', place=0, position=0, phase='', finish=False), **extra)


class FixtureTests(unittest.TestCase):
    def test_same_hands_reproduce_the_old_fixture(self):
        deck = E.v.deck_cards('Artmage')
        hand = F.HP.HANDS['BV1-B-1cost']['hand']
        old = E.v.fixture(dict(hand=hand), 'Artmage', 1112103)
        self.assertEqual(F.fixture(deck, hand, hand, 1112103), old)

    def test_player0_rows_unchanged_by_the_opponent_hand(self):
        deck = E.v.deck_cards('Artmage')
        hand = F.HP.HANDS['BV1-B-2cost']['hand']
        old = [r for r in E.v.fixture(dict(hand=hand), 'Artmage', 1112103) if r[1] == 0]
        new = [r for r in F.fixture(deck, hand, F.OPPONENT_HAND, 1112103) if r[1] == 0]
        self.assertEqual(old, new)
        opp = [r[0] for r in F.fixture(deck, hand, F.OPPONENT_HAND, 1112103) if r[1] == 1 and r[3] == 2]
        self.assertEqual(opp, F.OPPONENT_HAND)


class PrefixTests(unittest.TestCase):
    def test_prefixes_end_before_the_graphrea_prompt(self):
        for hand, n in [('BV1-B-1cost', 14), ('BV1-B-2cost', 15)]:
            prefix, src = F.prefix_of(hand)
            self.assertEqual(len(prefix), n)
            self.assertEqual(prefix[-1]['message'], 'select_yesno')
            self.assertEqual(src['source_choice'], F.VARNISH)
        self.assertEqual(len(F.conditions()), 8)


class OpponentTests(unittest.TestCase):
    def test_infinite_target_priority_and_zone_order(self):
        op = F.Opponent('OI', E.v)
        op.state['activated'] = True
        s = dict(message='select_card', turn_count=2, turn_player=1, phase=4, to_play=1, cards=[], public_chain=[],
                 actions=[act(0, 'None', F.FINEMELT, spec='om3'), act(1, 'None', F.MAIDEN, spec='om2'), act(2, 'None', F.MAIDEN, spec='om1')])
        self.assertEqual(op(s)[0], 2)
        op2 = F.Opponent('OI', E.v)
        op2.state['activated'] = True
        s['actions'] = [act(0, 'None', 97434754, spec='om2'), act(1, 'None', F.FINEMELT, spec='om4')]
        self.assertEqual(op2(s)[0], 1)

    def test_medius_script_sequence(self):
        op = F.Opponent('OM', E.v)
        base = dict(turn_count=2, turn_player=1, phase=4, to_play=1, cards=[], public_chain=[])
        self.assertEqual(op(dict(base, message='select_idlecmd', actions=[act(0, 'Summon', F.MEDIUS), act(1, 'None', phase='End')]))[0], 0)
        self.assertEqual(op(dict(base, message='select_effectyn', actions=[act(0, 'Activate', F.MEDIUS), act(1, 'Cancel')]))[0], 0)
        self.assertEqual(op(dict(base, message='select_card', actions=[act(0, 'None', 17473466), act(1, 'None', F.BEAST)]))[0], 1)
        self.assertEqual(op(dict(base, message='select_option', actions=[act(0, 'Activate', effect=1190), act(1, 'Activate', effect=1152)]))[0], 1)
        idle = dict(base, message='select_idlecmd', actions=[act(0, 'Summon', 12345), act(1, 'None', phase='End')])
        self.assertEqual(op(idle)[0], 1)


if __name__ == '__main__':
    unittest.main()
