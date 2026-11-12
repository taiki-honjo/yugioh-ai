"""Offline tests for the Artmage hand-branch study (no engine duel, no training)."""
import collections
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np  # noqa: E402

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-hand-branch-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import hb_plans as HP  # noqa: E402
import hb_data as HD  # noqa: E402
import batch_plans as BP  # noqa: E402
import pilot_data as PD  # noqa: E402
import artmage_teacher as T  # noqa: E402


def act(index, a, code=0, **extra):
    return dict(dict(index=index, act=a, code=code, effect=-1, spec='', place=0, position=0, phase='', finish=False), **extra)


def snap(message, actions):
    return dict(message=message, actions=list(actions), cards=[], turn_count=1, turn_player=0, phase=4, to_play=0, public_chain=[])


class PairTests(unittest.TestCase):
    def test_pairs_differ_by_one_card_and_share_a_seed(self):
        for p in HP.PAIRS:
            (va, ha), (vb, hb) = list(p['hands'].items())
            self.assertEqual(sorted(ha[:-1]), sorted(hb[:-1]))
            self.assertNotEqual(ha[-1], hb[-1])
            self.assertEqual(HP.HANDS[f"{p['pair']}-{va}"]['seed'], HP.HANDS[f"{p['pair']}-{vb}"]['seed'])
            if p['group'] == 'A':
                self.assertNotIn(p['differing']['A-nocost'], HP.COST_CARDS)
                self.assertIn(p['differing']['A-cost'], HP.COST_CARDS)
            else:
                self.assertEqual(sum(c in HP.COST_CARDS for c in p['hands']['B-1cost']), 1)
                self.assertEqual(sum(c in HP.COST_CARDS for c in p['hands']['B-2cost']), 2)

    def test_counts_and_split(self):
        self.assertEqual(sum(p['split'] == 'train' for p in HP.PAIRS), 8)
        self.assertEqual(sum(p['split'] == 'validation' for p in HP.PAIRS), 4)
        keys = [HP.hand_key(x['hand']) for x in HP.HANDS.values()]
        self.assertEqual(len(keys), len(set(keys)))
        for x in list(BP.TRAIN.values()) + list(BP.VALIDATION.values()):
            self.assertNotIn(HP.hand_key(x['hand']), keys)
        train_cards = {tuple(sorted(p['hands'][v])) for p in HP.PAIRS if p['split'] == 'train' for v in p['hands']}
        val_cards = {tuple(sorted(p['hands'][v])) for p in HP.PAIRS if p['split'] == 'validation' for v in p['hands']}
        self.assertFalse(train_cards & val_cards)


class TeacherTests(unittest.TestCase):
    def test_cost_order_for_two_different_costs(self):
        t = HP.BranchTeacher([HP.a('Activate', HP.GRAPH)])
        opts = [act(0, 'None', HP.FUNSAI, spec='h1'), act(1, 'None', HP.SANSEN, spec='h2')]
        self.assertEqual(t(snap('select_card', opts)), 1)
        self.assertEqual(t.log[-1]['why']['kind'], 'cost-order')
        with self.assertRaises(T.Unresolved):
            HP.BranchTeacher([HP.a('Activate', HP.GRAPH)])(snap('select_card', [act(0, 'None', HP.FUNSAI, spec='h1'), act(1, 'None', HP.LITERA, spec='h2')]))

    def test_old_route_uses_the_medius_cost_rule(self):
        self.assertNotIn(HP.a('None', 59438930), HP.OLD_ROUTE)
        self.assertTrue(any(e.get('_rule') == 'medius_cost' for e in HP.OLD_ROUTE))


class BranchTests(unittest.TestCase):
    def test_branch_steps_on_existing_demos(self):
        import artmage_entry as E
        c1, _ = PD.load_demo('C1')
        found = HP.branch_steps(c1, E._IDS)
        # C1 contains both studied choices: Medius① → ネルヴェド and Graphrea sets 改変
        self.assertEqual([(b['branch'], b['label']) for b in found], [('A: Medius①', HP.NERVED), ('B: Graphrea set', HP.VARNISH)])
        c2, _ = PD.load_demo('C2')
        found = HP.branch_steps(c2, E._IDS)
        self.assertIn(('B: Graphrea set', HP.VARNISH), [(b['branch'], b['label']) for b in found])
        self.assertNotIn(('B: Graphrea set', HP.ASSAULT), [(b['branch'], b['label']) for b in found])


class SamplerTests(unittest.TestCase):
    def test_blocks_hold_two_existing_and_two_paired_minibatches(self):
        existing = {g: [0] * n for g, n in [('F0', 12), ('F1', 2), ('F2', 2)]}
        pairs = {'A': [dict(pair='A1'), dict(pair='A2')], 'B': [dict(pair='B1')]}
        s = HD.BlockSampler(existing, pairs, 30131001)
        kinds = [s.next()[0] for _ in range(400)]
        for i in range(0, 400, 4):
            self.assertEqual(collections.Counter(kinds[i:i + 4]), {'existing': 2, 'pair': 2})
        c = s.counts()
        self.assertEqual(c['existing_minibatches'], 200)
        self.assertEqual(c['pair_minibatches'], 200)


if __name__ == '__main__':
    unittest.main()
