"""Pure tests for the Maxx C / Droll condition analysis (no engine). Uses the existing P2 teacher record
(experiments/capability-probe-20261109, train split, read only) as the no-hand-trap baseline shape."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('handtrap_analysis', P / 'experiments/handtrap-continuing-20261109/code/handtrap_analysis.py')
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
RECORD = P / 'experiments/capability-probe-20261109/records/P2-elf-rukina-regina-g00-p0.json'


def card(code, controller, location, sequence=0):
    return dict(code=code, controller=controller, location=location, sequence=sequence, position=10, level=0)


class HandtrapAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RECORD.read_text())

    def base(self):
        return copy.deepcopy(self.r['trace']), copy.deepcopy(self.r['own_end'])

    def test_baseline_shape(self):
        trace, end = self.base()
        s = H.summarize(trace, end, trace[0]['snapshot'])
        self.assertEqual([e['step'] for e in s['p0_add_events']], [11, 32])   # Regina, then Dina reach the hand
        self.assertEqual((s['p0_adds_from_hidden'], s['p1_gained_total'], s['p0_monster_entries']), (2, 0, 10))
        self.assertIsNone(s['p0_adds_after_droll'])                             # no Droll: not zero, not measured
        self.assertEqual(s['p1_hand_start'], s['p1_hand_end'])

    def test_player1_gains_are_counted_only_from_hidden_zones(self):
        trace, end = self.base()
        for row in trace[40:]:
            row['snapshot']['cards'].append(card(H.G, 1, H.HAND, 9))          # e.g. a Maxx C draw from the deck at step 40
        end['cards'].append(card(H.G, 1, H.HAND, 9))
        s = H.summarize(trace, end, trace[0]['snapshot'])
        self.assertEqual((s['p1_gained_total'], s['p1_gain_events'][0]['step']), (1, 39))
        # a card moving from player 1's graveyard to hand is not a draw
        trace, end = self.base()
        for row in trace[40:]:
            row['snapshot']['cards'].append(card(H.DROLL, 1, H.HAND, 9))
        for row in trace[:40]:
            row['snapshot']['cards'].append(card(H.DROLL, 1, H.GRAVE, 0))
        end['cards'].append(card(H.DROLL, 1, H.HAND, 9))
        self.assertEqual(H.summarize(trace, end, trace[0]['snapshot'])['p1_gained_total'], 0)

    def test_counts_after_droll_start_when_its_chain_has_resolved(self):
        trace, end = self.base()
        row = trace[13]                                                        # a player-1 window after Regina was added
        self.assertEqual(row['snapshot']['to_play'], 1)
        row['selected'] = dict(row['selected'], act='Activate', code=H.DROLL, effect=0)
        s = H.summarize(trace, end, trace[0]['snapshot'])
        self.assertEqual(s['droll_activation_step'], 13)
        self.assertGreater(s['droll_in_effect_from_step'], 13)
        self.assertEqual(s['p0_adds_after_droll'], 1)                         # the later Dina add would be a violation
        self.assertEqual(s['p0_adds_from_hidden'], 2)

    def test_compare_finds_divergence_and_board_difference(self):
        trace, end = self.base()
        same = H.compare(trace, end, trace, end)
        self.assertTrue(same['identical_deploying_actions'])
        short = [r for r in trace if r['step'] < 30]
        end2 = copy.deepcopy(end)
        end2['cards'] = [c for c in end2['cards'] if not (c['controller'] == 0 and c['code'] == 84815190 and c['location'] == H.MZONE)]
        diff = H.compare(trace, end, short, end2)
        self.assertFalse(diff['identical_deploying_actions'])
        self.assertEqual(diff['first_divergence_index'], diff['decisions'])
        self.assertEqual(diff['end_board_missing_vs_baseline'], {84815190: 1})

    def test_legal_in_baseline_uses_action_meaning(self):
        trace, _ = self.base()
        strel = next(r['selected'] for r in trace if r['selected']['act'] == 'Activate' and r['selected']['code'] == 42302563)
        steps = H.legal_in_baseline(trace, strel)
        self.assertIn(33, steps)


if __name__ == '__main__':
    unittest.main()
