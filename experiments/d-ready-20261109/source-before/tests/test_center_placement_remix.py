"""Placement adapter v2 (Kewl Tune Remix tracking): regression tests.

v1 (pinned in existing snapshots) is loaded read-only to show the original failure and
to check that v2 is identical to v1 wherever v1 did not fail.
"""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

P = Path('/home/hatk8/projects/yugioh-ai')
V1_FILES = [P / 'evaluation/battle-position-20261108/code/center_placement.py',
            P / 'experiments/continuous-elfnotes-openings-20261109/snapshot/policy-code/center_placement.py']
V1_SHA256 = '805e68213710ad411ae4c5a3fc47bd998a8eb7ed2113a3335386eeed7db00b3c'
V2_FILE = P / 'evaluation/placement-remix-fix-20261109/code/center_placement.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


v1, v2 = load(V1_FILES[0], 'center_placement_v1'), load(V2_FILE, 'center_placement_v2')
REMIX, REG, STREL = 88170262, 56651978, 42302563
MIX, RECO, CUE, LOUDNESS, ASH = 16509007, 89392810, 16387555, 41069676, 14558127
ACTIVATE, NONE, CANCEL = 8, 0, 9

# Recorded from the A-777 candidate battle that stopped (experiments/placement-remix-fix-20261109/
# repro/remix-crash-v1.json): (msg, act, code, from_hand, public_chain_length).
REPRODUCED = [(2, ACTIVATE, REMIX, False, 1), (2, CANCEL, 0, False, 2), (9, NONE, MIX, False, 2), (9, NONE, RECO, False, 2),
              (3, NONE, MIX, False, 2), (11, NONE, 0, False, 2), (5, NONE, RECO, False, 2),
              (7, ACTIVATE, REMIX, False, 2),  # "Synchro Summon now?" after Remix resolved
              (3, NONE, LOUDNESS, False, 2), (9, NONE, RECO, False, 2), (9, NONE, CUE, True, 2)]


def replay(memory, events, chain=True):
    """chain=False mimics callers that cannot report the public chain (teacher demonstrations, v1)."""
    reports_chain = chain and 'chain_length' in inspect.signature(memory.observe).parameters
    for msg, act, code, hand, length in events:
        if reports_chain:
            memory.observe(msg, act, code, hand, length)
        else:
            memory.observe(msg, act, code, hand)
    return memory


def state(memory):
    return memory.incoming, memory.locked, None if memory.remix_targets is None else list(memory.remix_targets)


class RemixTrackingTests(unittest.TestCase):
    def test_v1_is_left_unchanged_in_existing_snapshots(self):
        for path in V1_FILES:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), V1_SHA256)

    def test_reproduced_battle_sequence_no_longer_raises(self):
        with self.assertRaises(ValueError):
            replay(v1.PlacementMemory(), REPRODUCED)
        memory = replay(v2.PlacementMemory(), REPRODUCED[:7])
        self.assertEqual(memory.events, [])
        replay(memory, REPRODUCED[7:])
        # The Synchro choice and its materials follow the unchanged generic rule.
        self.assertEqual(state(memory), (CUE, False, None))

    def test_remix_resolution_marks_the_unpicked_card_as_arriving(self):
        for module in (v1, v2):
            memory = replay(module.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (9, NONE, ASH, False, 2),
                                                       (9, NONE, REG, False, 2), (3, NONE, ASH, False, 2)])
            self.assertEqual(state(memory), (REG, True, [ASH, REG]))
            places = [3, 1, 2]  # raw choice: center
            self.assertEqual(memory.choose(places, 0, np.array([3., 2., 1.])), 0)
            self.assertEqual(memory.events[-1]['reason'], 'regina-center-trigger')
            memory.observe(11, NONE, 0)
            self.assertEqual(state(memory), (0, False, None))

    def test_follow_up_synchro_prompt_does_not_restart_collection(self):
        memory = replay(v2.PlacementMemory(), [(7, ACTIVATE, REMIX, False, 2), (3, NONE, LOUDNESS, False, 2)])
        self.assertEqual(state(memory), (LOUDNESS, False, None))
        memory = replay(v2.PlacementMemory(), [(6, ACTIVATE, REMIX, False, 1)])
        self.assertIsNone(memory.remix_targets)

    def test_pick_outside_the_two_recorded_cards_falls_back_with_a_diagnostic(self):
        with self.assertRaises(ValueError):
            replay(v1.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (9, NONE, MIX, False, 1),
                                          (9, NONE, RECO, False, 1), (3, NONE, CUE, False, 1)])
        memory = replay(v2.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (9, NONE, MIX, False, 1),
                                               (9, NONE, RECO, False, 1), (3, NONE, CUE, False, 1)])
        self.assertEqual(state(memory), (CUE, False, None))
        self.assertEqual(memory.diagnostics[-1]['reason'], 'pick-outside-remix-picks')
        self.assertEqual(memory.diagnostics[-1]['picks'], [MIX, RECO])

    def test_chain_that_ends_before_the_picks_stops_collection(self):
        memory = replay(v2.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (1, NONE, 0, False, 0),
                                               (3, NONE, MIX, False, 0)])
        self.assertEqual(state(memory), (MIX, False, None))
        self.assertEqual(memory.diagnostics[-1]['reason'], 'chain-ended')
        # A new Remix activation that starts a chain is not mistaken for a chain end.
        memory = replay(v2.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 0)])
        self.assertEqual(memory.remix_targets, [])
        # Teacher demonstrations do not report the chain; v1 behavior is kept there.
        memory = replay(v2.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (9, NONE, MIX, False, 1)], chain=False)
        self.assertEqual(memory.remix_targets, [MIX])

    def test_another_activation_in_the_chain_ends_collection_as_in_v1(self):
        events = [(2, ACTIVATE, REMIX, False, 1), (2, ACTIVATE, ASH, True, 2), (9, NONE, MIX, False, 2)]
        old, new = replay(v1.PlacementMemory(), events), replay(v2.PlacementMemory(), events)
        self.assertEqual(state(old), state(new))
        self.assertEqual(new.diagnostics[-1]['reason'], 'other-activation')

    def test_reset_and_later_placements_start_clean(self):
        memory = replay(v2.PlacementMemory(), [(2, ACTIVATE, REMIX, False, 1), (9, NONE, MIX, False, 1),
                                               (9, NONE, RECO, False, 1), (3, NONE, CUE, False, 1)])
        memory.observe(11, NONE, 0, False, 1)
        self.assertEqual(state(memory), (0, False, None))
        self.assertEqual(memory.choose([3, 1], 0, np.array([2., 1.])), 0)
        self.assertEqual(memory.events[-1]['reason'], 'unknown-arrival-defer-to-model')
        memory.reset()
        self.assertEqual((state(memory), memory.events, memory.diagnostics), ((0, False, None), [], []))

    def test_identical_to_v1_wherever_v1_did_not_fail(self):
        rng = random.Random(20261109)
        codes = [0, REMIX, REG, STREL, MIX, RECO, CUE, ASH, LOUDNESS]
        compared = 0
        for _ in range(3000):
            old, new = v1.PlacementMemory(), v2.PlacementMemory()
            for _ in range(rng.randint(1, 25)):
                # Command prompts (idle/chain/battle) carry summon/activate/cancel actions; card, sum,
                # position and place prompts carry plain picks. The public chain is never reported empty.
                msg = rng.choice([1, 2, 3, 5, 8, 9, 11, 12])
                act = rng.choice([0, 3, 4, 5, 8, 9]) if msg in (1, 2, 8) else 0
                code, hand, length = rng.choice(codes), rng.random() < .3, rng.randint(1, 3)
                try:
                    old.observe(msg, act, code, hand)
                except ValueError:
                    break  # the v1 failure path; covered by the tests above
                new.observe(msg, act, code, hand, length)
                self.assertEqual(state(old), state(new))
                compared += 1
                if msg == 11:
                    places, raw = [3, 1, 2], rng.randrange(3)
                    self.assertEqual(old.choose(places, raw), new.choose(places, raw))
            self.assertEqual(old.events, new.events)
        self.assertGreater(compared, 10000)

    def test_policy_reports_the_public_chain_length(self):
        class Base:
            base = head = state = None

            def __init__(self, logits):
                self.logits = logits

            def infer(self, *args):
                return None, self.logits

        def obs(msg, act_code):
            actions = np.zeros((1, 4, 12), np.uint8)
            actions[0, :2, 3] = msg
            actions[0, 0, 4], actions[0, 0, 2] = act_code, 7  # option 0: Activate card id 7
            return dict(actions_=actions, cards_=np.zeros((1, 160, 41), np.uint8))
        logits = np.array([[5., 1., 0., 0.]])
        with patch.dict(sys.modules, all_openings_common=SimpleNamespace(features=lambda info: np.zeros((1, 4)))):
            policy = v2.PlacementPolicy(Base(logits), {7: REMIX})
            policy.act(obs(2, ACTIVATE), dict(num_options=np.array([2]), public_chain_length=np.array([1])))
            self.assertEqual(policy.memory.remix_targets, [])
            policy.act(obs(1, NONE), dict(num_options=np.array([2]), public_chain_length=np.array([0])))
            self.assertIsNone(policy.memory.remix_targets)
            self.assertEqual(policy.memory.diagnostics[-1]['reason'], 'chain-ended')


if __name__ == '__main__':
    unittest.main(verbosity=2)
