"""Pure tests for the capability-probe analysis (no engine, no model).

Uses an existing reference-route record (evaluation/jj-ash-release-20261108, spec-development witness, read only)
and small synthetic variations of it."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('capability_analysis', P / 'experiments/capability-probe-20261109/code/analysis.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
PROOF = P / 'evaluation/jj-ash-release-20261108/cue-reco-jj-zaren-s0.json'


def card(code, controller, location, sequence=0, position=1):
    return dict(code=code, controller=controller, location=location, sequence=sequence, position=position, level=0)


def activate(code, effect_index, spec='m1', index=0):
    return dict(index=index, act='Activate', code=code, effect=A.CARD_EFFECT_OFFSET + effect_index, spec=spec, place=0,
                position=0, number=0, phase='None', finish=False, response=0)


def snap(turn_player, to_play, actions, cards=(), chain=(), message='select_chain', phase=4):
    return dict(turn_player=turn_player, to_play=to_play, message=message, phase=phase, chain_size=len(chain),
                public_chain=list(chain), cards=list(cards), actions=list(actions), turn_count=1 + turn_player)


class O1StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(PROOF.read_text())

    def test_reference_route_preparation_response_and_outcome_are_separate(self):
        st = A.o1_stages(self.record['trace'], self.record['final'])
        self.assertEqual(len(st['preparation']), 1)
        prep = st['preparation'][0]
        self.assertEqual((prep['step'], prep['phase'], prep['action']['code'], prep['chain_at_activation']), (80, 1, A.REMIX, []))
        self.assertEqual(prep['own_hand'], dict(added={A.ASH: 1}, removed={}))       # Ash returned to hand in the draw phase
        self.assertLess(prep['resolved_at_step'], st['search']['step'])               # preparation finished before the search
        self.assertEqual(st['search']['step'], 100)
        self.assertEqual([(r['step'], r['chain_link'], r['action']['code']) for r in st['responses']], [(101, 2, A.ASH)])
        self.assertEqual(st['outcome'], 'search_not_applied')
        self.assertEqual(st['evidence']['opponent_hand'], dict(added={}, removed={}))

    def test_a_card_reaching_the_opponent_hand_means_the_search_resolved(self):
        final = copy.deepcopy(self.record['final'])
        final['cards'].append(card(A.SPIRIT, 1, A.HAND, 9))
        st = A.o1_stages(self.record['trace'], final)
        self.assertEqual(st['outcome'], 'search_resolved')
        self.assertEqual(st['evidence']['added_from_hidden_zone'], {A.SPIRIT: 1})

    def test_removing_the_source_is_not_stopping_the_search(self):
        final = copy.deepcopy(self.record['final'])
        ruk = next(c for c in final['cards'] if c['controller'] == 1 and c['code'] == A.RUK and c['location'] == A.MZONE)
        ruk['location'] = A.GRAVE
        final['cards'].append(card(A.SPIRIT, 1, A.HAND, 9))
        st = A.o1_stages(self.record['trace'], final)
        self.assertEqual(st['outcome'], 'search_resolved')
        self.assertTrue(st['evidence']['search_source_left_field'])
        self.assertTrue(any('not counted as stopping' in n for n in st['notes']))

    def test_unattributable_hand_change_stays_unverified(self):
        final = copy.deepcopy(self.record['final'])
        lost = next(c for c in final['cards'] if c['controller'] == 1 and c['location'] == A.HAND)
        lost['location'] = A.GRAVE  # e.g. a discard: hand total changes but no search result is visible
        st = A.o1_stages(self.record['trace'], final)
        self.assertEqual(st['outcome'], 'unverified')

    def test_unresolved_chain_or_missing_search_stays_unverified(self):
        st = A.o1_stages(self.record['trace'][:101], self.record['trace'][101]['snapshot'])
        self.assertEqual(st['outcome'], 'unverified')
        own_turn_only = [r for r in self.record['trace'] if r['snapshot']['turn_player'] == 0]
        self.assertEqual(A.o1_stages(own_turn_only, self.record['own_end'])['outcome'], 'unverified')


class BaronneTests(unittest.TestCase):
    def trace(self, opponent_turn_actions):
        own = snap(0, 0, [activate(A.BARONNE, 0, index=0), activate(A.CRYSTAL, 0, index=1)], message='select_idlecmd')
        opp = snap(1, 0, opponent_turn_actions, chain=[dict(code=A.RUK, player=1, description=A.RUK * 16)])
        return [dict(step=1, snapshot=own, selected=own['actions'][0]), dict(step=2, snapshot=opp, selected=dict(act='Cancel', code=0))]

    def test_baronne_2_in_the_opponent_turn_is_fine_and_1_is_controlled_in_own_turn(self):
        result = A.baronne_checks(self.trace([activate(A.BARONNE, 1), activate(A.CRYSTAL, 0, index=1)]))
        self.assertTrue(result['ok'])
        self.assertEqual((result['own_turn_control_steps'], result['baronne_2_in_opponent_turn']), ([1], [2]))

    def test_baronne_1_offered_in_the_opponent_turn_is_flagged(self):
        result = A.baronne_checks(self.trace([activate(A.BARONNE, 0)]))
        self.assertFalse(result['ok'])
        self.assertEqual(result['baronne_1_in_opponent_turn'], [2])

    def test_without_the_own_turn_control_or_opponent_windows_the_check_does_not_pass(self):
        trace = self.trace([activate(A.BARONNE, 1)])
        trace[0]['snapshot']['actions'] = []
        self.assertFalse(A.baronne_checks(trace)['ok'])
        self.assertFalse(A.baronne_checks(trace[:1])['ok'])

    def test_effect_index_decodes_the_native_offset(self):
        self.assertEqual([A.effect_index(dict(effect=e)) for e in [10010, 10011, 0, -1]], [0, 1, None, None])


class ReplayAndLedgerTests(unittest.TestCase):
    def test_step_comparison_uses_snapshot_and_action_meaning(self):
        row = json.loads(PROOF.read_text())['trace'][80]
        self.assertEqual(A.compare_step(row, copy.deepcopy(row['snapshot']), copy.deepcopy(row['selected'])), [])
        changed = copy.deepcopy(row['snapshot']); changed['lp1'] -= 100
        self.assertEqual(A.compare_step(row, changed, row['selected']), ['lp1'])
        other = dict(row['selected'], code=A.ASH)  # same index, different card: must not count as reproduced
        self.assertEqual(A.compare_step(row, row['snapshot'], other), ['selected_meaning'])

    def test_ledger_refuses_the_thirteenth_engine_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = A.Ledger(Path(tmp) / 'runs.jsonl')
            for i in range(12):
                self.assertEqual(ledger.start('record', 'x', 't'), i + 1)
                ledger.finish(i + 1, 'ok', 't')
            with self.assertRaises(RuntimeError):
                ledger.start('record', 'x', 't')
            self.assertEqual(len(ledger.started()), 12)


if __name__ == '__main__':
    unittest.main()
