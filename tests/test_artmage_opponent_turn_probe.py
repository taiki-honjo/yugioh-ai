"""Offline tests for the opponent-turn probe (no engine duel, no model)."""
import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-opponent-turn-probe-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import otp_runs as O  # noqa: E402
import otp_reuse as R  # noqa: E402

MED = O.F.MEDIUS


def act(index, a, code=0, effect=-1, spec=''):
    return dict(index=index, act=a, code=code, effect=effect, spec=spec, place=0, position=0, phase='None', finish=False)


def snap(message='select_chain', phase=4, chain=None, actions=None, to_play=0, turn=2, turn_player=1):
    chain = [dict(code=MED, player=1, description=MED * 16)] if chain is None else chain
    return dict(message=message, phase=phase, public_chain=chain, chain_size=len(chain), to_play=to_play, turn_count=turn,
                turn_player=turn_player, actions=actions or [act(0, 'Cancel')], cards=[], lp0=8000, lp1=8000, _step=0)


class ImportTests(unittest.TestCase):
    def test_frozen_helpers_come_from_this_root(self):
        # sys.path itself is not checked: other test modules in the same process add the old root on purpose
        self.assertEqual(Path(O.F.__file__).resolve().parent, ROOT / 'code')
        self.assertEqual(Path(O.hb_plans.__file__).resolve().parent, ROOT / 'code')
        self.assertEqual(O.hashlib.sha256(Path(O.F.__file__).read_bytes()).hexdigest()[:8], '8af7cbae')
        self.assertEqual(O.NEW, ['S1-D1R0', 'S1-D0R1', 'S1-D1R1', 'S2-D1R0', 'S2-D1R1'])
        self.assertEqual(O.parse('S2-D1R0'), dict(site='S2', old='BV1-2cost-ASSAULT-OM', d1=True, r1=False))


class WindowTests(unittest.TestCase):
    def test_medius_window_only_for_the_opponents_medius_effect1_alone(self):
        self.assertTrue(O.is_medius_window(snap()))
        self.assertFalse(O.is_medius_window(snap(chain=[dict(code=MED, player=0, description=MED * 16)])))  # own Medius
        self.assertFalse(O.is_medius_window(snap(chain=[dict(code=MED, player=1, description=MED * 16 + 1)])))  # Medius②
        self.assertFalse(O.is_medius_window(snap(chain=[dict(code=MED, player=1, description=MED * 16), dict(code=1, player=0, description=16)])))
        self.assertFalse(O.is_medius_window(snap(phase=1)))
        self.assertFalse(O.is_medius_window(snap(message='select_card')))
        self.assertFalse(O.is_medius_window(snap(to_play=1)))
        self.assertFalse(O.is_medius_window(snap(turn=3, turn_player=0)))

    def test_r1_priority_effect_number_and_zone_order(self):
        both = [act(0, 'Activate', O.DIAKTOROS, 10011, 'm3'), act(1, 'Activate', O.FINEMELT, 10011, 'm4'), act(2, 'Activate', O.FINEMELT, 10011, 'm2'), act(3, 'Cancel')]
        pick, label = O.r1_pick(snap(actions=both))
        self.assertEqual((pick['index'], label), (2, 'ファインメルト③'))
        only_d = [act(0, 'Activate', O.FINEMELT, 10010, 'h1'), act(1, 'Activate', O.DIAKTOROS, 10011, 'm3'), act(2, 'Cancel')]
        pick, label = O.r1_pick(snap(actions=only_d))
        self.assertEqual((pick['index'], label), (1, 'ディアクトロス②'))
        self.assertEqual(O.r1_pick(snap(actions=[act(0, 'Activate', O.DIAKTOROS, 10010, 'm3'), act(1, 'Cancel')])), (None, None))


class IntervenerTests(unittest.TestCase):
    def test_r1_fires_once_and_records_the_model_choice(self):
        iv = O.Intervener(False, True)
        s = snap(actions=[act(0, 'Activate', O.DIAKTOROS, 10011, 'm3'), act(1, 'Cancel')])
        self.assertEqual(iv(s, 10, 1), 0)
        self.assertEqual(iv.log[0]['model_choice']['act'], 'Cancel')
        self.assertFalse(iv.log[0]['model_agrees'])
        self.assertIsNone(iv(s, 12, 1))
        self.assertEqual(len(iv.log), 1)

    def test_r1_not_established_executes_the_model(self):
        iv = O.Intervener(False, True)
        self.assertIsNone(iv(snap(actions=[act(0, 'Activate', 31822037, 10011, 'm6'), act(1, 'Cancel')]), 5, 0))
        self.assertFalse(iv.log[0]['established'])

    def test_d1_cancels_only_in_the_draw_phase_and_leaves_forced_prompts(self):
        iv = O.Intervener(True, False)
        acts = [act(0, 'Activate', O.F.MAIDEN, 10011, 'm1'), act(1, 'Cancel')]
        self.assertEqual(iv(snap(phase=1, chain=[], actions=acts), 1, 0), 1)
        self.assertIsNone(iv(snap(phase=1, chain=[], message='select_card', actions=[act(0, 'None', 1), act(1, 'None', 2)]), 2, 0))
        self.assertTrue(iv.log[-1]['forced'])
        self.assertIsNone(iv(snap(phase=2, chain=[], actions=acts), 3, 0))  # standby: model
        self.assertIsNone(iv(snap(phase=4, chain=[], actions=acts), 4, 0))
        self.assertEqual(len(iv.log), 2)


class OldTraceTests(unittest.TestCase):
    def trace(self, old):
        return json.loads((R.OLD / 'runs' / old / 'record.json').read_text())['trace']

    def test_s2_d0r1_is_the_old_decision_sequence(self):
        div, log, cp = R.first_divergence(self.trace('BV1-2cost-ASSAULT-OM'), False, True)
        self.assertIsNone(div)
        self.assertEqual([(x['rule'], x['step'], x['executed'], x['model_agrees']) for x in log], [('R1', 169, 'ファインメルト③', True)])

    def test_s1_d0r1_differs_at_the_medius_window(self):
        div, log, cp = R.first_divergence(self.trace('BV1-1cost-VARNISH-OM'), False, True)
        self.assertEqual((div['step'], div['rule'], div['rule_executes']['code']), (149, 'R1', O.DIAKTOROS))
        self.assertEqual(cp['medius_response_window']['step'], 149)

    def test_d1_differs_at_the_first_draw_phase_activation(self):
        for old, step in [('BV1-1cost-VARNISH-OM', 119), ('BV1-2cost-ASSAULT-OM', 133)]:
            div, _, _ = R.first_divergence(self.trace(old), True, False)
            self.assertEqual((div['step'], div['rule'], div['rule_executes']['act']), (step, 'D1', 'Cancel'))


if __name__ == '__main__':
    unittest.main()
