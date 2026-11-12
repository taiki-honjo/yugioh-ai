"""Offline tests for the Imperm / Diaktoros② trade-off (no engine duel)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-imperm-tradeoff-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import it_runs as IT  # noqa: E402
import it_summarize as IS  # noqa: E402

MED, INF, MAIDEN, D = IT.MEDIUS, IT.INFINITE, IT.MAIDEN, IT.DIAKTOROS


def act(i, a, code=0, effect=-1, spec='', position=0, finish=False):
    return dict(index=i, act=a, code=code, effect=effect, spec=spec, place=0, position=position, phase='None', finish=finish)


def snap(message, actions, chain=(), to_play=0, turn=2, phase=4, step=0, turn_player=1):
    ch = list(chain)
    return dict(message=message, actions=actions, public_chain=ch, chain_size=len(ch), to_play=to_play, turn_count=turn, phase=phase, _step=step,
                turn_player=turn_player, cards=[])


class FakeV:
    @staticmethod
    def passive(s):
        c = next((a['index'] for a in s['actions'] if a['act'] == 'Cancel' or (a['act'] == 'None' and a.get('phase') == 'End')), None)
        if c is None:
            raise RuntimeError
        return c


class OpponentTests(unittest.TestCase):
    def test_imperm_then_medius_only_after_the_imperm_chain_resolved(self):
        o = IT.Opponent('IM', FakeV)
        idle = [act(0, 'Activate', INF), act(1, 'Summon', MED), act(2, 'None', 0, spec='', finish=False)]
        idle[2]['phase'] = 'End'
        self.assertEqual(o(snap('select_idlecmd', idle, to_play=1))[0], 0)
        tg = [act(0, 'None', IT.FINEMELT, spec='om4'), act(1, 'None', MAIDEN, spec='om2'), act(2, 'None', MAIDEN, spec='om1')]
        self.assertEqual(o(snap('select_card', tg, chain=[dict(player=1, code=INF, description=0)], to_play=1))[0], 2)  # Maiden, lowest zone
        # while the Imperm chain is open the opponent does not summon
        self.assertFalse(o.st['imperm_resolved'])
        o(snap('select_chain', [act(0, 'Cancel')], chain=[dict(player=1, code=INF, description=0)], to_play=1))
        self.assertFalse(o.st['imperm_resolved'])
        self.assertEqual(o(snap('select_idlecmd', idle, to_play=1))[0], 1)  # chain size 0 now -> resolved -> Medius
        self.assertTrue(o.st['imperm_resolved'])
        self.assertEqual(o(snap('select_effectyn', [act(0, 'Activate', MED), act(1, 'Cancel')], to_play=1))[0], 0)
        self.assertEqual(o(snap('select_card', [act(0, 'None', 17473466), act(1, 'None', IT.BEAST)], to_play=1))[0], 1)

    def test_imperm_only_mode_never_summons(self):
        o = IT.Opponent('I', FakeV)
        o.st.update(imperm=True, target=True, imperm_resolved=True)
        idle = [act(0, 'Summon', MED), act(1, 'None')]
        idle[1]['phase'] = 'End'
        self.assertEqual(o(snap('select_idlecmd', idle, to_play=1))[0], 1)


class OwnTests(unittest.TestCase):
    def test_imperm_window_rules(self):
        w = [act(0, 'Activate', D, 10011, 'm3'), act(1, 'Activate', IT.BEAST, 10010, 'm5'), act(2, 'Cancel')]
        ch = [dict(player=1, code=INF, description=0)]
        self.assertEqual(IT.Own(IT.parse('A-I-N'))(snap('select_chain', w, ch))[0], 0)
        self.assertEqual(IT.Own(IT.parse('A-I-P'))(snap('select_chain', w, ch))[0], 2)
        idx, why = IT.Own(IT.parse('A-I-N'))(snap('select_chain', [act(0, 'Cancel')], ch))
        self.assertIsNone(idx)
        self.assertIn('not established', why)

    def test_medius_window_uses_diaktoros_only_if_legal(self):
        own = IT.Own(IT.parse('A-IM-P'))
        own.st['responded'] = True
        med = [dict(player=1, code=MED, description=MED * 16)]
        self.assertEqual(own(snap('select_chain', [act(0, 'Activate', 59438930, 0), act(1, 'Activate', D, 10011, 'm3'), act(2, 'Cancel')], med))[0], 1)
        own2 = IT.Own(IT.parse('A-IM-N'))
        own2.st['responded'] = True
        self.assertEqual(own2(snap('select_chain', [act(0, 'Activate', 59438930, 0), act(1, 'Cancel')], med))[0], 1)

    def test_fusion_uses_the_original_maiden_zone_not_the_name(self):
        own = IT.Own(IT.parse('B-P'))
        own.st.update(responded=True, maiden_zone=3)
        window = [act(0, 'Activate', MAIDEN, 10011, 'm1'), act(1, 'Activate', MAIDEN, 10011, 'm4'), act(2, 'Cancel')]
        self.assertEqual(own(snap('select_chain', window))[0], 1)  # zone 3 = m4, not the other Maiden in m1
        self.assertEqual(own(snap('select_card', [act(0, 'None', 27184601, spec='x1'), act(1, 'None', MAIDEN, spec='x2')]))[0], 1)
        mats = [act(0, 'None', IT.LITERA, spec='m2'), act(1, 'None', IT.FINEMELT, spec='m5')]
        self.assertEqual(own(snap('select_unselect_card', mats))[0], 1)
        self.assertEqual(own(snap('select_unselect_card', mats))[0], 0)
        self.assertEqual(own(snap('select_unselect_card', [act(0, 'None', 1), act(1, 'None', 0, finish=True)]))[0], 1)
        # another Maiden only (different zone) is never activated
        own2 = IT.Own(IT.parse('B-N'))
        own2.st.update(responded=True, maiden_zone=3)
        self.assertEqual(own2(snap('select_chain', [act(0, 'Activate', MAIDEN, 10011, 'm1'), act(1, 'Cancel')]))[0], 1)


class SummaryTests(unittest.TestCase):
    def test_chain_end_detection_waits_for_the_chain_to_start(self):
        tr = [dict(step=s, snapshot=dict(chain_size=c)) for s, c in [(10, 0), (11, 1), (12, 2), (13, 1), (14, 0), (15, 0)]]
        self.assertEqual(IS.chain_end_after(tr, 10), 14)
        self.assertIsNone(IS.chain_end_after(tr[:3], 10))

    def test_fusion_classification_separates_activation_and_selection(self):
        rec = dict(maiden_zone_B=3, own_rule_log=[dict(step=5, rule='original Maiden② (fusion)')],
                   trace=[dict(step=5, snapshot=dict(to_play=0, turn_count=2, actions=[act(0, 'Activate', MAIDEN, 10011, 'm4')]))],
                   checkpoints=dict(after_imperm_chain=dict(monsters=[MAIDEN, IT.FINEMELT, IT.LITERA]), after_fusion_chain=dict(monsters=[MAIDEN, IT.FINEMELT, IT.LITERA])))
        c = IS.classify_fusion(rec)
        self.assertTrue(c['activated'])
        self.assertFalse(c['selection_shown'])
        self.assertEqual(c['materials_left_field'], [])
        self.assertFalse(c['new_fusion_maiden'])

    def test_replay_path_steps_the_opponent_and_own_scripts(self):
        src = (ROOT / 'code/it_runs.py').read_text()
        self.assertIn("o_idx, _ = opponent(s)", src)
        self.assertIn("own rule differs from the record", src)


if __name__ == '__main__':
    unittest.main()
