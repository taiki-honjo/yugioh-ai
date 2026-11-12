"""Pure tests of the scripted B2B decisions (no engine). Remix prompts are the real recorded snapshots of the
P1 teacher record (experiments/capability-probe-20261109, train split, read only); B2B and battle prompts are
synthetic in the native action format."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('b2b_policy', P / 'experiments/b2b-probe-20261109/code/b2b_policy.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
RECORD = P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json'


def passive(s):  # same rule as verify_opponent_article_basics.passive
    for a in s['actions']:
        if a['act'] == 'Cancel' or a['finish']:
            return a['index']
    for phase in ['End', 'Main2', 'Battle']:
        for a in s['actions']:
            if a['phase'] == phase:
                return a['index']
    if len(s['actions']) == 1:
        return 0
    raise RuntimeError('Unresolved passive choice')


def act(index, act, code=0, effect=-1, spec='', phase='None', position=0, finish=False):
    return dict(index=index, act=act, code=code, effect=effect, spec=spec, phase=phase, place=0, position=position, number=0, finish=finish, response=0)


def snap(message, actions, turn_player=1, phase=4, chain=(), turn_count=2):
    return dict(message=message, actions=list(actions), turn_player=turn_player, phase=phase, public_chain=list(chain), turn_count=turn_count, to_play=0,
                chain_size=len(chain), lp0=8000, lp1=8000)


class RemixRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = json.loads(RECORD.read_text())['trace']

    def run_recorded(self, condition, steps):
        st = B.new_state(condition)
        return st, [B.decide(self.trace[i]['snapshot'], st, i, passive) for i in steps]

    def test_reco_to_hand_ash_summoned_then_synchro_yes(self):
        st, out = self.run_recorded('R2', range(72, 81))
        labels = [l for _, l in out if not l.startswith('pass')]
        chosen = {i: self.trace[i]['snapshot']['actions'][idx] for i, (idx, _) in zip(range(72, 81), out)}
        self.assertEqual(chosen[72]['code'], B.REMIX)                                   # draw phase, Remix (2)
        self.assertEqual(out[74 - 72][1], 'pass')                                       # our response window: pass
        self.assertEqual([chosen[75]['code'], chosen[76]['code']], [B.ASH, B.RECO])      # the two graveyard tuners
        self.assertEqual(chosen[77]['code'], B.RECO)                                    # Reco goes to hand (Ash is summoned)
        self.assertEqual(chosen[79]['position'], 1)
        self.assertEqual((chosen[80]['act'], out[80 - 72][1], st['ctx']), ('Activate', 'remix-synchro-yes', 'remix-synchro'))
        self.assertIn('remix', labels)

    def test_a0_does_not_use_remix(self):
        _, out = self.run_recorded('A0', [72])
        self.assertEqual(self.trace[72]['snapshot']['actions'][out[0][0]]['act'], 'Cancel')


class B2BTests(unittest.TestCase):
    def chain_window(self):
        return snap('select_chain', [act(0, 'Activate', B.B2B, 10010, 'm1'), act(1, 'Activate', 78058681, 10010, 's1'), act(2, 'Cancel')],
                    chain=[dict(player=1, code=B.RUK, description=B.RUK * 16)])

    def flow(self, condition):
        st = B.new_state(condition)
        st['ctx'] = 'remix-synchro'
        steps = [
            ('synchro-monster', snap('select_card', [act(0, 'None', 99999), act(1, 'None', B.B2B, spec='x3')], phase=1)),
            ('synchro-material-1', snap('select_unselect_card', [act(0, 'None', B.ZAREN, spec='m1'), act(1, 'None', B.ASH, spec='m3')], phase=1)),
            ('synchro-material-2', snap('select_unselect_card', [act(0, 'None', B.ZAREN, spec='m1')], phase=1)),
            ('chain', self.chain_window()),
            ('target', snap('select_card', [act(0, 'None', B.ZAREN, spec='g7'), act(1, 'None', B.CUE, spec='g5'), act(2, 'None', 17209452, spec='g1')])),
            ('option', snap('select_option', [act(0, 'Activate', 0, B.TO_HAND), act(1, 'Activate', 0, B.SPECIAL_SUMMON)])),
            ('yesno', snap('select_yesno', [act(0, 'Activate', B.B2B, 10011), act(1, 'Cancel')])),
            ('track', snap('select_card', [act(0, 'None', B.TRACK, spec='x1')])),
            ('materials-1', snap('select_unselect_card', [act(0, 'None', B.CUE, spec='m2'), act(1, 'None', B.VEILER, spec='h1')])),
            ('materials-2', snap('select_unselect_card', [act(0, 'None', B.VEILER, spec='h1'), act(1, 'None', B.VEILER, spec='h2')])),
        ]
        return st, {name: (s, s['actions'][B.decide(s, st, i, passive)[0]]) for i, (name, s) in enumerate(steps)}

    def test_remix_synchro_makes_b2b_from_ash_and_zaren(self):
        _, got = self.flow('R1')
        self.assertEqual([got[k][1]['code'] for k in ['synchro-monster', 'synchro-material-1', 'synchro-material-2']], [B.B2B, B.ASH, B.ZAREN])

    def test_conditions_choose_their_target_option_and_follow_up(self):
        expect = {'R0': (2, None, None, None), 'R1': (0, B.CUE, B.TO_HAND, 'Cancel'), 'R2': (0, B.CUE, B.SPECIAL_SUMMON, 'Cancel'),
                  'R3': (0, B.CUE, B.SPECIAL_SUMMON, 'Activate'), 'A1': (0, B.ZAREN, B.SPECIAL_SUMMON, 'Cancel')}
        for condition, (window, target, option, yes) in expect.items():
            with self.subTest(condition):
                st, got = self.flow(condition)
                self.assertEqual(got['chain'][1]['index'], window)
                if target is None:
                    self.assertIsNone(st['b2b'])
                    continue
                self.assertEqual((got['target'][1]['code'], got['option'][1]['effect'], got['yesno'][1]['act']), (target, option, yes))
                if condition == 'R3':
                    self.assertEqual([got[k][1]['code'] for k in ['track', 'materials-1', 'materials-2']], [B.TRACK, B.CUE, B.VEILER])

    def test_missing_b2b_activation_is_logged_not_forced(self):
        st = B.new_state('R1')
        window = self.chain_window(); window['actions'] = [act(0, 'Cancel')]
        self.assertEqual(B.decide(window, st, 5, passive), (0, 'pass'))
        self.assertTrue(any(e['label'] == 'b2b-NOT-OFFERED' for e in st['log']))


class BattleTests(unittest.TestCase):
    def test_a1_attacks_twice_when_offered_and_a0_stops(self):
        for condition, second in [('A1', True), ('A0', False)]:
            with self.subTest(condition):
                st = B.new_state(condition)
                idle = snap('select_idlecmd', [act(0, 'Summon', 1), act(1, 'None', phase='Battle'), act(2, 'None', phase='End')], turn_player=0, turn_count=3)
                self.assertEqual(B.decide(idle, st, 1, passive), (1, 'enter-battle'))
                first = snap('select_battlecmd', [act(0, 'Attack', B.ZAREN, spec='m1'), act(1, 'Attack', 65961304, spec='m3'),
                                                  act(2, 'None', phase='Main2')], turn_player=0, phase=0x8, turn_count=3)
                self.assertEqual(B.decide(first, st, 2, passive), (0, 'attack'))
                target = snap('select_card', [act(0, 'None', B.RUK, spec='om3')], turn_player=0, phase=0x8, turn_count=3)
                self.assertEqual(B.decide(target, st, 3, passive), (0, 'attack-pick'))
                again = [act(0, 'DirectAttack', B.ZAREN, spec='m1')] if second else []
                after = snap('select_battlecmd', again + [act(len(again), 'None', phase='Main2')], turn_player=0, phase=0x8, turn_count=3)
                idx, label = B.decide(after, st, 4, passive)
                self.assertEqual(label, 'attack' if second else 'end-battle')
                self.assertEqual(len(st['attacks']), 2 if second else 1)
                self.assertTrue(B.finished(dict(turn_count=3, phase=0x100), st))
                self.assertFalse(B.finished(dict(turn_count=3, phase=0x80), st))

    def test_r_conditions_stop_at_our_turn_three(self):
        st = B.new_state('R2')
        self.assertFalse(B.finished(dict(turn_count=2, phase=4), st))
        self.assertTrue(B.finished(dict(turn_count=3, phase=1), st))


if __name__ == '__main__':
    unittest.main()
