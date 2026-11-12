"""Pure tests of the B2B demonstration scripts (no engine; synthetic prompts in the native action format)."""
import importlib.util
from pathlib import Path
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('demo_policy', P / 'experiments/b2b-demo-20261109/code/demo_policy.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


def passive(s):
    for a in s['actions']:
        if a['act'] == 'Cancel' or a['finish']:
            return a['index']
    for phase in ['End', 'Main2', 'Battle']:
        for a in s['actions']:
            if a['phase'] == phase:
                return a['index']
    return 0


def act(index, act, code=0, effect=-1, phase='None', position=0, finish=False):
    return dict(index=index, act=act, code=code, effect=effect, spec='', phase=phase, place=0, position=position, number=0, finish=finish, response=0)


def snap(message, actions, turn_player=1, turn_count=2, chain=(), cards=(), phase=4):
    return dict(message=message, actions=list(actions), turn_player=turn_player, turn_count=turn_count, public_chain=list(chain),
                cards=list(cards), phase=phase, lp0=8000, lp1=8000, to_play=0)


SEARCH = [dict(player=1, code=D.RUK, description=D.RUK * 16)]
RUKINA_UP = dict(code=D.RUK, controller=1, location=4, position=1)


class DemoTests(unittest.TestCase):
    def opponent_turn(self, condition):
        st = D.new_state(condition)
        seq = [
            snap('select_chain', [act(0, 'Activate', D.VEILER, 10010), act(1, 'Activate', D.B2B, 10010), act(2, 'Cancel')], chain=SEARCH),
            snap('select_card', [act(0, 'None', D.ZAREN), act(1, 'None', D.CUE), act(2, 'None', 17209452)], chain=SEARCH),
            snap('select_option', [act(0, 'Activate', 0, 1190), act(1, 'Activate', 0, D.SPECIAL_SUMMON)], chain=SEARCH),
            snap('select_position', [act(0, 'None', D.CUE, position=1), act(1, 'None', D.CUE, position=4)], chain=SEARCH),
            snap('select_yesno', [act(0, 'Activate', D.B2B, 10011), act(1, 'Cancel')], chain=SEARCH),
            snap('select_card', [act(0, 'None', 41069676), act(1, 'None', D.TRACK)], chain=SEARCH),
            snap('select_unselect_card', [act(0, 'None', D.VEILER), act(1, 'None', D.VEILER)], chain=SEARCH),
            snap('select_chain', [act(0, 'Activate', D.CUE, 10011), act(1, 'Activate', D.TRACK, 10010), act(2, 'Cancel')]),
            snap('select_card', [act(0, 'None', 1234), act(1, 'None', D.SYNC)]),
        ]
        return st, [(s, s['actions'][D.decide(s, st, i, passive)[0]]) for i, s in enumerate(seq)]

    def test_dm_a_recovers_cue_makes_trackmaker_and_searches(self):
        st, out = self.opponent_turn('DM-A')
        chosen = [a for _, a in out]
        self.assertEqual((chosen[0]['code'], chosen[1]['code'], chosen[2]['effect'], chosen[3]['position']), (D.B2B, D.CUE, D.SPECIAL_SUMMON, 1))
        self.assertEqual((chosen[4]['act'], chosen[5]['code'], chosen[6]['code']), ('Activate', D.TRACK, D.VEILER))
        self.assertEqual((chosen[7]['code'], chosen[8]['code']), (D.TRACK, D.SYNC))

    def test_dm_b_revives_zaren_without_s_summon(self):
        st, out = self.opponent_turn('DM-B')
        chosen = [a for _, a in out]
        self.assertEqual((chosen[1]['code'], chosen[2]['effect']), (D.ZAREN, D.SPECIAL_SUMMON))
        self.assertEqual(chosen[4]['act'], 'Cancel')
        self.assertIsNone(st['track_search'])

    def test_attack_rule_only_attacks_weaker_monsters_then_direct(self):
        st = D.new_state('DM-B')
        idle = snap('select_idlecmd', [act(0, 'Summon', 1), act(1, 'None', phase='Battle'), act(2, 'None', phase='End')], turn_player=0, turn_count=3)
        self.assertEqual(D.decide(idle, st, 1, passive), (1, 'enter-battle'))
        bc = snap('select_battlecmd', [act(0, 'Attack', D.ZAREN), act(1, 'Attack', D.B2B), act(2, 'Attack', D.TRACK), act(3, 'None', phase='Main2')],
                  turn_player=0, turn_count=3, cards=[RUKINA_UP], phase=0x10)
        self.assertEqual(D.decide(bc, st, 2, passive), (1, 'attack-monster'))      # B2B 3200 > Rukina 2500
        strong = dict(RUKINA_UP, code=50954680)                                      # unknown ATK: never attack it
        bc2 = snap('select_battlecmd', [act(0, 'Attack', D.ZAREN), act(1, 'None', phase='Main2')], turn_player=0, turn_count=3, cards=[strong], phase=0x10)
        self.assertEqual(D.decide(bc2, st, 3, passive), (1, 'end-battle'))
        bc3 = snap('select_battlecmd', [act(0, 'DirectAttack', D.TRACK), act(1, 'DirectAttack', D.ZAREN), act(2, 'None', phase='Main2')],
                   turn_player=0, turn_count=3, phase=0x10)
        self.assertEqual(D.decide(bc3, st, 4, passive), (1, 'attack-direct'))      # zero-ATK Trackmaker never attacks
        self.assertEqual([a['kind'] for a in st['attacks']], ['monster', 'direct'])

    def test_missing_b2b_is_logged_and_passed(self):
        st = D.new_state('DM-A')
        s = snap('select_chain', [act(0, 'Cancel')], chain=SEARCH)
        self.assertEqual(D.decide(s, st, 1, passive), (0, 'pass'))
        self.assertTrue(any(e['label'] == 'b2b-NOT-OFFERED' for e in st['log']))


if __name__ == '__main__':
    unittest.main()
