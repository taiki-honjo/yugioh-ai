"""Offline tests for the unseen-response probe (no engine duel)."""
import hashlib
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-unseen-response-probe-20261111')
CV = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-response-coverage-20261111')
sys.path.insert(0, str(ROOT / 'code'))
import up_runs as U  # noqa: E402
import sa_model as SM  # noqa: E402

MED = U.MEDIUS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def act(i, a, code=0, effect=-1, position=0):
    return dict(index=i, act=a, code=code, effect=effect, spec='', place=0, position=position, phase='None', finish=False)


def snap(message, chain, actions, step=0):
    return dict(message=message, public_chain=chain, actions=actions, turn_count=1, phase=4, to_play=1, _step=step)


class FakeV:
    @staticmethod
    def passive(s):
        c = next((a['index'] for a in s['actions'] if a['act'] == 'Cancel'), None)
        if c is None:
            raise RuntimeError('no pass')
        return c


class OpponentTests(unittest.TestCase):
    def test_target_is_our_first_medius1_at_chain_link_1(self):
        for mode, code in [('ASH', U.ASH), ('GAMMA', U.GAMMA)]:
            o = U.Opponent(mode, FakeV)
            acts = [act(0, 'Activate', U.ASH), act(1, 'Activate', U.GAMMA), act(2, 'Cancel')]
            self.assertEqual(o(snap('select_chain', [], acts))[0], 2)  # before the target: pass
            idx, who = o(snap('select_chain', [dict(player=0, code=MED, description=MED * 16)], acts, 5))
            self.assertEqual((idx, who), ([a['index'] for a in acts if a['code'] == code][0], 'script'))
            self.assertEqual(o.state['target'], 'established')
            # Gamma resolution: Driver, face-up attack, first zone; then pass
            self.assertEqual(o(snap('select_card', [], [act(0, 'None', 1234), act(1, 'None', U.DRIVER)]))[0], 1)
            self.assertEqual(o(snap('select_position', [], [act(0, 'None', U.DRIVER, position=4), act(1, 'None', U.DRIVER, position=1)]))[0], 1)
            self.assertEqual(o(snap('select_chain', [], acts))[0], 2)

    def test_not_established_cases(self):
        o = U.Opponent('ASH', FakeV)
        o(snap('select_chain', [dict(player=1, code=1, description=16), dict(player=0, code=MED, description=MED * 16)], [act(0, 'Cancel')]))
        self.assertIn('not at chain link 1', o.state['target'])
        o2 = U.Opponent('GAMMA', FakeV)
        o2(snap('select_chain', [dict(player=0, code=MED, description=MED * 16)], [act(0, 'Activate', U.ASH), act(1, 'Cancel')]))
        self.assertEqual(o2.state['target'], 'opponent effect not legal')
        o3 = U.Opponent('ASH', FakeV)  # own Medius② or the opponent's own Medius are not the target
        o3(snap('select_chain', [dict(player=0, code=MED, description=MED * 16 + 1)], [act(0, 'Activate', U.ASH), act(1, 'Cancel')]))
        self.assertFalse(o3.state['seen_medius'])


class SetupTests(unittest.TestCase):
    def test_names_policies_and_files(self):
        self.assertEqual(len(U.COMPARISON), 8)
        self.assertEqual(len(U.CONTROLS), 4)
        self.assertEqual(U.parse('H1-GAMMA-CTRL-G'), dict(hand='H1', opp='GAMMA', policy='BASE', control='G'))
        self.assertEqual(U.parse('H2-ASH-CAND'), dict(hand='H2', opp='ASH', policy='CAND', control=None))
        self.assertEqual(tuple(sha(f) for f in U.BASE), SM.FROZEN_SHA)
        m, paths = SM.read_manifest(U.CAND)
        self.assertEqual(m['adapter']['sha256'], '00977941aab573d624a7e739dfbf3e9b2ba3502e9b22c2e0fbe8537072528b69')
        self.assertEqual(sha(paths['base']), sha(U.BASE[0]))
        self.assertEqual(sha(ROOT / 'code/sa_model.py'), sha(CV / 'code/sa_model.py'))
        self.assertEqual(U.HANDS['H1']['hand'], [74733322, 97045737, 23434538, 59438930, 49036338])
        self.assertEqual(U.HANDS['H2']['hand'], [74733322, 97045737, 23434538, 59438930, 10045474])
        self.assertEqual(U.HANDS['H1']['seed'], 1112103)

    def test_fixture_keeps_driver_in_the_opponent_deck(self):
        import artmage_entry as E
        deck = E.v.deck_cards('Artmage')
        rows = U.F.fixture(deck, U.HANDS['H1']['hand'], U.OPPONENT_HAND, 1112103)
        opp_hand = [r[0] for r in rows if r[1] == 1 and r[3] == 2]
        opp_deck = [r[0] for r in rows if r[1] == 1 and r[3] == 1]
        self.assertEqual(opp_hand, U.OPPONENT_HAND)
        self.assertEqual(opp_deck.count(U.DRIVER), 1)
        self.assertEqual(len([r for r in rows if r[1] == 1 and r[3] in (1, 2)]), 40)


if __name__ == '__main__':
    unittest.main()
