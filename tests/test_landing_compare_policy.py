"""Pure tests for the two-landing comparison scripts (no engine)."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('compare_policy', P / 'experiments/landing-compare-20261109/code/compare_policy.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
RECORD = P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json'


def passive(s):
    for a in s['actions']:
        if a['act'] == 'Cancel' or a['finish']:
            return a['index']
    for phase in ['End', 'Main2', 'Battle']:
        for a in s['actions']:
            if a['phase'] == phase:
                return a['index']
    return 0


def act(index, act, code=0, phase='None', place=0):
    return dict(index=index, act=act, code=code, effect=-1, spec='', phase=phase, place=place, position=0, number=0, finish=False, response=0)


def snap(message, actions, phase=4, turn_count=2):
    return dict(message=message, actions=list(actions), phase=phase, turn_count=turn_count, turn_player=1, to_play=1, public_chain=[], cards=[], lp0=8000, lp1=8000)


class MirrorTests(unittest.TestCase):
    def test_mirror_swaps_perspective_and_is_an_involution(self):
        s = json.loads(RECORD.read_text())['trace'][93]['snapshot']   # chain [(opponent) Rukina]
        m = M.mirror(s)
        own = lambda x: sorted(c['code'] for c in x['cards'] if c['controller'] == 0 and c['location'] == 4)
        opp = lambda x: sorted(c['code'] for c in x['cards'] if c['controller'] == 1 and c['location'] == 4)
        self.assertEqual(own(m), opp(s))
        self.assertEqual([c['player'] for c in m['public_chain']], [1 - c['player'] for c in s['public_chain']])
        self.assertEqual(m['actions'], s['actions'])
        self.assertEqual(M.mirror(m), s)


class OpponentTests(unittest.TestCase):
    def test_search_script_then_block_hands_over_to_the_fallback(self):
        calls = []
        opp = M.Opponent(M.SearchScript(), lambda obs, info: calls.append(obs) or 7, passive)
        self.assertEqual(opp.decide(snap('select_idlecmd', [act(0, 'SpSummon', M.RUK), act(1, 'None', phase='End')]), 'o', 'i', 1), (0, 'script'))
        blocked = snap('select_idlecmd', [act(0, 'Summon', 5), act(1, 'None', phase='End')])   # Activate Rukina not offered
        self.assertEqual(opp.decide(blocked, 'o', 'i', 2), (7, 'fallback-model'))
        self.assertEqual((opp.mode, opp.blocked['step'], opp.blocked['next_plan_steps']), ('fallback', 2, [dict(act='Activate', code=M.RUK)]))
        self.assertEqual(opp.decide(snap('select_chain', [act(0, 'Cancel')]), 'o2', 'i2', 3), (7, 'fallback-model'))
        self.assertEqual(calls, ['o', 'o2'])

    def test_completed_script_ends_the_turn_and_turn_three_is_passive(self):
        opp = M.Opponent(M.SearchScript(), lambda obs, info: 99, passive)
        opp.script.steps = []
        self.assertEqual(opp.decide(snap('select_idlecmd', [act(0, 'Summon', 5), act(1, 'None', phase='End')]), None, None, 1), (1, 'script-pass'))
        self.assertEqual(opp.decide(snap('select_chain', [act(0, 'Activate', 23434538), act(1, 'Cancel')], turn_count=3), None, None, 2), (1, 'turn3-pass'))

    def test_teacher_script_sees_the_mirrored_snapshot_and_raises_into_fallback(self):
        seen = []
        class Teacher:
            steps = [dict(act='SpSummon', code=42302563)]
            def __call__(self, s):
                seen.append(s)
                if s['message'] == 'select_idlecmd':
                    raise RuntimeError('Unresolved recovery witness')
                return 0
        opp = M.Opponent(M.TeacherScript(Teacher()), lambda obs, info: 3, passive)
        raw = snap('select_chain', [act(0, 'Cancel')]); raw['cards'] = [dict(code=1, controller=1, location=4)]
        self.assertEqual(opp.decide(raw, None, None, 1), (0, 'script'))
        self.assertEqual(seen[0]['cards'][0]['controller'], 0)
        self.assertEqual(opp.decide(snap('select_idlecmd', [act(0, 'Summon', 5)]), None, None, 2), (3, 'fallback-model'))
        self.assertEqual(opp.blocked['next_plan_steps'], [dict(act='SpSummon', code=42302563)])

    def test_stop_rule(self):
        self.assertFalse(M.finished(dict(turn_count=3)))
        self.assertTrue(M.finished(dict(turn_count=4)))


if __name__ == '__main__':
    unittest.main()
