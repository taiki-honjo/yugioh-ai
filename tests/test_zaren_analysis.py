"""Pure tests for the Zaren chain-condition analysis (no engine). Uses the P1 teacher record of
experiments/capability-probe-20261109 (train split, read only) plus synthetic edits."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path('/home/hatk8/projects/yugioh-ai')
spec = importlib.util.spec_from_file_location('zaren_analysis', P / 'experiments/zaren-chain-20261109/code/zaren_analysis.py')
Z = importlib.util.module_from_spec(spec); spec.loader.exec_module(Z)
RECORD = P / 'experiments/capability-probe-20261109/records/P1-cue-reco-jj-zaren-g00-p0.json'


def option(effect_index, index):
    return dict(index=index, act='Activate', code=Z.ZAREN, effect=Z.A.CARD_EFFECT_OFFSET + effect_index, spec='', place=0,
                position=0, number=0, phase='None', finish=False, response=0)


class ZarenAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(RECORD.read_text())

    def test_zaren_is_offered_only_from_chain_length_two(self):
        legal = Z.zaren_legality(self.r['trace'])
        self.assertEqual(legal[1]['zaren_legal_at_steps'], [])
        self.assertGreaterEqual(legal[1]['windows'], 1)
        self.assertEqual(legal[2]['zaren_legal_at_steps'], [95])

    def test_option_prompt_is_read_from_descriptions(self):
        trace = copy.deepcopy(self.r['trace'])
        row = trace[95]
        row['snapshot']['message'] = 'select_option'
        row['snapshot']['actions'] = [option(2, 0), option(3, 1)]
        row['selected'] = row['snapshot']['actions'][1]
        self.assertEqual(Z.zaren_option(trace), dict(step=95, offered=[2, 3], chosen=3))
        self.assertIsNone(Z.zaren_option(self.r['trace']))

    def test_outcome_reports_card_locations_and_opponent_turn_chain_only(self):
        o = Z.outcome(self.r['trace'], self.r['final'])
        self.assertEqual(o['search'], 'search_not_applied')
        self.assertEqual((o['rukina_locations_before'], o['rukina_locations_after']), ([4], [4]))
        self.assertEqual([c['code'] for c in o['longest_chain']], [Z.RUK, 14558127])
        end = copy.deepcopy(self.r['final'])
        next(c for c in end['cards'] if c['controller'] == 1 and c['code'] == Z.RUK)['location'] = Z.GRAVE
        self.assertEqual(Z.outcome(self.r['trace'], end)['rukina_locations_after'], [Z.GRAVE])


if __name__ == '__main__':
    unittest.main()
