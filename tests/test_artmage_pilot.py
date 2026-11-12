"""Offline tests for the Artmage learning pilot (no engine duel, no training). Imports the pinned engine library."""
import collections
import os
import sys
import unittest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np  # noqa: E402

ROOT = '/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110'
sys.path.insert(0, ROOT + '/code')
import pilot_data as D  # noqa: E402
import pilot_train as PT  # noqa: E402
import pilot_eval as EV  # noqa: E402
import pilot_model as M  # noqa: E402
import artmage_teacher as T  # noqa: E402

SEQS, STATS = D.build()


class DataTests(unittest.TestCase):
    def test_loss_counts_from_the_final_masks(self):
        self.assertEqual((STATS['old']['decisions'], STATS['old']['loss_steps']), (1077, 969))
        self.assertEqual((STATS['C1']['decisions'], STATS['C1']['loss_steps']), (113, 94))
        self.assertEqual((STATS['C2']['decisions'], STATS['C2']['loss_steps']), (100, 85))
        for name in D.SOURCES:
            self.assertNotIn(str(D.POSITION_MESSAGE), STATS[name]['messages_in_loss'])
            self.assertIn('11', STATS[name]['messages_in_loss'])  # select_place stays in the loss

    def test_batch_padding_has_no_loss_and_repeats_last_step(self):
        old = SEQS['old'][4]  # length 80
        b = D.batch([old, SEQS['C1'][0]])
        L = len(old['labels'])
        self.assertEqual(b['labels'].shape, (D.LENGTH, 2))
        self.assertTrue(np.all(b['loss'][L:, 0] == 0) and np.all(b['real'][L:, 0] == 0))
        self.assertTrue(np.array_equal(b['cards_'][L:, 0], np.repeat(old['cards_'][L - 1][None], D.LENGTH - L, 0)))

    def test_schedule_fourteen_one_one_per_sixteen(self):
        s = D.Schedule(30111001)
        for _ in range(8):
            s.next()
        self.assertEqual(s.counts(), {'old': 14, 'C1': 1, 'C2': 1})
        for _ in range(8 * 15):
            s.next()
        self.assertEqual(s.counts(), {'old': 224, 'C1': 16, 'C2': 16})
        s2 = D.Schedule(30111001)
        self.assertEqual([s2.next() for _ in range(8)], s.drawn[:16] and [s.drawn[i:i + 2] for i in range(0, 16, 2)])


class AdvantageTests(unittest.TestCase):
    def test_gae_terminal_and_bootstrap(self):
        adv, tgt = M.gae([0.5, 0.2], [0.0, 1.0], [False, True], bootstrap=99.0, gamma=1.0, lam=0.95)
        self.assertAlmostEqual(adv[1], 0.8, places=6)
        self.assertAlmostEqual(adv[0], (0.2 - 0.5) + 0.95 * 0.8, places=6)
        adv, tgt = M.gae([0.3], [0.0], [False], bootstrap=0.3)
        self.assertAlmostEqual(adv[0], 0.0, places=6)
        adv, _ = M.gae([0.1, 0.4, 0.2], [-1.0, 0.0, 0.0], [True, False, False], bootstrap=0.2)
        self.assertAlmostEqual(adv[0], -1.1, places=6)  # episode boundary cuts the trace


class ChunkTests(unittest.TestCase):
    def test_chunk_masks_padding_and_adapter_steps(self):
        s = SEQS['C1'][0]
        steps = []
        for t in range(5):
            steps.append(dict(obs={k: s[k][t] for k in M.OBS_KEYS}, chain=s['chain'][t], action=int(s['labels'][t]), logits=np.zeros(128, np.float32),
                              value=0.0, controlled=(t == 2), first=(t == 0), pre_state=(np.full((1, 512), t, np.float32), np.zeros((1, 512), np.float32)),
                              advantage=1.0, target=0.5))
        c = PT.chunk_arrays(steps, steps[0])
        self.assertEqual(c['mask'][:, 0].tolist()[:6], [1, 1, 0, 1, 1, 0])
        self.assertEqual(int(c['mask'].sum()), 4)
        self.assertTrue(c['first'][0, 0] and not c['first'][1:, 0].any())
        self.assertEqual(float(c['init_state'][0][0, 0]), 0.0)
        empty = PT.chunk_arrays([], steps[0])
        self.assertEqual(int(empty['mask'].sum()), 0)


class ObservationTests(unittest.TestCase):
    def test_privacy_and_own_hand_from_observations(self):
        s = SEQS['C1'][0]
        obs = {k: s[k][0:1] for k in M.OBS_KEYS}
        self.assertTrue(PT.privacy_ok(obs))
        self.assertEqual(PT.own_hand(obs), sorted(T.CONDITIONS['C1']['hand']))
        leak = {k: v.copy() for k, v in obs.items()}
        rows = np.where((leak['cards_'][0, :, 4] == 1) & (leak['cards_'][0, :, 2] == 2))[0]
        self.assertTrue(len(rows) > 0)
        leak['cards_'][0, rows[0], 1] = 7
        self.assertFalse(PT.privacy_ok(leak))

    def test_visibility_audit_classifies_and_flags_clear_exposure(self):
        # synthetic, not an engine observation: an opponent hand card made visible in a real C1 observation
        s = SEQS['C1'][0]
        t = int(np.where(s['message'] != D.POSITION_MESSAGE)[0][0])
        obs = {k: s[k][t:t + 1].copy() for k in M.OBS_KEYS}
        self.assertEqual(PT.visibility_events(obs), [])
        rows = np.where((obs['cards_'][0, :, 4] == 1) & (obs['cards_'][0, :, 2] == 2))[0]
        act_rows = np.where(obs['actions_'][0, :, 3] != 0)[0]
        a = obs['actions_'][0, act_rows[0]]
        obs['cards_'][0, rows[0], 0], obs['cards_'][0, rows[0], 1] = a[1], a[2]          # same id as a legal action
        obs['cards_'][0, rows[1], 0], obs['cards_'][0, rows[1], 1] = 0, 200              # id not among the actions
        shown = PT.visibility_events(obs)
        origins = sorted(c['origin'] for c in shown)
        self.assertEqual(origins, ['selection_prompt', 'unverified'] if a[1] * 256 + a[2] != 200 else origins)
        self.assertIsNotNone(PT.clear_exposure(shown, 0))                                 # before any action of the episode
        self.assertIsNone(PT.clear_exposure(shown, 3))
        self.assertIsNotNone(PT.clear_exposure([shown[0]] * 16, 3))                      # more than 15 at once
        self.assertFalse(PT.privacy_ok(obs))

    def test_position_adapter_keeps_non_position_actions(self):
        import artmage_entry as E
        s = SEQS['C1'][0]
        module = E._adapter()
        policy = module.PositionPolicy(PT.Preset(), E._IDS)
        for t in range(len(s['labels'])):
            if s['message'][t] == D.POSITION_MESSAGE:
                continue
            obs = {k: s[k][t:t + 1] for k in M.OBS_KEYS}
            policy.base.raw = int(s['labels'][t])
            info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]))
            self.assertEqual(policy.act(obs, info), int(s['labels'][t]))

    def test_episode_seeds_are_deterministic_and_distinct(self):
        a, b = PT.episode_seeds(10111001, 200), PT.episode_seeds(10111001, 200)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 200)


class VariantTests(unittest.TestCase):
    def test_variant_hands_are_legal_and_new(self):
        recipe = collections.Counter(int(x) for x in open('/home/hatk8/projects/yugioh-ai/experiments/artmage-parallel-20261110/decks/Artmage.ydk').read().split('!side')[0].split('#extra')[0].split() if x.isdigit())
        for name, spec in EV.VARIANTS.items():
            self.assertEqual(len(spec['hand']), 5)
            for code, n in collections.Counter(spec['hand']).items():
                self.assertLessEqual(n, recipe[code], (name, code))
            self.assertNotIn(tuple(sorted(spec['hand'])), {tuple(sorted(c['hand'])) for c in T.CONDITIONS.values()})

    def test_variant_medius_cost_rule_uses_variant_fillers(self):
        def card(code, loc=2):
            return dict(code=code, location=loc, sequence=0, controller=0, position=1)
        hand = [card(T.USAGI), card(T.PSY_GAMMA), card(T.PSY_DRIVER)]
        t = EV.VariantTeacher([T.mark(T.a('Activate', T.FINEMELT), 'before_draw'), T.rule('medius_cost', optional=True)], EV.VARIANTS['C1v']['filler_priority'])
        snap = lambda msg, acts, cards: dict(message=msg, actions=acts, cards=cards, turn_count=1, turn_player=0, phase=4, to_play=0, public_chain=[])
        act = lambda i, a, c=0, **k: dict(dict(index=i, act=a, code=c, effect=-1, spec='', place=0, position=0, phase='', finish=False), **k)
        t(snap('select_yesno', [act(0, 'Activate', T.FINEMELT), act(1, 'Cancel')], hand))
        cands = [act(0, 'None', T.USAGI, spec='h1'), act(1, 'None', T.PSY_GAMMA, spec='h2'), act(2, 'None', T.PSY_DRIVER, spec='h3'), act(3, 'None', T.BEAST, spec='h4')]
        self.assertEqual(t(snap('select_card', cands, hand + [card(T.BEAST)])), 2)


if __name__ == '__main__':
    unittest.main()
