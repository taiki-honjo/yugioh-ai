"""Offline tests for the Artmage basic-opening batch (no engine duel, no training)."""
import collections
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np  # noqa: E402

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-basic-openings-batch-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import batch_plans as BP  # noqa: E402
import batch_data as BD  # noqa: E402
import batch_model as BM  # noqa: E402
import pilot_data as PD  # noqa: E402
import artmage_teacher as T  # noqa: E402


def act(index, a, code=0, **extra):
    return dict(dict(index=index, act=a, code=code, effect=-1, spec='', place=0, position=0, phase='', finish=False), **extra)


def snap(message, actions, cards=(), turn=1, turn_player=0, phase=4):
    return dict(message=message, actions=list(actions), cards=list(cards), turn_count=turn, turn_player=turn_player, phase=phase,
                to_play=0, public_chain=[])


class TeacherTests(unittest.TestCase):
    def test_phase_constraint(self):
        t = BP.BatchTeacher([BP.opp(BP.phase(BP.a('Activate', BP.MAIDEN), 1))])
        opts = [act(0, 'Activate', BP.MAIDEN, effect=10011), act(1, 'Cancel')]
        self.assertEqual(t(snap('select_chain', opts, turn=2, turn_player=1, phase=2)), 1)   # standby: not matched
        self.assertEqual(t(snap('select_chain', opts, turn=2, turn_player=1, phase=1)), 0)   # draw phase: matched

    def test_identical_copies_resolve_to_first(self):
        t = BP.BatchTeacher([BP.a('Activate', BP.GRAPH)])
        opts = [act(0, 'None', BP.INFINITE, spec='h1'), act(1, 'None', BP.INFINITE, spec='h2')]
        self.assertEqual(t(snap('select_card', opts)), 0)
        self.assertEqual(t.log[-1]['why']['kind'], 'identical-copies')
        mixed = [act(0, 'None', BP.FINEMELT, spec='m1'), act(1, 'None', BP.FINEMELT, spec='h1')]
        with self.assertRaises(T.Unresolved):
            BP.BatchTeacher([BP.a('Activate', BP.GRAPH)])(snap('select_card', mixed))

    def test_three_trigger_group_lifo(self):
        t = BP.BatchTeacher([g for g in BP.F6_PLAN if '_group' in g])
        offered = [act(0, 'Activate', BP.NERVED, effect=10012), act(1, 'Activate', BP.MAIDEN, effect=10010), act(2, 'Activate', BP.BEAST, effect=10011), act(3, 'Cancel')]
        self.assertEqual(t(snap('select_chain', offered)), 2)   # 獄神獣 listed first
        self.assertEqual(t(snap('select_chain', offered[:2] + [offered[3]])), 1)
        self.assertEqual(t(snap('select_chain', [offered[0], offered[3]])), 0)
        self.assertEqual([p['code'] for p in t.pending], [BP.FINEMELT, BP.FIELD, BP.ASSAULT])

    def test_medius_cost_uses_batch_priority(self):
        hand = [dict(code=c, location=2, sequence=0, controller=0, position=1) for c in [BP.ASH, BP.WARASHI, BP.DRIVER]]
        t = BP.BatchTeacher([BP.mark(BP.a('Activate', BP.FINEMELT), 'before_draw'), BP.rule('medius_cost', optional=True)])
        t(snap('select_yesno', [act(0, 'Activate', BP.FINEMELT), act(1, 'Cancel')], hand))
        cands = [act(0, 'None', BP.ASH, spec='h1'), act(1, 'None', BP.WARASHI, spec='h2'), act(2, 'None', BP.DRIVER, spec='h3'), act(3, 'None', BP.BEAST, spec='h4')]
        drawn = hand + [dict(code=BP.BEAST, location=2, sequence=0, controller=0, position=1)]
        self.assertEqual(t(snap('select_card', cands, drawn)), 2)


class PlanTests(unittest.TestCase):
    def test_plans_use_recipe_cards_and_f2_is_c1_from_the_summon(self):
        main, sec = collections.Counter(), None
        for line in open('/home/hatk8/projects/yugioh-ai/experiments/artmage-parallel-20261110/decks/Artmage.ydk'):
            line = line.strip()
            if line in ('#main', '#extra', '!side'):
                sec = line
            elif line.isdigit():
                main[(sec, int(line))] += 1
        codes = {c for (_, c) in main} | {0}
        for name, c in BP.CONDITIONS.items():
            if isinstance(c['plan'], str):
                continue
            for e in c['plan']:
                steps = [m['step'] for m in e['_group']] + [p for m in e['_group'] for p in m['picks']] if '_group' in e else [e]
                for s in steps:
                    if 'code' in s:
                        self.assertIn(s['code'], codes, (name, s))
        self.assertEqual(BP.F2_PLAN, T.CONDITIONS['C1']['plan'][3:])

    def test_hands_distinct_and_new(self):
        keys = [BP.hand_key(x['hand']) for x in list(BP.TRAIN.values()) + list(BP.VALIDATION.values())]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(BP.hand_key(T.CONDITIONS['C1']['hand']), keys)
        for h in BP.PAST_EVALUATION_HANDS.values():
            self.assertNotIn(BP.hand_key(h), keys)


class DataTests(unittest.TestCase):
    def test_group_sampler_is_uniform_over_groups(self):
        seqs = {'F0': [0] * 12, 'F1': [0, 0], 'F6': [0]}
        s = BD.GroupSampler(seqs, 30121001)
        for _ in range(3000):
            s.next()
        c = s.counts()
        for g in seqs:
            self.assertAlmostEqual(c[g] / 6000, 1 / 3, delta=0.03)
        s2 = BD.GroupSampler(seqs, 30121001)
        self.assertEqual([s2.next() for _ in range(5)], [s.drawn[i:i + 2] for i in range(0, 10, 2)])

    def test_load_new_uses_the_pilot_mask_rule(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / 'demos/X').mkdir(parents=True)
            for f in ['observations.npz', 'record.json']:
                shutil.copy(PD.DEMO_DIR / 'C1' / f, tmp / 'demos/X' / f)
            old_root = BD.ROOT
            BD.ROOT = tmp
            try:
                new = BD.load_new('X')
            finally:
                BD.ROOT = old_root
            ref, _ = PD.load_demo('C1')
            self.assertTrue(np.array_equal(new['loss'], ref['loss']))
            self.assertEqual(int(new['loss'].sum()), 94)
        finally:
            shutil.rmtree(tmp)

    def test_conflict_detector_needs_same_history(self):
        import batch_train as BT
        base = {k: np.zeros((3, 2), np.uint8) for k in PD.OBS_KEYS}
        base['chain'] = np.zeros((3, 4), np.int32)
        a = dict(base, labels=np.array([0, 1, 0]), loss=np.ones(3, np.float32))
        b = dict(base, labels=np.array([0, 1, 1]), loss=np.ones(3, np.float32))     # same history, last label differs
        c = dict({k: v.copy() for k, v in base.items()}, labels=np.array([1, 1, 0]), loss=np.ones(3, np.float32))
        c['cards_'][0, 0] = 9                                                          # different history
        out = BT.conflicts([a, b], ['a', 'b'])
        self.assertGreaterEqual(out['full_history_conflicts'], 1)
        out2 = BT.conflicts([a, c], ['a', 'c'])
        self.assertEqual(out2['full_history_conflicts'], 0)


class LossTests(unittest.TestCase):
    def test_sequence_loss_weights_each_sequence_equally(self):
        import jax.numpy as jnp
        T_, B, A = 4, 2, 3
        logits = np.zeros((T_, B, A), np.float32)
        logits[:, 0, 0] = 5.0      # sequence 0: confident on label 0
        orig = BM.PM.unroll
        BM.PM.unroll = lambda *a, **k: (jnp.asarray(logits), None)
        try:
            batch = dict(labels=jnp.zeros((T_, B), jnp.int32), loss=jnp.asarray(np.array([[1, 1], [0, 1], [0, 1], [0, 1]], np.float32)))
            loss, aux = BM.sequence_loss(None, None, batch)
        finally:
            BM.PM.unroll = orig
        ce0 = -np.log(np.exp(5) / (np.exp(5) + 2))
        ce1 = np.log(3)
        self.assertAlmostEqual(float(loss), (ce0 + ce1) / 2, places=5)    # not (1*ce0 + 4*ce1) / 5


if __name__ == '__main__':
    unittest.main()
