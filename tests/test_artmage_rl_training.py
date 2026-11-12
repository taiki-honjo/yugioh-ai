"""Offline tests for the Artmage RL training (no engine duel): value loss, episode-end rewards, windows/bootstrap,
adapter-owned steps, timeouts / step limits, seeds, rehearsal wiring, battle classification and pair summary."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np  # noqa: E402

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
sys.path.insert(0, str(ROOT / 'code'))
import rlt_train as RT  # noqa: E402


# ------------------------------------------------------------------ fake engine / model for collect()
class FakeEnv:
    """script: list of (to_play, controlled) per decision; end = (reward, term, trunc, termination_reason) after the last."""
    def __init__(self, script, end):
        self.script, self.end, self.i = script, end, 0

    def info(self, tp, tr=0):
        return {'to_play': np.array([tp]), 'num_options': np.array([1]), 'termination_reason': np.array([tr])}


class FakeHooks:
    def __init__(self, episodes, values=None, boot=0.5):
        self.episodes, self.e = episodes, -1
        self.updates, self.boot_calls, self.values, self.boot = [], 0, values or {}, boot
        self.step_count = 0

    def make_env(self, seed):
        self.e += 1
        script, end = self.episodes[self.e % len(self.episodes)]
        env = FakeEnv(script, end)
        return env, {'i': 0}, env.info(script[0][0])

    def step(self, env, idx):
        env.i += 1
        self.step_count += 1
        if env.i >= len(env.script):
            r, term, trunc, tr = env.end
            return {'i': env.i}, np.array([r]), np.array([term]), np.array([trunc]), env.info(0, tr)
        return {'i': env.i}, np.array([0.0]), np.array([False]), np.array([False]), env.info(env.script[env.i][0])

    def close_env(self, env):
        pass

    def init_state(self):
        return ('zero',)

    def infer(self, role, state, obs, info):
        return ('s',), np.zeros(4, np.float32), self.values.get(self.step_count, 0.0)

    def sample(self, logits):
        return 0

    def bootstrap(self, state, obs, info):
        self.boot_calls += 1
        return self.boot

    def reset_adapters(self):
        pass

    def position_final(self, seat, raw, obs, info):
        return raw

    def message_type(self, obs, idx):
        env_script = self.episodes[self.e % len(self.episodes)][0]
        return 5 if env_script[obs['i']][1] else 3

    def card_code(self, obs, idx):
        return 0

    def step_row(self, obs, info):
        return dict(obs=None, chain=None, env_i=obs['i'])

    def audit(self, obs, info, n):
        return [], None

    def describe(self, obs, info):
        return {}

    def update(self, steps):
        self.updates.append([dict(episode=s['episode'], env_i=s['env_i'], reward=s['reward'], done=s['done'], controlled=s['controlled'],
                                  advantage=s['advantage'], target=s['target'], value=s['value']) for s in steps])
        return [dict(ppo_steps=sum(not s['controlled'] for s in steps), rehearsal=['F0:0', 'F0:1'], reh_steps=10, reh_context_steps=12)]

    def window_extra(self):
        return {}

    def episode_extra(self):
        return {}

    def stop_requested(self):
        return False

    def elapsed(self):
        return 0.0


def run(episodes, **kw):
    H = FakeHooks(episodes, **{k: kw.pop(k) for k in ['values', 'boot'] if k in kw})
    with tempfile.TemporaryDirectory() as d:
        stats = RT.collect(H, Path(d), seeds=list(range(1, 300)), **kw)
        eps = [json.loads(l) for l in (Path(d) / 'episodes.jsonl').read_text().splitlines()]
        wins = [json.loads(l) for l in (Path(d) / 'windows.jsonl').read_text().splitlines()]
    return H, stats, eps, wins


class EndTests(unittest.TestCase):
    def test_reward_sign_for_both_seats_and_both_actors(self):
        # r is the reward of the actor (native: winner == player ? 1 : -1)
        for seat in (0, 1):
            self.assertEqual(RT.classify_end(1.0, True, False, 0, actor=seat, learner_seat=seat)['learner_reward'], 1.0)
            self.assertEqual(RT.classify_end(-1.0, True, False, 0, actor=seat, learner_seat=seat)['learner_reward'], -1.0)
            # the frozen side made the last decision (game ends on the opponent's move)
            self.assertEqual(RT.classify_end(1.0, True, False, 0, actor=1 - seat, learner_seat=seat)['learner_reward'], -1.0)
            self.assertEqual(RT.classify_end(-1.0, True, False, 0, actor=1 - seat, learner_seat=seat)['learner_reward'], 1.0)
        self.assertEqual(RT.classify_end(1.0, True, False, 0, actor=1, learner_seat=0)['winner'], 1)

    def test_step_limit_and_timeout_are_not_natural(self):
        e = RT.classify_end(0.0, True, True, 1, actor=0, learner_seat=0)
        self.assertEqual((e['kind'], e['learner_reward']), ('step_limit', 0.0))
        e = RT.classify_end(0.0, True, False, 1, actor=0, learner_seat=0)
        self.assertEqual(e['kind'], 'step_limit')
        e = RT.classify_end(0.0, True, False, 2, actor=1, learner_seat=0)
        self.assertEqual((e['kind'], e['learner_reward']), ('timeout', None))
        e = RT.classify_end(0.0, True, False, 0, actor=1, learner_seat=0)
        self.assertEqual((e['kind'], e['learner_reward']), ('natural', 0.0))
        with self.assertRaises(RuntimeError):
            RT.classify_end(1.0, False, False, 0, actor=0, learner_seat=0)


class CollectTests(unittest.TestCase):
    def test_game_ending_on_the_frozen_move_rewards_the_learner_last_step(self):
        # learner seat 0: L F L F(ends, frozen wins: r=+1 for actor 1)
        ep = ([(0, False), (1, False), (0, False), (1, False)], (1.0, True, False, 0))
        H, stats, eps, wins = run([ep], cap=4, episode_cap=1, window_size=100)
        steps = H.updates[0]
        self.assertEqual([s['reward'] for s in steps], [0.0, -1.0])
        self.assertEqual([s['done'] for s in steps], [False, True])
        self.assertEqual(eps[0]['end']['kind'], 'natural')
        self.assertEqual(eps[0]['end']['learner_reward'], -1.0)
        self.assertEqual(wins[0]['bootstrap'], 'terminal')
        self.assertEqual(stats['counts']['natural_loss'], 1)

    def test_seat_alternates_and_learner_seat_1_wins_on_its_own_move(self):
        ep0 = ([(0, False), (1, False)], (1.0, True, False, 0))  # actor 1 wins
        H, stats, eps, wins = run([ep0], cap=100, episode_cap=2, window_size=1, first_seat=0)
        self.assertEqual([e['learner_seat'] for e in eps], [0, 1])
        self.assertEqual(eps[0]['end']['learner_reward'], -1.0)  # learner 0, actor 1 won
        self.assertEqual(eps[1]['end']['learner_reward'], 1.0)   # learner 1 = actor, won
        rewards = [s['reward'] for u in H.updates for s in u]
        self.assertEqual(rewards, [-1.0, 1.0])

    def test_exact_bootstrap_when_the_learner_moves_next(self):
        # window_size 2: after L F the learner is next -> exact bootstrap V=0.5; then L F ends (learner wins as actor 0? no: actor 1, r=-1 -> learner +1)
        ep = ([(0, False), (1, False), (0, False), (1, False)], (-1.0, True, False, 0))
        H, stats, eps, wins = run([ep], cap=100, episode_cap=1, window_size=2, boot=0.5)
        self.assertEqual([w['bootstrap'] for w in wins], ['exact', 'terminal'])
        self.assertEqual(H.boot_calls, 1)
        first = H.updates[0]
        self.assertEqual(len(first), 1)
        self.assertAlmostEqual(first[0]['target'], 0.5)  # r 0 + V_boot (values are 0)
        self.assertEqual(H.updates[1][0]['reward'], 1.0)

    def test_cap_reached_while_the_frozen_side_moves_drops_the_unfinished_tail(self):
        # episode 0 (learner seat 0): L F, ends. Episode 1 (learner seat 1): L L | cap 4 reached, the frozen side (0) is next
        short = ([(0, False), (1, False)], (1.0, True, False, 0))
        ep = ([(1, False), (1, False), (0, False), (1, False)], (1.0, True, False, 0))
        H, stats, eps, wins = run([short, ep], cap=4, episode_cap=10, window_size=100)
        self.assertEqual(wins[-1]['bootstrap'], 'cut_dropped')
        self.assertEqual(wins[-1]['dropped'], 2)  # the two learner steps of the unfinished episode
        self.assertEqual([s['episode'] for s in H.updates[-1]], [0])
        self.assertEqual(stats['counts']['dropped_cut'], 2)
        self.assertEqual(eps[-1]['end'], None)
        self.assertEqual(eps[-1]['cut_by'], 'decision_cap')
        self.assertEqual(stats['total'], 4)

    def test_cap_reached_with_the_learner_next_bootstraps_exactly(self):
        ep = ([(0, False), (1, False), (0, False), (1, False)], (1.0, True, False, 0))
        H, stats, eps, wins = run([ep], cap=2, episode_cap=10, window_size=100, boot=0.25)
        self.assertEqual(wins[-1]['bootstrap'], 'exact')
        self.assertAlmostEqual(H.updates[-1][0]['target'], 0.25)
        self.assertEqual(stats['stop_reason'], 'decision_cap')

    def test_adapter_owned_steps_stay_in_history_but_not_in_the_loss(self):
        ep = ([(0, False), (0, True), (0, False), (1, False)], (1.0, True, False, 0))
        H, stats, eps, wins = run([ep], cap=100, episode_cap=1, window_size=100)
        steps = H.updates[0]
        self.assertEqual([s['controlled'] for s in steps], [False, True, False])
        self.assertEqual(wins[0]['loss_steps'], 2)
        dummy = dict(obs={k: np.zeros(3) for k in ['a']}, chain=np.zeros(4), action=0, logits=np.zeros(4), controlled=True, first=True,
                     pre_state=(np.zeros((1, 2)),))
        rows = [dict(dummy, controlled=c, first=i == 0, advantage=1.0, target=0.0) for i, c in enumerate([False, True, False])]
        chunk = RT.chunk_arrays(rows, dummy, 4, ['a'])
        self.assertEqual(chunk['mask'][:, 0].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(chunk['first'][:, 0].tolist(), [True, False, False, False])

    def test_timeout_drops_the_episode_steps_of_the_open_window_and_invents_no_outcome(self):
        ep = ([(0, False), (1, False), (0, False), (1, False)], (0.0, True, False, 2))
        ok = ([(0, False), (1, False)], (-1.0, True, False, 0))
        H, stats, eps, wins = run([ok, ep, ok], cap=100, episode_cap=3, window_size=100)
        self.assertEqual(eps[1]['end']['kind'], 'timeout')
        self.assertEqual(eps[1]['end']['dropped_learner_steps'], 2)
        self.assertEqual(stats['counts']['end_timeout'], 1)
        used = H.updates[-1]
        self.assertEqual([s['episode'] for s in used], [0, 2])
        self.assertTrue(all(s['done'] for s in used))

    def test_step_limit_gives_zero_and_is_counted_separately(self):
        ep = ([(0, False), (1, False)], (0.0, True, True, 1))
        H, stats, eps, wins = run([ep], cap=100, episode_cap=1, window_size=100)
        self.assertEqual(eps[0]['end']['kind'], 'step_limit')
        self.assertEqual(H.updates[0][0]['reward'], 0.0)
        self.assertTrue(H.updates[0][0]['done'])
        self.assertEqual(stats['counts']['end_step_limit'], 1)
        self.assertNotIn('natural_draw', stats['counts'])

    def test_last_allowed_episode_flushes_a_short_window(self):
        ep = ([(0, False), (1, False)], (-1.0, True, False, 0))
        H, stats, eps, wins = run([ep], cap=100, episode_cap=3, window_size=100)
        self.assertEqual(stats['stop_reason'], 'episode_cap')
        self.assertEqual(len(H.updates), 1)
        self.assertEqual(len(H.updates[0]), 3)
        self.assertEqual(stats['episodes'], 3)

    def test_engine_error_stops_without_a_crash_and_is_recorded(self):
        class Broken(FakeHooks):
            def step(self, env, idx):
                if self.step_count == 3:
                    raise RuntimeError('native failure')
                return super().step(env, idx)
        ep = ([(0, False), (1, False)], (1.0, True, False, 0))
        H = Broken([ep])
        with tempfile.TemporaryDirectory() as d:
            stats = RT.collect(H, Path(d), cap=100, episode_cap=10, window_size=100, seeds=list(range(1, 300)))
            eps = [json.loads(l) for l in (Path(d) / 'episodes.jsonl').read_text().splitlines()]
        self.assertEqual(stats['stop_reason'], 'error')
        self.assertIn('native failure', stats['error'])
        self.assertEqual(eps[-1]['cut_by'], 'error')
        self.assertEqual(len(eps), 2)

    def test_gae_matches_hand_computation(self):
        adv, tgt = RT.gae([0.1, 0.2, 0.3], [0.0, 0.0, 1.0], [False, False, True], 0.0, 1.0, 0.95)
        d2 = 1.0 - 0.3
        d1 = 0.0 + 0.3 - 0.2
        d0 = 0.0 + 0.2 - 0.1
        a2, a1 = d2, d1 + 0.95 * d2
        a0 = d0 + 0.95 * a1
        np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-6)
        np.testing.assert_allclose(tgt, np.array([a0, a1, a2]) + [0.1, 0.2, 0.3], rtol=1e-6)


class SeedTests(unittest.TestCase):
    def test_seed_lists(self):
        tr = RT.training_seeds()
        ev = RT.evaluation_seeds(tr)
        self.assertEqual(len(tr), 256)
        self.assertEqual(len(ev), 64)
        self.assertEqual(len(set(ev)), 64)
        self.assertFalse(set(ev) & set(tr))
        self.assertEqual(RT.evaluation_seeds(RT.training_seeds()), ev)
        self.assertEqual(tr, RT.episode_seeds(10111201, 256))
        if (ROOT / 'data/seeds.json').exists():
            reg = json.loads((ROOT / 'data/seeds.json').read_text())
            self.assertEqual(reg['training'], tr)
            self.assertEqual(reg['evaluation'], ev)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        jax.config.update('jax_default_matmul_precision', 'highest')
        import artmage_entry  # noqa: F401  (pinned model code on sys.path)
        import rlt_model as M
        assert (M.CONFIG['gamma'], M.CONFIG['gae_lambda']) == (RT.GAMMA, RT.GAE_LAMBDA)
        cls.M, cls.jax = M, jax

    def test_value_loss_has_a_gradient_for_large_errors(self):
        import jax.numpy as jnp
        g = self.jax.grad(lambda v: jnp.sum(self.M.value_loss(v, jnp.array([0.0, 0.0]))))(jnp.array([0.5, 5.0]))
        np.testing.assert_allclose(np.asarray(g), [0.5, 5.0])
        old = self.jax.grad(lambda v: jnp.sum(jnp.minimum(0.5 * (v - 0.0) ** 2, 1.0)))(jnp.array([0.5, 5.0]))
        np.testing.assert_allclose(np.asarray(old), [0.5, 0.0])  # the pilot's cap: zero gradient at error 5
        src = (ROOT / 'code/rlt_model.py').read_text()
        self.assertNotIn("CONFIG['vloss_clip']", src)
        self.assertNotIn('jnp.minimum(0.5', src)
        self.assertIn("v = value_loss(values, chunk['targets'])", src)

    def test_real_model_value_gradient_and_one_update_step(self):
        """Real start model on CPU: with a target far from V, the value term still moves the parameters toward the
        target; one update_step with a 40-sequence rehearsal batch runs and logs 9 sources."""
        import jax.numpy as jnp
        M, jax = self.M, self.jax
        params, bstats, _ = M.load_pair(*RT.START)
        import hb_data as HD
        existing, names, pairs, stats = HD.load()
        length = max([len(s['labels']) for g in existing for s in existing[g]] + [len(s['labels']) for g in pairs for p in pairs[g] for _, _, s in p['hands']])
        self.assertEqual(length, 115)
        self.assertEqual(sum(len(v) for v in existing.values()) + sum(len(p['hands']) for g in pairs for p in pairs[g]), 40)
        seq = existing['F0'][0]
        T = 4
        steps = []
        state = M.init_state(1)
        for t in range(T):
            o = {k: jnp.asarray(seq[k][t:t + 1]) for k in M.OBS_KEYS}
            new_state, logits, value = M.infer(params, bstats, state, o, jnp.asarray(seq['chain'][t:t + 1]))
            steps.append(dict(obs={k: seq[k][t] for k in M.OBS_KEYS}, chain=seq['chain'][t], action=int(seq['labels'][t]),
                              logits=np.asarray(logits)[0], value=float(np.asarray(value)[0]), controlled=False, first=t == 0,
                              pre_state=tuple(np.asarray(x) for x in state), advantage=0.0))
            state = new_state
        values = np.array([s['value'] for s in steps])
        for far in (50.0, 60.0):
            for s, v in zip(steps, values):
                s['target'] = float(v + far)
        chunk = RT.chunk_arrays(steps, steps[0], M.CONFIG['chunk_length'], M.OBS_KEYS)
        chunk = jax.tree.map(jnp.asarray, chunk)

        def vterm(p, c):
            return M.ppo_loss(p, bstats, c)[1]['v']
        g = jax.grad(lambda p: M.ppo_loss(p, bstats, chunk)[0])(params)
        self.assertGreater(float(M.tree_norm(g['base'])), 0.0)
        # gradient step on the value term alone lowers |V - target| although the error (~50) is far above sqrt(2)
        gv = jax.grad(lambda p: vterm(p, chunk))(params)
        moved = jax.tree.map(lambda a, b: a - 1e-4 * b, params, gv)
        self.assertLess(float(vterm(moved, chunk)), float(vterm(params, chunk)))
        # update_step with rehearsal (sources = 9)
        sampler = HD.BlockSampler(existing, pairs, 30111201)
        items = [sampler.next() for _ in range(4)]
        self.assertEqual(sorted(k for k, _ in items), ['existing', 'existing', 'pair', 'pair'])
        reh, labels = HD.make_batch(existing, pairs, items[0], length)
        reh['source'] = np.array(RT.source_index(items[0]), np.int32)
        opt = M.make_optimizer()
        new_params, _, metrics = M.update_step(params, opt.init(params), bstats, chunk, jax.tree.map(jnp.asarray, reh), opt, len(RT.SOURCES))
        self.assertEqual(np.asarray(metrics['per_source']).shape, (9, 3))
        self.assertGreater(float(metrics['base_update']), 0.0)
        self.assertGreater(float(metrics['head_update']), 0.0)
        self.assertTrue(np.isfinite(float(metrics['loss'])))
        self.assertAlmostEqual(float(np.asarray(metrics['per_source'])[:, 0].sum()), float(metrics['reh_steps']), places=3)

    def test_source_index(self):
        self.assertEqual(RT.source_index(('existing', [('F0', 3), ('F6', 1)])), [0, 6])
        self.assertEqual(RT.source_index(('pair', ('B', 2))), [8, 8])


class RealHooksTests(unittest.TestCase):
    """The real hooks without any duel (CPU): inference/sampling/bootstrap on a stored observation, one real window
    update through the BlockSampler, save + reload through the isolated entry, rehearsal fit, audit."""

    def test_real_hooks_offline(self):
        H = RT.RealHooks(dict(training=dict(seed=RT.TRAIN_SEED, rehearsal_seed=RT.REHEARSAL_SEED)))
        np_ = H.np
        self.assertEqual(H.length, 115)
        seq = H.existing['F1'][0]

        def obs_info(t):
            obs = {k: seq[k][t:t + 1] for k in H.M.OBS_KEYS}
            obs['mask_'] = None
            ch = seq['chain'][t]
            info = dict(num_options=np_.array([int(np_.sum(seq['actions_'][t, :, 3] != 0))]), to_play=np_.zeros(1, np_.int32),
                        public_chain_length=np_.array([ch[0]]), public_chain_player=np_.array([1 if ch[1] == 2 else 0]),
                        public_chain_card_id=np_.array([ch[2]]), public_chain_effect=np_.array([ch[3]]))
            return obs, info
        state = H.init_state()
        steps = []
        H.reset_adapters()
        for t in range(40):
            obs, info = obs_info(t)
            new_state, logits, value = H.infer('learner', state, obs, info)
            raw = H.sample(logits)
            self.assertTrue(0 <= raw < int(info['num_options'][0]))
            final = H.position_final(0, raw, obs, info)
            controlled = H.message_type(obs, raw) == RT.POSITION_MESSAGE
            self.assertTrue(final == raw or controlled)
            row = dict(H.step_row(obs, info), action=raw, logits=logits, value=value, controlled=controlled, first=t == 0, pre_state=state,
                       reward=0.0, done=False, episode=0)
            np_.testing.assert_array_equal(row['chain'], seq['chain'][t])
            steps.append(row)
            state = new_state
        self.assertIsInstance(H.bootstrap(state, *obs_info(40)), float)
        _, frozen_logits, _ = H.infer('frozen', H.init_state(), *obs_info(0))
        steps[-1].update(reward=1.0, done=True)
        kept, binfo = RT.close_window(steps, None, False, lambda: 0.0)
        self.assertEqual(binfo['bootstrap'], 'terminal')
        metrics = H.update(kept)
        self.assertEqual(len(metrics), 4)
        self.assertEqual(sum(m['chunk_steps'] for m in metrics), 40)
        self.assertEqual(sorted(m['rehearsal_kind'] for m in metrics), ['existing', 'existing', 'pair', 'pair'])
        self.assertTrue(all(m['reh_context_steps'] > 0 and m['reh_steps'] > 0 for m in metrics))
        self.assertEqual(H.window_extra()['total_notfinite'], 0)
        # frozen side unchanged, learner changed
        _, frozen_after, _ = H.infer('frozen', H.init_state(), *obs_info(0))
        np_.testing.assert_array_equal(frozen_logits, frozen_after)
        _, learner_after, _ = H.infer('learner', H.init_state(), *obs_info(0))
        self.assertGreater(float(np_.abs(learner_after - frozen_after).max()), 0.0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'candidate').mkdir()
            H.M.save_pair(H.params, H.original_base, out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
            # only 2 sequences for speed: same code path as the 40-sequence check
            ex, pairs = H.existing, H.pairs
            H.existing, H.pairs = {'F1': ex['F1'][:1]}, {'A': [dict(pair='x', hands=pairs['A'][0]['hands'][:1])]}
            try:
                chk = RT.saved_checks(H, out)
                fit = RT.rehearsal_fit(H, H.params)
            finally:
                H.existing, H.pairs = ex, pairs
        self.assertEqual(chk['sequences_checked'], 2)
        self.assertEqual(chk['max_abs_logit_diff_memory_vs_saved'], 0.0)
        self.assertTrue(chk['argmax_equal'] and chk['adapter_final_actions_equal'] and chk['both_parts_changed'])
        self.assertEqual(set(fit), {'F1', 'pairs-A'})
        # audit: an opponent hand identity equal to a legal action is 'selection_prompt'; before any action it stops
        obs, info = obs_info(5)
        cards = obs['cards_'].copy()
        act = obs['actions_'][0][obs['actions_'][0][:, 3] != 0][0]
        cards[0, 0, :] = 0
        cards[0, 0, 0], cards[0, 0, 1], cards[0, 0, 2], cards[0, 0, 4] = act[1], act[2], RT.HAND, 1
        shown, reason = H.audit(dict(obs, cards_=cards), info, 3)
        self.assertEqual([c['origin'] for c in shown], ['selection_prompt'])
        self.assertIsNone(reason)
        self.assertIsNotNone(H.audit(dict(obs, cards_=cards), info, 0)[1])


class PromotionTests(unittest.TestCase):
    def test_record_checks_on_the_training_record(self):
        if not (ROOT / 'train/result.json').exists():
            self.skipTest('no training record')
        import rlt_promote as PR
        checks, res, tb = PR.record_checks()
        self.assertTrue(all(checks.values()), checks)
        self.assertIn('close_env', tb)

    def test_close_env_bug_is_the_reference_order(self):
        src = (ROOT / 'code/rlt_train.py').read_text()
        body = src[src.index('    def close_env(self, env):'):src.index('    # model')]
        self.assertIn("assert self.v.native.get_env_resource_stats()['live_duels'] == 0", body)
        loop = src[src.index('                try:\n                    H.close_env(env)'):]
        self.assertLess(loop.index('H.close_env(env)'), loop.index('del env'))  # the env was still referenced when checked

    def test_eval_uses_the_promoted_pair_through_the_hash_checked_lookup(self):
        src = (ROOT / 'code/rlt_eval.py').read_text()
        self.assertIn("HR.models = lambda which: (pair, want) if which == 'candidate' else original(which)", src)
        self.assertIn("promo['candidate']['base_sha256']", src)


class BattleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rlt_eval as EV
        cls.EV = EV
        cls.ev = EV.evaluate_module()

    def test_classify_game(self):
        c = self.EV.classify_game(1.0, True, False, 0, 1, actor=1, candidate_seat=0, max_step_wall=0.1)
        self.assertEqual((c['outcome'], c['reason_label'], c['abnormal']), ('loss', 'normal_lp0', False))
        c = self.EV.classify_game(-1.0, True, False, 0, -1, actor=1, candidate_seat=0, max_step_wall=0.1)
        self.assertEqual((c['outcome'], c['reason_label']), ('win', 'normal_deckout_or_special'))
        c = self.EV.classify_game(0.0, True, True, 1, 1, actor=0, candidate_seat=0, max_step_wall=0.1)
        self.assertEqual((c['outcome'], c['truncated'], c['abnormal']), ('draw', True, True))
        c = self.EV.classify_game(0.0, True, False, 2, 1, actor=1, candidate_seat=0, max_step_wall=31)
        self.assertEqual((c['outcome'], c['engine_timeout'], c['reason_label']), ('draw', True, 'engine_timeout'))

    def test_public_board_hides_face_down_identities_and_hand_contents(self):
        card = lambda code, c, loc, seq, pos: dict(code=code, controller=c, location=loc, sequence=seq, position=pos)
        s = dict(turn_count=3, turn_player=0, phase=4, lp0=8000, lp1=7000,
                 cards=[card(11, 0, 2, 0, 10), card(12, 1, 2, 0, 10), card(13, 1, 4, 2, 8), card(14, 1, 4, 0, 1), card(15, 0, 8, 1, 10),
                        card(16, 0, 16, 0, 5), card(17, 1, 32, 0, 5), card(18, 1, 32, 1, 10), card(19, 0, 1, 0, 10)])
        b = self.EV.public_board(s)
        self.assertEqual(b['lp'], [8000, 7000])
        self.assertEqual((b['p0']['hand'], b['p1']['hand'], b['p0']['deck']), (1, 1, 1))
        self.assertEqual([(f['code'], f['face_up']) for f in b['p1']['field']], [(14, True), (None, False)])
        self.assertEqual([(f['code'], f['face_up']) for f in b['p0']['field']], [(None, False)])
        self.assertEqual(b['p0']['graveyard'], [16])
        self.assertEqual((b['p1']['banished_face_up'], b['p1']['banished_face_down']), ([17], 1))
        self.assertNotIn('11', json.dumps(b))

    def test_check_records_mismatches_and_stops_only_on_clear_exposure(self):
        import collections

        class V:
            ID_CODES = {1 * 256 + 2: 555}
        actions = np.zeros((1, 128, 12), np.int64)
        actions[0, 0, 1:4] = [1, 2, 3]
        cards = np.zeros((1, 160, 41), np.int64)
        obs = dict(actions_=actions, cards_=cards)
        info = dict(raw_num_options=np.array([130]), public_chain_length=np.array([0]))
        s = dict(actions=[dict(index=0, code=999, act='Activate')], public_chain=[], turn_count=1, phase=4, to_play=0, message='select_idlecmd')
        notes, audit = collections.Counter(), []
        self.EV.check(V, obs, info, s, audit, notes, 5, None)
        self.assertEqual((notes['action_identity_mismatch'], notes['raw_options_over_128']), (1, 1))
        cards[0, 0, [0, 1, 2, 4]] = [1, 2, 2, 1]  # opponent hand card shown, same id as a legal action
        self.EV.check(V, obs, info, s, audit, notes, 5, None)
        self.assertEqual(audit[-1]['cards'][0]['origin'], 'selection_prompt')
        with self.assertRaises(RuntimeError):
            self.EV.check(V, obs, info, s, audit, notes, 0, None)

    def test_pair_summary_excludes_uncertain_pairs_and_uses_the_reviewed_interval(self):
        mk = lambda pair, seat, outcome, label='normal_lp0': dict(pair=pair, candidate_seat=seat, error=None, result=dict(
            outcome=outcome, win=int(outcome == 'win'), loss=int(outcome == 'loss'), draw=int(outcome == 'draw'), engine_timeout=label == 'engine_timeout',
            timeout_suspected=False, truncated=label == 'truncated', reason_label=label))
        games = [mk(0, 0, 'win'), mk(0, 1, 'win'), mk(1, 0, 'win'), mk(1, 1, 'loss'), mk(2, 0, 'loss'), mk(2, 1, 'draw', 'truncated'),
                 mk(3, 0, 'draw', 'engine_timeout'), mk(3, 1, 'win')]
        s = self.EV.pair_summary(games, self.ev)
        self.assertEqual(s['verified_pairs'], 2)
        self.assertEqual((s['wins'], s['losses'], s['draws']), (3, 1, 0))
        self.assertAlmostEqual(s['win_rate'], 0.75)
        low, high, _, _ = self.ev.paired_wilson_ci95([1.0, 0.5])
        self.assertEqual(s['ci95_pairs'], [low, high])
        self.assertEqual(len(s['excluded_pairs']), 2)
        self.assertEqual(s['raw']['games'], 8)
        self.assertEqual(s['pair_kinds'], {'2-0': 1, 'first-seat-both': 1})


if __name__ == '__main__':
    unittest.main()
