"""Artmage RL training (spec artmage-rl-training-plan-2026-11-12): one candidate, PPO on an Artmage mirror against the
frozen start model + rehearsal of the 40 old sequences. Based on the learning pilot's pilot_train.py (d39754a3, the
version after its visibility-audit amendment); the collection rules are the pilot's unless noted.

Collection (natural duels: normal dealing and draws, benchmark fixture reset to empty, a fresh env per episode, player
0 moves first):
  - learner seat alternates per episode (first learner seat registered); learner = current params, frozen side = the
    start params (never updated); separate RNN states and separate position adapters per seat, reset every episode;
    both sides sample from their own final (base + chain-head) logits with one registered PRNG key;
  - learner decisions owned by the position adapter (select_position, message 5) stay as RNN history but are excluded
    from the PPO loss (mask 0);
  - opponent hand/deck identities in the observation are audited (origin selection_prompt / unverified, the pilot's
    amended audit); a clear exposure stops the run (= not established). Audit values never reach the model.
Caps: 32,768 environment decisions (learner + frozen) or 256 episodes, whichever comes first; per episode 4,000
decisions (native max_steps) and a native step timeout of 30 s. Graceful wall-time stop (see WALL_LIMIT).
Episode ends (classify_end):
  natural   termination_reason 0 and not truncated: learner reward = sign(r) for the actor's view (r is the reward of the
            player who made the last decision), +1 / -1 / 0;
  step_limit termination_reason 1 (or truncated): reward 0, recorded separately (the native LP judgement is not used);
  timeout   termination_reason 2 (native dummy end state): the episode's learner steps in the open window are dropped
            from the update (no outcome is invented), recorded.
Updates: a window closes at the first learner decision or episode end after >= 128 environment decisions, at the
decision cap, or at the end of the last allowed episode. Bootstrap: terminal (last step done) = 0; learner to move
next = V of that observation (current params, the learner's RNN state); otherwise (decision cap reached while the
frozen side moves) the unfinished episode's steps of the window are dropped from the update and counted. One PPO
epoch over the window's learner decisions in 4 contiguous chunks (<= 32 decisions, started from the stored rollout
RNN state), each chunk = one minibatch + one rehearsal batch of 2 sequences from hb_data.BlockSampler (2 existing-
group batches + 2 pair batches per block of 4). GAE(gamma 1.0, lambda 0.95) over the learner's own decisions.
The candidate is the state after the last window (no checkpoint selection).
"""
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

T_PROCESS = time.time()

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
START = (ROOT / 'models/start/base.flax_model', ROOT / 'models/start/chain-head.flax_model')
START_SHA = ('b9c7751f5c1f7179faa4911e1a52e8603e332fd9700a0f301dd4e759f8208f66', '77191d5d9a9aaa65d63902c93e47c0e3274009e7a91ee1e8e8abb48fcd73c6a2')
CAP = 32768
EPISODE_CAP = 256
WINDOW = 128
MAX_GAME_DECISIONS = 4000
NATIVE_TIMEOUT = 30
WALL_LIMIT = 3300  # seconds since process start: no new window after this (the job is killed at 3,600 s)
POSITION_MESSAGE = 5
HAND, DECK = 2, 1
TRAIN_SEED, REHEARSAL_SEED, EVAL_SEED = 10111201, 30111201, 80111201
EVAL_PAIRS = 64
SOURCES = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'pairs-A', 'pairs-B')
MAX_VISIBLE = 15
GAMMA, GAE_LAMBDA = 1.0, 0.95  # = rlt_model.CONFIG (asserted when the model is loaded)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=1, default=float) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


# ------------------------------------------------------------------ seeds (registered before training)
def episode_seeds(seed, n=10000):
    """Same generator as the pilot."""
    rng = random.Random(seed * 1000003 + 77)
    return [rng.randint(1, 2 ** 31 - 2) for _ in range(n)]


def training_seeds():
    return episode_seeds(TRAIN_SEED, EPISODE_CAP)


def evaluation_seeds(exclude):
    """Fixed rule: walk the 80111201 stream in order, skip seeds in `exclude` (all 256 training seeds) and repeats,
    keep the first 64."""
    out, seen = [], set(exclude)
    for s in episode_seeds(EVAL_SEED, 100000):
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) == EVAL_PAIRS:
            return out
    raise RuntimeError('not enough evaluation seeds')


# ------------------------------------------------------------------ episode end / windows (pure)
def classify_end(reward, terminated, truncated, termination_reason, actor, learner_seat):
    """reward = env reward of the step that ended the duel, from the view of `actor` (the player who made that
    decision; native ygopro.h: reward = winner == player ? 1 : -1, 0 when termination_reason == 1)."""
    r = float(reward)
    tr = None if termination_reason is None else int(termination_reason)
    if tr == 2:
        return dict(kind='timeout', learner_reward=None, raw_reward=r, termination_reason=tr, winner=None)
    if tr == 1 or bool(truncated):
        return dict(kind='step_limit', learner_reward=0.0, raw_reward=r, termination_reason=tr, winner=None)
    if not bool(terminated):
        raise RuntimeError('episode end without terminated/truncated')
    if tr != 0:
        raise RuntimeError(f'unknown termination_reason {tr}')
    sign = float((r > 0) - (r < 0))
    winner = None if sign == 0 else (actor if sign > 0 else 1 - actor)
    learner_reward = sign * (1.0 if actor == learner_seat else -1.0)
    return dict(kind='natural', learner_reward=learner_reward, raw_reward=r, termination_reason=tr, winner=winner)


def gae(values, rewards, dones, bootstrap, gamma=1.0, lam=0.95):
    """One learner segment (time order). dones[i] = episode ended after decision i. bootstrap = V after the last step."""
    import numpy as np
    n = len(values)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for i in reversed(range(n)):
        next_value = bootstrap if i == n - 1 else values[i + 1]
        nonterminal = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + np.asarray(values, np.float32)


def close_window(window, open_episode, learner_next, bootstrap_fn):
    """Decide the bootstrap of a closing window and which steps enter the update.

    open_episode: index of the episode still running (None if it ended at this step). Returns (steps, info)."""
    if not window:
        return [], dict(bootstrap='empty', dropped=0, bootstrap_value=None)
    if window[-1]['done']:
        steps, kind, boot, dropped = window, 'terminal', 0.0, 0
    elif learner_next:
        steps, kind, boot, dropped = window, 'exact', float(bootstrap_fn()), 0
    else:
        steps = [s for s in window if s['episode'] != open_episode]
        dropped = len(window) - len(steps)
        if steps and not steps[-1]['done']:
            raise RuntimeError('a window step of a finished episode is not terminal')
        kind, boot = 'cut_dropped', 0.0
    if steps:
        adv, tgt = gae([s['value'] for s in steps], [s['reward'] for s in steps], [s['done'] for s in steps], boot, GAMMA, GAE_LAMBDA)
        for s, a_, t_ in zip(steps, adv, tgt):
            s['advantage'], s['target'] = float(a_), float(t_)
    return steps, dict(bootstrap=kind, dropped=dropped, bootstrap_value=boot)


# ------------------------------------------------------------------ collection loop (engine and model via hooks)
def collect(H, out, cap=CAP, episode_cap=EPISODE_CAP, window_size=WINDOW, first_seat=0, seeds=None, wall_limit=WALL_LIMIT):
    """H: hooks (RealHooks in training, fakes in the offline tests). Writes episodes/windows/minibatches/progress."""
    seeds = seeds or training_seeds()
    counts = collections.Counter()
    total, window_start, window, window_index, minibatch_index = 0, 0, [], 0, 0
    stop_reason, error = None, None
    e = -1
    t0 = time.time()
    while stop_reason is None:
        if total >= cap:
            stop_reason = 'decision_cap'
            break
        if e + 1 >= episode_cap:
            stop_reason = 'episode_cap'
            break
        e += 1
        seat = (e + first_seat) % 2
        ep = dict(episode=e, env_seed=seeds[e], learner_seat=seat, start_total=total)
        ep_learner, learner_first, ep_decisions, prev_action = [], True, collections.Counter(), None
        end, env = None, None
        try:
            env, obs, info = H.make_env(seeds[e])
            states = {0: H.init_state(), 1: H.init_state()}
            H.reset_adapters()
            while True:
                tp = int(info['to_play'][0])
                role = 'learner' if tp == seat else 'frozen'
                shown, reason = H.audit(obs, info, ep_decisions['total'])
                if shown:
                    counts['visibility_events'] += 1
                    counts['visible_cards'] += len(shown)
                    counts['visible_unverified'] += sum(c['origin'] == 'unverified' for c in shown)
                    counts['visible_selection_prompt'] += sum(c['origin'] == 'selection_prompt' for c in shown)
                    append(out / 'visibility-events.jsonl', dict(episode=e, env_seed=seeds[e], total=total, viewer_seat=tp, role=role,
                                                               previous_action=prev_action, cards=shown, stopped=reason, **H.describe(obs, info)))
                    if reason:
                        raise RuntimeError('clear exposure: ' + reason)
                pre = states[tp]
                t_inf = time.time()
                new_state, logits, value = H.infer(role, pre, obs, info)
                counts[f'{role}_infer_seconds'] += time.time() - t_inf
                n = int(info['num_options'][0])
                raw = H.sample(logits)
                if not 0 <= raw < n:
                    raise RuntimeError('sampled an illegal option')
                final = H.position_final(tp, raw, obs, info)
                controlled = H.message_type(obs, raw) == POSITION_MESSAGE
                if final != raw and not controlled:
                    raise RuntimeError('adapter changed a non-position action')
                states[tp] = new_state
                if role == 'learner':
                    row = dict(H.step_row(obs, info), action=raw, logits=logits, value=float(value), controlled=controlled,
                               first=learner_first, pre_state=pre, reward=0.0, done=False, episode=e)
                    learner_first = False
                    window.append(row)
                    ep_learner.append(row)
                    counts['learner'] += 1
                    counts['learner_adapter_controlled'] += int(controlled)
                    counts['learner_adapter_changed'] += int(final != raw)
                else:
                    counts['frozen'] += 1
                    counts['frozen_adapter_changed'] += int(final != raw)
                ep_decisions[role] += 1
                ep_decisions['total'] += 1
                prev_action = dict(seat=tp, role=role, index=final, message_type=H.message_type(obs, final), card_code=H.card_code(obs, final))
                obs, reward, term, trunc, info = H.step(env, final)
                total += 1
                ended = bool(term[0] or trunc[0])
                if ended:
                    tr = info.get('termination_reason')
                    end = classify_end(float(reward[0]), term[0], trunc[0], None if tr is None else int(tr[0]), tp, seat)
                    counts[f"end_{end['kind']}"] += 1
                    if end['kind'] == 'timeout':
                        before = len(window)
                        window = [s for s in window if s['episode'] != e]
                        counts['dropped_timeout'] += before - len(window)
                        end['dropped_learner_steps'] = before - len(window)
                    elif ep_learner:
                        ep_learner[-1]['reward'] = end['learner_reward']
                        ep_learner[-1]['done'] = True
                    if end['kind'] == 'natural':
                        counts['natural_' + {1.0: 'win', -1.0: 'loss', 0.0: 'draw'}[end['learner_reward']]] += 1
                    ep.update(end=end, last_actor=tp, last_actor_role=role)
                learner_next = not ended and int(info['to_play'][0]) == seat
                at_cap = total >= cap
                last_episode = ended and e + 1 >= episode_cap
                if (total - window_start >= window_size and (ended or learner_next)) or at_cap or last_episode:
                    steps, binfo = close_window(window, None if ended else e, learner_next,
                                                lambda: H.bootstrap(states[seat], obs, info))
                    counts['dropped_cut'] += binfo['dropped'] if binfo['bootstrap'] == 'cut_dropped' else 0
                    wrow = dict(window=window_index, start_total=window_start, end_total=total, decisions=total - window_start,
                                learner_steps=len(window), update_steps=len(steps), loss_steps=sum(not s['controlled'] for s in steps),
                                minibatches=[], seconds=time.time() - t0, **binfo)
                    if steps:
                        for m in H.update(steps):
                            m.update(minibatch=minibatch_index, window=window_index)
                            append(out / 'minibatches.jsonl', m)
                            wrow['minibatches'].append(minibatch_index)
                            counts['ppo_loss_steps'] += int(m['ppo_steps'])
                            counts['rehearsal_sequences'] += len(m['rehearsal'])
                            counts['rehearsal_loss_steps'] += int(m['reh_steps'])
                            counts['rehearsal_context_steps'] += int(m['reh_context_steps'])
                            minibatch_index += 1
                    wrow.update(H.window_extra())
                    append(out / 'windows.jsonl', wrow)
                    window_index += 1
                    window, window_start = [], total
                    write_json(out / 'progress.json', dict(total=total, episodes_started=e + 1, windows=window_index, updates=minibatch_index,
                                                           seconds=time.time() - t0, process_seconds=time.time() - T_PROCESS, counts=dict(counts)))
                    if H.stop_requested():
                        stop_reason = 'STOP=now requested during the job'
                    elif H.elapsed() >= wall_limit:
                        stop_reason = 'wall_limit'
                if ended or at_cap or stop_reason:
                    break
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            stop_reason = 'error'
        finally:
            if end is None:
                ep.update(end=None, cut_by=stop_reason or ('decision_cap' if total >= cap else 'unknown'))
                counts['episodes_cut'] += 1
            ep.update(end_total=total, decisions=total - ep['start_total'], learner_decisions=ep_decisions['learner'],
                      frozen_decisions=ep_decisions['frozen'], learner_steps=len(ep_learner), **H.episode_extra())
            append(out / 'episodes.jsonl', ep)
            if env is not None:
                try:
                    H.close_env(env)
                except Exception as exc:
                    error = error or f'close: {type(exc).__name__}: {exc}'
                    stop_reason = 'error'
                    traceback.print_exc()
            del env
            gc.collect()
        print('EPISODE', json.dumps(dict(episode=e, seat=seat, decisions=ep['decisions'], end=(ep['end'] or {}).get('kind'),
                                         reward=(ep['end'] or {}).get('learner_reward'))), 'total', total, 'updates', minibatch_index,
              f'{time.time() - t0:.0f}s', flush=True)
    if stop_reason is None:
        stop_reason = 'decision_cap' if total >= cap else 'episode_cap'
    return dict(total=total, episodes=e + 1, windows=window_index, minibatches=minibatch_index, stop_reason=stop_reason, error=error,
                seconds=time.time() - t0, counts=dict(counts))


# ------------------------------------------------------------------ real hooks (engine + JAX)
def make_env(E, seed):
    """Natural duel (pilot make_env): benchmark fixture reset to empty = normal dealing and draws; per-game caps."""
    v = E.v
    v.native.set_benchmark_fixture([])
    env = v.ygoenv.make('YGOPro-v1', env_type='gymnasium', num_envs=1, num_threads=1, seed=seed, deck1=E.DECK_KEY, deck2=E.DECK_KEY,
                        player=-1, play_mode='self', max_options=128, n_history_actions=32, oppo_info=False, async_reset=False,
                        max_steps=MAX_GAME_DECISIONS, max_episode_steps=MAX_GAME_DECISIONS, timeout=NATIVE_TIMEOUT, greedy_reward=False)
    env.num_envs = 1
    env = v.EnvPreprocess(env, skip_mask=True)
    return (env, *env.reset())


class RealHooks:
    def __init__(self, reg):
        import jax
        jax.config.update('jax_default_matmul_precision', 'highest')
        import jax.numpy as jnp
        import numpy as np
        sys.path.insert(0, str(ROOT / 'code'))
        import artmage_entry as E
        import hb_data as HD
        import rlt_model as M
        from opponent_article_bc_common import features
        self.jax, self.jnp, self.np, self.E, self.HD, self.M, self.features = jax, jnp, np, E, HD, M, features
        self.v = E.v
        assert (M.CONFIG['gamma'], M.CONFIG['gae_lambda']) == (GAMMA, GAE_LAMBDA)
        assert (sha(START[0]), sha(START[1])) == START_SHA
        self.params, self.bstats, self.original_base = M.load_pair(*START)
        self.frozen = jax.tree.map(lambda x: x.copy(), self.params)
        self.optimizer = M.make_optimizer()
        self.opt_state = self.optimizer.init(self.params)
        self.existing, self.names, self.pairs, self.data_stats = HD.load()
        self.length = max([len(s['labels']) for g in self.existing for s in self.existing[g]] +
                          [len(s['labels']) for g in self.pairs for p in self.pairs[g] for _, _, s in p['hands']])
        self.sampler = HD.BlockSampler(self.existing, self.pairs, reg['training']['rehearsal_seed'])
        self.key = jax.random.PRNGKey(reg['training']['seed'])
        seq0 = self.existing['F0'][0]
        self.dummy = dict(obs={k: seq0[k][0] for k in M.OBS_KEYS}, chain=seq0['chain'][0], action=0, logits=np.zeros(128, np.float32),
                          controlled=True, first=True, pre_state=tuple(np.zeros((1, 512), np.float32) for _ in range(2)))
        bstats, optimizer = self.bstats, self.optimizer
        self.step_fn = jax.jit(lambda p, o, c, r: M.update_step(p, o, bstats, c, r, optimizer, len(SOURCES)))
        self.adapter_module = E._adapter()
        self.adapters = {}
        self.t0 = time.time()
        self.position_counts = collections.Counter()

    # env
    def make_env(self, seed):
        return make_env(self.E, seed)

    def step(self, env, idx):
        return env.step(self.np.array([idx], self.np.int32))

    def close_env(self, env):
        self.E.close(env)
        gc.collect()
        assert self.v.native.get_env_resource_stats()['live_duels'] == 0

    # model
    def init_state(self):
        return self.M.init_state(1)

    def _obs(self, obs):
        return {k: self.jnp.asarray(self.np.asarray(obs[k])) for k in self.M.OBS_KEYS}

    def infer(self, role, state, obs, info):
        params = self.params if role == 'learner' else self.frozen
        new_state, logits, value = self.M.infer(params, self.bstats, state, self._obs(obs), self.jnp.asarray(self.features(info)))
        self._logits = logits
        return new_state, self.np.asarray(logits)[0].copy(), float(self.np.asarray(value)[0])

    def sample(self, logits):
        self.key, sub = self.jax.random.split(self.key)
        return int(self.jax.random.categorical(sub, self._logits[0]))

    def bootstrap(self, state, obs, info):
        _, _, value = self.M.infer(self.params, self.bstats, state, self._obs(obs), self.jnp.asarray(self.features(info)))
        return float(self.np.asarray(value)[0])

    # adapter
    def reset_adapters(self):
        class Preset:
            def __init__(self):
                self.raw = None

            def act(self, obs, info):
                return self.raw

            def reset(self):
                self.raw = None
        self.adapters = {p: self.adapter_module.PositionPolicy(Preset(), self.E._IDS) for p in (0, 1)}

    def position_final(self, seat, raw, obs, info):
        self.adapters[seat].base.raw = raw
        return int(self.adapters[seat].act(obs, info))

    def message_type(self, obs, idx):
        return int(self.np.asarray(obs['actions_'])[0, idx, 3])

    def card_code(self, obs, idx):
        row = self.np.asarray(obs['actions_'])[0, idx]
        return self.E._IDS.get(int(row[1]) * 256 + int(row[2]), 0)

    def step_row(self, obs, info):
        return dict(obs={k: self.np.asarray(obs[k])[0].copy() for k in self.M.OBS_KEYS}, chain=self.np.asarray(self.features(info))[0].copy())

    # audit (pilot amendment-1 rules)
    def audit(self, obs, info, actions_so_far):
        np = self.np
        cards = np.asarray(obs['cards_'])[0]
        actions = np.asarray(obs['actions_'])[0]
        action_ids = {int(x[1]) * 256 + int(x[2]) for x in actions[actions[:, 3] != 0]}
        hidden = (cards[:, 4] == 1) & np.isin(cards[:, 2], [DECK, HAND])
        shown = []
        for i in np.where(hidden & ((cards[:, 0] > 0) | (cards[:, 1] > 0)))[0]:
            cid = int(cards[i, 0]) * 256 + int(cards[i, 1])
            shown.append(dict(row=int(i), location='deck' if int(cards[i, 2]) == DECK else 'hand', card_id=cid, code=self.E._IDS.get(cid, 0),
                              origin='selection_prompt' if cid in action_ids else 'unverified'))
        reason = None
        if shown and actions_so_far == 0:
            reason = 'opponent hand/deck identity visible before any action of the episode'
        elif len(shown) > MAX_VISIBLE:
            reason = f'{len(shown)} opponent hand/deck identities visible at once'
        return shown, reason

    def describe(self, obs, info):
        g = self.np.asarray(obs['global_'])[0]
        return dict(turn=int(g[4]), phase=int(g[5]), message_type=int(self.np.asarray(obs['actions_'])[0, 0, 3]),
                    public_chain=self.np.asarray(self.features(info))[0].tolist())

    # update
    def update(self, steps):
        np, M = self.np, self.M
        out = []
        for part in np.array_split(np.arange(len(steps)), M.CONFIG['minibatches_per_window']):
            chunk = chunk_arrays([steps[i] for i in part], self.dummy, M.CONFIG['chunk_length'], M.OBS_KEYS)
            item = self.sampler.next()
            reh, names = self.HD.make_batch(self.existing, self.pairs, item, self.length)
            reh['source'] = np.array(source_index(item), np.int32)
            params, opt_state, metrics = self.step_fn(self.params, self.opt_state, to_device(self.jnp, chunk), to_device(self.jnp, reh))
            self.params, self.opt_state = params, opt_state
            m = {k: np.asarray(x).tolist() for k, x in metrics.items()}
            m.update(rehearsal=names, rehearsal_kind=item[0], ppo_steps=int(chunk['mask'].sum()), chunk_steps=len(part),
                     reh_context_steps=int(reh['real'].sum()))
            out.append(m)
        return out

    def window_extra(self):
        finite_state = self.opt_state
        try:
            return dict(notfinite_count=int(self.np.asarray(finite_state.notfinite_count)), total_notfinite=int(self.np.asarray(finite_state.total_notfinite)))
        except AttributeError:
            return {}

    def episode_extra(self):
        return dict(position_events={str(p): len(a.positions.events) for p, a in self.adapters.items()},
                    position_changed={str(p): sum(1 for x in a.positions.events if x.get('changed')) for p, a in self.adapters.items()})

    def stop_requested(self):
        path = ROOT / 'STOP'
        return path.exists() and path.read_text().strip() == 'now'

    def elapsed(self):
        return time.time() - T_PROCESS


def source_index(item):
    kind, v = item
    if kind == 'existing':
        return [SOURCES.index(g) for g, _ in v]
    return [SOURCES.index('pairs-' + v[0])] * 2


def chunk_arrays(steps, dummy, T, obs_keys):
    import numpy as np
    real = len(steps)
    if real > T:
        raise RuntimeError(f'chunk longer than {T} learner decisions')
    src = steps if steps else [dummy]
    idx = [min(t, len(src) - 1) for t in range(T)]
    out = {k: np.stack([src[i]['obs'][k] for i in idx])[:, None] for k in obs_keys}
    out['chain'] = np.stack([src[i]['chain'] for i in idx])[:, None]
    out['actions'] = np.array([src[i]['action'] for i in idx], np.int32)[:, None]
    out['old_logits'] = np.stack([src[i]['logits'] for i in idx])[:, None]
    out['advantages'] = np.array([src[i].get('advantage', 0.0) for i in idx], np.float32)[:, None]
    out['targets'] = np.array([src[i].get('target', 0.0) for i in idx], np.float32)[:, None]
    mask = np.array([(t < real) and not src[idx[t]]['controlled'] for t in range(T)], np.float32)
    out['mask'] = mask[:, None]
    out['first'] = np.array([(t < real) and src[idx[t]]['first'] for t in range(T)], bool)[:, None]
    out['init_state'] = tuple(np.asarray(x) for x in src[0]['pre_state'])
    return out


def to_device(jnp, tree):
    import jax
    return jax.tree.map(jnp.asarray, tree)


# ------------------------------------------------------------------ after training (saved-file checks)
def saved_checks(H, out):
    """Reload the saved pair through the isolated entry (hash-checked): logits, neural argmax and adapter final actions
    on all 40 rehearsal histories must equal the in-memory params; weights finite; both parts changed."""
    jax, jnp, np, E, M = H.jax, H.jnp, H.np, H.E, H.M
    features = H.features
    cand = (out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    want = (sha(cand[0]), sha(cand[1]))
    reloaded = E.create_policy(cand[0], cand[1], positions=False, expected=want)
    bstats = M.load_pair(*cand)[1]
    items = [s for g in sorted(H.existing) for s in H.existing[g]] + [s for g in sorted(H.pairs) for p in H.pairs[g] for _, _, s in p['hands']]
    max_diff, argmax_equal = 0.0, True
    for s in items:
        st_mem, st_file = M.init_state(1), reloaded.state
        for t in range(len(s['labels'])):
            o = {k: jnp.asarray(s[k][t:t + 1]) for k in M.OBS_KEYS}
            ch = jnp.asarray(s['chain'][t:t + 1])
            st_mem, l_mem, _ = M.infer(H.params, bstats, st_mem, o, ch)
            st_file, l_file = reloaded.infer(reloaded.base, reloaded.head, st_file, dict(o, mask_=None), ch)
            max_diff = max(max_diff, float(jnp.max(jnp.abs(l_mem - l_file))))
            argmax_equal &= int(jnp.argmax(l_mem[0])) == int(jnp.argmax(l_file[0]))

    class MemoryPolicy:
        def __init__(self):
            self.reset()

        def reset(self):
            self.state = M.init_state(1)

        def act(self, obs, info):
            o = {k: jnp.asarray(obs[k]) for k in M.OBS_KEYS}
            self.state, logits, _ = M.infer(H.params, bstats, self.state, o, jnp.asarray(features(info)))
            return int(np.asarray(logits)[0].argmax())
    module = E._adapter()
    adapter_equal = True
    for s in items:
        finals = []
        for pol in [E.create_policy(cand[0], cand[1], positions=True, expected=want), module.PositionPolicy(MemoryPolicy(), E._IDS)]:
            pol.reset()
            acts = []
            for t in range(len(s['labels'])):
                obs = {k: s[k][t:t + 1] for k in M.OBS_KEYS}
                obs['mask_'] = None
                ch = s['chain'][t]
                info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                            public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                            public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
                acts.append(int(pol.act(obs, info)))
            finals.append(acts)
        adapter_equal &= finals[0] == finals[1]
    import flax.serialization as ser
    finite = {}
    for name, path in zip(['base', 'head'], cand):
        tree = ser.msgpack_restore(path.read_bytes())
        finite[name] = all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree))
    delta = {part: float(M.tree_norm(jax.tree.map(lambda a, b: a - b, H.params[part], H.frozen[part]))) for part in ['base', 'head']}
    return dict(candidate_base_sha256=want[0], candidate_head_sha256=want[1], reload_through_isolated_entry=True,
                max_abs_logit_diff_memory_vs_saved=max_diff, argmax_equal=argmax_equal, adapter_final_actions_equal=adapter_equal,
                saved_weights_finite=finite, sequences_checked=len(items), parameter_change_norm=delta,
                both_parts_changed=all(x > 0 for x in delta.values()))


def rehearsal_fit(H, params):
    """Teacher-forced top-1 / CE on the loss steps of the 40 rehearsal sequences (diagnostic only)."""
    jnp, np, M = H.jnp, H.np, H.M
    out = {}
    import pilot_data as PD
    fn = H.jax.jit(lambda p, b: M.rehearsal_step_ce(p, H.bstats, b))
    groups = [(g, H.existing[g]) for g in sorted(H.existing)] + [('pairs-' + g, [s for p in H.pairs[g] for _, _, s in p['hands']]) for g in sorted(H.pairs)]
    for name, seqs in groups:
        ce, hit = [], []
        for s in seqs:
            b = PD.batch([s], length=H.length)
            c, h = fn(params, {k: jnp.asarray(v) for k, v in b.items()})
            m = b['loss'][:, 0] > 0
            ce.extend(np.asarray(c)[:, 0][m].tolist())
            hit.extend(np.asarray(h)[:, 0][m].tolist())
        out[name] = dict(loss_steps=len(ce), ce=float(np.mean(ce)), top1=float(np.mean(hit)))
    return out


def registered():
    reg = json.loads((ROOT / 'registration.json').read_text())
    amendments = sorted(ROOT.glob('amendment-*.json'))
    code = json.loads(amendments[-1].read_text())['code'] if amendments else reg['code']
    for name, digest in code.items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    if json.loads((ROOT / 'data/seeds.json').read_text())['training'] != training_seeds():
        raise RuntimeError('registered training seeds differ')
    return reg


def main():
    reg = registered()
    out = ROOT / 'train'
    out.mkdir(exist_ok=True)
    if (out / 'result.json').exists() or (out / 'episodes.jsonl').exists():
        raise SystemExit('a training attempt exists; one candidate only')
    H = RealHooks(reg)
    H.E.verify_pins()
    print('DEVICES', H.jax.devices(), flush=True)
    fit_start = rehearsal_fit(H, H.params)
    stats = collect(H, out, first_seat=reg['training']['first_learner_seat'], seeds=json.loads((ROOT / 'data/seeds.json').read_text())['training'])
    stats.update(rehearsal_counts=H.sampler.counts(), data=H.data_stats, rehearsal_length=H.length, adapter_window_extra=H.window_extra(),
                 native_lua_errors=H.v.native.get_lua_error_count(), live_duels=H.v.native.get_env_resource_stats()['live_duels'])
    finite = all(bool(H.np.isfinite(H.np.asarray(x)).all()) for x in H.jax.tree.leaves(H.params))
    if stats['error'] or not finite:
        status = 'not_established'
        target = out / 'stopped-state'
    else:
        status = 'complete'
        target = out / 'candidate'
    target.mkdir(exist_ok=True)
    H.M.save_pair(H.params, H.original_base, target / 'base.flax_model', target / 'chain-head.flax_model')
    for name in ['base.flax_model', 'chain-head.flax_model']:
        os.chmod(target / name, 0o444)
    result = dict(status=status, params_finite=finite, **stats, rehearsal_fit=dict(start=fit_start))
    if status == 'complete':
        result['saved'] = saved_checks(H, out)
        result['rehearsal_fit']['candidate'] = rehearsal_fit(H, H.params)
        ok = (result['saved']['argmax_equal'] and result['saved']['adapter_final_actions_equal'] and all(result['saved']['saved_weights_finite'].values())
              and result['saved']['max_abs_logit_diff_memory_vs_saved'] == 0.0)
        if not ok:
            result['status'] = 'not_established'
            result['why'] = 'saved files do not reproduce the in-memory policy'
    else:
        result['stopped_state'] = dict(base_sha256=sha(target / 'base.flax_model'), head_sha256=sha(target / 'chain-head.flax_model'))
    result['process_seconds'] = time.time() - T_PROCESS
    write_json(out / 'result.json', result)
    print('RESULT', json.dumps({k: result[k] for k in ['status', 'total', 'episodes', 'windows', 'minibatches', 'stop_reason', 'error']}), flush=True)
    if 'saved' in result:
        print('SAVED', json.dumps({k: result['saved'][k] for k in ['candidate_base_sha256', 'candidate_head_sha256', 'max_abs_logit_diff_memory_vs_saved',
                                                                   'argmax_equal', 'adapter_final_actions_equal', 'parameter_change_norm']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
