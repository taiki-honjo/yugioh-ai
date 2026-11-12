"""Artmage learning pilot: one candidate, PPO on an Artmage mirror against the frozen start model + rehearsal.

Collection (natural duels: normal dealing and draws, a fresh env per episode, player 0 always moves first):
  - the learner seat alternates per episode (first seat registered), so learner first/second is balanced;
  - learner = current params, frozen side = the start params (never updated); separate RNN states per seat,
    reset every episode; both sides sample from their own final (base + chain-head) logits;
  - both sides use the common position adapter (own memory per seat, reset per episode); learner decisions it
    controls (select_position, message 5) are kept as history but excluded from the PPO loss;
  - observations are checked every step: the opponent's hand/deck card identities are zero (public/own only).
Cap: 4,096 environment decisions in total (learner + frozen, logged separately).
Updates: a window closes at the first learner decision or episode end after >= 128 environment decisions (or at
the cap); one PPO epoch over the window's learner decisions in 4 contiguous chunks (<= 32 decisions each, started
from the stored rollout RNN state), each chunk = one minibatch + 2 rehearsal sequences (14 old / 1 C1 / 1 C2 per
16, shuffled by the registered seed). GAE(gamma 1.0, lambda 0.95) on the learner's own decisions; reward = learner
outcome (+1 / -1, 0 for draws and step-limit ends) at its last decision of the episode.
The candidate is the state after the last window (no checkpoint selection).
"""
import argparse
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

import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-learning-pilot-20261110')
sys.path.insert(0, str(ROOT / 'code'))
import artmage_entry as E  # noqa: E402
import pilot_data as D  # noqa: E402
import pilot_model as M  # noqa: E402
from opponent_article_bc_common import features  # noqa: E402

v = E.v
START = (ROOT / 'models/initial/base.flax_model', ROOT / 'models/initial/chain-head.flax_model')
START_SHA = ('12500c7724b2d723983a2963a960d75172a6df51fb55ea4f3acf102eeb34132e', '0bc7534cab102cd87586e5d22bd4ad0cd3b55d52d08e284146976506ae6193fc')
CAP = 4096
WINDOW = 128
POSITION_MESSAGE = 5
HAND, DECK = 2, 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')


def privacy_ok(obs):
    cards = np.asarray(obs['cards_'])[0]
    hidden = (cards[:, 4] == 1) & np.isin(cards[:, 2], [DECK, HAND])
    return bool(np.all(cards[hidden, :2] == 0))


def own_hand(obs):
    cards = np.asarray(obs['cards_'])[0]
    rows = cards[(cards[:, 4] == 0) & (cards[:, 2] == HAND)]
    return sorted(E._IDS.get(int(r[0]) * 256 + int(r[1]), 0) for r in rows)


def episode_seeds(seed, n=10000):
    rng = random.Random(seed * 1000003 + 77)
    return [rng.randint(1, 2 ** 31 - 2) for _ in range(n)]


def make_env(seed):
    v.native.set_benchmark_fixture([])
    env = v.ygoenv.make('YGOPro-v1', env_type='gymnasium', num_envs=1, num_threads=1, seed=seed, deck1=E.DECK_KEY, deck2=E.DECK_KEY,
                        player=-1, play_mode='self', max_options=128, n_history_actions=32, oppo_info=False, async_reset=False,
                        max_steps=2048, max_episode_steps=2048, timeout=10, greedy_reward=False)
    env.num_envs = 1
    env = v.EnvPreprocess(env, skip_mask=True)
    return (env, *env.reset())


class Preset:
    """Base for the position adapter: returns the action already sampled from the neural policy."""
    def __init__(self):
        self.raw = None

    def act(self, obs, info):
        return self.raw

    def reset(self):
        self.raw = None


def chunk_arrays(steps, dummy):
    T = M.CONFIG['chunk_length']
    real = len(steps)
    if real > T:
        raise RuntimeError(f'chunk longer than {T} learner decisions')
    src = steps if steps else [dummy]
    idx = [min(t, len(src) - 1) for t in range(T)]
    out = {k: np.stack([src[i]['obs'][k] for i in idx])[:, None] for k in M.OBS_KEYS}
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


def to_device(tree):
    return jax.tree.map(jnp.asarray, tree)


def stop_requested():
    path = ROOT / 'STOP'
    return path.exists() and path.read_text().strip() == 'now'


def train(reg, out):
    seed = reg['training']['seed']
    log, episodes_log, windows_log = out / 'minibatches.jsonl', out / 'episodes.jsonl', out / 'windows.jsonl'
    assert (sha(START[0]), sha(START[1])) == START_SHA
    params, bstats, original_base = M.load_pair(*START)
    frozen = jax.tree.map(lambda x: x.copy(), params)
    optimizer = M.make_optimizer()
    opt_state = optimizer.init(params)
    seqs, data_stats = D.build()
    schedule = D.Schedule(reg['training']['rehearsal_seed'])
    dummy_seq = seqs['C1'][0]
    dummy = dict(obs={k: dummy_seq[k][0] for k in M.OBS_KEYS}, chain=dummy_seq['chain'][0], action=0,
                 logits=np.zeros(128, np.float32), controlled=True, first=True,
                 pre_state=tuple(np.zeros((1, 512), np.float32) for _ in range(2)))
    step_fn = jax.jit(lambda p, o, c, r: M.update_step(p, o, bstats, c, r, optimizer))
    key = jax.random.PRNGKey(seed)
    seeds = episode_seeds(seed)
    adapter_module = E._adapter()
    counts = collections.Counter()
    total, window_start, window, window_index, minibatch_index = 0, 0, [], 0, 0
    first_seat = reg['training']['first_learner_seat']
    t_start = time.time()
    e = -1
    aborted = None
    while total < CAP and aborted is None:
        e += 1
        seat = (e + first_seat) % 2
        env, obs, info = make_env(seeds[e])
        states = {0: M.init_state(1), 1: M.init_state(1)}
        adapters = {p: adapter_module.PositionPolicy(Preset(), E._IDS) for p in (0, 1)}
        hands, first_turn = {}, {}
        learner_first = True
        ep_learner_steps = []
        ep = dict(episode=e, env_seed=seeds[e], learner_seat=seat, start_total=total)
        ended = False
        try:
            while True:
                tp = int(info['to_play'][0])
                if not privacy_ok(obs):
                    raise RuntimeError('opponent private card identity visible in the observation')
                g = np.asarray(obs['global_'])[0]
                if tp not in hands:
                    hands[tp], first_turn[tp] = own_hand(obs), int(g[4])
                chain = jnp.asarray(features(info))
                o = {k: jnp.asarray(np.asarray(obs[k])) for k in M.OBS_KEYS}
                learner = tp == seat
                pre = states[tp]
                new_state, logits, value = M.infer(params if learner else frozen, bstats, pre, o, chain)
                key, sub = jax.random.split(key)
                raw = int(jax.random.categorical(sub, logits[0]))
                n = int(info['num_options'][0])
                if not 0 <= raw < n:
                    raise RuntimeError('sampled an illegal option')
                adapters[tp].base.raw = raw
                final = int(adapters[tp].act(obs, info))
                controlled = int(np.asarray(obs['actions_'])[0, raw, 3]) == POSITION_MESSAGE
                if final != raw and not controlled:
                    raise RuntimeError('adapter changed a non-position action')
                states[tp] = new_state
                if learner:
                    row = dict(obs={k: np.asarray(obs[k])[0].copy() for k in M.OBS_KEYS}, chain=np.asarray(features(info))[0].copy(),
                               action=raw, logits=np.asarray(logits)[0].copy(), value=float(np.asarray(value)[0]), controlled=controlled,
                               first=learner_first, pre_state=tuple(np.asarray(x).copy() for x in pre), reward=0.0, done=False, episode=e)
                    learner_first = False
                    window.append(row)
                    ep_learner_steps.append(row)
                    counts['learner'] += 1
                    counts['learner_adapter_controlled'] += int(controlled)
                    counts['learner_adapter_changed'] += int(final != raw)
                else:
                    counts['frozen'] += 1
                obs, reward, term, trunc, info = env.step(np.array([final], np.int32))
                total += 1
                ended = bool(term[0] or trunc[0])
                if ended:
                    r = float(np.asarray(reward).reshape(-1)[0])
                    outcome = float(np.sign(r)) * (1.0 if tp == seat else -1.0)
                    if ep_learner_steps:
                        ep_learner_steps[-1]['reward'] = outcome
                        ep_learner_steps[-1]['done'] = True
                    ep.update(end_total=total, decisions=total - ep['start_total'], learner_outcome=outcome, terminated=bool(term[0]),
                              truncated=bool(trunc[0]), win_reason=int(np.asarray(info.get('win_reason', [0])).reshape(-1)[0]) if 'win_reason' in info else None)
                learner_next = not ended and int(info['to_play'][0]) == seat
                if (total - window_start >= WINDOW and (ended or learner_next)) or total >= CAP:
                    if window and not window[-1]['done']:
                        if learner_next:
                            o2 = {k: jnp.asarray(np.asarray(obs[k])) for k in M.OBS_KEYS}
                            _, _, boot = M.infer(params, bstats, states[seat], o2, jnp.asarray(features(info)))
                            bootstrap, kind = float(np.asarray(boot)[0]), 'exact'
                        else:
                            bootstrap, kind = window[-1]['value'], 'neutral'
                    else:
                        bootstrap, kind = 0.0, 'terminal'
                    adv, tgt = M.gae([s['value'] for s in window], [s['reward'] for s in window], [s['done'] for s in window], bootstrap)
                    for s, a_, t_ in zip(window, adv, tgt):
                        s['advantage'], s['target'] = float(a_), float(t_)
                    parts = np.array_split(np.arange(len(window)), M.CONFIG['minibatches_per_window'])
                    wrow = dict(window=window_index, start_total=window_start, end_total=total, decisions=total - window_start,
                                learner_steps=len(window), loss_steps=sum(not s['controlled'] for s in window), bootstrap=kind,
                                minibatches=[])
                    for part in parts:
                        chunk = chunk_arrays([window[i] for i in part], dummy)
                        picked = schedule.next()
                        reh = D.rehearsal_batch(seqs, picked)
                        params, opt_state, metrics = step_fn(params, opt_state, to_device(chunk), to_device(reh))
                        m = {k: np.asarray(x).tolist() for k, x in metrics.items()}
                        m.update(minibatch=minibatch_index, window=window_index, rehearsal=[f'{s}:{i}' for s, i in picked], ppo_steps=int(chunk['mask'].sum()))
                        append(log, m)
                        wrow['minibatches'].append(minibatch_index)
                        minibatch_index += 1
                    append(windows_log, wrow)
                    window_index += 1
                    window, window_start = [], total
                    if stop_requested():
                        aborted = 'STOP=now requested during the job'
                if ended or total >= CAP or aborted:
                    break
        finally:
            if not ended:
                ep.update(end_total=total, decisions=total - ep['start_total'], learner_outcome=None, cut_by_cap=total >= CAP)
            ep.update(hands={str(p): h for p, h in hands.items()}, hand_first_turn={str(p): t for p, t in first_turn.items()},
                      learner_steps=len(ep_learner_steps))
            append(episodes_log, ep)
            E.close(env)
            del env
            gc.collect()
            assert v.native.get_env_resource_stats()['live_duels'] == 0
        print('EPISODE', json.dumps({k: ep[k] for k in ['episode', 'learner_seat', 'decisions', 'learner_outcome']}), 'total', total,
              'updates', window_index, f'{time.time() - t_start:.0f}s', flush=True)
    return params, original_base, dict(total=total, learner=counts['learner'], frozen=counts['frozen'],
                                       learner_adapter_controlled=counts['learner_adapter_controlled'],
                                       learner_adapter_changed=counts['learner_adapter_changed'], episodes=e + 1, windows=window_index,
                                       minibatches=minibatch_index, rehearsal_counts=schedule.counts(), aborted=aborted,
                                       seconds=time.time() - t_start, data=data_stats), frozen


def verify_saved(params, frozen, out):
    """Reload the saved pair through the isolated entry (hash-checked) and compare logits on the C1/C2 histories."""
    cand = (out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    want = (sha(cand[0]), sha(cand[1]))
    reloaded = E.create_policy(cand[0], cand[1], positions=False, expected=want)
    seqs, _ = D.build()
    bstats = M.load_pair(*cand)[1]
    rows = {}
    for name in ['C1', 'C2']:
        s = seqs[name][0]
        st_mem, st_file, st_start = M.init_state(1), reloaded.state, M.init_state(1)
        diff_mem_file, diff_start, agree_start, agree_cand, raw_equal = 0.0, 0.0, 0, 0, True
        for t in range(len(s['labels'])):
            o = {k: jnp.asarray(s[k][t:t + 1]) for k in M.OBS_KEYS}
            ch = jnp.asarray(s['chain'][t:t + 1])
            st_mem, l_mem, _ = M.infer(params, bstats, st_mem, o, ch)
            st_file, l_file = reloaded.infer(reloaded.base, reloaded.head, st_file, dict(o, mask_=None), ch)
            st_start, l_start, _ = M.infer(frozen, bstats, st_start, o, ch)
            diff_mem_file = max(diff_mem_file, float(jnp.max(jnp.abs(l_mem - l_file))))
            diff_start = max(diff_start, float(jnp.max(jnp.abs(l_mem - l_start))))
            raw_equal &= int(jnp.argmax(l_mem[0])) == int(jnp.argmax(l_file[0]))
            agree_start += int(int(jnp.argmax(l_start[0])) == int(s['labels'][t]))
            agree_cand += int(int(jnp.argmax(l_mem[0])) == int(s['labels'][t]))
        rows[name] = dict(decisions=len(s['labels']), max_abs_logit_diff_memory_vs_saved=diff_mem_file, argmax_equal_memory_vs_saved=raw_equal,
                          max_abs_logit_diff_candidate_vs_start=diff_start, label_argmax_agreement_start=agree_start, label_argmax_agreement_candidate=agree_cand)
    delta = {part: float(M.tree_norm(jax.tree.map(lambda a, b: a - b, params[part], frozen[part]))) for part in ['base', 'head']}
    return dict(candidate_base_sha256=want[0], candidate_head_sha256=want[1], reload_through_isolated_entry=True, histories=rows,
                parameter_change_norm=delta, both_parts_changed=all(x > 0 for x in delta.values()))


class MemoryPolicy:
    """Neural argmax of an in-memory params tree with its own RNN state (same interface as Policy.act)."""
    def __init__(self, params, batch_stats):
        self.params, self.batch_stats = params, batch_stats
        self.reset()

    def reset(self):
        self.state = M.init_state(1)

    def act(self, obs, info):
        o = {k: jnp.asarray(obs[k]) for k in M.OBS_KEYS}
        self.state, logits, _ = M.infer(self.params, self.batch_stats, self.state, o, jnp.asarray(features(info)))
        return int(np.asarray(logits)[0].argmax())


def adapter_check(params, out):
    """Position adapter on top of (a) the saved candidate loaded through the isolated entry and (b) the in-memory
    params: the final actions on the C1/C2 histories must be identical; neural argmax is checked in verify_saved."""
    cand = (out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    want = (sha(cand[0]), sha(cand[1]))
    bstats = M.load_pair(*cand)[1]
    seqs, _ = D.build()
    module = E._adapter()
    result = {}
    for name in ['C1', 'C2']:
        s = seqs[name][0]
        saved = E.create_policy(cand[0], cand[1], positions=True, expected=want)
        memory = module.PositionPolicy(MemoryPolicy(params, bstats), E._IDS)
        finals = {'saved': [], 'memory': []}
        for label, pol in [('saved', saved), ('memory', memory)]:
            pol.reset()
            for t in range(len(s['labels'])):
                obs = {k: s[k][t:t + 1] for k in M.OBS_KEYS}
                obs['mask_'] = None
                ch = s['chain'][t]
                info = dict(num_options=np.array([int(np.sum(s['actions_'][t, :, 3] != 0))]), to_play=np.zeros(1, np.int32),
                            public_chain_length=np.array([ch[0]]), public_chain_player=np.array([1 if ch[1] == 2 else 0]),
                            public_chain_card_id=np.array([ch[2]]), public_chain_effect=np.array([ch[3]]))
                finals[label].append(int(pol.act(obs, info)))
        result[name] = dict(final_actions_equal=finals['saved'] == finals['memory'], position_decisions=len(saved.positions.events),
                            position_changed_by_adapter=sum(1 for e in saved.positions.events if e.get('changed')))
    return result


def main():
    reg = json.loads((ROOT / 'registration.json').read_text())
    for name, digest in reg['code'].items():
        if sha(ROOT / 'code' / name) != digest:
            raise RuntimeError(f'registered code changed: {name}')
    E.verify_pins()
    out = ROOT / 'train'
    if (out / 'result.json').exists():
        raise SystemExit('training result exists; one candidate only')
    print('DEVICES', jax.devices(), flush=True)
    params, original_base, stats, frozen = train(reg, out)
    if stats['aborted']:
        (out / 'result.json').write_text(json.dumps(dict(status='aborted', **stats), ensure_ascii=False, indent=1, default=float) + '\n')
        print('ABORTED', stats['aborted'], flush=True)
        return
    (out / 'candidate').mkdir(exist_ok=True)
    M.save_pair(params, original_base, out / 'candidate/base.flax_model', out / 'candidate/chain-head.flax_model')
    for name in ['base.flax_model', 'chain-head.flax_model']:
        os.chmod(out / 'candidate' / name, 0o444)
    check = verify_saved(params, frozen, out)
    result = dict(status='complete', **stats, saved=check, adapter=adapter_check(params, out),
                  native_lua_errors=v.native.get_lua_error_count(), live_duels=v.native.get_env_resource_stats()['live_duels'])
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float) + '\n')
    print('RESULT', json.dumps({k: result[k] for k in ['status', 'total', 'learner', 'frozen', 'episodes', 'windows', 'minibatches', 'rehearsal_counts']}),
          json.dumps(check['parameter_change_norm']), flush=True)
    print('NATIVE_LUA_ERRORS', result['native_lua_errors'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
