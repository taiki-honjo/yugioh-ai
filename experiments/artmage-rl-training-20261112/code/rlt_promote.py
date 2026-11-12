"""Amendment 1: verify the saved final training state from files only and promote it to the candidate.

The training job used the full budget (32,768 decisions, 996 updates); after the last update, the harness's own
resource check (RealHooks.close_env asserted live_duels == 0 while collect() still referenced the cut episode's env)
raised, so the job wrote status 'not_established' and saved the final state to train/stopped-state/. result.json is
kept as written. This script (CPU, no duel) checks that record and the files, then copies them to train/candidate/:
  - result.json: total 32,768, 996 minibatches, 249 windows, the last window ends at 32,768 and its 4 minibatches exist,
    error = the close_env assertion only, live_duels 0 after the reference was dropped, params finite, no non-finite update
  - file hashes = result.json stopped_state; weights finite; both parts changed vs the start model
  - on all 40 rehearsal histories: logits of the file loaded by rlt_model.load_pair equal the isolated entry's
    Policy (hash-pinned) exactly, argmax equal, and the position adapter's final actions equal
    (the in-memory params of the finished job no longer exist, so memory-vs-file cannot be compared)
  - rehearsal fit of the candidate (diagnostic)
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/home/hatk8/projects/yugioh-ai/experiments/artmage-rl-training-20261112')
sys.path.insert(0, str(ROOT / 'code'))
import rlt_train as RT  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_checks():
    res = json.loads((ROOT / 'train/result.json').read_text())
    wins = [json.loads(l) for l in (ROOT / 'train/windows.jsonl').read_text().splitlines()]
    mbs = [json.loads(l) for l in (ROOT / 'train/minibatches.jsonl').read_text().splitlines()]
    log = (ROOT / 'logs/train.log').read_text()
    tb = log[log.index('Traceback'):log.index('RESULT')]
    checks = dict(total=res['total'] == RT.CAP, minibatches=res['minibatches'] == 996 == len(mbs), windows=res['windows'] == 249 == len(wins),
                  last_window_at_cap=wins[-1]['end_total'] == RT.CAP and all(i in {m['minibatch'] for m in mbs} for i in wins[-1]['minibatches']),
                  error_is_close_assertion=res['error'] == 'close: AssertionError: ' and 'in close_env' in tb and "['live_duels'] == 0" in tb and tb.count('Traceback') == 1,
                  live_duels_after=res['live_duels'] == 0, params_finite=res['params_finite'], no_nonfinite_update=res['adapter_window_extra']['total_notfinite'] == 0,
                  lua_errors=res['native_lua_errors'] == 0, stopped_state_hashes=(sha(ROOT / 'train/stopped-state/base.flax_model'), sha(ROOT / 'train/stopped-state/chain-head.flax_model'))
                  == (res['stopped_state']['base_sha256'], res['stopped_state']['head_sha256']))
    return checks, res, tb


def file_checks(H, pair, want):
    jax, jnp, np, E, M = H.jax, H.jnp, H.np, H.E, H.M
    params, bstats, _ = M.load_pair(*pair)
    entry = E.create_policy(pair[0], pair[1], positions=False, expected=want)
    items = [s for g in sorted(H.existing) for s in H.existing[g]] + [s for g in sorted(H.pairs) for p in H.pairs[g] for _, _, s in p['hands']]
    max_diff, argmax_equal = 0.0, True
    for s in items:
        st_a, st_b = M.init_state(1), entry.state
        for t in range(len(s['labels'])):
            o = {k: jnp.asarray(s[k][t:t + 1]) for k in M.OBS_KEYS}
            ch = jnp.asarray(s['chain'][t:t + 1])
            st_a, l_a, _ = M.infer(params, bstats, st_a, o, ch)
            st_b, l_b = entry.infer(entry.base, entry.head, st_b, dict(o, mask_=None), ch)
            max_diff = max(max_diff, float(jnp.max(jnp.abs(l_a - l_b))))
            argmax_equal &= int(jnp.argmax(l_a[0])) == int(jnp.argmax(l_b[0]))
    features = H.features

    class Loaded:
        def __init__(self):
            self.reset()

        def reset(self):
            self.state = M.init_state(1)

        def act(self, obs, info):
            o = {k: jnp.asarray(obs[k]) for k in M.OBS_KEYS}
            self.state, logits, _ = M.infer(params, bstats, self.state, o, jnp.asarray(features(info)))
            return int(np.asarray(logits)[0].argmax())
    module = E._adapter()
    adapter_equal = True
    for s in items:
        finals = []
        for pol in [E.create_policy(pair[0], pair[1], positions=True, expected=want), module.PositionPolicy(Loaded(), E._IDS)]:
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
    finite = {name: all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ser.msgpack_restore(path.read_bytes())))
              for name, path in zip(['base', 'head'], pair)}
    delta = {part: float(M.tree_norm(jax.tree.map(lambda a, b: a - b, params[part], H.frozen[part]))) for part in ['base', 'head']}
    return params, dict(candidate_base_sha256=want[0], candidate_head_sha256=want[1], reload_through_isolated_entry=True,
                        max_abs_logit_diff_load_pair_vs_entry=max_diff, argmax_equal=argmax_equal, adapter_final_actions_equal=adapter_equal,
                        saved_weights_finite=finite, sequences_checked=len(items), parameter_change_norm=delta,
                        both_parts_changed=all(x > 0 for x in delta.values()),
                        not_checked='memory-vs-file (the in-memory params of the finished job no longer exist)')


def main():
    amend = json.loads((ROOT / 'amendment-1.json').read_text())
    for n, digest in amend['code'].items():
        if sha(ROOT / 'code' / n) != digest:
            raise RuntimeError(f'registered code changed: {n}')
    if (ROOT / 'train/promotion.json').exists() or (ROOT / 'train/candidate').exists():
        raise SystemExit('promoted already')
    t0 = time.time()
    checks, res, tb = record_checks()
    if not all(checks.values()):
        raise SystemExit(f'record checks failed: {checks}')
    src = (ROOT / 'train/stopped-state/base.flax_model', ROOT / 'train/stopped-state/chain-head.flax_model')
    want = (res['stopped_state']['base_sha256'], res['stopped_state']['head_sha256'])
    H = RT.RealHooks(json.loads((ROOT / 'registration.json').read_text()))  # loads the start params (frozen copy) and the 40 sequences; no duel
    H.E.verify_pins()
    params, saved = file_checks(H, src, want)
    ok = (saved['max_abs_logit_diff_load_pair_vs_entry'] == 0.0 and saved['argmax_equal'] and saved['adapter_final_actions_equal']
          and all(saved['saved_weights_finite'].values()) and saved['both_parts_changed'])
    fit = RT.rehearsal_fit(H, params)
    out = dict(amendment=1, status='promoted' if ok else 'not_promoted', record_checks=checks, traceback=tb.strip().splitlines(), saved=saved,
               rehearsal_fit=dict(start=res['rehearsal_fit']['start'], candidate=fit), source='train/stopped-state', seconds=time.time() - t0)
    if ok:
        (ROOT / 'train/candidate').mkdir()
        for p in src:
            shutil.copyfile(p, ROOT / 'train/candidate' / p.name)
            os.chmod(ROOT / 'train/candidate' / p.name, 0o444)
        out['candidate'] = dict(base=str(ROOT / 'train/candidate/base.flax_model'), head=str(ROOT / 'train/candidate/chain-head.flax_model'),
                                base_sha256=sha(ROOT / 'train/candidate/base.flax_model'), head_sha256=sha(ROOT / 'train/candidate/chain-head.flax_model'))
        assert (out['candidate']['base_sha256'], out['candidate']['head_sha256']) == want
    RT.write_json(ROOT / 'train/promotion.json', out)
    print('PROMOTION', json.dumps(dict(status=out['status'], checks=checks, saved={k: saved[k] for k in ['max_abs_logit_diff_load_pair_vs_entry', 'argmax_equal',
                                                                                                     'adapter_final_actions_equal', 'saved_weights_finite', 'parameter_change_norm']})), flush=True)


if __name__ == '__main__':
    main()
