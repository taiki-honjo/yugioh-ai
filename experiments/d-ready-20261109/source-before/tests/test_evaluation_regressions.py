#!/usr/bin/env python3
"""Regression test suite for scripts/evaluate.py."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path("/home/hatk8/projects/yugioh-ai")
SPEC = importlib.util.spec_from_file_location("evaluated", PROJECT / "scripts" / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def make_context(factory, timeout=600, max_steps=1000):
    c = m.EvalContext.__new__(m.EvalContext)
    c.num_envs = 2
    c.env_threads = 1
    c.env_id = "YGOPro-v1"
    c.deck_name = "fixture"
    c.max_options = 24
    c.n_history_actions = 32
    c.max_steps = max_steps
    c.timeout = timeout
    c.warmup_seconds = 0.0
    c.EnvPreprocess = lambda env, skip_mask=True: env
    c.ygoenv = SimpleNamespace(make=lambda **kw: factory())
    c.jax = SimpleNamespace(device_put=lambda x: x)
    c.jnp = np
    c.agent = SimpleNamespace(init_rnn_state=lambda n: np.zeros((n, 1)))
    c._step_model = lambda params, state, obs, active, done: (state, np.zeros(2, dtype=int))
    c.load_params = lambda spec: None
    return c


class TwoLengths:
    def reset(self):
        self.t = 0
        return {}, {"to_play": np.zeros(2, dtype=int), "num_options": np.ones(2, dtype=int)}

    def step(self, actions):
        self.t += 1
        done = np.array([self.t == 1, self.t == 3])
        return (
            {},
            done.astype(float),
            done,
            np.zeros(2, dtype=bool),
            {
                "to_play": np.zeros(2, dtype=int),
                "num_options": np.ones(2, dtype=int),
                "win_reason": np.ones(2, dtype=int),
                "step_time": np.zeros((2, 2)),
                "termination_reason": np.zeros(2, dtype=int),
            },
        )

    def close(self):
        pass


# 1. Batch calls/rows vs active decision counts and timing separation
random_spec = {"kind": "builtin", "strategy": "random"}
c = make_context(TwoLengths)
episodes, timing = c._run_single_wave_seat(
    {"kind": "checkpoint"}, random_spec, {"params": True}, None, 0, 1, 0
)
assert [e["env_steps"] for e in episodes] == [1, 3]
assert timing["cand_model_calls"] == 3 and timing["cand_model_rows"] == 6
assert timing["env_slot_steps_submitted"] == 6 and timing["env_steps_total"] == 4
assert timing["cand_decision_steps_total"] == 4
assert 0.0 <= timing["cand_model_sec"] <= timing["cand_model_batch_wall_sec"]
assert 0.0 <= timing["env_step_sec"] <= timing["env_step_batch_wall_sec"]


# 2. Explicit native termination_reason (0: natural, 1: step limit, 2: timeout)
class Terminal(TwoLengths):
    reason = 0

    def step(self, actions):
        obs, reward, term, trunc, info = super().step(actions)
        term[:] = True
        reward[:] = 1.0
        info["termination_reason"][:] = self.reason
        return obs, reward, term, trunc, info


class TimeoutEnv(Terminal):
    reason = 2


class LimitEnv(Terminal):
    reason = 1


c_to = make_context(TimeoutEnv)
rows_to, _ = c_to._run_single_wave_seat(random_spec, random_spec, None, None, 0, 1, 0)
assert all(
    e["abnormal"]
    and e["engine_timeout"]
    and e["outcome"] == "draw"
    and e["reason_label"] == "engine_timeout"
    for e in rows_to
)

c_lim = make_context(LimitEnv)
rows_lim, _ = c_lim._run_single_wave_seat(random_spec, random_spec, None, None, 0, 1, 0)
assert all(
    e["abnormal"] and e["truncated"] and e["reason_label"] == "truncated" for e in rows_lim
)

c_nat = make_context(Terminal, max_steps=1)
rows_nat, _ = c_nat._run_single_wave_seat(random_spec, random_spec, None, None, 0, 1, 0)
assert all(
    (not e["abnormal"]) and e["outcome"] == "win" and e["reason_label"] == "normal_lp0"
    for e in rows_nat
), "Natural win at step budget boundary with explicit termination_reason=0 must be preserved"


# 3. Paired Clopper-Pearson confidence interval at K=1 and K=16
lo_k1_0, hi_k1_0, se_k1_0, n_k1_0 = m.paired_wilson_ci95([0.0], wins=0, total=2)
lo_k1_1, hi_k1_1, se_k1_1, n_k1_1 = m.paired_wilson_ci95([1.0], wins=2, total=2)
assert n_k1_0 == 1.0 and n_k1_1 == 1.0
assert abs(hi_k1_0 - 0.975) < 1e-6 and abs(lo_k1_1 - 0.025) < 1e-6
# At p=0.2, both P1=0 ([0.0, 0.975]) and P1=1 ([0.025, 1.0]) cover p=0.2 -> 100% coverage
assert (lo_k1_0 <= 0.2 <= hi_k1_0) and (lo_k1_1 <= 0.2 <= hi_k1_1)

# At K=16, all-split 1-1 pairs must not collapse to zero width, and X=1 boundary must cover >= 95%
lo_split, hi_split, se_split, n_split = m.paired_wilson_ci95([0.5] * 16, wins=16, total=32)
assert n_split == 16.0 and se_split == 0.0
assert lo_split < 0.25 and hi_split > 0.75
lo_x1, hi_x1, _, _ = m.paired_wilson_ci95([1.0] + [0.0] * 15, wins=2, total=32)
p_just_below = max(0.0, lo_x1 - 1e-6)
assert (1.0 - p_just_below) ** 16 >= 0.95


# 4. Suspected timeout separation (including C++ handle_timeout() dummy state signature)
class SuspectedDummyEnv:
    def reset(self):
        return {}, {"to_play": np.zeros(2, dtype=int), "num_options": np.ones(2, dtype=int)}

    def step(self, actions):
        time.sleep(0.015)
        return (
            {},
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=bool),
            np.zeros(2, dtype=bool),
            {
                "num_options": np.ones(2, dtype=np.int32),
                "to_play": np.ones(2, dtype=np.int32),
                "is_selfplay": np.ones(2, dtype=np.int32),
                "win_reason": np.ones(2, dtype=np.int32),
                "reward": np.ones(2, dtype=np.float32),
                "step_time": np.zeros((2, 2), dtype=np.float64),
            },
        )

    def close(self):
        pass


c_susp = make_context(SuspectedDummyEnv, timeout=0.01)
s_susp = c_susp.evaluate_matchup("random", "random", PROJECT, num_episodes=4)
assert s_susp["terminations"]["timeout_suspected"] == 4
assert s_susp["terminations"]["normal_total"] == 0
assert s_susp["terminations"]["normal_lp0"] == 0
assert s_susp["terminations"]["abnormal_total"] == 4
assert s_susp["results"]["natural_only"]["episodes"] == 0
assert s_susp["results"]["natural_only"]["win_rate"] == 0.0
assert s_susp["results"]["verified_episodes"] == 0
assert s_susp["results"]["excluded_suspected_episodes"] == 4
assert s_susp["results"]["wins"] == 0 and s_susp["results"]["win_rate"] == 0.0
assert s_susp["results"]["seed_pairs"]["verified_pairs"] == 0
assert s_susp["results"]["seed_pairs"]["suspected_or_timeout_pairs"] == 2
assert s_susp["results"]["seed_pairs"]["sweep_2_0_wins"] == 0
assert s_susp["results"]["raw_including_suspected"]["wins"] == 4
assert all(
    e["timeout_suspected"]
    and e["abnormal"]
    and e["reason_label"] == "timeout_suspected"
    and e["outcome"] == "win"
    for e in s_susp["episodes"]
)


# 5. Confirmed engine timeouts (termination_reason=2 / timeout=True) are excluded pair-wise
class ConfirmedTimeoutEnv:
    extra: dict = {}

    def reset(self):
        return {}, {"to_play": np.zeros(2, dtype=int), "num_options": np.ones(2, dtype=int)}

    def step(self, actions):
        info = {
            "to_play": np.zeros(2, dtype=int),
            "num_options": np.ones(2, dtype=int),
            "is_selfplay": np.zeros(2, dtype=int),
            "win_reason": np.ones(2, dtype=int),
            "step_time": np.zeros((2, 2)),
        }
        info.update({k: np.array(v) for k, v in self.extra.items()})
        return {}, np.array([1.0, -1.0]), np.ones(2, dtype=bool), np.zeros(2, dtype=bool), info

    def close(self):
        pass


def run_confirmed(extra):
    cls = type("Env", (ConfirmedTimeoutEnv,), {"extra": extra})
    ctx = make_context(cls, timeout=10)
    return ctx.evaluate_matchup("random", "random", PROJECT, num_episodes=4)


# 5a. All slots termination_reason=2 -> no verified pairs at all
s_all = run_confirmed({"termination_reason": [2, 2]})
r_all = s_all["results"]
assert s_all["terminations"]["engine_timeout"] == 4
assert r_all["verified_episodes"] == 0 and r_all["excluded_timeout_or_suspected_episodes"] == 4
assert (r_all["wins"], r_all["losses"], r_all["draws"]) == (0, 0, 0)
assert r_all["seed_pairs"]["verified_pairs"] == 0
assert r_all["seed_pairs"]["suspected_or_timeout_pairs"] == 2
assert r_all["win_rate_ci95"] == [0.0, 1.0] and r_all["effective_sample_size"] == 0.0
assert r_all["requires_rerun_or_exclusion"] is True
assert r_all["natural_only"]["episodes"] == 0
assert r_all["raw_including_suspected"]["draws"] == 4
assert all(e["included_in_verified_score"] is False for e in s_all["episodes"])

# 5b/5c. Only slot 0 timed out (termination_reason or timeout flag) -> pair e00 excluded, e01 kept
for extra in ({"termination_reason": [2, 0]}, {"timeout": [True, False]}):
    s_one = run_confirmed(extra)
    r_one = s_one["results"]
    assert r_one["verified_episodes"] == 2, extra
    assert r_one["seed_pairs"]["verified_pairs"] == 1, extra
    assert r_one["seed_pairs"]["suspected_or_timeout_pairs"] == 1, extra
    # Slot 1 receives reward -1 in both seats -> verified pair is a 0-2 sweep loss, no draws
    assert (r_one["wins"], r_one["losses"], r_one["draws"]) == (0, 2, 0), extra
    assert r_one["seed_pairs"]["sweep_0_2_losses"] == 1, extra
    assert r_one["effective_sample_size"] == 1.0, extra
    assert abs(r_one["win_rate_ci95"][1] - 0.975) < 1e-4, extra
    flags = {(e["pair_id"], e["candidate_seat"]): e["included_in_verified_score"] for e in s_one["episodes"]}
    assert flags == {
        ("w000_e00", 0): False,
        ("w000_e00", 1): False,
        ("w000_e01", 0): True,
        ("w000_e01", 1): True,
    }, extra


# 6. Large-K binomial CI: no overflow and agreement with exact references
def exact_cp_reference(scores):
    """Old exact comb-based CI (valid for moderate K) used as reference."""
    k = len(scores)
    s = sum(scores)
    j_low, j_high = math.floor(s + 1e-12), math.ceil(s - 1e-12)

    def cdf(j, p):
        return sum(math.comb(k, i) * p ** i * (1 - p) ** (k - i) for i in range(j + 1))

    def bisect(fn, target):
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if fn(mid) > target:
                lo = mid
            else:
                hi = mid
        return lo, hi

    low = 0.0 if j_low <= 0 else bisect(lambda p: cdf(j_low - 1, p), 0.975)[0]
    high = 1.0 if j_high >= k else bisect(lambda p: cdf(j_high, p), 0.025)[1]
    return low, high


for k in (1, 2, 16, 64, 256):
    for scores in ([0.0] * k, [1.0] * k, [0.5] * k, [1.0] + [0.0] * (k - 1), [0.0] + [1.0] * (k - 1)):
        lo, hi, _, _ = m.paired_wilson_ci95(scores)
        ref_lo, ref_hi = exact_cp_reference(scores)
        assert abs(lo - ref_lo) < 1e-9 and abs(hi - ref_hi) < 1e-9, (k, scores[:2], lo, ref_lo, hi, ref_hi)

for k in (1024, 2048, 4096, 8192):
    for label, scores in (
        ("all_split", [0.5] * k),
        ("all_loss", [0.0] * k),
        ("all_win", [1.0] * k),
        ("one_win", [1.0] + [0.0] * (k - 1)),
        ("one_loss", [0.0] + [1.0] * (k - 1)),
        ("quarter", [1.0] * (k // 4) + [0.0] * (k - k // 4)),
    ):
        lo, hi, _, n = m.paired_wilson_ci95(scores)
        mean = sum(scores) / k
        assert n == float(k)
        assert 0.0 <= lo <= mean <= hi <= 1.0, (k, label, lo, mean, hi)
        # Defining equations of Clopper-Pearson bounds hold at the returned endpoints
        s = sum(scores)
        j_low, j_high = math.floor(s + 1e-12), math.ceil(s - 1e-12)
        if j_low > 0:
            assert abs(m._binom_sf(j_low, k, lo) - 0.025) < 1e-6, (k, label, "low")
        if j_high < k:
            assert abs(m._binom_cdf(j_high, k, hi) - 0.025) < 1e-6, (k, label, "high")
    # Endpoint closed forms: all-loss upper = 1 - 0.025**(1/k), all-win lower = 0.025**(1/k)
    assert abs(m.paired_wilson_ci95([0.0] * k)[1] - (1 - 0.025 ** (1 / k))) < 1e-9
    assert abs(m.paired_wilson_ci95([1.0] * k)[0] - 0.025 ** (1 / k)) < 1e-9
    # Symmetry: all-split interval is symmetric around 0.5
    lo_s, hi_s, _, _ = m.paired_wilson_ci95([0.5] * k)
    assert abs((0.5 - lo_s) - (hi_s - 0.5)) < 1e-9

result = {
    "checks": [
        "full_batch_rows_6_vs_useful_decisions_4",
        "native_termination_reason_handling_and_boundary_win_preserved",
        "paired_clopper_pearson_ci95_k1_and_k16_boundary_coverage",
        "timeout_suspected_and_native_dummy_separated_from_natural_and_verified_results",
        "confirmed_engine_timeout_pairs_excluded_from_verified_results",
        "large_k_binomial_ci_stable_and_matches_exact_reference",
    ],
    "passed": 6,
}
print(json.dumps(result, indent=2))
