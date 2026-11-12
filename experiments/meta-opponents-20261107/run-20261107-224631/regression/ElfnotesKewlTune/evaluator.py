#!/usr/bin/env python3
"""
yugioh-ai 評価スクリプト (scripts/evaluate.py)

目的:
  - 学習に使っていない乱数シード（デフォルト: seed=1000）を使用
  - 同一シードで先攻（Player 0）・後攻（Player 1）を入れ替えたペア対戦を実施
  - 並列環境の「短い対戦への偏り（short-game bias）」を防ぐため、各ウェーブで
    各環境スロットから最初の1戦のみを収集
  - 相手の非公開手札・山札情報をマスク（oppo_info=False, skip_mask=True）
  - 「勝敗（総合・先攻・後攻・先後ペア内訳）」「決着理由・異常終了」「判断時間（ウォームアップ・モデル推論・エンジン処理）」を分離して JSON / CSV / Markdown に記録
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 学習時に使用されたシード（誤用防止の警告用）
TRAINING_SEEDS_USED = {0}
TRIAL_EVAL_SEEDS_USED = {100}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci95(wins: int, total: int) -> Tuple[float, float]:
    """独立な二項試行（勝率）の95%信頼区間（Wilson score interval）を計算する。"""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% CI
    p = wins / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def _binom_log_pmf(i: int, k: int, log_p: float, log_q: float, lgamma_k1: float) -> float:
    """二項分布 Binomial(k, p) の対数確率質量関数 log P(X = i) を返す。"""
    return (
        lgamma_k1
        - math.lgamma(i + 1)
        - math.lgamma(k - i + 1)
        + i * log_p
        + (k - i) * log_q
    )


def _binom_cdf(j: int, k: int, p: float) -> float:
    """
    二項分布 Binomial(k, p) の累積分布関数 P(X <= j) を数値的に安定に計算する。

    math.comb(k, i) の直接 float 変換を行わず lgamma / log1p を用いるため、
    K=2048 以上の大標本でも OverflowError やアンダーフローを起こさない。
    """
    if j < 0:
        return 0.0
    if j >= k:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    lgamma_k1 = math.lgamma(k + 1)
    mode = int(math.floor((k + 1) * p))

    if j <= mode:
        total = 0.0
        for i in range(j, -1, -1):
            term = math.exp(_binom_log_pmf(i, k, log_p, log_q, lgamma_k1))
            total += term
            if i < mode and term < total * 1e-16:
                break
        return min(1.0, max(0.0, total))

    upper = 0.0
    for i in range(j + 1, k + 1):
        term = math.exp(_binom_log_pmf(i, k, log_p, log_q, lgamma_k1))
        upper += term
        if i > mode and term < upper * 1e-16:
            break
    return min(1.0, max(0.0, 1.0 - upper))


def _binom_sf(j: int, k: int, p: float) -> float:
    """
    二項分布 Binomial(k, p) の生存関数（上側尾確率）P(X >= j) を数値的に安定に計算する。
    """
    if j <= 0:
        return 1.0
    if j > k:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    lgamma_k1 = math.lgamma(k + 1)
    mode = int(math.floor((k + 1) * p))

    if j > mode:
        upper = 0.0
        for i in range(j, k + 1):
            term = math.exp(_binom_log_pmf(i, k, log_p, log_q, lgamma_k1))
            upper += term
            if i > mode and term < upper * 1e-16:
                break
        return min(1.0, max(0.0, upper))

    lower = 0.0
    for i in range(j - 1, -1, -1):
        term = math.exp(_binom_log_pmf(i, k, log_p, log_q, lgamma_k1))
        lower += term
        if i < mode and term < lower * 1e-16:
            break
    return min(1.0, max(0.0, 1.0 - lower))


def paired_wilson_ci95(
    pair_scores: List[float],
    wins: int = 0,
    total: int = 0,
) -> Tuple[float, float, float, float]:
    """
    独立なシードペア K 組 (P_i in [0, 1]) に基づく勝率の保守的95%信頼区間を計算する。

    - K=1 を含む任意のペア数 K >= 1 について、個別対戦数 (total=2K) の独立二項試行へは戻さず、
      常にシードペア数 K (n_eff = K) を独立な観測単位として扱う。
    - ペアスコア合計 S = sum_{i=1}^K P_i in [0, K] に対し、下端には j_low = floor(S)、
      上端には j_high = ceil(S) を用いた Clopper-Pearson 正確二項区間（両側 alpha=0.05）を算出する。
    - これにより、2値ペアスコア P_i in {0, 1} では任意の K >= 1 および任意の母平均 p in (0, 1) で
      被覆率 >= 95% が成立し、3値ペアスコア P_i in {0, 0.5, 1} でも全ペア1勝1敗 (S=K/2)・全勝・全敗・
      まれな2連勝を含む境界走査（K=1, 2, 3, 4, 8, 16, 32, 64, 128）で被覆率 >= 95% を満たす。
      （※一般の連続 [0, 1] 分布における二項分布との裾確率比較には Bentkus (2004) 等の定数因子が
       関与するため、本区間は {0, 0.5, 1} ペア標本に対する保守的 Clopper-Pearson 区間として定義する。）

    Returns:
        (ci_low, ci_high, pair_se, n_eff)
    """
    k = len(pair_scores)
    if k <= 0:
        return 0.0, 1.0, 0.0, 0.0

    s_sum = max(0.0, min(float(k), float(sum(pair_scores))))
    p = s_sum / k
    if k > 1:
        s2 = sum((float(x) - p) ** 2 for x in pair_scores) / (k - 1)
        pair_se = math.sqrt(s2 / k)
    else:
        pair_se = 0.0

    n_eff = float(k)
    j_low = int(math.floor(s_sum + 1e-12))
    j_high = int(math.ceil(s_sum - 1e-12))

    # 下端: P(Binomial(k, low) >= j_low) = 0.025
    if j_low <= 0:
        low = 0.0
    else:
        target_tail = 0.025
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if _binom_sf(j_low, k, mid) < target_tail:
                lo = mid
            else:
                hi = mid
        low = lo

    # 上端: P(Binomial(k, high) <= j_high) = 0.025
    if j_high >= k:
        high = 1.0
    else:
        target_high = 0.025
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if _binom_cdf(j_high, k, mid) > target_high:
                lo = mid
            else:
                hi = mid
        high = hi

    return low, high, pair_se, n_eff


def resolve_agent_spec(spec: str, proj_dir: Path) -> Dict[str, Any]:
    """
    エージェント指定文字列を解釈する。
      - "random": エンジン内蔵ランダム行動
      - "greedy": エンジン内蔵ヒューリスティック（bot）
      - "init":   学習前の初期モデル（seed=0の初期重み、0判断時点）
      - その他:   .flax_model チェックポイントへのファイルパス
    """
    lower = spec.strip().lower()
    if lower in ("random", "greedy"):
        return {
            "kind": "builtin",
            "strategy": lower,
            "label": lower,
            "path": None,
            "sha256": None,
        }
    if lower == "init":
        return {
            "kind": "model_init",
            "strategy": "init",
            "label": "init (0-step untrained)",
            "path": None,
            "sha256": None,
        }
    p = Path(spec)
    if not p.is_absolute():
        cand = proj_dir / p
        if cand.exists():
            p = cand
        else:
            p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {spec} (解決先: {p})")
    return {
        "kind": "checkpoint",
        "strategy": "checkpoint",
        "label": p.name,
        "path": str(p),
        "sha256": sha256_file(p),
    }


def make_wave_seed(base_seed: int, wave_idx: int) -> int:
    """
    ベースシードとウェーブ番号から決定論的に環境シードを生成する。
    先攻パスと後攻パスで同じ wave_idx を渡すことで、完全に同じ初期乱数状態を共有する。
    """
    rng = random.Random(base_seed * 1000003 + wave_idx * 9973 + 100000)
    return rng.randint(1, 100_000_000)


class EvalContext:
    """JAXモデルとYGOPro環境初期化を保持し、複数マッチアップを高速に評価するコンテキスト。"""

    def __init__(
        self,
        runtime_scripts_dir: Path,
        deck_path: Path,
        env_id: str = "YGOPro-v1",
        lang: str = "english",
        code_list_file: str = "code_list.txt",
        max_options: int = 24,
        n_history_actions: int = 32,
        num_envs: int = 8,
        env_threads: int = 4,
        max_steps: int = 1000,
        timeout: int = 600,
        init_train_seed: int = 0,
    ):
        self.runtime_scripts_dir = runtime_scripts_dir
        self.deck_path = deck_path
        self.env_id = env_id
        self.lang = lang
        self.code_list_file = code_list_file
        self.max_options = max_options
        self.n_history_actions = n_history_actions
        self.num_envs = num_envs
        self.env_threads = min(env_threads, num_envs)
        self.max_steps = max_steps
        self.timeout = timeout
        self.init_train_seed = init_train_seed

        # YGOProエンジンはカレントディレクトリの ./script 等を参照するため移動
        os.chdir(self.runtime_scripts_dir)
        if str(self.runtime_scripts_dir) not in sys.path:
            sys.path.insert(0, str(self.runtime_scripts_dir))

        import ygoenv
        import jax
        import jax.numpy as jnp
        import flax
        from jax.experimental.compilation_cache import compilation_cache as cc
        from ygoai.utils import init_ygopro
        from ygoai.rl.utils import EnvPreprocess
        from ygoai.rl.jax.agent import RNNAgent, ModelArgs

        self.ygoenv = ygoenv
        self.jax = jax
        self.jnp = jnp
        self.flax = flax
        self.EnvPreprocess = EnvPreprocess
        self.RNNAgent = RNNAgent
        self.ModelArgs = ModelArgs

        cc.set_cache_dir(os.path.expanduser("~/.cache/jax"))

        deck_name, deck_names = init_ygopro(
            self.env_id,
            self.lang,
            str(self.deck_path),
            self.code_list_file,
            return_deck_names=True,
        )
        self.deck_name = deck_name
        self.deck_names = sorted(deck_names)

        # 観測空間取得用のダミー環境を作成してすぐ閉じる
        dummy_envs = ygoenv.make(
            task_id=self.env_id,
            env_type="gymnasium",
            num_envs=2,
            num_threads=1,
            seed=1,
            deck1=self.deck_name,
            deck2=self.deck_name,
            player=-1,
            max_options=self.max_options,
            n_history_actions=self.n_history_actions,
            play_mode="self",
            async_reset=False,
            oppo_info=False,
            max_steps=self.max_steps,
            max_episode_steps=self.max_steps,
            timeout=self.timeout,
        )
        dummy_envs.num_envs = 2
        dummy_envs = EnvPreprocess(dummy_envs, skip_mask=True)
        obs_space = dummy_envs.observation_space
        sample_obs = jax.tree.map(lambda x: jnp.array([x]), obs_space.sample())
        dummy_envs.close()
        del dummy_envs

        self.model_args = ModelArgs()
        self.agent = RNNAgent(**asdict(self.model_args), embedding_shape=None)

        # cleanba.py の --seed 0 と同一の初期重み生成
        init_rng = random.Random(self.init_train_seed)
        init_seed_val = init_rng.randint(0, int(1e8))
        self.init_seed_val = init_seed_val
        init_key = jax.random.PRNGKey(init_seed_val)
        rstate_1 = self.agent.init_rnn_state(1)
        self.base_variables = jax.jit(self.agent.init)(init_key, sample_obs, rstate_1)
        self.base_variables = jax.device_put(self.base_variables)

        # パラメータキャッシュ
        self._params_cache: Dict[str, Any] = {}

        # JITコンパイル済みの1ステップ推論関数（単一モデル用）
        agent_ref = self.agent

        @jax.jit
        def _step_model(params, rstate, obs, active_mask, prev_done):
            # 直前ステップで終了していた環境のRNN状態を0にリセット
            rstate = jax.tree.map(lambda x: jnp.where(prev_done[:, None], 0, x), rstate)
            next_rstate, logits, value = agent_ref.apply(params, obs, rstate)[:3]
            # このプレイヤーの手番である環境のみRNN状態を更新
            rstate = jax.tree.map(
                lambda nxt, cur: jnp.where(active_mask[:, None], nxt, cur),
                next_rstate,
                rstate,
            )
            actions = jnp.argmax(logits, axis=-1)
            return rstate, actions

        self._step_model = _step_model
        self._jit_warmed_up = False
        self.warmup_seconds = 0.0

    def load_params(self, agent_spec: Dict[str, Any]) -> Any:
        kind = agent_spec["kind"]
        if kind == "builtin":
            return None
        if kind == "model_init":
            return self.base_variables
        ckpt_path = agent_spec["path"]
        if ckpt_path not in self._params_cache:
            with open(ckpt_path, "rb") as f:
                loaded = self.flax.serialization.from_bytes(self.base_variables, f.read())
            self._params_cache[ckpt_path] = self.jax.device_put(loaded)
        return self._params_cache[ckpt_path]

    def warmup_jit(self) -> float:
        """初回推論のXLAコンパイル時間を測定し、本計測から分離する。"""
        if self._jit_warmed_up:
            return self.warmup_seconds
        envs = self.ygoenv.make(
            task_id=self.env_id,
            env_type="gymnasium",
            num_envs=self.num_envs,
            num_threads=self.env_threads,
            seed=999999,
            deck1=self.deck_name,
            deck2=self.deck_name,
            player=0,
            max_options=self.max_options,
            n_history_actions=self.n_history_actions,
            play_mode="random",
            async_reset=False,
            oppo_info=False,
            max_steps=self.max_steps,
            max_episode_steps=self.max_steps,
            timeout=self.timeout,
        )
        envs.num_envs = self.num_envs
        envs = self.EnvPreprocess(envs, skip_mask=True)
        obs, _ = envs.reset()
        rstate = self.jax.device_put(self.agent.init_rnn_state(self.num_envs))
        active = self.jnp.ones(self.num_envs, dtype=self.jnp.bool_)
        dones = self.jnp.zeros(self.num_envs, dtype=self.jnp.bool_)

        t0 = time.perf_counter()
        rstate, actions = self._step_model(self.base_variables, rstate, obs, active, dones)
        _ = np.asarray(actions)
        self.warmup_seconds = time.perf_counter() - t0
        self._jit_warmed_up = True
        envs.close()
        del envs
        return self.warmup_seconds

    def _run_single_wave_seat(
        self,
        cand_spec: Dict[str, Any],
        opp_spec: Dict[str, Any],
        cand_params: Any,
        opp_params: Any,
        wave_idx: int,
        wave_seed: int,
        candidate_seat: int,  # 0: 先攻(Player 0), 1: 後攻(Player 1)
        max_steps_per_wave: int = 4000,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        指定した wave_seed と candidate_seat（0=先攻 または 1=後攻）で
        num_envs 個の並列環境から各1戦ずつ（計 num_envs 戦）を偏りなく収集する。
        """
        num_envs = self.num_envs
        opp_is_builtin = opp_spec["kind"] == "builtin"
        cand_is_builtin = cand_spec["kind"] == "builtin"

        if opp_is_builtin:
            play_mode = "bot" if opp_spec["strategy"] == "greedy" else "random"
            env_player = candidate_seat
        elif cand_is_builtin:
            # 候補側が組み込みボット、相手がモデルの場合
            play_mode = "bot" if cand_spec["strategy"] == "greedy" else "random"
            env_player = 1 - candidate_seat
        else:
            play_mode = "self"
            env_player = -1

        envs = self.ygoenv.make(
            task_id=self.env_id,
            env_type="gymnasium",
            num_envs=num_envs,
            num_threads=self.env_threads,
            seed=wave_seed,
            deck1=self.deck_name,
            deck2=self.deck_name,
            player=env_player,
            max_options=self.max_options,
            n_history_actions=self.n_history_actions,
            play_mode=play_mode,
            async_reset=False,
            oppo_info=False,
            max_steps=self.max_steps,
            max_episode_steps=self.max_steps,
            timeout=self.timeout,
        )
        envs.num_envs = num_envs
        envs = self.EnvPreprocess(envs, skip_mask=True)

        # 乱数シード（候補または相手がPython側random戦略をとる場合のため固定）
        np_rng = np.random.default_rng(wave_seed + candidate_seat * 17)

        # 各環境スロットの累積統計（最初の1エピソード完了まで）
        ep_returns = np.zeros(num_envs, dtype=np.float64)
        ep_env_steps = np.zeros(num_envs, dtype=np.int32)
        ep_cand_decisions = np.zeros(num_envs, dtype=np.int32)
        ep_opp_decisions = np.zeros(num_envs, dtype=np.int32)
        ep_cand_step_ms = np.zeros(num_envs, dtype=np.float64)
        ep_opp_step_ms = np.zeros(num_envs, dtype=np.float64)
        ep_max_step_wall_ms = np.zeros(num_envs, dtype=np.float64)
        collected = np.zeros(num_envs, dtype=np.bool_)

        episodes: List[Optional[Dict[str, Any]]] = [None] * num_envs

        cand_model_sec = 0.0
        opp_model_sec = 0.0
        env_step_sec = 0.0
        cand_model_batch_wall_sec = 0.0
        opp_model_batch_wall_sec = 0.0
        env_step_batch_wall_sec = 0.0
        cand_decision_steps_total = 0
        opp_decision_steps_total = 0
        env_steps_total = 0
        cand_model_calls = 0
        opp_model_calls = 0
        env_step_calls = 0
        cand_model_rows = 0
        opp_model_rows = 0
        env_slot_steps_submitted = 0

        wave_t0 = time.perf_counter()
        step_count = 0
        wave_timeout_abort = False

        try:
            obs, infos = envs.reset()
        except RuntimeError as exc:
            if "timeout" in str(exc).lower():
                wave_timeout_abort = True
                obs, infos = None, {}
            else:
                envs.close()
                raise

        next_to_play = infos.get("to_play", np.zeros(num_envs, dtype=np.int32))
        prev_dones = np.zeros(num_envs, dtype=np.bool_)

        rstate_cand = (
            self.jax.device_put(self.agent.init_rnn_state(num_envs))
            if cand_params is not None and not wave_timeout_abort
            else None
        )
        rstate_opp = (
            self.jax.device_put(self.agent.init_rnn_state(num_envs))
            if opp_params is not None and not wave_timeout_abort
            else None
        )

        while not wave_timeout_abort and not np.all(collected) and step_count < max_steps_per_wave:
            active_uncollected = ~collected
            to_play = next_to_play

            if play_mode == "self":
                cand_turn_mask = to_play == candidate_seat
                opp_turn_mask = ~cand_turn_mask
            elif opp_is_builtin:
                # 相手が組み込みボットの場合、Python側に返ってくるステップはすべて候補の手番
                cand_turn_mask = np.ones(num_envs, dtype=np.bool_)
                opp_turn_mask = np.zeros(num_envs, dtype=np.bool_)
            else:
                # 候補が組み込みボット、相手がモデルの場合
                cand_turn_mask = np.zeros(num_envs, dtype=np.bool_)
                opp_turn_mask = np.ones(num_envs, dtype=np.bool_)

            cand_active_mask = cand_turn_mask & active_uncollected
            opp_active_mask = opp_turn_mask & active_uncollected
            n_cand_active = int(np.sum(cand_active_mask))
            n_opp_active = int(np.sum(opp_active_mask))
            n_env_active = int(np.sum(active_uncollected))

            # 1. 候補の行動選択（未収集の有効エピソードで手番がある場合のみ実行）
            actions = np.zeros(num_envs, dtype=np.int32)
            if n_cand_active > 0:
                if cand_params is not None:
                    t0 = time.perf_counter()
                    rstate_cand, cand_acts_jax = self._step_model(
                        cand_params,
                        rstate_cand,
                        obs,
                        self.jnp.asarray(cand_active_mask),
                        self.jnp.asarray(prev_dones),
                    )
                    cand_acts = np.asarray(cand_acts_jax)
                    dt = time.perf_counter() - t0
                    cand_model_batch_wall_sec += dt
                    # 固定バッチ(num_envs)推論のうち、有効な判断対象(n_cand_active)に対応する推論時間のみを計上し、
                    # 推論時間と判断数の集計対象を一致させる
                    cand_model_sec += dt * (n_cand_active / float(num_envs))
                    cand_decision_steps_total += n_cand_active
                    cand_model_calls += 1
                    cand_model_rows += num_envs
                    actions = np.where(cand_active_mask, cand_acts, actions)
                else:
                    if cand_spec["strategy"] == "random":
                        num_opts = np.maximum(1, infos["num_options"])
                        cand_acts = np_rng.integers(0, num_opts, size=num_envs, dtype=np.int32)
                    else:
                        cand_acts = np.zeros(num_envs, dtype=np.int32)
                    cand_decision_steps_total += n_cand_active
                    actions = np.where(cand_turn_mask, cand_acts, actions)

            # 2. 相手の行動選択（未収集の有効エピソードで手番がある場合のみ実行）
            if n_opp_active > 0:
                if opp_params is not None:
                    t0 = time.perf_counter()
                    rstate_opp, opp_acts_jax = self._step_model(
                        opp_params,
                        rstate_opp,
                        obs,
                        self.jnp.asarray(opp_active_mask),
                        self.jnp.asarray(prev_dones),
                    )
                    opp_acts = np.asarray(opp_acts_jax)
                    dt = time.perf_counter() - t0
                    opp_model_batch_wall_sec += dt
                    opp_model_sec += dt * (n_opp_active / float(num_envs))
                    opp_decision_steps_total += n_opp_active
                    opp_model_calls += 1
                    opp_model_rows += num_envs
                    actions = np.where(opp_active_mask, opp_acts, actions)
                else:
                    if opp_spec["strategy"] == "random":
                        num_opts = np.maximum(1, infos["num_options"])
                        opp_acts = np_rng.integers(0, num_opts, size=num_envs, dtype=np.int32)
                    else:
                        opp_acts = np.zeros(num_envs, dtype=np.int32)
                    opp_decision_steps_total += n_opp_active
                    actions = np.where(opp_turn_mask, opp_acts, actions)

            # 3. 環境のステップ実行
            t0 = time.perf_counter()
            try:
                obs, rewards, terminated, truncated, infos = envs.step(actions)
            except RuntimeError as exc:
                if "timeout" in str(exc).lower():
                    step_wall_sec = time.perf_counter() - t0
                    env_step_batch_wall_sec += step_wall_sec
                    env_step_sec += step_wall_sec * (n_env_active / float(num_envs))
                    env_step_calls += 1
                    env_slot_steps_submitted += num_envs
                    ep_max_step_wall_ms = np.where(
                        active_uncollected,
                        np.maximum(ep_max_step_wall_ms, step_wall_sec * 1000.0),
                        ep_max_step_wall_ms,
                    )
                    wave_timeout_abort = True
                    break
                envs.close()
                raise

            step_wall_sec = time.perf_counter() - t0
            step_wall_ms = step_wall_sec * 1000.0
            per_env_step_ms = step_wall_ms / float(num_envs)
            env_step_batch_wall_sec += step_wall_sec
            env_step_sec += step_wall_sec * (n_env_active / float(num_envs))
            env_step_calls += 1
            env_slot_steps_submitted += num_envs
            next_to_play = infos["to_play"]

            dones = np.logical_or(terminated, truncated)
            step_rewards = infos.get("reward", rewards)

            ep_returns += np.where(active_uncollected, step_rewards, 0.0)
            ep_env_steps += np.where(active_uncollected, 1, 0)
            ep_cand_decisions += np.where(cand_active_mask, 1, 0)
            ep_opp_decisions += np.where(opp_active_mask, 1, 0)
            ep_cand_step_ms += np.where(cand_active_mask, per_env_step_ms, 0.0)
            ep_opp_step_ms += np.where(opp_active_mask, per_env_step_ms, 0.0)
            ep_max_step_wall_ms = np.where(
                active_uncollected,
                np.maximum(ep_max_step_wall_ms, step_wall_ms),
                ep_max_step_wall_ms,
            )
            env_steps_total += n_env_active
            step_count += 1

            # C++ 側のタイムアウト判定:
            # 1) ネイティブ環境がスロット単位の明示的なタイムアウト情報 (info['termination_reason'] == 2、
            #    info['timeout'] / info['engine_timeout']、または info['step_time'][idx] >= timeout) を返した場合のみ
            #    確定タイムアウト (engine_timeout=True) として扱い、引き分け(reward=0)へ補正する。
            # 2) ネイティブ側のスロット識別フラグが無い場合、バッチ全体の壁時計時間 (step_wall_sec >= self.timeout) は
            #    「タイムアウト疑い (timeout_suspected=True)」として記録する。
            #    このとき未確認スロットの生 outcome は診断用に保持して一括で引き分けへ書き換えない一方、
            #    C++ handle_timeout() のダミー終端状態が混入している可能性があるため abnormal=True・
            #    reason_label='timeout_suspected' として正常決着 (normal_total / normal_lp0 / natural_only) および
            #    確定シードペア集計 (seed_pairs.sweep_2_0_wins 等) からは厳格に除外する。
            step_wall_exceeded_timeout = (self.timeout > 0) and (step_wall_sec >= float(self.timeout))
            newly_done_indices = [
                i for i in range(num_envs) if (not collected[i]) and bool(dones[i])
            ]
            explicit_timeout_arr = infos.get("timeout", infos.get("engine_timeout"))
            term_reason_arr = infos.get("termination_reason")

            for idx in newly_done_indices:
                collected[idx] = True

                # エンジン内部の step_time 情報（C++側で 0 固定の場合は実測ステップ時間を使用）
                reported_cand_ms = 0.0
                reported_opp_ms = 0.0
                if "step_time" in infos:
                    reported_cand_ms = float(infos["step_time"][idx][candidate_seat]) * 1000.0
                    reported_opp_ms = float(infos["step_time"][idx][1 - candidate_seat]) * 1000.0

                cand_engine_ms = (
                    reported_cand_ms if reported_cand_ms > 0.0 else float(ep_cand_step_ms[idx])
                )
                opp_engine_ms = (
                    reported_opp_ms if reported_opp_ms > 0.0 else float(ep_opp_step_ms[idx])
                )

                explicit_term_reason = (
                    int(term_reason_arr[idx])
                    if term_reason_arr is not None and len(term_reason_arr) > idx
                    else None
                )
                has_explicit_timeout_flag = (
                    explicit_timeout_arr is not None and len(explicit_timeout_arr) > idx
                )
                explicit_slot_timeout = (
                    bool(explicit_timeout_arr[idx]) if has_explicit_timeout_flag else False
                )
                reported_timeout = (self.timeout > 0) and (
                    reported_cand_ms >= float(self.timeout) * 1000.0
                    or reported_opp_ms >= float(self.timeout) * 1000.0
                )
                engine_timeout = (
                    (explicit_term_reason == 2) or explicit_slot_timeout or reported_timeout
                )

                slot_wall_exceeded_timeout = step_wall_exceeded_timeout or (
                    self.timeout > 0
                    and float(ep_max_step_wall_ms[idx]) >= float(self.timeout) * 1000.0
                )
                timeout_suspected = (
                    (explicit_term_reason is None)
                    and (not has_explicit_timeout_flag)
                    and (not engine_timeout)
                    and slot_wall_exceeded_timeout
                )

                # YGOProEnvImpl::step は step_count_ >= max_steps で _duel_end(winner, 0x01) を呼ぶ。
                # ネイティブ側が明示的な termination_reason (0:自然決着, 1:ステップ上限, 2:タイムアウト) を返す場合はそれを優先し、
                # 未提供の場合は truncated[idx] または env_steps >= self.max_steps でステップ上限打ち切りを判定する。
                if explicit_term_reason is not None:
                    step_limit_reached = (not engine_timeout) and (
                        (explicit_term_reason == 1) or bool(truncated[idx])
                    )
                else:
                    step_limit_reached = (not engine_timeout) and (
                        bool(truncated[idx]) or int(ep_env_steps[idx]) >= self.max_steps
                    )
                is_trunc = step_limit_reached
                is_term = bool(terminated[idx]) and (not is_trunc) and (not engine_timeout)

                if engine_timeout:
                    # 明示的に確認されたタイムアウトのみ引き分け・報酬0として扱う
                    cand_reward = 0.0
                    outcome = "draw"
                    win_reason = 0
                else:
                    raw_ret = float(ep_returns[idx])
                    if play_mode == "self":
                        # play_mode='self' では最終手を打ったプレイヤー(to_play[idx])視点の報酬
                        cand_acted_last = bool(to_play[idx] == candidate_seat)
                        cand_reward = raw_ret if cand_acted_last else -raw_ret
                    elif opp_is_builtin:
                        # 相手がbuiltinボットの場合、常に候補(player=candidate_seat)視点の報酬
                        cand_reward = raw_ret
                    else:
                        # 候補がbuiltinボット、相手がモデルの場合、相手視点の報酬なので符号反転
                        cand_reward = -raw_ret

                    if cand_reward > 0:
                        outcome = "win"
                    elif cand_reward < 0:
                        outcome = "loss"
                    else:
                        outcome = "draw"

                    win_reason = int(infos["win_reason"][idx]) if "win_reason" in infos else 0

                is_abnormal = (
                    is_trunc
                    or (outcome == "draw")
                    or engine_timeout
                    or timeout_suspected
                    or (not is_term)
                )

                if engine_timeout:
                    reason_label = "engine_timeout"
                elif timeout_suspected:
                    reason_label = "timeout_suspected"
                elif is_trunc:
                    reason_label = "truncated"
                elif outcome == "draw":
                    reason_label = "draw_zero_reward"
                elif win_reason == 1:
                    reason_label = "normal_lp0"
                elif win_reason == -1:
                    reason_label = "normal_deckout_or_special"
                else:
                    reason_label = f"other_reason_{win_reason}"

                episodes[idx] = {
                    "wave_idx": wave_idx,
                    "env_slot": idx,
                    "pair_id": f"w{wave_idx:03d}_e{idx:02d}",
                    "wave_seed": wave_seed,
                    "candidate_seat": candidate_seat,
                    "candidate_seat_label": "first" if candidate_seat == 0 else "second",
                    "outcome": outcome,
                    "win": 1 if outcome == "win" else 0,
                    "loss": 1 if outcome == "loss" else 0,
                    "draw": 1 if outcome == "draw" else 0,
                    "candidate_reward": round(cand_reward, 6),
                    "env_steps": int(ep_env_steps[idx]),
                    "candidate_decisions": int(ep_cand_decisions[idx]),
                    "opponent_decisions": int(ep_opp_decisions[idx]),
                    "win_reason_code": win_reason,
                    "terminated": is_term,
                    "truncated": is_trunc,
                    "engine_timeout": engine_timeout,
                    "timeout_suspected": timeout_suspected,
                    "abnormal": is_abnormal,
                    "reason_label": reason_label,
                    "cand_engine_step_ms": round(cand_engine_ms, 3),
                    "opp_engine_step_ms": round(opp_engine_ms, 3),
                }

            prev_dones = dones

        # 万一タイムアウト例外または max_steps_per_wave 到達で未完了の環境があれば異常終了として記録
        for idx in range(num_envs):
            if episodes[idx] is None:
                is_timeout_abort = wave_timeout_abort
                suspected_wall = (not is_timeout_abort) and (
                    self.timeout > 0
                    and float(ep_max_step_wall_ms[idx]) >= float(self.timeout) * 1000.0
                )
                episodes[idx] = {
                    "wave_idx": wave_idx,
                    "env_slot": idx,
                    "pair_id": f"w{wave_idx:03d}_e{idx:02d}",
                    "wave_seed": wave_seed,
                    "candidate_seat": candidate_seat,
                    "candidate_seat_label": "first" if candidate_seat == 0 else "second",
                    "outcome": "draw",
                    "win": 0,
                    "loss": 0,
                    "draw": 1,
                    "candidate_reward": 0.0,
                    "env_steps": int(ep_env_steps[idx]),
                    "candidate_decisions": int(ep_cand_decisions[idx]),
                    "opponent_decisions": int(ep_opp_decisions[idx]),
                    "win_reason_code": 0,
                    "terminated": False,
                    "truncated": not is_timeout_abort,
                    "engine_timeout": is_timeout_abort,
                    "timeout_suspected": suspected_wall,
                    "abnormal": True,
                    "reason_label": "engine_timeout" if is_timeout_abort else "aborted_max_steps",
                    "cand_engine_step_ms": round(float(ep_cand_step_ms[idx]), 3),
                    "opp_engine_step_ms": round(float(ep_opp_step_ms[idx]), 3),
                }

        wave_wall_sec = time.perf_counter() - wave_t0
        envs.close()
        del envs

        timing = {
            "wave_wall_sec": wave_wall_sec,
            "cand_model_sec": cand_model_sec,
            "opp_model_sec": opp_model_sec,
            "env_step_sec": env_step_sec,
            "cand_model_batch_wall_sec": cand_model_batch_wall_sec,
            "opp_model_batch_wall_sec": opp_model_batch_wall_sec,
            "env_step_batch_wall_sec": env_step_batch_wall_sec,
            "cand_decision_steps_total": float(cand_decision_steps_total),
            "opp_decision_steps_total": float(opp_decision_steps_total),
            "env_steps_total": float(env_steps_total),
            "cand_model_calls": int(cand_model_calls),
            "opp_model_calls": int(opp_model_calls),
            "env_step_calls": int(env_step_calls),
            "cand_model_rows": int(cand_model_rows),
            "opp_model_rows": int(opp_model_rows),
            "env_slot_steps_submitted": int(env_slot_steps_submitted),
        }
        return [ep for ep in episodes if ep is not None], timing

    def evaluate_matchup(
        self,
        candidate_str: str,
        opponent_str: str,
        proj_dir: Path,
        num_episodes: int = 32,
        base_seed: int = 1000,
        max_steps_per_wave: int = 4000,
    ) -> Dict[str, Any]:
        """
        候補 (candidate) と 対戦相手 (opponent) を先後入れ替え・同一シードペアで対戦させ、
        勝敗・異常終了・判断時間の詳細な集計辞書を返す。
        """
        cand_spec = resolve_agent_spec(candidate_str, proj_dir)
        opp_spec = resolve_agent_spec(opponent_str, proj_dir)

        min_batch = self.num_envs * 2
        if num_episodes < min_batch or num_episodes % min_batch != 0:
            raise ValueError(
                f"num_episodes ({num_episodes}) は num_envs*2 ({min_batch}) の倍数で指定してください"
                "（先攻・後攻を同じシードで num_envs 戦ずつペア実行するため）。"
            )

        # モデルを使う場合は事前にJITウォームアップを完了させる
        if cand_spec["kind"] != "builtin" or opp_spec["kind"] != "builtin":
            self.warmup_jit()

        cand_params = self.load_params(cand_spec)
        opp_params = self.load_params(opp_spec)

        num_waves = num_episodes // min_batch
        all_episodes: List[Dict[str, Any]] = []

        total_wall_sec = 0.0
        total_cand_model_sec = 0.0
        total_opp_model_sec = 0.0
        total_env_step_sec = 0.0
        total_cand_model_batch_wall_sec = 0.0
        total_opp_model_batch_wall_sec = 0.0
        total_env_step_batch_wall_sec = 0.0
        total_cand_decisions_timed = 0.0
        total_opp_decisions_timed = 0.0
        total_env_steps_timed = 0.0
        wave_seeds_used: List[int] = []

        matchup_t0 = time.perf_counter()
        for w in range(num_waves):
            w_seed = make_wave_seed(base_seed, w)
            wave_seeds_used.append(w_seed)

            # Pass A: 候補 = 先攻 (seat=0), 相手 = 後攻 (seat=1)
            eps_first, t_first = self._run_single_wave_seat(
                cand_spec,
                opp_spec,
                cand_params,
                opp_params,
                wave_idx=w,
                wave_seed=w_seed,
                candidate_seat=0,
                max_steps_per_wave=max_steps_per_wave,
            )
            # Pass B: 候補 = 後攻 (seat=1), 相手 = 先攻 (seat=0) （同一 wave_seed）
            eps_second, t_second = self._run_single_wave_seat(
                cand_spec,
                opp_spec,
                cand_params,
                opp_params,
                wave_idx=w,
                wave_seed=w_seed,
                candidate_seat=1,
                max_steps_per_wave=max_steps_per_wave,
            )

            all_episodes.extend(eps_first)
            all_episodes.extend(eps_second)

            for t in (t_first, t_second):
                total_wall_sec += t["wave_wall_sec"]
                total_cand_model_sec += t["cand_model_sec"]
                total_opp_model_sec += t["opp_model_sec"]
                total_env_step_sec += t["env_step_sec"]
                total_cand_model_batch_wall_sec += t["cand_model_batch_wall_sec"]
                total_opp_model_batch_wall_sec += t["opp_model_batch_wall_sec"]
                total_env_step_batch_wall_sec += t["env_step_batch_wall_sec"]
                total_cand_decisions_timed += t["cand_decision_steps_total"]
                total_opp_decisions_timed += t["opp_decision_steps_total"]
                total_env_steps_timed += t["env_steps_total"]

        matchup_wall_sec = time.perf_counter() - matchup_t0

        # --- 1. 勝敗の集計（全体・先攻・後攻・先後ペア） ---
        # timeout_suspected（C++ handle_timeout() ダミー状態混入の疑いあり）または
        # engine_timeout（確定タイムアウト、引き分け補正済み）を含むペアは、
        # 個別エピソード辞書および raw_including_suspected に生の outcome を保持しつつ、
        # 確定の勝率・信頼区間・先後ペア集計からはペアごと除外して混入を防ぐ。
        total_eps = len(all_episodes)
        raw_wins = sum(e["win"] for e in all_episodes)
        raw_losses = sum(e["loss"] for e in all_episodes)
        raw_draws = sum(e["draw"] for e in all_episodes)
        raw_win_rate = raw_wins / total_eps if total_eps > 0 else 0.0

        # 同一シードの先後ペア集計 (pair_id ごとに先攻結果と後攻結果を照合)
        pairs_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for e in all_episodes:
            pid = e["pair_id"]
            pairs_by_id.setdefault(pid, {})[e["candidate_seat"]] = e

        pair_sweep_wins = 0              # 先攻〇・後攻〇 (2勝0敗、タイムアウト疑いなし)
        pair_split_first = 0             # 先攻〇・後攻● (先攻側がどちらも勝利、タイムアウト疑いなし)
        pair_split_second = 0            # 先攻●・後攻〇 (後攻側がどちらも勝利、タイムアウト疑いなし)
        pair_sweep_losses = 0            # 先攻●・後攻● (0勝2敗、タイムアウト疑いなし)
        pair_other_draws = 0             # 引き分けを含むペア（タイムアウト疑いなし）
        pair_suspected_or_timeout = 0    # タイムアウトまたはタイムアウト疑いを含むペア（確定比較から分離）
        raw_pair_scores: List[float] = []
        verified_pair_scores: List[float] = []
        verified_episodes: List[Dict[str, Any]] = []

        for pid, seat_map in sorted(pairs_by_id.items()):
            e0 = seat_map.get(0)
            e1 = seat_map.get(1)
            if not e0 or not e1:
                continue
            p_score = 0.5 * (float(e0["win"]) + float(e1["win"]))
            raw_pair_scores.append(p_score)

            has_suspected = bool(
                e0.get("timeout_suspected", False) or e1.get("timeout_suspected", False)
            )
            has_engine_timeout = bool(
                e0.get("engine_timeout", False) or e1.get("engine_timeout", False)
            )
            # 確定タイムアウト (engine_timeout) もタイムアウト疑い (timeout_suspected) も、
            # 引き分け補正スコアや未確認の勝敗が実力比較を歪めるため、ペアごと確認済み集計から除外する。
            if has_suspected or has_engine_timeout:
                pair_suspected_or_timeout += 1
                e0["included_in_verified_score"] = False
                e1["included_in_verified_score"] = False
                continue

            e0["included_in_verified_score"] = True
            e1["included_in_verified_score"] = True
            verified_pair_scores.append(p_score)
            verified_episodes.append(e0)
            verified_episodes.append(e1)
            if e0["win"] == 1 and e1["win"] == 1:
                pair_sweep_wins += 1
            elif e0["win"] == 1 and e1["loss"] == 1:
                pair_split_first += 1
            elif e0["loss"] == 1 and e1["win"] == 1:
                pair_split_second += 1
            elif e0["loss"] == 1 and e1["loss"] == 1:
                pair_sweep_losses += 1
            else:
                pair_other_draws += 1

        scored_eps = len(verified_episodes)
        excluded_suspected_eps = total_eps - scored_eps
        first_eps = [e for e in verified_episodes if e["candidate_seat"] == 0]
        second_eps = [e for e in verified_episodes if e["candidate_seat"] == 1]

        wins = sum(e["win"] for e in verified_episodes)
        losses = sum(e["loss"] for e in verified_episodes)
        draws = sum(e["draw"] for e in verified_episodes)
        win_rate = wins / scored_eps if scored_eps > 0 else 0.0

        first_wins = sum(e["win"] for e in first_eps)
        first_losses = sum(e["loss"] for e in first_eps)
        first_draws = sum(e["draw"] for e in first_eps)
        first_win_rate = first_wins / len(first_eps) if first_eps else 0.0
        first_ci_low, first_ci_high = wilson_ci95(first_wins, len(first_eps))

        second_wins = sum(e["win"] for e in second_eps)
        second_losses = sum(e["loss"] for e in second_eps)
        second_draws = sum(e["draw"] for e in second_eps)
        second_win_rate = second_wins / len(second_eps) if second_eps else 0.0
        second_ci_low, second_ci_high = wilson_ci95(second_wins, len(second_eps))

        # 総合勝率の95%信頼区間は、タイムアウト疑いを除いた独立なシードペア K 個に対する
        # Floor/Ceil Clopper-Pearson 保守区間から算出
        ci_low, ci_high, pair_se, n_eff = paired_wilson_ci95(
            verified_pair_scores, wins, scored_eps
        )
        raw_ci_low, raw_ci_high, _, _ = paired_wilson_ci95(
            raw_pair_scores, raw_wins, total_eps
        )

        # --- 2. 決着理由・異常終了の集計 ---
        normal_lp0_count = sum(1 for e in all_episodes if e["reason_label"] == "normal_lp0")
        normal_deckout_count = sum(
            1 for e in all_episodes if e["reason_label"] == "normal_deckout_or_special"
        )
        truncated_count = sum(1 for e in all_episodes if e["truncated"])
        engine_timeout_count = sum(1 for e in all_episodes if e["engine_timeout"])
        timeout_suspected_count = sum(1 for e in all_episodes if e.get("timeout_suspected", False))
        aborted_max_steps_count = sum(
            1 for e in all_episodes if e["reason_label"] == "aborted_max_steps"
        )
        abnormal_total = sum(1 for e in all_episodes if e["abnormal"])
        normal_total = total_eps - abnormal_total
        truncation_rate = truncated_count / total_eps if total_eps > 0 else 0.0

        # 自然決着（打ち切り・タイムアウト・タイムアウト疑い等の異常終了を除いた対戦）のみの記述統計
        natural_eps = [e for e in all_episodes if not e["abnormal"]]
        nat_wins = sum(e["win"] for e in natural_eps)
        nat_losses = sum(e["loss"] for e in natural_eps)
        nat_draws = sum(e["draw"] for e in natural_eps)
        nat_total = len(natural_eps)
        nat_win_rate = nat_wins / nat_total if nat_total > 0 else 0.0

        # --- 3. 判断数・判断時間の集計 ---
        cand_dec_arr = np.array([e["candidate_decisions"] for e in all_episodes], dtype=np.float64)
        env_step_arr = np.array([e["env_steps"] for e in all_episodes], dtype=np.float64)
        cand_eng_ms_arr = np.array(
            [e["cand_engine_step_ms"] for e in all_episodes], dtype=np.float64
        )

        # (A) 有効スロット均等按分による推定値 (dt * n_active / num_envs の合計 / 有効判断数)
        cand_ms_per_decision = (
            (total_cand_model_sec * 1000.0) / total_cand_decisions_timed
            if total_cand_decisions_timed > 0
            else 0.0
        )
        opp_ms_per_decision = (
            (total_opp_model_sec * 1000.0) / total_opp_decisions_timed
            if total_opp_decisions_timed > 0
            else 0.0
        )
        # (B) バッチ推論の実測総時間を有効判断数で割った実行コスト (dt の合計 / 有効判断数)
        cand_batch_wall_ms_per_decision = (
            (total_cand_model_batch_wall_sec * 1000.0) / total_cand_decisions_timed
            if total_cand_decisions_timed > 0
            else 0.0
        )
        opp_batch_wall_ms_per_decision = (
            (total_opp_model_batch_wall_sec * 1000.0) / total_opp_decisions_timed
            if total_opp_decisions_timed > 0
            else 0.0
        )
        env_ms_per_step = (
            (total_env_step_sec * 1000.0) / total_env_steps_timed
            if total_env_steps_timed > 0
            else 0.0
        )
        env_batch_wall_ms_per_step = (
            (total_env_step_batch_wall_sec * 1000.0) / total_env_steps_timed
            if total_env_steps_timed > 0
            else 0.0
        )
        sps = total_env_steps_timed / total_wall_sec if total_wall_sec > 0 else 0.0

        summary = {
            "matchup": f"{cand_spec['label']} vs {opp_spec['label']}",
            "candidate": cand_spec,
            "opponent": opp_spec,
            "config": {
                "deck": self.deck_name,
                "num_episodes": total_eps,
                "num_envs": self.num_envs,
                "env_threads": self.env_threads,
                "max_steps": self.max_steps,
                "timeout": self.timeout,
                "num_waves": num_waves,
                "base_seed": base_seed,
                "wave_seeds": wave_seeds_used,
                "unused_in_training": base_seed not in TRAINING_SEEDS_USED,
                "oppo_info": False,
                "skip_mask": True,
                "seat_swapped": True,
            },
            "results": {
                "total_episodes": total_eps,
                "verified_episodes": scored_eps,
                "excluded_suspected_episodes": excluded_suspected_eps,
                "excluded_timeout_or_suspected_episodes": excluded_suspected_eps,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(win_rate, 4),
                "win_rate_includes_truncated_lp": True,
                "truncation_rate": round(truncation_rate, 4),
                "win_rate_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "win_rate_se_paired": round(pair_se, 4),
                "effective_sample_size": round(n_eff, 2),
                "avg_reward": (
                    round(float(np.mean([e["candidate_reward"] for e in verified_episodes])), 4)
                    if verified_episodes
                    else 0.0
                ),
                "has_suspected_timeout": timeout_suspected_count > 0,
                "requires_rerun_or_exclusion": (timeout_suspected_count > 0) or (engine_timeout_count > 0),
                "timeout_suspected_policy": (
                    "exclude_engine_timeout_and_suspected_pairs_from_verified_and_natural_results_keep_raw_in_episodes_and_raw_including_suspected_flag_rerun"
                ),
                "natural_only": {
                    "episodes": nat_total,
                    "wins": nat_wins,
                    "losses": nat_losses,
                    "draws": nat_draws,
                    "win_rate": round(nat_win_rate, 4),
                },
                "raw_including_suspected": {
                    "episodes": total_eps,
                    "wins": raw_wins,
                    "losses": raw_losses,
                    "draws": raw_draws,
                    "win_rate": round(raw_win_rate, 4),
                    "win_rate_ci95": [round(raw_ci_low, 4), round(raw_ci_high, 4)],
                },
                "first_player": {
                    "episodes": len(first_eps),
                    "wins": first_wins,
                    "losses": first_losses,
                    "draws": first_draws,
                    "win_rate": round(first_win_rate, 4),
                    "win_rate_ci95": [round(first_ci_low, 4), round(first_ci_high, 4)],
                },
                "second_player": {
                    "episodes": len(second_eps),
                    "wins": second_wins,
                    "losses": second_losses,
                    "draws": second_draws,
                    "win_rate": round(second_win_rate, 4),
                    "win_rate_ci95": [round(second_ci_low, 4), round(second_ci_high, 4)],
                },
                "seed_pairs": {
                    "total_pairs": len(pairs_by_id),
                    "verified_pairs": len(verified_pair_scores),
                    "suspected_or_timeout_pairs": pair_suspected_or_timeout,
                    "sweep_2_0_wins": pair_sweep_wins,
                    "split_1_1_first_player_won_both": pair_split_first,
                    "split_1_1_second_player_won_both": pair_split_second,
                    "sweep_0_2_losses": pair_sweep_losses,
                    "other_with_draws": pair_other_draws,
                },
            },
            "terminations": {
                "normal_total": normal_total,
                "normal_lp0": normal_lp0_count,
                "normal_deckout_or_special": normal_deckout_count,
                "abnormal_total": abnormal_total,
                "truncated": truncated_count,
                "engine_timeout": engine_timeout_count,
                "timeout_suspected": timeout_suspected_count,
                "aborted_max_steps": aborted_max_steps_count,
                "draws": draws,
            },
            "timing_and_decisions": {
                "jit_warmup_seconds_once": round(self.warmup_seconds, 4),
                "matchup_wall_seconds": round(matchup_wall_sec, 4),
                "active_wave_wall_seconds": round(total_wall_sec, 4),
                "candidate_model_total_seconds": round(total_cand_model_sec, 4),
                "opponent_model_total_seconds": round(total_opp_model_sec, 4),
                "env_step_total_seconds": round(total_env_step_sec, 4),
                "candidate_model_batch_wall_seconds": round(total_cand_model_batch_wall_sec, 4),
                "opponent_model_batch_wall_seconds": round(total_opp_model_batch_wall_sec, 4),
                "env_step_batch_wall_seconds": round(total_env_step_batch_wall_sec, 4),
                "candidate_ms_per_decision": round(cand_ms_per_decision, 4),
                "opponent_ms_per_decision": round(opp_ms_per_decision, 4),
                "candidate_allocated_ms_per_decision": round(cand_ms_per_decision, 4),
                "opponent_allocated_ms_per_decision": round(opp_ms_per_decision, 4),
                "candidate_batch_wall_ms_per_decision": round(cand_batch_wall_ms_per_decision, 4),
                "opponent_batch_wall_ms_per_decision": round(opp_batch_wall_ms_per_decision, 4),
                "env_ms_per_step": round(env_ms_per_step, 4),
                "env_batch_wall_ms_per_step": round(env_batch_wall_ms_per_step, 4),
                "sps": round(sps, 1),
                "candidate_decisions_per_duel": {
                    "mean": round(float(np.mean(cand_dec_arr)), 2),
                    "median": round(float(np.median(cand_dec_arr)), 2),
                    "min": int(np.min(cand_dec_arr)),
                    "max": int(np.max(cand_dec_arr)),
                },
                "env_steps_per_duel": {
                    "mean": round(float(np.mean(env_step_arr)), 2),
                    "median": round(float(np.median(env_step_arr)), 2),
                    "min": int(np.min(env_step_arr)),
                    "max": int(np.max(env_step_arr)),
                },
                "engine_reported_candidate_ms_per_duel": {
                    "mean": round(float(np.mean(cand_eng_ms_arr)), 3),
                    "max": round(float(np.max(cand_eng_ms_arr)), 3),
                },
            },
            "episodes": all_episodes,
        }
        return summary


def format_markdown_report(
    run_meta: Dict[str, Any],
    summaries: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# 青眼（Blue-Eyes）モデル評価レポート")
    lines.append("")
    lines.append(f"- **実行日時（JST）**: {run_meta['timestamp_jst']}")
    lines.append(f"- **乱数シード（base_seed）**: `{run_meta['base_seed']}` （学習時 `seed=0`・初回確認 `seed=100` と重複しない未使用シード）")
    lines.append(f"- **1カードあたりの対戦数**: 各 `{run_meta['num_episodes']}` 戦（先攻 `{run_meta['num_episodes']//2}` 戦 ＋ 後攻 `{run_meta['num_episodes']//2}` 戦、同一シードで先後入れ替えペア実行）")
    lines.append(f"- **1戦あたりステップ上限（max_steps）**: `{run_meta.get('max_steps', 1000)}` ステップ（上限到達時は `truncated=True` の打ち切りとして正常決着と分離集計）")
    lines.append(f"- **非公開情報の保護**: `oppo_info=False`, `skip_mask=True`")
    lines.append(f"- **初回JITコンパイル（ウォームアップ）時間**: `{run_meta['jit_warmup_seconds']:.3f}` 秒（以下の推論時間・対戦時間からは分離済み）")
    lines.append("")
    lines.append("## 1. 勝敗・先攻後攻別成績一覧")
    lines.append("")
    lines.append("| 候補 (Candidate) | 対戦相手 (Opponent) | 総合勝率 [LP判定込] (勝-敗-分) | 自然決着のみ (打ち切り率) | 95%信頼区間 (ペア単位保守区間) | 先攻勝率 | 後攻勝率 | 先後ペア (2連勝 / 1勝1敗 / 2連敗) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for s in summaries:
        cand_lbl = s["candidate"]["label"]
        opp_lbl = s["opponent"]["label"]
        r = s["results"]
        nat = r.get(
            "natural_only",
            {
                "episodes": r["total_episodes"],
                "wins": r["wins"],
                "losses": r["losses"],
                "draws": r["draws"],
                "win_rate": r["win_rate"],
            },
        )
        trunc_rate = r.get("truncation_rate", 0.0)
        fp = r["first_player"]
        sp = r["second_player"]
        pairs = r["seed_pairs"]
        split_1_1 = pairs["split_1_1_first_player_won_both"] + pairs["split_1_1_second_player_won_both"]
        ci_l, ci_h = r["win_rate_ci95"]

        lines.append(
            f"| `{cand_lbl}` | `{opp_lbl}` | "
            f"**{r['win_rate']*100:.1f}%** ({r['wins']}-{r['losses']}-{r['draws']}) | "
            f"{nat['win_rate']*100:.1f}% ({nat['wins']}-{nat['losses']}) / {trunc_rate*100:.1f}% | "
            f"{ci_l*100:.1f}%–{ci_h*100:.1f}% | "
            f"{fp['win_rate']*100:.1f}% ({fp['wins']}/{fp['episodes']}) | "
            f"{sp['win_rate']*100:.1f}% ({sp['wins']}/{sp['episodes']}) | "
            f"{pairs['sweep_2_0_wins']} / {split_1_1} / {pairs['sweep_0_2_losses']} |"
        )

    lines.append("")
    lines.append("## 2. 決着理由・異常終了の内訳")
    lines.append("")
    lines.append("| 候補 vs 対戦相手 | 正常決着 (計) | LP0決着 (`reason=1`) | デッキ切れ等 (`reason=-1`) | 異常終了・打ち切り (計) | ステップ上限/Truncated | タイムアウト (確定) | タイムアウト疑い |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for s in summaries:
        t = s["terminations"]
        lines.append(
            f"| `{s['candidate']['label']}` vs `{s['opponent']['label']}` | "
            f"{t['normal_total']} | {t['normal_lp0']} | {t['normal_deckout_or_special']} | "
            f"**{t['abnormal_total']}** | {t['truncated']} | {t['engine_timeout']} | {t.get('timeout_suspected', 0)} |"
        )

    lines.append("")
    lines.append("## 3. 判断時間・処理速度の内訳")
    lines.append("")
    lines.append("| 候補 vs 対戦相手 | 候補推論 按分推定 (ms/判断) | 候補推論 実測コスト (ms/判断) | 相手推論 按分/実測 (ms/判断) | エンジン処理 按分 (ms/step) | 候補の平均判断数/戦 (最小–最大) | 実測時間 (秒) | SPS |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for s in summaries:
        td = s["timing_and_decisions"]
        cd = td["candidate_decisions_per_duel"]
        cand_batch_ms = td.get("candidate_batch_wall_ms_per_decision", td["candidate_ms_per_decision"])
        opp_batch_ms = td.get("opponent_batch_wall_ms_per_decision", td["opponent_ms_per_decision"])
        if s["candidate"]["kind"] != "builtin":
            cand_alloc_str = f"{td['candidate_ms_per_decision']:.3f} ms"
            cand_batch_str = f"{cand_batch_ms:.3f} ms"
        else:
            cand_alloc_str = "0.000 ms"
            cand_batch_str = "0.000 ms"
        if s["opponent"]["kind"] != "builtin":
            opp_ms_str = f"{td['opponent_ms_per_decision']:.3f} / {opp_batch_ms:.3f} ms"
        else:
            opp_ms_str = "-"
        lines.append(
            f"| `{s['candidate']['label']}` vs `{s['opponent']['label']}` | "
            f"{cand_alloc_str} | {cand_batch_str} | {opp_ms_str} | "
            f"{td['env_ms_per_step']:.3f} ms | "
            f"{cd['mean']:.1f} ({cd['min']}–{cd['max']}) | "
            f"{td['active_wave_wall_seconds']:.2f} s | {td['sps']:.0f} |"
        )

    lines.append("")
    lines.append("## 4. 評価結果の解釈上の注意点")
    lines.append("")
    lines.append("- **先後ペア比較**: 同じ乱数シードで候補モデルが先攻（Player 0）と後攻（Player 1）の両方をプレイしています。「2連勝」は初手運や先攻・後攻の有利不利に関わらず候補モデルが両方勝ったペア数、「1勝1敗」は片方の席でのみ勝ったペア数です（タイムアウトまたはタイムアウト疑いを含むペアは `suspected_or_timeout_pairs` として分離し、確定ペア集計には含めません）。")
    lines.append("- **95%信頼区間（ペア単位 Clopper-Pearson 保守区間）**: 独立な観測単位はシードペア $K$ 組（各ペアスコア $P_i \\in \\{0, 0.5, 1\\}$）です。$K=1$ を含む全サンプルサイズで有効サンプルサイズを $K$ 組のまま扱い（2戦の独立試行に水増しせず）、下限に `j_low = floor(sum(P_i))`・上限に `j_high = ceil(sum(P_i))` を用いた正確二項（Clopper-Pearson）区間を表示しています。$\\{0, 1\\}$ 値分布（および $P_i \\equiv 0.5$ の退化分布）で被覆率 $\\ge 95\\%$ が保証され、3値 $\\{0, 0.5, 1\\}$ の単体格子全体でも数値的に被覆率 $\\ge 95\\%$ を確認しています（なお一般の $[0, 1]$ 分布では凸順序 $\\le_{cx}$ から直接点ごとの二項尾確率支配は従わず Bentkus 2004 の定数因子が伴うため、本評価の3値ペアスコアに対しては上記 Floor/Ceil Clopper-Pearson と格子検証を根拠としています）。信頼区間が重なっている場合は「統計的に明確な強さの差はまだ断定できない」ことを示します。")
    lines.append("- **ステップ上限打ち切り（Truncated）と総合勝率**: 1戦の環境ステップ数が `max_steps`（デフォルト `1000`）に達した場合、C++エンジンはLP比較で勝敗を付与して終了します。表1の「総合勝率 [LP判定込]」および95%信頼区間にはこの打ち切り時のLP判定が含まれます。一方、通常決着（`normal_lp0` 等）とは区別して `truncated=True`（異常終了・打ち切り）に分類し、表1に「自然決着のみ（打ち切り・異常終了除外）」の記述統計と「打ち切り率」を併記しています（※打ち切り除外のみの勝率は長期戦を除いた選択バイアスを含むため、自然決着の勝率を主指標とする場合は `max_steps` を増やした再評価を検討します）。")
    lines.append("- **タイムアウト（確定 vs 疑い）と採用ルール**: ネイティブ側からスロット単位のタイムアウト通知（`termination_reason == 2`、`timeout=True`、`step_time >= timeout` 等）または例外が確認された場合は `engine_timeout`（確定タイムアウト）として個別エピソード上は引き分け（報酬0）に補正しますが、この補正スコアは実力を反映しないため、確定タイムアウトを含むシードペアもペアごと確定の勝率・信頼区間・シードペア集計から除外します。ネイティブ側のスロット識別情報が無くバッチ全体の壁時計時間が `timeout` 秒以上に達した場合は、未確認スロットの生 `outcome` を一括で引き分けに書き換えず個別エピソードおよび `raw_including_suspected` に診断用として保持する一方、C++ `handle_timeout()` のダミー終端状態が混入している可能性があるため `abnormal=True`・`reason_label='timeout_suspected'` に分類し、正常決着（`normal_total` / `normal_lp0` / `natural_only`）および確定の勝率・シードペア集計（`wins` / `win_rate` / `win_rate_ci95` / `seed_pairs.verified_pairs` / `seed_pairs.sweep_2_0_wins` 等）から除外した上で `requires_rerun_or_exclusion=True`（要再実行または採用除外）と判定します。")
    lines.append("- **判断時間（均等按分推定値 vs 実測コスト）**: 初回XLAコンパイル時間（ウォームアップ）を除外しています。「候補推論 按分推定 (ms/判断)」は固定サイズ（`num_envs`）のバッチ推論時間 `dt` をそのステップの有効手番スロット割合（`n_active / num_envs`）で均等按分した推定値であり、個別スロット単体の実測時間ではありません。「候補推論 実測コスト (ms/判断)」はバッチ推論の実測総時間（`candidate_model_batch_wall_seconds`）を有効判断総数で割った実行全体の有効判断あたりコストです。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="yugioh-ai 評価スクリプト（先後入れ替え・未使用シード・勝敗/異常終了/判断時間分離記録）")
    parser.add_argument("--candidate", type=str, default=None, help="評価対象のモデルパス、または 'init' / 'random' / 'greedy'")
    parser.add_argument("--opponent", type=str, default="random", help="対戦相手のモデルパス、または 'random' / 'greedy' / 'init'")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="標準ベースライン評価スイート（random / init / 4096 / 4096+512 の各対戦カード）を一括実行する",
    )
    parser.add_argument("--num-episodes", type=int, default=32, help="1対戦カードあたりの総対戦数（num_envs*2 の倍数、デフォルト: 32）")
    parser.add_argument("--num-envs", type=int, default=8, help="並列環境数（デフォルト: 8）")
    parser.add_argument("--env-threads", type=int, default=4, help="環境スレッド数（デフォルト: 4）")
    parser.add_argument("--max-steps", type=int, default=1000, help="1戦あたりの最大環境ステップ数（デフォルト: 1000）")
    parser.add_argument("--timeout", type=int, default=600, help="エンジン1ステップのタイムアウト秒数（デフォルト: 600）")
    parser.add_argument("--seed", type=int, default=1000, help="評価用の乱数ベースシード（デフォルト: 1000、学習時の0とは別シード）")
    parser.add_argument("--deck", type=str, default="assets/deck/BlueEyes.ydk", help="使用するデッキファイルのパス")
    parser.add_argument("--out-dir", type=str, default=None, help="結果の出力先ディレクトリ（省略時は records/eval-日時）")
    parser.add_argument("--runtime-dir", type=str, default="/opt/ygo-ai", help="ygo-ai ランタイムディレクトリ")
    args = parser.parse_args()

    proj_dir = Path(__file__).resolve().parent.parent
    runtime_dir = Path(args.runtime_dir)
    runtime_scripts_dir = runtime_dir / "current" / "scripts"

    deck_path = Path(args.deck)
    if not deck_path.is_absolute():
        deck_path = (proj_dir / deck_path).resolve()
    if not deck_path.is_file():
        raise FileNotFoundError(f"デッキファイルが見つかりません: {deck_path}")

    if args.seed in TRAINING_SEEDS_USED:
        print(
            f"[WARN] 指定された --seed {args.seed} は学習時に使用されたシードです。"
            "公平な評価のため別シード（例: 1000）の使用を推奨します。",
            file=sys.stderr,
        )

    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst)
    stamp = now_jst.strftime("%Y%m%d-%H%M%S")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (proj_dir / out_dir).resolve()
    else:
        out_dir = proj_dir / "records" / f"eval-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 評価カード一覧の決定
    ckpt_4096 = str(proj_dir / "checkpoints/trial-2026-10-27/blueeyes-4096.flax_model")
    ckpt_4608 = str(proj_dir / "checkpoints/trial-2026-10-27/blueeyes-4096-plus512.flax_model")

    if args.suite:
        matchups = [
            ("random", "random"),
            ("init", "random"),
            (ckpt_4096, "random"),
            (ckpt_4608, "random"),
            ("init", "greedy"),
            (ckpt_4096, "greedy"),
            (ckpt_4608, "greedy"),
            (ckpt_4096, "init"),
            (ckpt_4608, ckpt_4096),
        ]
    else:
        if not args.candidate:
            raise ValueError("--candidate を指定するか、--suite を指定してください。")
        matchups = [(args.candidate, args.opponent)]

    print("==========================================")
    print(" yugioh-ai 評価実行 (scripts/evaluate.py)")
    print("==========================================")
    print(f"出力ディレクトリ : {out_dir}")
    print(f"使用デッキ       : {deck_path}")
    print(f"評価ベースシード : {args.seed} (学習シード {sorted(TRAINING_SEEDS_USED)} とは独立)")
    print(f"対戦数 / カード  : {args.num_episodes} 戦 (先攻 {args.num_episodes // 2} 戦 + 後攻 {args.num_episodes // 2} 戦)")
    print(f"並列環境数       : num_envs={args.num_envs}, env_threads={args.env_threads}")
    print(f"ステップ上限     : max_steps={args.max_steps}, timeout={args.timeout}s")
    print(f"対戦カード数     : {len(matchups)}")
    print("")

    ctx = EvalContext(
        runtime_scripts_dir=runtime_scripts_dir,
        deck_path=deck_path,
        num_envs=args.num_envs,
        env_threads=args.env_threads,
        max_steps=args.max_steps,
        timeout=args.timeout,
    )

    summaries: List[Dict[str, Any]] = []
    all_episodes_flat: List[Dict[str, Any]] = []

    for idx, (cand_str, opp_str) in enumerate(matchups, start=1):
        print(f"[{idx}/{len(matchups)}] 対戦開始: {Path(cand_str).name} vs {Path(opp_str).name} ...")
        summary = ctx.evaluate_matchup(
            candidate_str=cand_str,
            opponent_str=opp_str,
            proj_dir=proj_dir,
            num_episodes=args.num_episodes,
            base_seed=args.seed,
        )
        r = summary["results"]
        t = summary["terminations"]
        td = summary["timing_and_decisions"]
        print(
            f"  -> 勝率: {r['win_rate']*100:.1f}% ({r['wins']}勝 {r['losses']}敗 {r['draws']}分) "
            f"[先攻 {r['first_player']['win_rate']*100:.1f}%, 後攻 {r['second_player']['win_rate']*100:.1f}%] "
            f"| 異常終了: {t['abnormal_total']} (上限:{t['truncated']}, TO:{t['engine_timeout']}) "
            f"| 推論: {td['candidate_ms_per_decision']:.3f} ms/判断 "
            f"| 時間: {td['active_wave_wall_seconds']:.2f}s"
        )

        # フラットなエピソード一覧にもメタ情報を付与して保存
        for ep in summary["episodes"]:
            ep_row = {
                "matchup_index": idx,
                "candidate": summary["candidate"]["label"],
                "opponent": summary["opponent"]["label"],
                **ep,
            }
            all_episodes_flat.append(ep_row)

        # summary.json をコンパクトに保つため episodes フィールドは別ファイル (episodes.csv) に分離可能にするが、
        # 完全な再現性のため summary_full.json には含め、summary.json からは episodes を除いた集計を保存
        summaries.append(summary)

    run_meta = {
        "timestamp_jst": now_jst.isoformat(),
        "base_seed": args.seed,
        "num_episodes": args.num_episodes,
        "num_envs": args.num_envs,
        "env_threads": args.env_threads,
        "max_steps": args.max_steps,
        "timeout": args.timeout,
        "deck_path": str(deck_path),
        "jit_warmup_seconds": round(ctx.warmup_seconds, 4),
        "init_model_rng_seed": ctx.init_seed_val,
    }

    # 1. summary.json (エピソード配列を除いた見やすい集計)
    summaries_compact = []
    for s in summaries:
        sc = dict(s)
        sc.pop("episodes", None)
        summaries_compact.append(sc)

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": run_meta, "matchups": summaries_compact},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 2. episodes.csv (全エピソードの1戦ごとの詳細記録)
    episodes_csv_path = out_dir / "episodes.csv"
    if all_episodes_flat:
        fieldnames = list(all_episodes_flat[0].keys())
        with open(episodes_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_episodes_flat)

    # 3. report.md (人間が読みやすいMarkdownレポート)
    report_md = format_markdown_report(run_meta, summaries)
    report_md_path = out_dir / "report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md + "\n")

    print("")
    print("==========================================")
    print(" 評価完了・保存ファイル")
    print("==========================================")
    print(f"- 集計JSON     : {summary_json_path}")
    print(f"- 全対戦CSV    : {episodes_csv_path}")
    print(f"- レポートMD   : {report_md_path}")
    print("")
    print(report_md)


if __name__ == "__main__":
    main()
