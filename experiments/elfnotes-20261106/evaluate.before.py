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


def paired_wilson_ci95(
    pair_scores: List[float],
    wins: int,
    total: int,
) -> Tuple[float, float, float, float]:
    """
    同一シード先後ペアの相関を反映した勝率の95%信頼区間を計算する。

    各シードペア i (i=1..K) の平均勝率 P_i = (W_{i,first} + W_{i,second}) / 2 を
    独立なクラスタ標本として扱い、ペア間標本分散
      s_P^2 = (1 / (K - 1)) * sum((P_i - p_hat)^2)
            = (1 / 4) * (Var(W_first) + Var(W_second) + 2 * Cov(W_first, W_second))
    から標準誤差 SE = sqrt(s_P^2 / K) と有効サンプルサイズ n_eff = p_hat * (1 - p_hat) / (s_P^2 / K)
    を算出し、Wilson score interval に適用する。

    Returns:
        (ci_low, ci_high, pair_se, n_eff)
    """
    k = len(pair_scores)
    if total <= 0 or k <= 0:
        return 0.0, 0.0, 0.0, 0.0
    if k <= 1:
        low, high = wilson_ci95(wins, total)
        p = wins / total
        se = math.sqrt(p * (1.0 - p) / total)
        return low, high, se, float(total)

    z = 1.959963984540054  # 95% CI
    p = float(sum(pair_scores)) / k
    s2 = sum((x - p) ** 2 for x in pair_scores) / (k - 1)
    v_pair = s2 / k
    pair_se = math.sqrt(v_pair)

    if p <= 0.0 or p >= 1.0:
        # 全勝または全敗時は独立なシードペア数 K を有効サンプルサイズとする
        n_eff = float(k)
    else:
        # 全ペアが 1勝1敗 (P_i = 0.5) 等で標本分散が 0 になる場合の連続性下限
        min_v_pair = ((0.5 / k) ** 2) / k
        v_eff = max(v_pair, min_v_pair)
        n_eff = (p * (1.0 - p)) / v_eff

    denom = 1.0 + (z * z) / n_eff
    center = (p + (z * z) / (2.0 * n_eff)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n_eff)) / n_eff)) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
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

            # C++ 側の YGOProEnv::Step は pool.wait_for(seconds(timeout_)) がタイムアウトすると
            # handle_timeout() を呼び、info:step_time を 0 のまま done=True, reward=1.0,
            # to_play=1, is_selfplay=1, win_reason=1, num_options=1 を書き込む。
            # そのため、ステップ実測時間が timeout に達したかでタイムアウト発生を検出する。
            step_wall_timed_out = (self.timeout > 0) and (step_wall_sec >= self.timeout * 0.95)
            newly_done_indices = [
                i for i in range(num_envs) if (not collected[i]) and bool(dones[i])
            ]
            timeout_signature_indices = set()
            if step_wall_timed_out and newly_done_indices:
                for i in newly_done_indices:
                    sig_match = (
                        float(step_rewards[i]) == 1.0
                        and int(infos["to_play"][i]) == 1
                        and int(infos.get("win_reason", np.ones(num_envs, dtype=np.int32))[i]) == 1
                        and int(infos.get("num_options", np.ones(num_envs, dtype=np.int32))[i]) == 1
                        and int(infos.get("is_selfplay", np.ones(num_envs, dtype=np.int32))[i]) == 1
                    )
                    if sig_match:
                        timeout_signature_indices.add(i)
                if not timeout_signature_indices:
                    timeout_signature_indices.update(newly_done_indices)

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

                reported_timeout = (self.timeout > 0) and (
                    reported_cand_ms >= self.timeout * 1000.0
                    or reported_opp_ms >= self.timeout * 1000.0
                )
                engine_timeout = reported_timeout or (idx in timeout_signature_indices)

                # YGOProEnvImpl::step は step_count_ >= max_steps で _duel_end(winner, 0x01) を呼ぶ。
                # max_episode_steps=self.max_steps の設定に加えて、env_steps >= self.max_steps でも
                # 確実にステップ上限打ち切り (truncated) として判定し、通常決着(normal_lp0)と区別する。
                step_limit_reached = (not engine_timeout) and (
                    bool(truncated[idx]) or int(ep_env_steps[idx]) >= self.max_steps
                )
                is_trunc = step_limit_reached
                is_term = bool(terminated[idx]) and (not is_trunc) and (not engine_timeout)

                if engine_timeout:
                    # handle_timeout() が書き込むダミー報酬(reward=1.0, win_reason=1)を無効化
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

                is_abnormal = is_trunc or (outcome == "draw") or engine_timeout or (not is_term)

                if engine_timeout:
                    reason_label = "engine_timeout"
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
                    "abnormal": is_abnormal,
                    "reason_label": reason_label,
                    "cand_engine_step_ms": round(cand_engine_ms, 3),
                    "opp_engine_step_ms": round(opp_engine_ms, 3),
                }

            prev_dones = dones

        # 万一タイムアウト例外または max_steps_per_wave 到達で未完了の環境があれば異常終了として記録
        for idx in range(num_envs):
            if episodes[idx] is None:
                is_timeout_abort = wave_timeout_abort or (
                    self.timeout > 0 and float(ep_max_step_wall_ms[idx]) >= self.timeout * 950.0
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
        total_eps = len(all_episodes)
        first_eps = [e for e in all_episodes if e["candidate_seat"] == 0]
        second_eps = [e for e in all_episodes if e["candidate_seat"] == 1]

        wins = sum(e["win"] for e in all_episodes)
        losses = sum(e["loss"] for e in all_episodes)
        draws = sum(e["draw"] for e in all_episodes)
        win_rate = wins / total_eps if total_eps > 0 else 0.0

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

        # 同一シードの先後ペア集計 (pair_id ごとに先攻結果と後攻結果を照合)
        pairs_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for e in all_episodes:
            pid = e["pair_id"]
            pairs_by_id.setdefault(pid, {})[e["candidate_seat"]] = e

        pair_sweep_wins = 0      # 先攻〇・後攻〇 (2勝0敗)
        pair_split_first = 0     # 先攻〇・後攻● (先攻側がどちらも勝利)
        pair_split_second = 0    # 先攻●・後攻〇 (後攻側がどちらも勝利)
        pair_sweep_losses = 0    # 先攻●・後攻● (0勝2敗)
        pair_other_draws = 0     # 引き分けを含むペア
        pair_scores: List[float] = []

        for pid, seat_map in sorted(pairs_by_id.items()):
            e0 = seat_map.get(0)
            e1 = seat_map.get(1)
            if not e0 or not e1:
                continue
            pair_scores.append(0.5 * (float(e0["win"]) + float(e1["win"])))
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

        # 総合勝率の95%信頼区間は、同一シード先後ペアの相関を反映したペア単位クラスタ分散から算出
        ci_low, ci_high, pair_se, n_eff = paired_wilson_ci95(pair_scores, wins, total_eps)

        # --- 2. 決着理由・異常終了の集計 ---
        normal_lp0_count = sum(1 for e in all_episodes if e["reason_label"] == "normal_lp0")
        normal_deckout_count = sum(
            1 for e in all_episodes if e["reason_label"] == "normal_deckout_or_special"
        )
        truncated_count = sum(1 for e in all_episodes if e["truncated"])
        engine_timeout_count = sum(1 for e in all_episodes if e["engine_timeout"])
        aborted_max_steps_count = sum(
            1 for e in all_episodes if e["reason_label"] == "aborted_max_steps"
        )
        abnormal_total = sum(1 for e in all_episodes if e["abnormal"])
        normal_total = total_eps - abnormal_total

        # --- 3. 判断数・判断時間の集計 ---
        cand_dec_arr = np.array([e["candidate_decisions"] for e in all_episodes], dtype=np.float64)
        env_step_arr = np.array([e["env_steps"] for e in all_episodes], dtype=np.float64)
        cand_eng_ms_arr = np.array(
            [e["cand_engine_step_ms"] for e in all_episodes], dtype=np.float64
        )

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
        env_ms_per_step = (
            (total_env_step_sec * 1000.0) / total_env_steps_timed
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
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(win_rate, 4),
                "win_rate_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "win_rate_se_paired": round(pair_se, 4),
                "effective_sample_size": round(n_eff, 2),
                "avg_reward": round(float(np.mean([e["candidate_reward"] for e in all_episodes])), 4),
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
                "env_ms_per_step": round(env_ms_per_step, 4),
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
    lines.append("| 候補 (Candidate) | 対戦相手 (Opponent) | 総合勝率 (勝-敗-分) | 95%信頼区間 (ペア相関考慮) | 先攻勝率 | 後攻勝率 | 先後ペア (2連勝 / 1勝1敗 / 2連敗) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for s in summaries:
        cand_lbl = s["candidate"]["label"]
        opp_lbl = s["opponent"]["label"]
        r = s["results"]
        fp = r["first_player"]
        sp = r["second_player"]
        pairs = r["seed_pairs"]
        split_1_1 = pairs["split_1_1_first_player_won_both"] + pairs["split_1_1_second_player_won_both"]
        ci_l, ci_h = r["win_rate_ci95"]

        lines.append(
            f"| `{cand_lbl}` | `{opp_lbl}` | "
            f"**{r['win_rate']*100:.1f}%** ({r['wins']}-{r['losses']}-{r['draws']}) | "
            f"{ci_l*100:.1f}%–{ci_h*100:.1f}% | "
            f"{fp['win_rate']*100:.1f}% ({fp['wins']}/{fp['episodes']}) | "
            f"{sp['win_rate']*100:.1f}% ({sp['wins']}/{sp['episodes']}) | "
            f"{pairs['sweep_2_0_wins']} / {split_1_1} / {pairs['sweep_0_2_losses']} |"
        )

    lines.append("")
    lines.append("## 2. 決着理由・異常終了の内訳")
    lines.append("")
    lines.append("| 候補 vs 対戦相手 | 正常決着 (計) | LP0決着 (`reason=1`) | デッキ切れ等 (`reason=-1`) | 異常終了・打ち切り (計) | ステップ上限/Truncated | タイムアウト |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for s in summaries:
        t = s["terminations"]
        lines.append(
            f"| `{s['candidate']['label']}` vs `{s['opponent']['label']}` | "
            f"{t['normal_total']} | {t['normal_lp0']} | {t['normal_deckout_or_special']} | "
            f"**{t['abnormal_total']}** | {t['truncated']} | {t['engine_timeout']} |"
        )

    lines.append("")
    lines.append("## 3. 判断時間・処理速度の内訳")
    lines.append("")
    lines.append("| 候補 vs 対戦相手 | 候補推論 (ms/判断) | 相手推論 (ms/判断) | エンジン処理 (ms/step) | 候補の平均判断数/戦 (最小–最大) | 実測時間 (秒) | SPS |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    for s in summaries:
        td = s["timing_and_decisions"]
        cd = td["candidate_decisions_per_duel"]
        opp_ms_str = f"{td['opponent_ms_per_decision']:.3f}" if s["opponent"]["kind"] != "builtin" else "-"
        lines.append(
            f"| `{s['candidate']['label']}` vs `{s['opponent']['label']}` | "
            f"{td['candidate_ms_per_decision']:.3f} ms | {opp_ms_str} | "
            f"{td['env_ms_per_step']:.3f} ms | "
            f"{cd['mean']:.1f} ({cd['min']}–{cd['max']}) | "
            f"{td['active_wave_wall_seconds']:.2f} s | {td['sps']:.0f} |"
        )

    lines.append("")
    lines.append("## 4. 評価結果の解釈上の注意点")
    lines.append("")
    lines.append("- **先後ペア比較**: 同じ乱数シードで候補モデルが先攻（Player 0）と後攻（Player 1）の両方をプレイしています。「2連勝」は初手運や先攻・後攻の有利不利に関わらず候補モデルが両方勝ったペア数、「1勝1敗」は片方の席でのみ勝ったペア数です。")
    lines.append("- **95%信頼区間（ペア相関考慮）**: 各シードペアの平均勝率 $P_i \\in \\{0, 0.5, 1\\}$ を独立なクラスタ標本としてペア間標本分散から標準誤差・有効サンプルサイズを算出し、先後ペア内の正負の相関を反映した Wilson 信頼区間を表示しています。信頼区間が重なっている場合は「統計的に明確な強さの差はまだ断定できない」ことを示します。")
    lines.append("- **ステップ上限打ち切り（Truncated）**: 1戦の環境ステップ数が `max_steps`（デフォルト `1000`）に達した場合、C++エンジンはLP比較で勝敗を付与して終了しますが、本評価では通常決着（`normal_lp0`）とは区別し、`truncated=True`（異常終了・打ち切り）として計上しています。")
    lines.append("- **判断時間**: 初回XLAコンパイル時間（ウォームアップ）を除外し、未収集の有効エピソードで実際に行われた判断（`active_uncollected` かつ該当プレイヤー手番）に対応する推論時間と判断数の対象を一致させた定常状態の1判断あたりミリ秒（ms/判断）です。")
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
