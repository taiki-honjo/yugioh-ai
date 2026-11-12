#!/usr/bin/env bash
# Ubuntu-22.04 内で実行する小規模学習・追加学習用スクリプト
# 使い方:
#   新規学習(4096判断): bash scripts/train-blueeyes.sh 4096
#   重み引き継ぎ追加学習: bash scripts/train-blueeyes.sh 4096 checkpoints/trial-2026-10-27/blueeyes-4096-plus512.flax_model
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="/opt/ygo-ai"

steps=${1:-4096}
checkpoint=${2:-}

if [[ ! "$steps" =~ ^[0-9]+$ ]] || ((steps < 128 || steps % 128 != 0)); then
  echo "エラー: 判断数(steps)は 128 の倍数の正の整数で指定してください（例: 512, 4096）。"
  exit 2
fi

extra=()
if [[ -n "$checkpoint" ]]; then
  # 相対パスの場合は実行元またはプロジェクトルートから解決
  if [[ "$checkpoint" != /* && -f "${PROJ_DIR}/${checkpoint}" ]]; then
    checkpoint="${PROJ_DIR}/${checkpoint}"
  else
    checkpoint="$(realpath "$checkpoint")"
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "エラー: チェックポイントが存在しないか空です: $checkpoint"
    exit 2
  fi
  extra=(--checkpoint "$checkpoint")
fi

source "${RUNTIME_DIR}/.venv/bin/activate"
cd "${RUNTIME_DIR}/current/scripts"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

stamp=$(date +%Y%m%d-%H%M%S)
run="${PROJ_DIR}/experiments/manual-$stamp"
mkdir -p "$run"
updates=$((steps / 128))

python -u cleanba.py \
  --actor-device-ids 0 --learner-device-ids 0 \
  --deck "${PROJ_DIR}/assets/deck/BlueEyes.ydk" \
  --local-num-envs 4 --local-env-threads 2 --num-actor-threads 1 \
  --num-steps 32 --num-minibatches 4 \
  --learning-rate 1e-4 --vloss-clip 1.0 \
  --save-interval "$updates" --local-eval-episodes 4 --eval-interval "$updates" \
  --total-timesteps "$steps" --seed 0 --log-frequency 1 \
  --ckpt-dir "$run/checkpoints" --tb-dir "$run/tensorboard" \
  "${extra[@]}" 2>&1 | tee "$run/train.log"

echo "保存先: $run"
