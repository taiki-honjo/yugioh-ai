#!/usr/bin/env bash
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJ_DIR/experiments/elfnotes-20261106/runtime"
steps=${1:-4096}
checkpoint=${2:-}
if [[ ! "$steps" =~ ^[0-9]+$ ]] || ((steps < 128 || steps % 128 != 0)); then
  echo '判断数は128の倍数を指定してください。' >&2
  exit 2
fi
stamp=$(TZ=Asia/Tokyo date +%Y%m%d-%H%M%S)
run="$PROJ_DIR/experiments/elfnotes-20261106/train-$stamp-$steps"
mkdir -p "$run/checkpoints"
export PYTHONPATH="$RUNTIME_DIR/agent:$RUNTIME_DIR/agent/ygoenv"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
updates=$((steps / 128))
extra=()
if [[ -n "$checkpoint" ]]; then
  checkpoint=$(realpath "$checkpoint")
  [[ -s "$checkpoint" ]] || { echo '重みファイルが見つかりません。' >&2; exit 2; }
  extra=(--checkpoint "$checkpoint")
fi
check_args=()
if [[ -n "$checkpoint" ]]; then check_args=(--checkpoint "$checkpoint"); fi
/opt/ygo-ai/.venv/bin/python "$PROJ_DIR/scripts/check-elfnotes-runtime.py" "${check_args[@]}"
cp "$PROJ_DIR/env/elfnotes/runtime.json" "$run/checkpoints/runtime.json"
cd "$RUNTIME_DIR/agent/scripts"
printf '%s\n' "$run" > "$PROJ_DIR/experiments/elfnotes-20261106/latest-training-path.txt"
/opt/ygo-ai/.venv/bin/python -u cleanba.py \
  --actor-device-ids 0 --learner-device-ids 0 \
  --deck "$PROJ_DIR/assets/deck/ElfnotesKewlTune.ydk" \
  --max-options 128 --timeout 10 \
  --local-num-envs 4 --local-env-threads 2 --num-actor-threads 1 \
  --num-steps 32 --num-minibatches 4 --learning-rate 1e-4 --vloss-clip 1.0 \
  --save-interval "$updates" --local-eval-episodes 4 --eval-interval "$updates" \
  --total-timesteps "$steps" --seed 0 --log-frequency 1 \
  --ckpt-dir "$run/checkpoints" --tb-dir "$run/tensorboard" \
  "${extra[@]}" 2>&1 | tee "$run/train.log"
printf '保存先: %s\n' "$run"
