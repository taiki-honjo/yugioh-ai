#!/usr/bin/env bash
# Ubuntu-22.04 内で実行するモデル評価用補助スクリプト
#
# 使い方:
#   1. 標準ベースライン評価スイート一括実行（引数なし、または --suite）:
#      bash scripts/eval-blueeyes.sh --suite
#
#   2. 個別のモデルと対戦相手を指定して評価（例: 32戦、シード1000）:
#      bash scripts/eval-blueeyes.sh \
#        --candidate checkpoints/trial-2026-10-27/blueeyes-4096-plus512.flax_model \
#        --opponent checkpoints/trial-2026-10-27/blueeyes-4096.flax_model \
#        --num-episodes 32 --seed 1000
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="/opt/ygo-ai"

source "${RUNTIME_DIR}/.venv/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"

if [[ $# -eq 0 ]]; then
  set -- --suite
fi

python -u "${PROJ_DIR}/scripts/evaluate.py" "$@"
