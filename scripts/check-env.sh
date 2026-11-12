#!/usr/bin/env bash
# 非破壊の環境・ファイル整合性チェック用スクリプト（学習や対戦は実行しません）
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="/opt/ygo-ai"

EXPECTED_COMMIT="5cfbe008ed110e32c2591fd9f4922d37d000b1a9"
EXPECTED_SO_SHA256="55ad5f03a65530df6d3e4aa56d928bc7c134070d1e178bc4fa7cc834e5726a57"
EXPECTED_CKPT_4096_SHA256="dd100d9f2e3a86cc5d292a36bacd78bd49d4375d380b2ee12747696abea1d49c"
EXPECTED_CKPT_4608_SHA256="a181b3304d7f95ce2140d950ddcca0257e2c167d96ea433250b0bbd1867e2185"

echo "=========================================="
echo " yugioh-ai プロジェクト接続・整合性確認"
echo "=========================================="
echo "プロジェクトパス: ${PROJ_DIR}"
echo "ランタイムパス  : ${RUNTIME_DIR}"
echo ""

# 1. Python venv check
if [[ -x "${RUNTIME_DIR}/.venv/bin/python" ]]; then
  PY_VER="$("${RUNTIME_DIR}/.venv/bin/python" --version 2>&1)"
  echo "[OK] Python仮想環境: ${PY_VER} (${RUNTIME_DIR}/.venv)"
else
  echo "[NG] Python仮想環境が見つかりません: ${RUNTIME_DIR}/.venv/bin/python"
  exit 1
fi

# 2. Source commit path check
CURRENT_TARGET="$(readlink -f "${RUNTIME_DIR}/current" || true)"
if [[ "${CURRENT_TARGET}" == *"${EXPECTED_COMMIT}"* ]]; then
  echo "[OK] ygo-agent 固定版ソース: ${CURRENT_TARGET}"
else
  echo "[WARN] /opt/ygo-ai/current のリンク先が想定と異なります: ${CURRENT_TARGET}"
fi

# 3. Native engine SHA256 check
SO_FILE="$(find -L "${RUNTIME_DIR}/current/" "${RUNTIME_DIR}/.venv/" -name "ygopro_ygoenv_cp310.so" 2>/dev/null | head -n 1 || true)"
if [[ -z "${SO_FILE}" ]]; then
  SO_FILE="$(find -L "${RUNTIME_DIR}" -maxdepth 5 -name "*ygoenv*.so" 2>/dev/null | head -n 1 || true)"
fi
if [[ -n "${SO_FILE}" && -f "${SO_FILE}" ]]; then
  SO_SHA="$(sha256sum "${SO_FILE}" | awk '{print $1}')"
  if [[ "${SO_SHA}" == "${EXPECTED_SO_SHA256}" ]]; then
    echo "[OK] ネイティブエンジン SHA256一致 (${SO_FILE})"
  else
    echo "[NG] ネイティブエンジン SHA256不一致: ${SO_SHA}"
    exit 1
  fi
else
  echo "[NG] ygopro_ygoenv_cp310.so が見つかりません"
  exit 1
fi

# 4. Deck CRLF check
DECK_FILE="${PROJ_DIR}/assets/deck/BlueEyes.ydk"
if [[ -f "${DECK_FILE}" ]]; then
  if grep -q $'\r' "${DECK_FILE}"; then
    echo "[NG] デッキファイルにCRLF改行が含まれています: ${DECK_FILE}"
    exit 1
  else
    MAIN_COUNT="$(awk '/^#main/{flag=1;next}/^#extra|^!side/{flag=0}flag && /^[0-9]+$/' "${DECK_FILE}" | wc -l)"
    echo "[OK] デッキファイル (LF改行・メイン${MAIN_COUNT}枚): ${DECK_FILE}"
  fi
else
  echo "[NG] デッキファイルが見つかりません: ${DECK_FILE}"
  exit 1
fi

# 5. Checkpoints SHA256 check
CKPT_4096="${PROJ_DIR}/checkpoints/trial-2026-10-27/blueeyes-4096.flax_model"
CKPT_4608="${PROJ_DIR}/checkpoints/trial-2026-10-27/blueeyes-4096-plus512.flax_model"

if [[ -f "${CKPT_4096}" ]]; then
  SHA1="$(sha256sum "${CKPT_4096}" | awk '{print $1}')"
  if [[ "${SHA1}" == "${EXPECTED_CKPT_4096_SHA256}" ]]; then
    echo "[OK] 初回学習重み (4,096判断) SHA256一致"
  else
    echo "[NG] 初回学習重み (4,096判断) SHA256不一致: ${SHA1}"
    exit 1
  fi
fi

if [[ -f "${CKPT_4608}" ]]; then
  SHA2="$(sha256sum "${CKPT_4608}" | awk '{print $1}')"
  if [[ "${SHA2}" == "${EXPECTED_CKPT_4608_SHA256}" ]]; then
    echo "[OK] 追加学習重み (4,096+512判断) SHA256一致"
  else
    echo "[NG] 追加学習重み (4,096+512判断) SHA256不一致: ${SHA2}"
    exit 1
  fi
fi

echo ""
echo "すべての静的整合性チェックに合格しました。"
