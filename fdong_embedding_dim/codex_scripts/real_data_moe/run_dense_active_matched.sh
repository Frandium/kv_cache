#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/Users/bytedance/Desktop/dclm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_16k}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/fdong_embedding_dim/outputs/real_data_moe_4layer_v2/dense_active_matched}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.train \
  --variant dense \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device mps \
  --max-steps 5000 \
  --save-every 250 \
  --keep-last 0 \
  --log-every 10 \
  --sequence-length 512 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --seed "${SEED}" \
  --no-orthogonalize-tail \
  --resume auto

echo "[done] dense active-matched output: ${OUTPUT_DIR}"
