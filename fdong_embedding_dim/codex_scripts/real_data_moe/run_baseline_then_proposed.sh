#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/Users/bytedance/Desktop/dclm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_16k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/fdong_embedding_dim/outputs/real_data_moe_4layer}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.prepare_tokenizer \
  --data-dir "${DATA_DIR}" \
  --output-dir "${TOKENIZER_DIR}" \
  --vocab-size 16384

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --device mps
  --max-steps 5000
  --save-every 250
  --keep-last 3
  --sequence-length 512
  --batch-size 2
  --gradient-accumulation 4
  --no-orthogonalize-tail
  --resume auto
)

"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.train \
  --variant baseline \
  --output-dir "${OUTPUT_ROOT}/baseline" \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.train \
  --variant proposed \
  --output-dir "${OUTPUT_ROOT}/proposed" \
  "${COMMON_ARGS[@]}"
