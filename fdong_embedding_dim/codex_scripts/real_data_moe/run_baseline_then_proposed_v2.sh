#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/Users/bytedance/Desktop/dclm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_16k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/fdong_embedding_dim/outputs/real_data_moe_4layer_v2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.prepare_tokenizer \
  --data-dir "${DATA_DIR}" \
  --output-dir "${TOKENIZER_DIR}" \
  --vocab-size 16384

# DCLMTokenStream uses its own Random(seed) for file order. Giving both runs
# the same seed makes optimizer step n consume the same token batch in both.
COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --device mps
  --max-steps 5000
  --save-every 250
  --keep-last 0
  --log-every 10
  --sequence-length 512
  --batch-size 2
  --gradient-accumulation 4
  --seed "${SEED}"
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

ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"
mkdir -p "${ANALYSIS_DIR}"

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_real_data_moe_v2 \
  "${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.plot_training_loss \
  --baseline-metrics "${OUTPUT_ROOT}/baseline/metrics.jsonl" \
  --proposed-metrics "${OUTPUT_ROOT}/proposed/metrics.jsonl" \
  --output-dir "${ANALYSIS_DIR}" \
  --smooth-window 50

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_real_data_moe_v2 \
  "${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.analyze_route_continuity \
  --baseline-checkpoint "${OUTPUT_ROOT}/baseline/latest.pt" \
  --proposed-checkpoint "${OUTPUT_ROOT}/proposed/latest.pt" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --device mps \
  --num-sequences 16 \
  --num-tokens 100

echo "[done] training and analysis outputs: ${OUTPUT_ROOT}"
