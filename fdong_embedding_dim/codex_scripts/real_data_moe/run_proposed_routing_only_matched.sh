#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/Users/bytedance/Desktop/dclm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_16k}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/fdong_embedding_dim/outputs/real_data_moe_4layer_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/proposed_routing_only_matched}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${EXPERIMENT_ROOT}/analysis_proposed_routing_only_matched}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"

cd "${ROOT_DIR}"

# This model has exactly the same common/tail expert shapes as baseline.
# Only the routing signal changes from residual to attention_mean.
"${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.train \
  --variant proposed \
  --common-intermediate-size 768 \
  --tail-intermediate-size 768 \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device mps \
  --max-steps 5000 \
  --save-every 250 \
  --keep-last 3 \
  --log-every 10 \
  --sequence-length 512 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --seed "${SEED}" \
  --no-orthogonalize-tail \
  --resume auto

mkdir -p "${ANALYSIS_DIR}"

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_routing_only_matched \
  "${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.plot_training_loss \
  --baseline-metrics "${EXPERIMENT_ROOT}/baseline/metrics.jsonl" \
  --proposed-metrics "${OUTPUT_DIR}/metrics.jsonl" \
  --output-dir "${ANALYSIS_DIR}" \
  --smooth-window 50

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_routing_only_matched \
  "${PYTHON_BIN}" -m fdong_embedding_dim.codex_scripts.real_data_moe.analyze_route_continuity \
  --baseline-checkpoint "${EXPERIMENT_ROOT}/baseline/latest.pt" \
  --proposed-checkpoint "${OUTPUT_DIR}/latest.pt" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --device mps \
  --num-sequences 16 \
  --num-tokens 100

echo "[done] routing-only matched experiment: ${OUTPUT_DIR}"
