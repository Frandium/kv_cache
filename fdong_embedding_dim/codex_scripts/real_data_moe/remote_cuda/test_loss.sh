#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
BASELINE_CKPT="${BASELINE_CKPT:-${OUTPUT_ROOT}/baseline/latest.pt}"
PROPOSED_CKPT="${PROPOSED_CKPT:-${OUTPUT_ROOT}/proposed_total_matched/latest.pt}"
ROUTING_CKPT="${ROUTING_CKPT:-${OUTPUT_ROOT}/routing_only/latest.pt}"
python3 -m fdong_embedding_dim.codex_scripts.real_data_moe.evaluate_checkpoints \
  --run "baseline=${BASELINE_CKPT}" \
  --run "proposed_total_matched=${PROPOSED_CKPT}" \
  --run "routing_only=${ROUTING_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output "${ANALYSIS_DIR}/test_loss.json" \
  --device cuda:0 \
  --num-sequences "${TEST_SEQUENCES:-16}" \
  --sequence-length 1024 \
  --batch-size 1
