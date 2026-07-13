#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
BASELINE_CKPT="${BASELINE_CKPT:-${OUTPUT_ROOT}/baseline/latest.pt}"
PROPOSED_CKPT="${PROPOSED_CKPT:-${OUTPUT_ROOT}/proposed_total_matched/latest.pt}"
ROUTING_CKPT="${ROUTING_CKPT:-${OUTPUT_ROOT}/routing_only/latest.pt}"
MPLBACKEND=Agg python3 -m fdong_embedding_dim.codex_scripts.real_data_moe.analyze_multi_continuity \
  --run "baseline=${BASELINE_CKPT}" \
  --run "proposed_total_matched=${PROPOSED_CKPT}" \
  --run "routing_only=${ROUTING_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${ANALYSIS_DIR}" \
  --device cuda:0 \
  --num-sequences "${CONTINUITY_SEQUENCES:-16}" \
  --num-tokens 100
