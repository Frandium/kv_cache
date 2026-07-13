#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
MPLBACKEND=Agg python3 -m fdong_embedding_dim.codex_scripts.real_data_moe.plot_multi_training_loss \
  --run "baseline=${OUTPUT_ROOT}/baseline/metrics.jsonl" \
  --run "proposed_total_matched=${OUTPUT_ROOT}/proposed_total_matched/metrics.jsonl" \
  --run "routing_only=${OUTPUT_ROOT}/routing_only/metrics.jsonl" \
  --output "${ANALYSIS_DIR}/training_loss.png" \
  --smooth-window 50
