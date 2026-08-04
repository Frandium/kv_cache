#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
MPLBACKEND=Agg python3 -m "${MOE_MODULE}.plot_multi_training_loss" \
  --run "baseline_lb=${OUTPUT_ROOT}/baseline_lb/metrics.jsonl" \
  --run "proposed_no_lb=${OUTPUT_ROOT}/proposed_no_lb/metrics.jsonl" \
  --run "proposed_lb=${OUTPUT_ROOT}/proposed_lb/metrics.jsonl" \
  --output "${ANALYSIS_DIR}/training_loss.png" \
  --smooth-window 50 \
  --truncate-common
