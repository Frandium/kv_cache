#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
COMMON_STEP="$(normalize_step "${CHECKPOINT_STEP:-$(latest_common_checkpoint_step)}")"
BASELINE_CKPT="${BASELINE_CKPT:-$(checkpoint_for_run baseline_lb "${COMMON_STEP}")}"
PROPOSED_NO_LB_CKPT="${PROPOSED_NO_LB_CKPT:-$(checkpoint_for_run proposed_no_lb "${COMMON_STEP}")}"
PROPOSED_LB_CKPT="${PROPOSED_LB_CKPT:-$(checkpoint_for_run proposed_lb "${COMMON_STEP}")}"
STEP_PADDED="${COMMON_STEP}"
MPLBACKEND=Agg python3 -m "${MOE_MODULE}.analyze_multi_continuity" \
  --run "baseline_lb=${BASELINE_CKPT}" \
  --run "proposed_no_lb=${PROPOSED_NO_LB_CKPT}" \
  --run "proposed_lb=${PROPOSED_LB_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${ANALYSIS_DIR}/continuity_step${STEP_PADDED}" \
  --device cuda:0 \
  --num-sequences "${CONTINUITY_SEQUENCES:-16}" \
  --num-tokens "${CONTINUITY_TOKENS:-1024}" \
  --cache-capacities "${CONTINUITY_CACHE_CAPACITIES:-1,2,3,4}"
