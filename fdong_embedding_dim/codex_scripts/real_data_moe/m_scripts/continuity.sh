#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
COMMON_STEP="$(normalize_step "${CHECKPOINT_STEP:-$(latest_common_checkpoint_step)}")"
BASELINE_CKPT="${BASELINE_CKPT:-$(checkpoint_for_run baseline "${COMMON_STEP}")}"
PROPOSED_CKPT="${PROPOSED_CKPT:-$(checkpoint_for_run proposed "${COMMON_STEP}")}"
MPLBACKEND=Agg python3 -m "${MOE_MODULE}.analyze_multi_continuity" \
  --run "baseline=${BASELINE_CKPT}" \
  --run "proposed=${PROPOSED_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output-dir "${ANALYSIS_DIR}/continuity_step${COMMON_STEP}" \
  --device cuda:0 \
  --num-sequences "${CONTINUITY_SEQUENCES:-1024}" \
  --num-tokens "${CONTINUITY_TOKENS:-1024}" \
  --cache-capacities "${CONTINUITY_CACHE_CAPACITIES:-1,2,3,4}"
