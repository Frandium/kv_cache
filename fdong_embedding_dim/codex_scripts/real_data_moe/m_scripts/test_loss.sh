#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_ROOT}/analysis}"
COMMON_STEP="$(normalize_step "${CHECKPOINT_STEP:-$(latest_common_checkpoint_step)}")"
BASELINE_CKPT="${BASELINE_CKPT:-$(checkpoint_for_run baseline "${COMMON_STEP}")}"
PROPOSED_CKPT="${PROPOSED_CKPT:-$(checkpoint_for_run proposed "${COMMON_STEP}")}"
python3 -m "${MOE_MODULE}.evaluate_checkpoints" \
  --run "baseline=${BASELINE_CKPT}" \
  --run "proposed=${PROPOSED_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --output "${ANALYSIS_DIR}/test_loss_step${COMMON_STEP}.json" \
  --device cuda:0 \
  --num-sequences "${TEST_SEQUENCES:-16}" \
  --sequence-length 1024 \
  --batch-size 1
