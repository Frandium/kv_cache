#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_NAME="${RUN_NAME:-proposed_no_lb}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_ROOT}/${RUN_NAME}/latest.pt}"
TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,siqa,race,winogrande}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/lm_eval}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "${EVAL_OUTPUT_DIR}"
python3 -m "${MOE_MODULE}.run_lm_eval" \
  --checkpoint "${CHECKPOINT}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --tasks "${TASKS}" \
  --device "${EVAL_DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --output "${EVAL_OUTPUT_DIR}/${RUN_NAME}_results.json"
