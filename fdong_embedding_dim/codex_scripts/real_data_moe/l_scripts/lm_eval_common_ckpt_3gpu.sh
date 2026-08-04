#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,siqa,race,winogrande}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/lm_eval}"

RUNS=(baseline_lb proposed_no_lb proposed_lb)
DEVICES=(cuda:7 cuda:6 cuda:5)

mkdir -p "${EVAL_OUTPUT_DIR}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

for run in "${RUNS[@]}"; do
  if [[ ! -d "${OUTPUT_ROOT}/${run}" ]]; then
    echo "[error] missing run directory: ${OUTPUT_ROOT}/${run}" >&2
    exit 1
  fi
  checkpoint_steps "${run}" > "${tmp_dir}/${run}.steps"
  if [[ ! -s "${tmp_dir}/${run}.steps" ]]; then
    echo "[error] no checkpoint-*.pt found in ${OUTPUT_ROOT}/${run}" >&2
    exit 1
  fi
done

comm -12 "${tmp_dir}/baseline_lb.steps" "${tmp_dir}/proposed_no_lb.steps" \
  | comm -12 - "${tmp_dir}/proposed_lb.steps" > "${tmp_dir}/common.steps"

if [[ ! -s "${tmp_dir}/common.steps" ]]; then
  echo "[error] no common checkpoint step found across: ${RUNS[*]}" >&2
  exit 1
fi

COMMON_STEP="$(tail -n 1 "${tmp_dir}/common.steps")"
STEP_PADDED="$(normalize_step "${COMMON_STEP}")"
echo "[info] common latest checkpoint step: ${COMMON_STEP}"
echo "[info] output dir: ${EVAL_OUTPUT_DIR}"
echo "[info] tasks: ${TASKS}"
echo "[info] batch size: ${EVAL_BATCH_SIZE}"

for index in "${!RUNS[@]}"; do
  run="${RUNS[$index]}"
  device="${DEVICES[$index]}"
  checkpoint="${OUTPUT_ROOT}/${run}/checkpoint-${STEP_PADDED}.pt"
  output="${EVAL_OUTPUT_DIR}/${run}_step${STEP_PADDED}_results.json"
  log="${EVAL_OUTPUT_DIR}/${run}_step${STEP_PADDED}.log"

  echo "[launch] ${run} ${checkpoint} on ${device}"
  nohup "${PYTHON_BIN}" -m "${MOE_MODULE}.run_lm_eval" \
    --checkpoint "${checkpoint}" \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --tasks "${TASKS}" \
    --device "${device}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --output "${output}" \
    > "${log}" 2>&1 < /dev/null &
  echo "[pid] $!  [log] ${log}  [json] ${output}"
done

echo "[info] monitor with:"
for run in "${RUNS[@]}"; do
  echo "  tail -f ${EVAL_OUTPUT_DIR}/${run}_step${STEP_PADDED}.log"
done
