#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../moe" ]]; then
  BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -d "${SCRIPT_DIR}/moe" ]]; then
  BUNDLE_ROOT="${SCRIPT_DIR}"
else
  BUNDLE_ROOT="$(pwd)"
fi
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/fmoe_cuda_2b_outputs}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${BUNDLE_ROOT}/moe/tokenizer_qwen3}"
TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,siqa,race,winogrande}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/lm_eval}"

RUNS=(baseline proposed_total_matched routing_only)
DEVICES=(cuda:7 cuda:6 cuda:5)

mkdir -p "${EVAL_OUTPUT_DIR}"

steps_for_run() {
  local run="$1"
  find "${OUTPUT_ROOT}/${run}" -maxdepth 1 -type f -name 'checkpoint-*.pt' \
    | sed -E 's/.*checkpoint-0*([0-9]+)\.pt$/\1/' \
    | sort -n
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

for run in "${RUNS[@]}"; do
  if [[ ! -d "${OUTPUT_ROOT}/${run}" ]]; then
    echo "[error] missing run directory: ${OUTPUT_ROOT}/${run}" >&2
    exit 1
  fi
  steps_for_run "${run}" > "${tmp_dir}/${run}.steps"
  if [[ ! -s "${tmp_dir}/${run}.steps" ]]; then
    echo "[error] no checkpoint-*.pt found in ${OUTPUT_ROOT}/${run}" >&2
    exit 1
  fi
done

comm -12 "${tmp_dir}/baseline.steps" "${tmp_dir}/proposed_total_matched.steps" \
  | comm -12 - "${tmp_dir}/routing_only.steps" > "${tmp_dir}/common.steps"

if [[ ! -s "${tmp_dir}/common.steps" ]]; then
  echo "[error] no common checkpoint step found across: ${RUNS[*]}" >&2
  exit 1
fi

COMMON_STEP="$(tail -n 1 "${tmp_dir}/common.steps")"
STEP_PADDED="$(printf '%07d' "${COMMON_STEP}")"
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
  nohup "${PYTHON_BIN}" -m moe.run_lm_eval \
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
