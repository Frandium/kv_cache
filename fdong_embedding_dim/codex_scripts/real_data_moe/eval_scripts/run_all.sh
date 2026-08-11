#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/mnt/workspace/dclm/global-shard_01_of_10}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${BUNDLE_ROOT}/moe/tokenizer_qwen3}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${BUNDLE_ROOT}/eval_outputs}"
RAW_DIR="${EVAL_OUTPUT_ROOT}/raw"
COMPACT_DIR="${EVAL_OUTPUT_ROOT}/compact"
LOG_DIR="${EVAL_OUTPUT_ROOT}/logs"
JOBS_DIR="${EVAL_OUTPUT_ROOT}/jobs"
MANIFEST_DIR="${EVAL_OUTPUT_ROOT}/manifest"
LATEST_MANIFEST="${MANIFEST_DIR}/checkpoint_manifest.json"
SCALING_MANIFEST="${MANIFEST_DIR}/scaling_manifest.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ROUTING_SEQUENCES="${ROUTING_SEQUENCES:-64}"
ROUTING_LENGTH="${ROUTING_LENGTH:-2048}"
PREDICT_TRAIN_SEQUENCES="${PREDICT_TRAIN_SEQUENCES:-512}"
PREDICT_TEST_SEQUENCES="${PREDICT_TEST_SEQUENCES:-64}"
PREDICT_LENGTH="${PREDICT_LENGTH:-256}"
PREDICT_EPOCHS="${PREDICT_EPOCHS:-4}"
SWAP_PROMPT_LENGTH="${SWAP_PROMPT_LENGTH:-32}"
SWAP_DECODE_TOKENS="${SWAP_DECODE_TOKENS:-2048}"
SWAP_REPEATS="${SWAP_REPEATS:-3}"
TTFT_PROMPT_LENGTHS="${TTFT_PROMPT_LENGTHS:-32,128,512,1024,2048}"
TTFT_REPEATS="${TTFT_REPEATS:-5}"
TEST_SEQUENCES="${TEST_SEQUENCES:-32}"
TEST_SEQUENCE_LENGTH="${TEST_SEQUENCE_LENGTH:-1024}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,siqa,race,winogrande}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-8}"
SCALING_PARALLEL_GPUS="${SCALING_PARALLEL_GPUS:-8}"

mkdir -p "${RAW_DIR}" "${COMPACT_DIR}" "${LOG_DIR}" "${JOBS_DIR}" "${MANIFEST_DIR}"
CURRENT_STAGE="initialization"
printf 'RUNNING\t%s\t%s\n' "$(date -Iseconds)" "${CURRENT_STAGE}" > "${EVAL_OUTPUT_ROOT}/STATUS.tsv"

on_error() {
  local exit_code=$?
  printf 'FAILED\t%s\t%s\texit_code=%s\n' "$(date -Iseconds)" "${CURRENT_STAGE}" "${exit_code}" \
    > "${EVAL_OUTPUT_ROOT}/STATUS.tsv"
  echo "[failed] stage=${CURRENT_STAGE} exit_code=${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

set_stage() {
  CURRENT_STAGE="$1"
  printf 'RUNNING\t%s\t%s\n' "$(date -Iseconds)" "${CURRENT_STAGE}" > "${EVAL_OUTPUT_ROOT}/STATUS.tsv"
  echo "[stage] ${CURRENT_STAGE} started at $(date -Iseconds)"
}

run_job() {
  local metric="$1" size="$2" method="$3" step="$4" checkpoint="$5" protocol="$6" config="$7" output="$8"
  shift 8
  local fingerprint marker
  fingerprint="$(python3 -m eval_scripts.job_key --checkpoint "${checkpoint}" --protocol "${protocol}" --config "${config}")"
  marker="${JOBS_DIR}/${metric}/${size}/${method}/step${step}/${fingerprint}.done"
  local managed="${output}.managed"
  if [[ -s "${output}" && -f "${marker}" ]]; then
    echo "[skip] ${metric} ${size}/${method} step=${step} key=${fingerprint}"
    return 0
  fi
  # Existing v1 results are imported once, then receive the same content key.
  if [[ -s "${output}" && ! -f "${managed}" ]]; then
    mkdir -p "$(dirname "${marker}")"
    touch "${marker}" "${managed}"
    echo "[reuse] ${metric} ${size}/${method} step=${step} output=${output}"
    return 0
  fi
  mkdir -p "$(dirname "${output}")" "$(dirname "${marker}")"
  echo "[job] ${metric} ${size}/${method} step=${step} key=${fingerprint}"
  "$@"
  [[ -s "${output}" ]] || { echo "missing job output: ${output}" >&2; return 1; }
  touch "${marker}" "${managed}"
}

set_stage preflight
python3 -m eval_scripts.smoke_test
python3 -m eval_scripts.preflight \
  --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" --required-gpus "${NPROC_PER_NODE}"

set_stage manifest
python3 -m eval_scripts.build_manifest --output-dir "${MANIFEST_DIR}"
python3 -m eval_scripts.migrate_legacy_outputs --manifest "${LATEST_MANIFEST}" --raw-dir "${RAW_DIR}"

set_stage latest_routing
while IFS=$'\t' read -r size method checkpoint step; do
  output="${RAW_DIR}/routing/v1/${size}/${method}/step$(printf '%07d' "${step}")"
  config="sequences=${ROUTING_SEQUENCES};length=${ROUTING_LENGTH};capacities=1,2,4,8"
  run_job routing "${size}" "${method}" "${step}" "${checkpoint}" routing_v1 "${config}" \
    "${output}/continuity_by_budget.csv" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --module eval_scripts.routing_eval \
      --checkpoint "${checkpoint}" --size "${size}" --method "${method}" \
      --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" --output-dir "${output}" \
      --num-sequences "${ROUTING_SEQUENCES}" --sequence-length "${ROUTING_LENGTH}" \
      --cache-capacities 1,2,4,8
done < <(python3 -m eval_scripts.list_manifest --manifest "${LATEST_MANIFEST}")

set_stage latest_predictability
while IFS=$'\t' read -r size method checkpoint step; do
  output="${RAW_DIR}/predictability/v1/${size}/${method}/step$(printf '%07d' "${step}")"
  config="train=${PREDICT_TRAIN_SEQUENCES};test=${PREDICT_TEST_SEQUENCES};length=${PREDICT_LENGTH};epochs=${PREDICT_EPOCHS};batch=2"
  run_job predictability "${size}" "${method}" "${step}" "${checkpoint}" predictability_v1 "${config}" \
    "${output}/predictability.csv" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --module eval_scripts.predictability_eval \
      --checkpoint "${checkpoint}" --size "${size}" --method "${method}" \
      --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" --output-dir "${output}" \
      --sequence-length "${PREDICT_LENGTH}" --train-sequences "${PREDICT_TRAIN_SEQUENCES}" \
      --test-sequences "${PREDICT_TEST_SEQUENCES}" --batch-size 2 --epochs "${PREDICT_EPOCHS}"
done < <(python3 -m eval_scripts.list_manifest --manifest "${LATEST_MANIFEST}")

set_stage latest_decode_latency
while IFS=$'\t' read -r size method checkpoint step; do
  for capacity in 1 2 4 8; do
    output="${RAW_DIR}/swap_latency/decode_v1/${size}/${method}/step$(printf '%07d' "${step}")/k${capacity}.csv"
    config="k=${capacity};prompt=${SWAP_PROMPT_LENGTH};decode=${SWAP_DECODE_TOKENS};repeats=${SWAP_REPEATS};device=cuda0"
    run_job decode_latency "${size}" "${method}" "${step}" "${checkpoint}" decode_v1 "${config}" "${output}" \
      python3 -m eval_scripts.swap_latency_eval --checkpoint "${checkpoint}" --size "${size}" \
        --method "${method}" --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" \
        --output "${output}" --cache-capacity "${capacity}" --prompt-length "${SWAP_PROMPT_LENGTH}" \
        --decode-tokens "${SWAP_DECODE_TOKENS}" --repeats "${SWAP_REPEATS}" --device cuda:0
  done
done < <(python3 -m eval_scripts.list_manifest --manifest "${LATEST_MANIFEST}")

set_stage latest_ttft
while IFS=$'\t' read -r size method checkpoint step; do
  for capacity in 1 2 4 8; do
    output="${RAW_DIR}/ttft/v1/${size}/${method}/step$(printf '%07d' "${step}")/k${capacity}.csv"
    config="k=${capacity};prompts=${TTFT_PROMPT_LENGTHS};repeats=${TTFT_REPEATS};cold+warm;device=cuda0"
    run_job ttft "${size}" "${method}" "${step}" "${checkpoint}" ttft_v1 "${config}" "${output}" \
      python3 -m eval_scripts.ttft_eval --checkpoint "${checkpoint}" --size "${size}" \
        --method "${method}" --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" \
        --output "${output}" --cache-capacity "${capacity}" --prompt-lengths "${TTFT_PROMPT_LENGTHS}" \
        --repeats "${TTFT_REPEATS}" --device cuda:0
  done
done < <(python3 -m eval_scripts.list_manifest --manifest "${LATEST_MANIFEST}")

scaling_one() {
  local size="$1" method="$2" checkpoint="$3" step="$4" gpu="$5" label="${size}_${method}_step${step}"
  local loss_output="${RAW_DIR}/scaling/test_loss_v1/${size}/${method}/step$(printf '%07d' "${step}")/result.json"
  local lm_output="${RAW_DIR}/scaling/lm_eval_v1/${size}/${method}/step$(printf '%07d' "${step}")/result.json"
  run_job scaling_test_loss "${size}" "${method}" "${step}" "${checkpoint}" test_loss_v1 \
    "sequences=${TEST_SEQUENCES};length=${TEST_SEQUENCE_LENGTH};batch=1" "${loss_output}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" python3 -m moe.evaluate_checkpoints --run "${label}=${checkpoint}" \
      --data-dir "${DATA_DIR}" --tokenizer-dir "${TOKENIZER_DIR}" --output "${loss_output}" \
      --device cuda:0 --num-sequences "${TEST_SEQUENCES}" --sequence-length "${TEST_SEQUENCE_LENGTH}" --batch-size 1
  run_job scaling_lm_eval "${size}" "${method}" "${step}" "${checkpoint}" lm_eval_v1 \
    "tasks=${LM_EVAL_TASKS};batch=${LM_EVAL_BATCH_SIZE};dtype=bfloat16" "${lm_output}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" python3 -m moe.run_lm_eval --checkpoint "${checkpoint}" \
      --tokenizer-dir "${TOKENIZER_DIR}" --tasks "${LM_EVAL_TASKS}" --device cuda:0 \
      --batch-size "${LM_EVAL_BATCH_SIZE}" --dtype bfloat16 --output "${lm_output}"
}

set_stage scaling_checkpoints
batch_count=0
gpu=0
pids=()
while IFS=$'\t' read -r size method checkpoint step; do
  scaling_one "${size}" "${method}" "${checkpoint}" "${step}" "${gpu}" &
  pids+=("$!")
  batch_count=$((batch_count + 1))
  gpu=$(((gpu + 1) % SCALING_PARALLEL_GPUS))
  if (( batch_count == SCALING_PARALLEL_GPUS )); then
    for pid in "${pids[@]}"; do wait "${pid}"; done
    pids=()
    batch_count=0
    gpu=0
  fi
done < <(python3 -m eval_scripts.list_manifest --manifest "${SCALING_MANIFEST}")
for pid in "${pids[@]}"; do wait "${pid}"; done

set_stage aggregate
python3 -m eval_scripts.aggregate_results --manifest "${LATEST_MANIFEST}" \
  --scaling-manifest "${SCALING_MANIFEST}" --raw-dir "${RAW_DIR}" --compact-dir "${COMPACT_DIR}"

set_stage package
python3 -m eval_scripts.package_results --compact-dir "${COMPACT_DIR}" \
  --output "${EVAL_OUTPUT_ROOT}/final_results.tar.gz" --max-bytes 5000000

trap - ERR
printf 'DONE\t%s\tall_jobs_completed\n' "$(date -Iseconds)" > "${EVAL_OUTPUT_ROOT}/STATUS.tsv"
echo "[done] result=${EVAL_OUTPUT_ROOT}/final_results.tar.gz"
