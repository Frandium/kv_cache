#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../moe" ]]; then
  BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  MOE_MODULE="moe"
  DEFAULT_TOKENIZER_DIR="${BUNDLE_ROOT}/moe/tokenizer_qwen3"
else
  BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
  MOE_MODULE="fdong_embedding_dim.codex_scripts.real_data_moe"
  DEFAULT_TOKENIZER_DIR="${BUNDLE_ROOT}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_qwen3"
fi
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/mnt/workspace/dclm/global-shard_01_of_10}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${DEFAULT_TOKENIZER_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/fmoe_cuda_2b_8e_outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
SEED="${SEED:-42}"
RESUME="${RESUME:-auto}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
RUN_NAMES=(baseline_lb proposed_no_lb proposed_lb)

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --data-pattern "*.txt"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --max-steps "${MAX_STEPS}"
  --save-every 4000
  --log-every 10
  --batch-size "${MICRO_BATCH_SIZE}"
  --sequence-length 1024
  --gradient-accumulation "${GRADIENT_ACCUMULATION}"
  --learning-rate 1e-4
  --min-learning-rate 5e-6
  --warmup-steps 200
  --decay-steps 50000
  --vocab-size 151936
  --hidden-size 1024
  --num-layers 28
  --num-heads 16
  --num-kv-heads 8
  --head-dim 128
  --num-tail-experts 8
  --num-experts-per-token 1
  --router-window 16
  --no-gradient-checkpointing
  --amp-dtype bfloat16
  --seed "${SEED}"
  --resume "${RESUME}"
)

run_distributed() {
  local variant="$1"
  local common_size="$2"
  local tail_size="$3"
  local output_dir="$4"
  local load_balance_weight="$5"
  mkdir -p "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --module \
    "${MOE_MODULE}.train_distributed" \
    --variant "${variant}" \
    --common-intermediate-size "${common_size}" \
    --tail-intermediate-size "${tail_size}" \
    --load-balance-weight "${load_balance_weight}" \
    --output-dir "${output_dir}" \
    "${COMMON_ARGS[@]}"
}

checkpoint_steps() {
  local run="$1"
  find "${OUTPUT_ROOT}/${run}" -maxdepth 1 -type f -name 'checkpoint-*.pt' \
    | sed -E 's/.*checkpoint-([0-9]+)\.pt$/\1/' \
    | sort
}

normalize_step() {
  local step="$1"
  printf '%07d' "$((10#${step}))"
}

latest_common_checkpoint_step() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local run
  for run in "${RUN_NAMES[@]}"; do
    checkpoint_steps "${run}" > "${tmp_dir}/${run}.steps"
    if [[ ! -s "${tmp_dir}/${run}.steps" ]]; then
      echo "[error] no checkpoint found for ${run}" >&2
      rm -rf "${tmp_dir}"
      return 1
    fi
  done
  comm -12 "${tmp_dir}/baseline_lb.steps" "${tmp_dir}/proposed_no_lb.steps" \
    | comm -12 - "${tmp_dir}/proposed_lb.steps" > "${tmp_dir}/common.steps"
  if [[ ! -s "${tmp_dir}/common.steps" ]]; then
    echo "[error] no common checkpoint step across ${RUN_NAMES[*]}" >&2
    rm -rf "${tmp_dir}"
    return 1
  fi
  tail -n 1 "${tmp_dir}/common.steps"
  rm -rf "${tmp_dir}"
}

checkpoint_for_run() {
  local run="$1"
  local step="$2"
  printf '%s/%s/checkpoint-%s.pt' "${OUTPUT_ROOT}" "${run}" "$(normalize_step "${step}")"
}
