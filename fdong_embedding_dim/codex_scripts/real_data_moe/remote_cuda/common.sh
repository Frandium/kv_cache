#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/mnt/workspace/dclm/global-shard_01_of_10}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${BUNDLE_ROOT}/fdong_embedding_dim/codex_scripts/real_data_moe/tokenizer_qwen3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/fmoe_cuda_2b_outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --data-pattern "*.txt"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --max-steps "${MAX_STEPS}"
  --save-every 4000
  --log-every 10
  --batch-size 4
  --sequence-length 1024
  --gradient-accumulation 8
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
  --num-tail-experts 4
  --router-window 16
  --no-gradient-checkpointing
  --amp-dtype bfloat16
  --seed "${SEED}"
  --resume auto
)

run_distributed() {
  local variant="$1"
  local common_size="$2"
  local tail_size="$3"
  local output_dir="$4"
  mkdir -p "${output_dir}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --module \
    fdong_embedding_dim.codex_scripts.real_data_moe.train_distributed \
    --variant "${variant}" \
    --common-intermediate-size "${common_size}" \
    --tail-intermediate-size "${tail_size}" \
    --output-dir "${output_dir}" \
    "${COMMON_ARGS[@]}"
}
