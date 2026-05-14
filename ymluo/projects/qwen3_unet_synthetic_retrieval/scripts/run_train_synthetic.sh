#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/train_unet_synthetic_retrieval.py" \
  --config_dir "${CONFIG_DIR:-/mnt/workspace/Qwen3-0.6B}" \
  --output_dir "${OUT_DIR:-${PROJECT_DIR}/outputs/train}" \
  --run_name "${RUN_NAME:-unet-4-variant-b-answer-only}" \
  --model_name "${MODEL_NAME:-unet-4}" \
  --variant "${VARIANT:-B}" \
  --total_steps "${TOTAL_STEPS:-10000}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-1e-4}" \
  --warmup_steps "${WARMUP_STEPS:-200}" \
  --train_mode "${TRAIN_MODE:-anchor_kv_decode}" \
  --save_interval "${SAVE_INTERVAL:-1000}" \
  --eval_interval "${EVAL_INTERVAL:-100}" \
  --eval_batches "${EVAL_BATCHES:-8}" \
  --seed "${SEED:-1234}" \
  --device "${DEVICE:-cuda}" \
  --use_bf16 "${USE_BF16:-true}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "$@"
