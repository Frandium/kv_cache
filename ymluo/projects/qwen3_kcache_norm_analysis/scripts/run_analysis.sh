#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kcache_norms}"

python "${PROJECT_DIR}/src/analyze_qwen3_kcache_norms.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-32768}" \
  --chunk_size "${CHUNK_SIZE:-1024}" \
  --max_chars "${MAX_CHARS:-4000000}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --percentiles "${PERCENTILES:-1,5,10,20,30,50,70,80,90,95,99}" \
  --histogram_bins "${HISTOGRAM_BINS:-100}" \
  --histogram_max "${HISTOGRAM_MAX:-0}" \
  --save_norm_tensors "${SAVE_NORM_TENSORS:-false}"
