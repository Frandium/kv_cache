#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-8B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kcache_value_delta}"

python "${PROJECT_DIR}/src/analyze_qwen3_kcache_value_delta.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-5000}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --percentiles "${PERCENTILES:-0.1,1,5,10,25,50,75,90,95,99,99.9}" \
  --histogram_bins "${HISTOGRAM_BINS:-200}" \
  --histogram_clip_percentile "${HISTOGRAM_CLIP_PERCENTILE:-99.9}" \
  --global_sample_size "${GLOBAL_SAMPLE_SIZE:-1000000}" \
  --sample_seed "${SAMPLE_SEED:-1234}" \
  --save_head_histograms "${SAVE_HEAD_HISTOGRAMS:-true}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_metrics "${PLOT_METRICS:-k_value,abs_k_value,delta_value,abs_delta_value,k_l2_norm,delta_l2_norm,relative_delta_l2,cosine_prev}" \
  --save_k_tensors "${SAVE_K_TENSORS:-false}" \
  --save_delta_tensors "${SAVE_DELTA_TENSORS:-false}"
