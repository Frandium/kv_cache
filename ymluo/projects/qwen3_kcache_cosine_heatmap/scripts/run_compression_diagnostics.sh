#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kv_compression_diagnostics}"

python "${PROJECT_DIR}/src/analyze_qwen3_kv_compression_diagnostics.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-5000}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_max_tokens "${REQUIRE_MAX_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --similarity_device "${SIMILARITY_DEVICE:-auto}" \
  --similarity_dtype "${SIMILARITY_DTYPE:-float32}" \
  --summary_percentiles "${SUMMARY_PERCENTILES:-1,5,25,50,75,95,99}" \
  --summary_sample_size "${SUMMARY_SAMPLE_SIZE:-1000000}" \
  --sample_seed "${SAMPLE_SEED:-1234}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_max_tokens "${PLOT_MAX_TOKENS:-5000}" \
  --figure_size "${FIGURE_SIZE:-7.5}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --cmap "${CMAP:-coolwarm}" \
  --vmin "${VMIN:--1.0}" \
  --vmax "${VMAX:-1.0}" \
  --write_token_csv "${WRITE_TOKEN_CSV:-true}" \
  --analyze_v_cache "${ANALYZE_V_CACHE:-true}" \
  --svd_energy_thresholds "${SVD_ENERGY_THRESHOLDS:-50,75,90,95,99}" \
  --svd_plot_top_n "${SVD_PLOT_TOP_N:-128}" \
  --attention_validation "${ATTENTION_VALIDATION:-true}" \
  --validation_query_count "${VALIDATION_QUERY_COUNT:-128}" \
  --validation_ranks "${VALIDATION_RANKS:-1,2,4,8,16,32,64,128}" \
  --validation_variants "${VALIDATION_VARIANTS:-raw,centered}" \
  --validation_cache_types "${VALIDATION_CACHE_TYPES:-k_only,kv}" \
  --strict_query_capture "${STRICT_QUERY_CAPTURE:-false}"
