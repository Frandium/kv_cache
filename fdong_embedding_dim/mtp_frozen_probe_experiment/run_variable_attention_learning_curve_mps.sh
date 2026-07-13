#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-fdong_embedding_dim/outputs/mtp_variable_attention_learning_curve_v1}"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-971,972,973}"
HIDDEN_SIZES="${HIDDEN_SIZES:-4}"

python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.learning_curve \
  --output-dir "${OUTPUT_DIR}" \
  --dataset variable_lookup \
  --backbones attention \
  --hidden-sizes "${HIDDEN_SIZES}" \
  --seeds "${SEEDS}" \
  --mtps 1,3 \
  --num-prefixes 8 \
  --num-bones 8 \
  --min-bone-length 1 \
  --max-bone-length 4 \
  --holdout-stride 4 \
  --train-steps 3000 \
  --learning-rate 3e-2 \
  --log-every 20 \
  --device "${DEVICE}"

MPLBACKEND="${MPLBACKEND:-Agg}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_mtp_variable_attention_learning_curve}" \
python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.plot_learning_curve \
  --input "${OUTPUT_DIR}/learning_curve.csv" \
  --output "${OUTPUT_DIR}/learning_curve.png"
