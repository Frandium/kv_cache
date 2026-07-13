#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-fdong_embedding_dim/outputs/mtp_frozen_probe_v1}"
DEVICE="${DEVICE:-cpu}"

python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.run \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --backbones linear,mlp,attention \
  --hidden-sizes 2,4,8 \
  --seeds 971,972,973 \
  --num-prefixes 8 \
  --num-bones 8 \
  --holdout-stride 4 \
  --train-steps 3000 \
  --probe-steps 1500 \
  --learning-rate 0.03 \
  --probe-learning-rate 0.03 \
  --probe-kinds linear,mlp

python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.plot_results \
  --input "${OUTPUT_DIR}/summary.csv" \
  --output "${OUTPUT_DIR}/summary.png"

