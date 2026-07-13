#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-fdong_embedding_dim/outputs/mtp_compositional_mlp_v2}"
DEVICE="${DEVICE:-cpu}"

python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.run \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --dataset compositional \
  --backbones mlp \
  --hidden-sizes 8 \
  --seeds 971,972,973 \
  --num-prefixes 8 \
  --num-bones 8 \
  --test-fraction 0.25 \
  --split-seed 20260705 \
  --train-steps 5000 \
  --probe-steps 2500 \
  --learning-rate 0.01 \
  --probe-learning-rate 0.01 \
  --probe-kinds linear,mlp

python3 -m fdong_embedding_dim.mtp_frozen_probe_experiment.plot_results \
  --input "${OUTPUT_DIR}/summary.csv" \
  --output "${OUTPUT_DIR}/summary.png"
