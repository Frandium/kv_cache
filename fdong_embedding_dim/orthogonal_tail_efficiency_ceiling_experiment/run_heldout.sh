#!/usr/bin/env bash
set -euo pipefail

ROOT="fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment"
SEEDS_D8="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"
SEEDS_D16="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 8 --seeds "$SEEDS_D8" --lrs 0.3 \
  --output-dir "$ROOT/results/three_way_raw_d8"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 16 --seeds "$SEEDS_D16" --lrs 0.3 --variants common_oracle \
  --output-dir "$ROOT/results/three_way_raw_d16_common"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 16 --seeds "$SEEDS_D16" --lrs 1.0 --variants residual_oracle \
  --output-dir "$ROOT/results/three_way_raw_d16_tail"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 16 --seeds "$SEEDS_D16" --lrs 0.1 --variants full_output_oracle \
  --output-dir "$ROOT/results/three_way_raw_d16_full"

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib-orthogonal-heldout \
python3 "$ROOT/analyze_three_way.py" \
  --tuning-summary "$ROOT/results/summary.csv" \
  --heldout-d8 "$ROOT/results/three_way_raw_d8/summary.csv" \
  --heldout-d16 \
    "$ROOT/results/three_way_raw_d16_common/summary.csv" \
    "$ROOT/results/three_way_raw_d16_tail/summary.csv" \
    "$ROOT/results/three_way_raw_d16_full/summary.csv" \
  --output-dir "$ROOT/results"
