#!/usr/bin/env bash
set -euo pipefail

ROOT="fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment"
SEEDS_D8="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
SEEDS_D16="5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 8 --seeds "$SEEDS_D8" --lrs 0.3 \
  --output-dir "$ROOT/results/heldout_raw_d8"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$ROOT/run_experiment.py" \
  --dims 16 --seeds "$SEEDS_D16" --lrs 0.3,1.0 \
  --output-dir "$ROOT/results/heldout_raw_d16"

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib-orthogonal-heldout \
python3 "$ROOT/analyze_heldout.py" \
  --tuning-summary "$ROOT/results/summary.csv" \
  --heldout-d8 "$ROOT/results/heldout_raw_d8/summary.csv" \
  --heldout-d16 "$ROOT/results/heldout_raw_d16/summary.csv" \
  --output-dir "$ROOT/results"
