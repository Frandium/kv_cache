# Detailed: A11_25 Curriculum Rank Boundary

## 0. Quick Recap

Purpose: test curriculum as a structural-rank repair. Hypothesis: a dose-matched schedule Pareto-improves static MTP. Conclusion: weakened in 5/5 seeds.

## 1. Setup

Exact A11_24 model. Static auxiliary weight is 2 for 1000 steps. Curriculum uses 0 for 250 steps and 8/3 for 750 steps, matching cumulative auxiliary dose. Primary criterion requires standard-loss gain above `0.1` with auxiliary-MSE cost at most `0.02`.

## 2. Results

No seed passes. Standard gains: `-0.002132, -0.000415, -0.000004, -0.001147, +0.000134`. Rank 2 makes both schedules equivalent at numerical zero.

## 3. Interpretation And Boundary

The rank-1 frontier is structural: scheduling changes optimization order but not representational dimension. Curriculum remains plausible only where capacity is sufficient and the blocker is optimization/path dependence.

## 4. Artifact Map

Runner: `active/synthetic_data_understanding/scripts/run_a11_25_curriculum_rank_boundary.py`; raw result `active/synthetic_data_understanding/results/a11_25_curriculum/`; table `tables/results.csv`; figure `figures/pareto.png`; local run.
