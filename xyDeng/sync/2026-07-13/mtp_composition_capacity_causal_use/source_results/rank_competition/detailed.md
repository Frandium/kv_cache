# Detailed: A11_24 Task-Rank Competition

## 0. Quick Recap

Purpose: causally isolate representation rank. Hypothesis: rank 1 causes future-vs-standard competition; rank 2 removes it. Conclusion: supported in 5/5 seeds.

## 1. Setup

$u,z\sim N(0,1)$; shared linear encoder rank 1 or 2; separate local, future, and auxiliary readouts; $L_N=2L_u+L_z$; $L_{MTP}=L_N+2L_A$; AdamW, 1000 steps, batch 256, seeds `971--975`.

## 2. Results

Rank-1 standard-loss gap mean `0.9630`, minimum `0.9084`; future MSE improves in 5/5. Rank-2 mean gap `4.90e-15`, maximum absolute gap `1.23e-13`.

## 3. Interpretation And Boundary

The intervention changes rank directly, supporting structural competition under the model assumptions. Target weights determine which factor is displaced. The result cannot identify task-related capacity in a Transformer.

## 4. Artifact Map

Runner: `active/synthetic_data_understanding/scripts/run_a11_24_task_rank_competition.py`; raw result: `active/synthetic_data_understanding/results/a11_24_rank/`; table `tables/results.csv`; figure `figures/rank_phase.png`; local run.
