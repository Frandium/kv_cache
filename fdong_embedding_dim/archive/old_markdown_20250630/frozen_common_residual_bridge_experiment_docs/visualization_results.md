# Visualization Results

No research conclusion is recorded until a run has produced:

- `results/history.csv`;
- `results/summary.csv`;
- `results/switch_diagnostics.png`;
- `results/learning_curves.png`;
- `results/geometry_curves.png`.

## Reading guide

`switch_diagnostics.png` must show the pretraining top-direction angle change, top singular-value relative change, K accuracy, and non-K accuracy. It establishes whether the automatic phase switch happened before or after the task was already solved.

`learning_curves.png` must compare post-switch loss and non-K accuracy from the identical checkpoint. The frozen no-bridge curve is the no-capacity lower control; the unconstrained bridge measures what a parameter-matched adapter can learn; the common-to-residual bridge tests the hypothesis.

`geometry_curves.png` must show adapter output common-energy fraction and frozen-base drift. A functional gain without these checks does not support the constrained-learning claim.

## Current run

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fdong_embedding_dim/frozen_common_residual_bridge_experiment/run_experiment.py
```

Configuration:

- dimensions: 2 and 3;
- seeds: 0, 1, 2;
- population loss over all 16 examples;
- base learning rate: 0.03;
- bridge learning rate: 0.05;
- maximum pretraining: 300 steps;
- post-switch continuation: 1500 steps;
- switch check every 3 steps;
- stable switch: K accuracy 1 and either maximum left/right direction change at most 0.5 degrees or relative top-singular-value change at most 0.002, for three consecutive checks.

## Observed results

### 1. Dimension 2 is a capacity-failure control

All three dimension-2 runs reached the 300-step pretraining limit without satisfying the switch rule. Their switch full accuracies were 0.25, 0.25, and 0.375. Even `baseline_continue` did not solve the task after another 1500 steps.

Therefore dimension 2 does not test the residual-bridge hypothesis cleanly under this tied one-layer architecture.

### 2. Dimension 3 supports the base task

The three switch steps were 108, 141, and 96. Switch full accuracies were 0.9375, 0.625, and 1.0. After continued end-to-end training, all three baseline runs reached full accuracy. One unfavorable seed required 1085 post-switch steps, which is why the default continuation is 1500 steps.

### 3. The geometric constraint is implemented correctly

For dimension 3:

- mean final adapter output common-energy fraction:
  - `common_to_residual`: \(5.84\times10^{-16}\);
  - `residual_to_residual`: \(8.56\times10^{-16}\);
- frozen-base relative drift: exactly 0 for all adapter variants;
- mean gradient block allocation over training:
  - `common_to_residual`: 0.9979 in the common-input to residual-output block;
  - `residual_to_residual`: 0.9991 in the residual-input to residual-output block.

This passes the implementation test: the constrained branch reads the selected input block and cannot write into the frozen common output direction.

### 4. Functional evidence is weak and negative for the strong hypothesis

Dimension-3 mean non-K loss:

- frozen no bridge: 0.3891 -> 0.3891;
- unconstrained bridge: 0.3891 -> 0.3197;
- common-to-residual: 0.3891 -> 0.3530;
- residual-to-residual: 0.3891 -> 0.3358;
- baseline continuation: 0.3891 -> 0.0084.

`common_to_residual` reduced remaining non-K loss while preserving exact output orthogonality, so a useful constrained update exists in this toy. However:

- it did not solve any unsolved run completely;
- it was weaker than the unconstrained bridge;
- the residual-input control was stronger than the common-input bridge;
- continued end-to-end training was much stronger than every frozen adapter.

Therefore the current run does **not** support the strong claim that the learned top input singular direction is the privileged or most efficient source for residual learning.

### 5. The selected common direction is only moderately K-related

At the dimension-3 switches, the mean alignment between the top output singular direction of \(B_{VO}\) and the K embedding was 0.598. This varies enough across seeds that “top \(B_{VO}\) direction equals the K/common feature” remains an uncertain operational assumption.

## Claim boundary and next uncertainty

Supported:

- the two-stage freeze and switch logic works;
- \(P_R^{\rm out}AP_C^{\rm in}\) produces the intended gradient block and exact output orthogonality;
- this branch can reduce some remaining loss without changing the base.

Not supported:

- dimension 2 is sufficient for this tied-attention task;
- common-input to residual-output is more efficient than residual-input to residual-output;
- the top singular direction of \(B_{VO}\) is a reliable definition of the K/common representation across seeds;
- freezing the base and adding this rank-one-input bridge can match continued training.

The next decisive experiment should compare three fixed definitions at the same switch checkpoint:

1. top singular directions of \(B_{VO}\);
2. top second-moment direction of the attention-weighted input embeddings;
3. the normalized K embedding direction.

That comparison will determine whether the weak result comes from the common-to-residual idea itself or from choosing the wrong common projector.
