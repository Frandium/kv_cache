# Experiment Design

## Primary comparison

Run dimensions 2 and 3 with matched initialization and the same detected switch checkpoint. Compare:

- `baseline_continue`;
- `frozen_no_bridge`;
- `unconstrained_bridge`;
- `common_to_residual`;
- `residual_to_residual`.

Use exact population loss over all 16 examples in the first experiment. Minibatching and Muon are intentionally excluded until the constrained mapping is shown to work at all.

The default continuation length is 1500 steps. A long-run capacity check showed that dimension 3 can solve all 16 examples but may need more than 1000 post-switch steps for an unfavorable seed; dimension 2 remained unsolved and is retained as a capacity-failure control.

## Algorithm

### Stage 1: learn and detect the common direction

Input:

- the 16 equal-weight cyclic trigram examples;
- tied-embedding one-layer attention model;
- one random seed and hidden dimension.

Procedure:

1. Train the base model with Adam.
2. Every `check_interval=3` steps, compute \(\sigma_1\) and the top left/right singular vectors of \(B_{VO}=W_OW_V\).
3. Compute sign-invariant left/right angle changes, record their maximum, and compute relative \(\sigma_1\) change.
4. Require `to_K_accuracy=1` and either angle at most `0.5` degrees or relative singular-value change at most `0.002`, for `stable_checks=3` consecutive checks after `min_pretrain_steps=15`. The OR rule follows the requested switch logic and avoids waiting for CE-driven margin growth to stop. `angle_and_sigma` is retained as a stricter control. The 0.5-degree default is loose enough to switch before every tail example is solved in the initial smoke trace; the exact angle is logged and must be swept later.
5. Save the switch checkpoint and detection evidence.
6. If no switch occurs by `max_pretrain_steps`, mark the switch as forced.

Debug artifacts:

- every checked angle and singular-value change in `history.csv`;
- switch reason and switch metrics in `summary.csv`;
- K/common alignment at the switch.

### Stage 2: matched continuation

1. Clone the exact switch checkpoint into every variant.
2. Freeze the base for all bridge variants.
3. Construct fixed \(P_C\) and \(P_R\) from the switch direction.
4. Train only adapter \(A\) for bridge variants.
5. Continue the original parameters only for `baseline_continue`.
6. Evaluate all 16 examples at every step.

## Metrics

Functional:

- population cross-entropy;
- full-example accuracy;
- `to_K_accuracy`;
- non-K accuracy;
- per-group accuracy;
- first post-switch step at which all examples remain correct for `stable_window` consecutive evaluations.

Geometry:

- top singular values and effective rank of both tied embedding and \(B_{VO}\);
- angle between current and frozen common direction;
- frozen-base parameter drift;
- adapter norm;
- adapter output common-energy fraction;
- adapter input common-energy fraction;
- K embedding alignment with the frozen common direction.

## Pass, fail, and insufficient-evidence conditions

Support for the existence claim requires all of:

1. `common_to_residual` improves non-K loss or accuracy over `frozen_no_bridge`.
2. Its adapter output common-energy fraction is below `1e-8` up to numerical precision.
3. Frozen-base parameter drift is below `1e-10`.
4. Adapter ablation restores the frozen-base predictions.

Stronger support occurs if `common_to_residual` approaches `unconstrained_bridge` while preserving exact output orthogonality.

The tested implementation fails if `unconstrained_bridge` learns but `common_to_residual` does not. This falsifies this rank-one output-residual parameterization at that dimension/checkpoint, not every possible common-to-residual mapping.

Evidence is insufficient if:

- the convergence detector is forced at `max_pretrain_steps`;
- the switch occurs only after every example is already stably correct;
- the frozen common direction has weak alignment with K and no other evidence that it represents the common feature;
- neither constrained nor unconstrained adapters improve over the frozen control.

## Expected failure modes

- In dimension 2, the residual output space is one-dimensional and may be too small for all remaining distinctions.
- The base may fully solve the toy before the common direction satisfies the stability rule.
- The top embedding singular direction may describe global token geometry rather than the K/common computation.
- Frozen output embeddings may not expose enough residual components for the adapter to change the required logits.

These failures must be reported as limitations of the operationalization rather than as proof that orthogonal learning is impossible.
