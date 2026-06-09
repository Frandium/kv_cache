# Round-2 Protocol: Dot-Product Common-Logit Source, Timing, and Causality

## 0. Scope

This protocol assumes the router score is dot-product:

```text
score_e(h) = w_e^T h
```

Therefore the hidden-state additive decomposition gives an exact additive logit decomposition up to numerical / decomposition residual error.

Study only sparse top-1 MoE:

```text
e_i = argmax_e score_e(h_i)
```

Do not use soft routing or multi-expert weighted forward passes.

## 1. Starting Evidence

Round-1 established:

```text
1. No-position moving-block baseline still collapses.
2. Routed hidden state contains slot information.
3. Logit decomposition suggests common component dominates most route decisions.
4. Slot-initialized positive control reaches slot-level specialization.
```

Round-2 asks:

```text
Is common-logit dominance already present at initialization, grown during early optimization, or amplified by sparse top-1 lock-in?
Can removing common-logit move random-init top-1 training toward slot-specialization?
Why does slot-initialized routing succeed?
```

## 2. Data and Baseline Must Stay Fixed

Use the same Round-1 minimal dataset:

```text
sequence = prefix filler + SLOT_s B_CONST Y_s + suffix filler
SLOT_s B_CONST Y_s is contiguous
B is fixed
position embedding disabled
block_start balanced across valid positions
loss only at B_CONST position
routing metrics only at B_CONST position
```

Do not introduce variable B, variable templates, or fillers between SLOT and B in the main run.

## 3. Shared Decomposition

For each checkpoint `t`, collect routed hidden states at the B position:

```text
m_{s,p,t}
```

where:

```text
s = slot id
p = block start / B position
```

Use:

```text
c_t = mean_{s,p}(m_{s,p,t})
r_{s,t} = mean_p(m_{s,p,t}) - c_t
u_{p,t} = mean_s(m_{s,p,t}) - c_t
residual_{s,p,t} = m_{s,p,t} - c_t - r_{s,t} - u_{p,t}
```

For dot-product router:

```text
score_e(m_{s,p,t}) = w_{e,t}^T c_t + w_{e,t}^T r_{s,t} + w_{e,t}^T u_{p,t} + w_{e,t}^T residual_{s,p,t}
```

For the actual top-1 expert `e*` and runner-up `e2`, decompose the margin:

```text
Delta_score = score_{e*}(m) - score_{e2}(m)
Delta_common = (w_{e*} - w_{e2})^T c_t
Delta_slot = (w_{e*} - w_{e2})^T r_{s,t}
Delta_position = (w_{e*} - w_{e2})^T u_{p,t}
Delta_residual = (w_{e*} - w_{e2})^T residual_{s,p,t}
```

Always report reconstruction error:

```text
abs(Delta_score - (Delta_common + Delta_slot + Delta_position + Delta_residual))
```

## 4. R2-P0: Dot-Product Decomposition Sanity Check

### Question

Is the common-dominance result numerically valid under dot-product score?

### Run

Use existing Round-1 checkpoints if available.

For steps:

```text
0, 1, 2, 5, 10, 20, 50, final
```

compute:

```text
actual top-2 margin
reconstructed top-2 margin
reconstruction error
common / slot / position / residual contribution
component-dominant label per cell
```

### Pass Criteria

```text
reconstruction error is near numerical precision or clearly much smaller than the margin scale
common-dominant classification remains consistent with previous result
```

### Deliverables

```text
r2_p0_decomposition_sanity.csv
r2_p0_reconstruction_error.png
r2_p0_summary.md
```

## 5. R2-P1: Common-Logit Timing Audit

### Question

Is the common-logit advantage already present at step 0, or does it grow during early optimization before lock-in?

### Run

For each seed and checkpoint:

```text
step 0, 1, 2, 5, 10, 20, 50, final
```

record:

```text
final dominant expert
step-t dominant expert
common margin
slot margin
position margin
residual margin
top-1 route by slot
slot_NMI
max_load
accuracy
```

### Primary Metrics

```text
common_predicts_final_expert_at_step_t
common_margin_growth_rate_0_to_10
slot_margin_growth_rate_0_to_10
first_step_common_dominates
global_lock_step
```

### Supports Initial-Common Hypothesis If

```text
step-0 common margin already predicts final dominant expert in most seeds / cells
```

### Supports Early-Growth Hypothesis If

```text
step-0 common is weak, but common margin grows before or by step 10 and then predicts final collapse
```

### Weakens Common-Cause Hypothesis If

```text
common margin appears only after route collapse is already stable
```

### Deliverables

```text
r2_p1_timing_by_seed.csv
r2_p1_common_slot_margin_trajectory.png
r2_p1_common_predicts_final_table.md
r2_p1_summary.md
```

## 6. R2-P2: Slot-Init Basin Audit

### Question

Why does slot-initialized routing succeed? Is there a margin threshold where initialization moves the router from common-collapse basin to slot-specialization basin?

### Run

Construct router initialization interpolation:

```text
W(alpha) = normalize((1 - alpha) W_random + alpha W_slot)
```

Recommended alpha values:

```text
0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
```

For each alpha and seed, train the same model and dataset.

At step 0 and early checkpoints, report:

```text
slot_NMI
max_load
accuracy
common margin
slot margin
slot_margin_minus_common_margin
lock_step
final slot_NMI
final max_load
```

### Supports Basin Hypothesis If

```text
there is a threshold alpha where final routing changes from collapse to stable slot-specialization
and successful runs have initial slot margin greater than common margin
```

### Weakens Basin Hypothesis If

```text
success does not correlate with initial slot/common margin geometry
```

### Deliverables

```text
r2_p2_alpha_sweep_metrics.csv
r2_p2_alpha_vs_final_slot_nmi.png
r2_p2_alpha_vs_margin_gap.png
r2_p2_basin_threshold_summary.md
```

## 7. R2-P3: Common-Logit Cancellation Intervention

### Question

Is common-logit dominance causally responsible for collapse?

### Intervention

Keep sparse top-1 forward. Modify router score only:

```text
score'_e(h) = w_e^T h - w_e^T c_t
```

where `c_t` is the current estimated common component at the B position.

Do not use soft routing.

### Variants

Minimum variants:

```text
baseline random-init dot-product top-1
common-cancel from step 0 to step 10
common-cancel from step 0 to final
```

Optional:

```text
common-cancel after step 10 only
```

### Primary Metrics

```text
slot_NMI
per-slot route purity
max_load
accuracy
route switch time
common margin after cancellation
slot margin after cancellation
```

### Supports Causal Common Hypothesis If

```text
common cancellation increases slot_NMI / purity and reduces collapse while preserving high accuracy
```

### Weakens Causal Common Hypothesis If

```text
common score is removed but routing still collapses with low slot_NMI
```

### Insufficient Evidence If

```text
max_load improves but slot_NMI does not improve
or accuracy drops significantly
```

### Deliverables

```text
r2_p3_common_cancel_metrics.csv
r2_p3_baseline_vs_cancel.png
r2_p3_route_heatmaps.png
r2_p3_summary.md
```

## 8. R2-P4: Common Source Audit

Run only after P1/P3 unless time is abundant.

### Question

After removing position embedding, where does common component come from?

### Minimal Variants

```text
A. original: fixed B_CONST, fixed filler
B. varied filler token identities, fixed B_CONST
C. varied B token identities, target still determined only by slot
D. route at SLOT_s instead of B_CONST, if implementation permits clean comparison
```

### Primary Metrics

```text
||c_t||
common margin
slot margin
slot_NMI
max_load
accuracy
```

### Interpretation

```text
If varying B reduces common: fixed routed B token is a major source.
If varying filler reduces common: shared context/template is a major source.
If neither reduces common: architecture/residual-stream mean or optimizer dynamics may be source.
```

### Deliverables

```text
r2_p4_source_ablation_metrics.csv
r2_p4_source_comparison.png
r2_p4_summary.md
```

## 9. Recommended Execution Order

Run first:

```text
R2-P0
R2-P1
R2-P2
```

Then run:

```text
R2-P3
```

Run last or only if needed:

```text
R2-P4
```

If compute is easy, run all. But report them in this order.

## 10. Final Report Required Questions

The final report must answer:

```text
1. Is dot-product decomposition exact enough to support the common-logit claim?
2. Is common-logit dominance present at step 0 or grown before step 10?
3. Does common-logit predict final dominant expert before lock-in?
4. Does slot-init success show a basin threshold?
5. Does common-logit cancellation increase slot-level specialization, not merely load balance?
6. What is the next decision: common-source audit, anti-lockin design, or representation redesign?
```

## 11. Forbidden Claims

Do not claim:

```text
common source is fully identified unless P4 supports it;
common-logit is causal unless P3 supports it;
load balance equals slot-specialization;
slot-init is a deployable method;
expert computation is causally slot-specialized;
soft routing was tested.
```
