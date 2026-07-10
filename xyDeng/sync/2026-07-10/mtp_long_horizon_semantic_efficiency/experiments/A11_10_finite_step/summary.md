# A11_10 Finite-Step Semantic Efficiency Summary

Status: completed full run.  
Run: `a11_10_finite_step_semantic_efficiency_4gpu_20260709_full`.  
Job id: `pt-h43wss54`.  
Source protocol: `protocol.md`.

## Conclusion First

A11_10 supports the sign-level finite-step story, but it does not support a tight finite-step efficiency theorem.

What is supported:

- In direct informative-horizon conditions, the early reference semantic velocity remains positive in `5/5` seeds.
- The no-information guard passes: `shared_only_k4` has zero semantic velocity, zero semantic margin growth, and no recovery.
- In all-position training, adding the direct decision-prefix `H3` term makes native `h_T -> S_Z` prediction recover to `1.0` in `5/5` seeds; masking/removing that direct term can still recover guarded `Q`, but native H3 stays `0.0`.

What is not yet supported:

- A tight, useful hitting-time upper bound. The certified lower-bound increment is often nonpositive after curvature correction, and when positive the bound is very loose.
- A universal finite-step theorem across aligned, low/conflict, decision-only, and all-position settings.

## Terminology / Definitions

| Term | Plain meaning | Formula / computation | Why it matters | Cannot prove |
|---|---|---|---|---|
| reference semantic margin | hidden state alignment with fixed semantic output directions | $M_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}h_z(t)$ | theorem target variable | recovery without calibration |
| hidden semantic velocity | gradient direction projected onto semantic direction | $G_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_K)$ | first-order source of margin growth | finite-step recovery alone |
| direct active term | current decision-prefix informative loss included in the training objective | $L_{direct}=\lambda_3\operatorname{CE}(q_3(\cdot\mid h_T),S_Z)$ with all-position scaling when it is part of an all-position average | separates direct supervision from background losses | absence of indirect transfer |
| non-direct term | all training loss not counted as the direct active term | $L_{non-direct}=L_{active}-L_{direct}$ | estimates shared / indirect perturbation | causal route by itself |
| guarded recovery score | conservative hidden-state recovery score | $Q=\min(A_{decoder},A_{probe},C_{swap})$ | protects against probe-only artifacts | direct supervision source |
| native H3 accuracy | model's own third-horizon prediction from `h_T` | $\Pr[\arg\max q_3(\cdot\mid h_T)=S_Z]$ | tests usable direct readout | hidden-state decodability |
| certified increment | empirical finite-step lower-bound term | $d_{cert}=\eta g_{min}-\eta^2\widehat B-\widehat\epsilon$ | tests whether one-step velocity can certify finite-step margin growth | tightness of the bound |

## Setup

Seeds: `971, 972, 973, 974, 975`.  
Training length: `2000` steps, evaluated every `100` steps.  
Primary threshold: `Q >= 0.9`; margin threshold: `gamma = 1.0`.

The run uses three minimal groups:

| Group | Conditions | Purpose |
|---|---|---|
| decision-only theorem window | `shared_only_k4`, `single_h3`, `aligned_h3_h4`, `low_conflict_h3_h4` with frozen heads | test velocity persistence and vector-sum geometry under clean control |
| head drift audit | `single_h3`, `low_conflict_h3_h4` with trainable heads | check whether reference geometry remains usable |
| all-position perturbation | `K3_active`, `K3_mask_decision_prefix_L3`, `K2_plus_decision_prefix_L3`, `K2_active` on `single_h3` | separate direct H3 supervision from indirect transfer |

## Main Results

### 1. Assumption Audit

| Condition | Scope / heads | `G_total_min` | `G_direct_min` | `G_non_direct_min` | velocity pass | calibration pass | bound pass | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `shared_only_k4` + `K4_active` | decision / frozen | `0.000000` | `0.000000` | `0.000000` | `1.0` | `0.0` | `0.0` | guard passes |
| `single_h3` + `K3_active` | decision / frozen | `0.000954` | `0.000954` | `~0` | `1.0` | `0.4` | `0.2` | partial |
| `single_h3` + `K3_active` | decision / trainable | `0.000947` | `0.000947` | `~0` | `1.0` | `0.4` | `0.4` | partial |
| `aligned_h3_h4` + `K4_active` | decision / frozen | `0.003816` | `0.001908` | `0.001908` | `1.0` | `0.2` | `0.4` | partial |
| `low_conflict_h3_h4` + `K4_active` | decision / frozen | `0.000153` | `0.000404` | `-0.000252` | `1.0` | `0.0` | `0.0` | partial / weak finite-step |
| `single_h3` + `K3_active` | all-position / trainable | `0.000036` | `0.000036` | `~0` | `1.0` | `0.8` | `0.8` | partial-to-support |
| `single_h3` + `K2_plus_direct_H3` | all-position / trainable | `0.000951` | `0.000951` | `~0` | `1.0` | `1.0` | `0.6` | support in `3/5` |
| `single_h3` + `K3_mask_direct_H3` | all-position / trainable | `~0` | `0.000000` | `~0` | `0.0` | `0.0` | `0.0` | direct mechanism absent |
| `single_h3` + `K2_active` | all-position / trainable | `0.000000` | `0.000000` | `0.000000` | `0.0` | `0.0` | `0.0` | direct mechanism absent |

Take-home: positive velocity persists where direct informative supervision exists, and it is zero in guards. The hitting-time bound is not tight enough to become a strong efficiency theorem.

### 2. Recovery and Local Learning

| Condition | Scope / heads | reach rate `Q>=0.9` | `AUC_Q` | early `AUC_Q` | final `Q` | final native H3 | final `M_Z_ref` |
|---|---|---:|---:|---:|---:|---:|---:|
| `shared_only_k4` + `K4_active` | decision / frozen | `0.0` | `0.250` | `0.250` | `0.250` | `nan` | `0.000` |
| `single_h3` + `K3_active` | decision / frozen | `0.4` | `0.513` | `0.475` | `0.550` | `0.550` | `2.771` |
| `aligned_h3_h4` + `K4_active` | decision / frozen | `0.2` | `0.489` | `0.478` | `0.500` | `0.500` | `5.189` |
| `low_conflict_h3_h4` + `K4_active` | decision / frozen | `0.2` | `0.464` | `0.428` | `0.500` | `0.500` | `0.180` |
| `K2_active` | all-position / trainable | `0.6` | `0.835` | `0.840` | `0.850` | `0.000` | `0.000` |
| `K3_mask_direct_H3` | all-position / trainable | `0.6` | `0.794` | `0.790` | `0.800` | `0.000` | `0.002` |
| `K2_plus_direct_H3` | all-position / trainable | `1.0` | `0.929` | `0.858` | `1.000` | `1.000` | `3.574` |
| `K3_active` | all-position / trainable | `1.0` | `0.974` | `0.948` | `1.000` | `1.000` | `2.065` |

Local next-token learning is not sacrificed: all conditions reach final `Y1_accuracy=1.0` and `Y2_accuracy=1.0`.

## Hypothesis Status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1 velocity persistence | supported for direct informative conditions | pass rate `1.0`; direct conditions keep positive early-window `G_total_ref` |
| H2 margin-to-recovery calibration | partial | all-position direct conditions pass more often; decision-only and low/conflict are seed-dependent |
| H3 head drift bound | supported as a bounded-audit condition | frozen drift is `0`; trainable alignment remains positive, but drift is nonzero |
| H4 perturbation bound | partially supported | direct all-position conditions keep positive net velocity; indirect-only controls recover `Q` without native H3 |
| finite-step hitting-time theorem | not fully closed | certified bounds are too loose or nonpositive in many seeds |

## Claim Boundary

Supported claim:

```text
In controlled A11 data, when the objective contains the direct informative horizon, it provides positive early semantic velocity for h_T. This velocity persists in the audited early window and explains why direct all-position conditions recover native H3 and guarded Q more reliably than masked or non-covering controls.
```

Not supported:

```text
A11_10 proves a tight finite-step learning-efficiency theorem.
MTP is generally more efficient in natural language.
Larger K is always better.
Q alone proves direct semantic supervision.
```

## Artifacts

- Full curves: `tables/curve_metrics.csv`
- Assumption audit: `tables/finite_step_assumption_summary.csv`
- Seed-level hitting-time bounds: `tables/seed_level_hitting_time_bound.csv`
- Recovery summary: `tables/condition_summary.csv`
- Local-loss guard: `tables/local_loss_guard_summary.csv`
- One-step geometry: `tables/one_step_summary.csv`
