# A11_10 All-Position Indirect-Transfer Dynamics Detailed Report

## 0. Quick Recap

Purpose: separate all-position indirect transfer from direct native `h_T -> S_Z` supervision.

Hypothesis: direct decision-prefix horizon-3 loss is the path that creates native H3 prediction and readout-effective margin; non-direct all-position losses can still create probe-readable recovery.

Experiment idea: run four clean `tau=3` all-position conditions over five seeds, with direct-add and direct-mask controls.

Conclusion: supported. Direct conditions have native H3 = 1.0 and large `M_Z_ref`; non-direct conditions can have nontrivial `Q` but native H3 = 0 and margin near zero.

Evidence: final `M_Z_ref` is 0.000 / 0.004 for K2 and masked K3, versus 2.430 / 3.766 for K3 full and K2 plus direct H3. Final native H3 is 0 / 0 versus 1 / 1.

## 1. Protocol Compliance

| Requirement | Status | Evidence |
|---|---|---|
| four mechanism-critical conditions | pass | K2, K3, K3 masked direct, K2 plus direct |
| five seeds | pass | 971-975 |
| all-position clean `tau=3` | pass | `objective_scope=all_positions_lm_control`, `data_regime=clean_tau3` |
| no-leakage initialization | pass | step0 `Q=0.25`, native H3 = 0, `M_Z_ref` near 0 |
| one-step hidden-gradient audit | pass | `tables/one_step_summary.csv` |
| gradient profile enabled | pass | `tables/gradient_usefulness_summary.csv` |
| local guard | pass | final Y1/Y2 accuracy = 1.0 in all conditions |

## 2. Data And Conditions

Synthetic sequence:

```text
BR_z F00 ... F15 U00 U01 U02 U03 A_SHARED C_SHARED S_z STOP
```

The decision prefix ends at `U03`. After that, the first two targets are shared and the third target `S_z` is the first informative future token.

| Condition | Active loss | Role |
|---|---|---|
| `K2_active` | all-position horizons 1 and 2 | non-covering baseline |
| `K3_active` | all-position horizons 1, 2, and 3 | full direct + indirect |
| `K3_mask_decision_prefix_L3` | K3 but remove direct decision-prefix H3 | indirect-only K3 control |
| `K2_plus_decision_prefix_L3` | K2 plus direct decision-prefix H3 | direct-term sufficiency control |

## 3. Metric Definitions

Guarded recovery:

$$
Q=\min\{A_{decoder},A_{probe},C_{swap}\}.
$$

Readout-effective reference margin:

$$
M_Z^{ref}=\frac1m\sum_z(u^0_{S_z}-\bar u^0)^\top h_z.
$$

Native horizon-3 accuracy:

$$
\Pr[\arg\max q_3(\cdot\mid h_T)=S_Z].
$$

One-step hidden semantic velocity:

$$
G_{hidden}^{ref}=\frac1m\sum_z(u^0_{S_z}-\bar u^0)^\top(-\nabla_{h_z}L).
$$

The primary judgment uses native H3 plus `M_Z_ref`. `Q` is secondary because it can be produced by probe-readable indirect transfer.

## 4. Main Result Table

| Condition | Reach rate | Mean reached `T_0.9` | Early AUC Q | Final Q | Native H3 | Early AUC `M_Z_ref` | Final `M_Z_ref` | Final `E_Z` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `K2_active` | 0.60 | 333 | 0.773 | 0.75 | 0.00 | 0.000 | 0.000 | 0.005 |
| `K3_mask_decision_prefix_L3` | 0.60 | 583 | 0.773 | 0.70 | 0.00 | 0.005 | 0.004 | 0.010 |
| `K3_active` | 1.00 | 300 | 0.965 | 1.00 | 1.00 | 1.887 | 2.430 | 50.915 |
| `K2_plus_decision_prefix_L3` | 1.00 | 300 | 0.965 | 1.00 | 1.00 | 3.386 | 3.766 | 83.262 |

Interpretation: direct conditions dominate on the direct-readout metrics. Non-direct conditions can recover `Q`, but they do not create native H3 or readout-effective margin.

## 5. One-Step Audit

At step 0, `K2_active` and masked K3 have near-zero one-step hidden semantic velocity:

| Condition | `G_M_hidden_mean` | Meaning |
|---|---:|---|
| `K2_active` | 0.000000 | no informative direct term |
| `K3_mask_decision_prefix_L3` | -0.000000 | all-position H3 without decision-prefix direct term does not move `h_T` along `S_Z` margin |
| `K3_active` | 0.000173 | full all-position H3 gives a small direct contribution because the direct position is averaged with all valid positions |
| `K2_plus_decision_prefix_L3` | 0.003816 | explicit direct decision-prefix H3 gives the strong direct velocity |

Interpretation: the one-step audit supports the same separation as the curve result. The full K3 all-position direct signal is diluted by averaging over all positions; the explicit direct-add condition shows the clean direct velocity.

## 6. Visualization Results

![Direct native margin split](figures/direct_native_margin_split.png)

Purpose: test whether guarded recovery and direct native readout are identical signals.

How to read: blue bars are final `Q`; orange bars are native H3 accuracy; green and purple lines are final and early `M_Z_ref`.

Observed result: non-direct conditions have nontrivial final `Q` but zero native H3 and near-zero margin. Direct conditions have final `Q=1.0`, native H3 = 1.0, and large margin.

Take-home: all-position training can make `Z` readable without training native decision-prefix H3. Direct supervision must be evaluated through native H3 and margin, not `Q` alone.

Limitation: the plot does not prove exact cross-position Jacobian kernels or natural-language benefit.

## 7. Seed Support

Direct conditions:

```text
K3_active: native H3 = 1.0 in 5/5 seeds; final Q = 1.0 in 5/5 seeds.
K2_plus_decision_prefix_L3: native H3 = 1.0 in 5/5 seeds; final Q = 1.0 in 5/5 seeds.
```

Non-direct conditions:

```text
K2_active: native H3 = 0.0 in 5/5 seeds; final Q is nontrivial in 4/5 seeds.
K3_mask_decision_prefix_L3: native H3 = 0.0 in 5/5 seeds; final Q is nontrivial in 4/5 seeds.
```

This is the key split: probe-readable recovery can occur without native H3.

## 8. Gradient Profile Boundary

The gradient profile tables show that some non-direct horizon gradients correlate with later `Q`, especially in K2 and masked-K3 controls. This supports the presence of all-position indirect transfer. However, these profiles do not prove exact cross-position kernels, and they do not override the primary direct-readout result: non-direct conditions remain at native H3 = 0 and near-zero `M_Z_ref`.

Therefore gradient profile is treated as secondary mechanism evidence, not as the final decision metric.

## 9. Hypothesis Updates

| Hypothesis | Verdict | Update |
|---|---|---|
| direct H3 is sufficient for native readout | supported | K2 plus direct H3 reaches native H3 = 1.0 and final `M_Z_ref` = 3.766 |
| direct H3 is necessary for K3 native/margin in this setting | supported | removing direct H3 leaves native H3 = 0 and final `M_Z_ref` = 0.004 |
| non-direct all-position training can make `Z` readable | supported | K2 and masked K3 keep nontrivial `Q` while native H3 remains 0 |
| `Q` alone identifies direct supervision | rejected | high or nontrivial `Q` appears without native H3 or margin |

## 10. Claim Boundary

Can claim:

```text
In the controlled clean all-position tau=3 setting, direct decision-prefix H3
supervision is the tested path that reliably creates native h_T -> S_Z readout
and large readout-effective margin.
```

Cannot claim:

```text
Indirect transfer is useless.
Direct supervision is the only possible recovery path.
This proves natural-language MTP benefit.
This proves MoE routing preservation or expert utility.
```

## 11. Artifact Map

```text
code workspace:
  /data/250010109/Research_System/Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments
runner:
  active/synthetic_data_understanding/scripts/run_a11_03_04_semantic_efficiency.py
config:
  active/synthetic_data_understanding/configs/a11_10_all_position_indirect_transfer_dynamics.json
run name:
  a11_10_all_position_indirect_transfer_full_20260709_164227
result dir:
  active/synthetic_data_understanding/results/a11_10_all_position_indirect_transfer_dynamics/a11_10_all_position_indirect_transfer_full_20260709_164227
local log dir:
  active/synthetic_data_understanding/logs/local/a11_10_all_position_indirect_transfer_full_20260709_164227
curated project dir:
  Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics
primary tables:
  tables/condition_summary.csv
  tables/efficiency_by_seed.csv
  tables/one_step_summary.csv
  tables/local_loss_guard_summary.csv
  tables/gradient_usefulness_summary.csv
figure:
  figures/direct_native_margin_split.png
```
