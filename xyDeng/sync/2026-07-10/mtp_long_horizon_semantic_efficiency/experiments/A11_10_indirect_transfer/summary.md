# A11_10 All-Position Indirect-Transfer Dynamics Summary

## Conclusion

A11_10 supports the native/probe split in clean all-position `tau=3` training. Non-direct conditions can recover nontrivial guarded `Q`, but they do not learn native `h_T -> S_Z` prediction and they keep readout-effective margin near zero. Direct decision-prefix horizon-3 supervision is the tested path that reliably creates native horizon-3 prediction and large `M_Z_ref`.

The result strengthens the A11 claim boundary: `Q` is a recovery metric, not direct-supervision evidence. Direct supervision must be identified by native H3 accuracy and readout-effective margin.

## Terminology / Definitions

| Term | Plain meaning | Concrete object / formula | Why it matters | Cannot prove |
|---|---|---|---|---|
| Direct decision-prefix H3 | the current decision state predicts the first informative future token | $\operatorname{CE}(q_3(\cdot\mid h_T),S_Z)$ | tests the short credit-assignment path | the only possible recovery path |
| Indirect transfer | other all-position losses make `Z` readable through shared parameters | $\dot h_T^{(s)}=-J_TJ_s^\top\nabla_{h_s}\ell_s$ | explains high `Q` without native H3 | native use of `S_Z` head |
| Guarded recovery `Q` | conservative recovery score | $Q=\min\{A_{decoder},A_{probe},C_{swap}\}$ | detects whether `Z` is readable | direct supervision |
| Native H3 accuracy | model's own H3 head predicts `S_Z` from `h_T` | $\Pr[\arg\max q_3(\cdot\mid h_T)=S_Z]$ | primary direct-readout metric | all semantic utility |
| Readout margin `M_Z_ref` | hidden state alignment with initial `S_Z` output rows | $\frac1m\sum_z(u^0_{S_z}-\bar u^0)^\top h_z$ | identifies readout-effective semantic direction | final language quality |

## Setup

```text
run name: a11_10_all_position_indirect_transfer_full_20260709_164227
scope: clean all-position LM-style tau=3
sequence: BR_z F00...F15 U00 U01 U02 U03 A_SHARED C_SHARED S_z STOP
conditions: K2_active, K3_active, K3_mask_decision_prefix_L3, K2_plus_decision_prefix_L3
seeds: 971, 972, 973, 974, 975
steps: 15000
eval interval: 250
shards: 4 completed
```

## Main Result

| Condition | Role | Reach rate | Early AUC Q | Final Q | Native H3 | Final `M_Z_ref` | Final `E_Z` |
|---|---|---:|---:|---:|---:|---:|---:|
| `K2_active` | non-covering baseline | 0.60 | 0.773 | 0.75 | 0.00 | 0.000 | 0.005 |
| `K3_mask_decision_prefix_L3` | indirect-only K3 | 0.60 | 0.773 | 0.70 | 0.00 | 0.004 | 0.010 |
| `K3_active` | full direct + indirect | 1.00 | 0.965 | 1.00 | 1.00 | 2.430 | 50.915 |
| `K2_plus_decision_prefix_L3` | direct-term sufficiency | 1.00 | 0.965 | 1.00 | 1.00 | 3.766 | 83.262 |

Direct-readout gap:

```text
K2_plus_direct - K2_active:
  native H3: +1.00
  final M_Z_ref: +3.766
  reach rate: +0.40

K3_active - K3_mask_decision_prefix_L3:
  native H3: +1.00
  final M_Z_ref: +2.425
  reach rate: +0.40
```

Local guard passed: all four conditions end with Y1 accuracy = 1.0 and Y2 accuracy = 1.0. Therefore the direct-readout gains are not explained by local target collapse.

## Central Figure

![Direct native margin split](figures/direct_native_margin_split.png)

This figure tests whether guarded recovery and direct native readout are the same signal. Blue bars show final `Q`; orange bars show native H3 accuracy; the green and purple lines show final and early readout margin. Non-direct conditions keep nontrivial `Q` but have zero native H3 and near-zero margin. Direct conditions have both high `Q` and high native/margin. The figure does not prove natural-language benefit or MoE routing utility.

## Hypothesis Judgment

| Hypothesis | Judgment | Evidence |
|---|---|---|
| Direct H3 is sufficient for native readout | supported | `K2_plus_decision_prefix_L3` reaches native H3 = 1.0 and final `M_Z_ref` = 3.766 |
| Direct H3 is necessary for native/margin in K3 | supported in this setting | removing the direct term gives native H3 = 0.0 and final `M_Z_ref` = 0.004 |
| Non-direct all-position training can recover probe-readable `Z` | supported | `K2_active` and masked K3 reach final `Q` 0.75 / 0.70 while native H3 remains 0 |
| `Q` alone proves direct supervision | rejected | high or nontrivial `Q` appears without native H3 or margin |

## Claim Boundary

Can claim:

```text
In the controlled clean all-position tau=3 setting, direct decision-prefix H3
supervision reliably creates native h_T -> S_Z readout and large readout
margin. Non-direct all-position losses can still make Z readable, so Q alone
is not direct-supervision evidence.
```

Cannot claim:

```text
Indirect recovery is useless.
Direct supervision is the only possible way to learn Z.
The result proves natural-language MTP benefit.
The result proves MoE expert utility or routing preservation.
```

## Artifacts

```text
protocol: Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/protocol.md
detailed: Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/detailed.md
config: Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/a11_10_all_position_indirect_transfer_dynamics.json
result dir: Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/a11_10_all_position_indirect_transfer_dynamics/a11_10_all_position_indirect_transfer_full_20260709_164227
curated tables: Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/tables/
figure: Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/figures/direct_native_margin_split.png
```
