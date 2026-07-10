# Protocol: A11_10 All-Position Indirect-Transfer Dynamics

Status: approved by user request; full run in progress on 2026-07-09.  
Anchor: `../../../problem_anchors/11_long_horizon_mtp_objective/11_10_all_position_indirect_transfer_dynamics_anchor.md`.

## 1. Purpose

Separate direct native `h_T -> S_Z` learning from all-position indirect transfer. The protocol asks whether non-direct recovery in all-position training is a probe-readable shared-parameter effect rather than native horizon-3 prediction from the decision state.

## 2. Dataset

Clean `tau=3` sequence:

```text
BR_z F00 ... F15 U00 U01 U02 U03 A_SHARED C_SHARED S_z STOP
```

The decision prefix ends at `U03`. The first two future targets after the decision prefix are shared; the third future target is a one-to-one code of the early variable `Z`.

## 3. Conditions

The full run contains four mechanism-critical conditions:

| Condition | Loss definition | Role |
|---|---|---|
| `K2_active` | all positions, horizons 1 and 2 | non-covering baseline |
| `K3_active` | all positions, horizons 1, 2, and 3 | full direct + indirect condition |
| `K3_mask_decision_prefix_L3` | K3 but remove only decision-prefix horizon-3 loss | indirect-only control |
| `K2_plus_decision_prefix_L3` | K2 plus decision-prefix horizon-3 loss | direct-term sufficiency control |

All runs use no-leakage initialization:

```text
init_mode: identical_branch
seeds: 971, 972, 973, 974, 975
steps: 15000
eval_every: 250
```

## 4. Metrics

Primary metric:

```text
native horizon-3 accuracy + M_Z_ref versus guarded recovery Q
```

Definitions:

$$
M_Z^{ref}=\frac1m\sum_z(u^0_{S_z}-\bar u^0)^\top h_z.
$$

$$
Q=\min\{A_{decoder},A_{probe},C_{swap}\}.
$$

One-step hidden semantic velocity:

$$
G_{hidden}^{ref}=\frac1m\sum_z(u^0_{S_z}-\bar u^0)^\top(-\nabla_{h_z}L).
$$

Secondary metrics:

```text
early AUC Q
early AUC M_Z_ref
early AUC G_hidden_ref
final Q
final native H3
final M_Z_ref
final E_Z
Y1/Y2 accuracy and CE
gradient_profile_summary
```

## 5. Procedure

1. Run a four-shard full run over the four conditions and five seeds.
2. Enable one-step hidden-gradient auditing.
3. Enable gradient profile at steps 0, 500, 1000, 1500, and 2000 for horizons 2 and 3.
4. Aggregate all shards into one result directory.
5. Write `summary.md`, `detailed.md`, and update the anchor with the supported / weakened / insufficient result.

## 6. Pass / Fail

Support:

```text
direct-add and full-K3 conditions produce high native H3 and large M_Z_ref;
K2 and masked-K3 can have nontrivial Q but low native H3 and near-zero M_Z_ref;
local Y1/Y2 guards pass;
one-step and early G_hidden_ref are positive mainly when the direct H3 term is present.
```

Weaken:

```text
masked-K3 matches full-K3 native H3 and M_Z_ref;
direct-add does not improve K2 native H3 or M_Z_ref;
Q, native H3, and M_Z_ref are indistinguishable across direct and non-direct conditions;
local target damage explains the gain.
```

## 7. Output Artifacts

Required:

```text
summary.md
detailed.md
tables/condition_summary.csv
tables/efficiency_by_seed.csv
tables/one_step_summary.csv
tables/one_step_metrics.csv
tables/gradient_profile_summary.csv
tables/gradient_usefulness_summary.csv
tables/local_loss_guard_summary.csv
run_manifest.json
```

## 8. Interpretation Boundary

If supported, the claim is about a controlled all-position synthetic setting. It does not prove that indirect transfer is bad, that direct supervision is the only recovery path, or that natural-language MTP has the same dynamics.
