# A11_08 General-K Readout-Margin Dynamics Summary

## What We Established

A11_08 supports the minimal general-K vector-sum mechanism in the controlled
decision-prefix setting. Under the same `K=4`, the same four active next-token
losses, and the same two informative horizons, the observed hidden-state
semantic velocity follows the output-row geometry:

```text
aligned H3+H4:      G_hidden = 0.015262
low/conflict H3+H4: G_hidden = 0.000954
aligned - low:      G_hidden = 0.014308
seed support:       5 / 5 seeds
verdict:            support, not falsified
```

The zero and positive controls also pass: `shared_only_k4` has zero semantic
velocity, while `single_h3` has positive semantic velocity
(`G_hidden=0.003816`). This supports the claim that next-K benefit is not "K is
larger", but "K covers informative future rows whose centered output directions
sum into a useful semantic update direction."

Claim boundary: this result is still decision-prefix, first-order, synthetic,
and geometry-controlled. It does not prove natural-language MTP benefit,
all-position direct supervision, MoE routing utility, or that larger K is
monotonically better.

## Terminology / Definitions

| Term | Plain meaning | Concrete object / formula | Why it matters | Cannot prove |
|---|---|---|---|---|
| `Z` | Early branch variable | Four-way synthetic branch id | Controlled long-horizon semantic variable | Natural-language semantics |
| `H3`, `H4` | Future prediction horizons 3 and 4 | Prediction heads for `Y3` and `Y4`, not Transformer layers | Names the future offsets that can carry `Z` | Layer-wise causality |
| `S_z`, `T_z` | Branch-coded future tokens | One token per branch at horizon 3 or 4 | Makes the future target informative about `Z` | Real token semantics |
| `shared_only_k4` | No future target carries `Z` | `Y1=A, Y2=C, Y3=A, Y4=C` | Zero semantic-update guard | Full LM behavior |
| `single_h3` | Only horizon 3 carries `Z` | `Y1=A, Y2=C, Y3=S_z, Y4=A` | Positive one-informative-horizon control | General K behavior |
| `aligned_h3_h4` | Horizons 3 and 4 both carry `Z`, and their centered output rows point the same way | `Y3=S_z, Y4=T_z`; head-4 `T_z` rows aligned to head-3 `S_z` rows | Should amplify semantic velocity | Natural output geometry |
| `low_conflict_h3_h4` | Horizons 3 and 4 both carry `Z`, but horizon 4 opposes/reduces horizon 3's centered direction | Same targets as aligned, but head-4 rows are low/conflicting | Minimal falsifier for "more informative horizons always help" | Natural target permutation behavior |
| `K4_active` | All four next-token losses are active | Active horizons `{1,2,3,4}` with equal loss weights | Controls for active loss count | Any other K value |
| `G_pred_batch_scaled` | Geometry-predicted first-order score after batch averaging | $\frac{1}{n_{branch}m^2}\sum_z\|\sum_{j\in I_K}\lambda_j(u_{j,z}-\bar u_j)\|^2$ | Checks whether the output geometry predicts a gap | Actual optimization convergence |
| `G_hidden` | Observed hidden-state semantic velocity | $\frac1m\sum_z v_z^{(K)\top}(-\nabla_{h_z}L_K)$ | Primary decision metric | Parameter-space sample efficiency |
| `Q_eval` | Conservative readability of `Z` from the current hidden state | $\min\{A(t),P(t),S(t)\}$, where `A` is suffix-decoder accuracy, `P` is independent probe accuracy, and `S` is branch-swap consistency | Secondary guard that `Z` becomes readable | Direct supervision or causality by itself |
| `M_Z_ref` | Fixed readout-effective semantic margin | $\frac1m\sum_z v_z^{(K)\top}h_z$ using the initial aggregate semantic rows | Checks movement toward theorem-facing output rows | Current-head adaptation quality |
| Local guard | Nearby shared-token modeling is preserved | Horizon-1 and horizon-2 cross-entropy / accuracy | Rules out semantic gain from damaging local targets | Full language-model quality |

## Setup

| Item | Value |
|---|---|
| Run name | `a11_08_general_k_readout_margin_full_20260709_051147` |
| Experiment | `general_k_readout_margin` |
| Scope | decision-only |
| Branches | 4 |
| `Kmax` | 4 |
| Active horizons | `{1,2,3,4}` |
| Seeds | `971, 972, 973, 974, 975` |
| One-step etas | `1e-4, 3e-4, 1e-3` |
| Training steps | 2000 |
| Evaluation interval | 100 |
| Model | one-layer Transformer, `d_model=64`, one attention head, dropout 0 |

| Condition | Future targets | Informative horizons | Mechanism role |
|---|---|---|---|
| `shared_only_k4` | `A, C, A, C` | none | zero guard |
| `single_h3` | `A, C, S_z, A` | 3 | positive single-horizon control |
| `aligned_h3_h4` | `A, C, S_z, T_z` | 3, 4 | two aligned informative horizons |
| `low_conflict_h3_h4` | `A, C, S_z, T_z` | 3, 4 | same informativeness, low/conflicting geometry |

## Primary Result

The geometry precondition passed for all etas:

$$
G^{pred}_{aligned}-G^{pred}_{low/conflict}
=0.014308
\ge 0.5G^{pred}_{single}
=0.001908.
$$

After that precondition passed, the observed hidden-state velocity separated in
the predicted direction in all five seeds.

| eta | `G_pred_single` | `G_pred_aligned` | `G_pred_low_conflict` | `G_hidden_single` | `G_hidden_shared` | `G_hidden_aligned` | `G_hidden_low_conflict` | aligned > low seeds | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0001 | 0.003816 | 0.015262 | 0.000954 | 0.003816 | 0.000000 | 0.015262 | 0.000954 | 5 / 5 | support |
| 0.0003 | 0.003816 | 0.015262 | 0.000954 | 0.003816 | 0.000000 | 0.015262 | 0.000954 | 5 / 5 | support |
| 0.0010 | 0.003816 | 0.015262 | 0.000954 | 0.003816 | 0.000000 | 0.015262 | 0.000954 | 5 / 5 | support |

![One-step predicted and observed semantic direction](figures/a11_08_one_step_prediction_vs_observation.png)

This figure tests the primary theorem-facing quantity. The gray bars are the
output-geometry prediction; the colored bars are observed hidden-state velocity.
The observed values match the geometry prediction almost exactly. The figure
does not show long-run optimization or natural-language behavior.

![Aligned versus low/conflict by seed](figures/a11_08_aligned_low_gap_by_seed.png)

This figure checks whether the result is a seed artifact. Each seed preserves a
large aligned-minus-low/conflict gap, so the main decision is not driven by one
outlier seed.

## Guards

Local shared-token modeling is not worse in the aligned condition; all
conditions learn `Y1` and `Y2` to accuracy 1.0 by the end.

| Condition | early `Y1` loss | early `Y2` loss | final `Y1` loss | final `Y2` loss | final `Y1/Y2` accuracy |
|---|---:|---:|---:|---:|---:|
| `aligned_h3_h4` | 0.393800 | 0.358616 | 0.000016 | 0.000016 | 1.0 / 1.0 |
| `low_conflict_h3_h4` | 0.394021 | 0.358784 | 0.000017 | 0.000017 | 1.0 / 1.0 |
| `shared_only_k4` | 0.393641 | 0.358459 | 0.000013 | 0.000012 | 1.0 / 1.0 |
| `single_h3` | 0.393683 | 0.358472 | 0.000011 | 0.000010 | 1.0 / 1.0 |

The short training curves are secondary. They are consistent with the one-step
mechanism, but they are not used as the primary proof:

| Condition | early AUC `Q_eval` | final `Q_eval` | final `M_Z_ref` |
|---|---:|---:|---:|
| `aligned_h3_h4` | 0.5350 | 0.55 | 2.922875 |
| `low_conflict_h3_h4` | 0.3875 | 0.50 | 0.487142 |
| `single_h3` | 0.3925 | 0.40 | 0.636958 |
| `shared_only_k4` | 0.2500 | 0.25 | 0.000000 |

![Training curve guard](figures/a11_08_training_curve_guard.png)

The curve shows that aligned geometry has stronger early and final
readout-facing margin than the low/conflict condition. It is a guard and
sanity check, not the decisive metric, because short training still includes
optimization noise and seed-level threshold effects.

## Interpretation

Result: the minimal falsifier did not fire. The same `K=4` objective can produce
large or small first-order semantic velocity depending on whether the covered
informative horizons' centered output directions align or conflict.

Interpretation: this strengthens the A11 story from "K must cover the first
informative horizon" to "when K covers multiple informative horizons, their
readout-effective directions combine approximately as a vector sum."

Claim: in this controlled first-order setting, general next-K semantic
efficiency is governed by the aggregate informative readout direction

$$
v_z^{(K)}=\sum_{j\in I_K}\lambda_j(u_{j,z}-\bar u_j).
$$

Speculation to keep separate: this may be the right bridge object for later
MoE-routing geometry, but A11_08 itself does not test routing.

## Next Decision

Open the next anchor on next-2 / next-3 / next-K inclusion law, not MoE yet. The
minimal next question should be:

```text
As K increases, does first-order semantic velocity turn on exactly when K first
includes an informative horizon, stay near zero for added uninformative horizons,
and change by the predicted vector increment when an additional informative
horizon is included?
```

This next anchor should vary K and informative-horizon placement while keeping
the same leakage, local-loss, output-geometry, and seed guards.

## Artifacts

- Protocol: `protocol.md`
- Detailed record: `detailed.md`
- Runner: `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/run_a11_03_04_semantic_efficiency.py`
- Config: `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/a11_08_general_k_readout_margin_dynamics.json`
- Result directory: `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/a11_08_general_k_readout_margin_dynamics/a11_08_general_k_readout_margin_full_20260709_051147/`
- Figures: `figures/`
