# A11_09 Next-K Inclusion Law Summary

## Conclusion

A11_09 supports the next-K inclusion law for direct hidden semantic velocity. In the frozen-head clean decision-only setting, semantic velocity is zero before K covers an informative horizon, turns on at the first informative horizon, does not increase when K only adds an uninformative shared horizon, and changes by the predicted vector increment when K adds a second informative horizon.

This result supports the narrow claim that K size alone is not the causal object. The causal object is whether K newly includes an informative future position and whether the new centered output direction aligns with the existing semantic direction.

## Terminology / Definitions

| Metric | Formula | Measures | Decision role | Cannot prove |
| --- | --- | --- | --- | --- |
| Informative horizon set `I_K` | $$\mathcal I_K=\{j\le K:Y_j\text{ one-to-one encodes }Z\}$$ | which future positions contribute branch information | defines when K should matter | natural-language horizon structure |
| Aggregate direction `v_z^(K)` | $$v_z^{(K)}=\sum_{j\in\mathcal I_K}\lambda_j(u_{j,z}-\bar u_j)$$ | sum of covered semantic readout directions | theory object | full convergence |
| Hidden semantic velocity `G_hidden` | $$\frac1m\sum_z v_z^{(K)\top}(-\nabla_{h_z}L_K)$$ | direct one-step push of `h_T` along semantic direction | primary metric | final recovery |
| Vector increment | $$\Delta G=\frac1{m^2}\sum_z(2v_z^\top a_z+\|a_z\|^2)$$ | effect of adding a new informative horizon | tests aligned vs low/conflict | all-position transfer |
| Readout margin `M_Z` | $$\frac1m\sum_z v_z^\top h_z$$ | accumulated hidden-state alignment | secondary curve metric | causality alone |
| Guarded recovery `Q_eval` | $$\min\{A_{decoder},A_{probe},C_{swap}\}$$ | conservative branch recovery | secondary guard | direct supervision proof |
| Y1/Y2 local guard | CE and accuracy on shared local targets | local prediction quality | excludes local damage | full LM quality |

## Setup

```text
run name: a11_09_next_k_inclusion_law_full_20260709_1228
job id: pt-9an690sp
scope: clean decision-only
head variant: frozen_heads
conditions: K2_active, K3_active, K4_active
data regimes: shared_only_k4, single_h3, aligned_h3_h4, low_conflict_h3_h4
seeds: 971, 972, 973, 974, 975
steps: 2000
eval every: 100
```

## Main One-Step Result

At `eta=0.0003`, the primary hidden semantic velocity is:

| Regime | K2 | K3 | K4 | K3-K2 | K4-K3 | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `shared_only_k4` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | no informative horizon, no semantic velocity |
| `single_h3` | 0.000000 | 0.003816 | 0.003816 | 0.003816 | -0.000000 | turns on at H3; shared H4 adds no direct increment |
| `aligned_h3_h4` | 0.000000 | 0.003816 | 0.015262 | 0.003816 | 0.011447 | aligned second informative horizon amplifies |
| `low_conflict_h3_h4` | 0.000000 | 0.003816 | 0.000954 | 0.003816 | -0.002862 | low/conflicting second horizon weakens |

Seed support:

| Claim | Seed support |
| --- | ---: |
| `single_h3`: K3 > K2 | 5 / 5 |
| `single_h3`: K4 > K3 | 0 / 5 |
| `aligned_h3_h4`: K4 > K3 | 5 / 5 |
| `low_conflict_h3_h4`: K4 > K3 | 0 / 5 |

## Curve Evidence

| Regime | Condition | early AUC `G_hidden_ref` | early AUC `M_Z_ref` | early AUC `Q` | final `Q` | final Y1/Y2 acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `single_h3` | K2 | 0.000000 | 0.000 | 0.250 | 0.250 | 1.0 / 1.0 |
| `single_h3` | K3 | 0.000666 | 1.397 | 0.475 | 0.550 | 1.0 / 1.0 |
| `single_h3` | K4 | 0.000783 | 0.758 | 0.392 | 0.400 | 1.0 / 1.0 |
| `aligned_h3_h4` | K4 | 0.002660 | 3.238 | 0.478 | 0.500 | 1.0 / 1.0 |
| `low_conflict_h3_h4` | K4 | 0.000184 | 0.128 | 0.428 | 0.500 | 1.0 / 1.0 |

Curve interpretation: the one-step inclusion law is cleanly supported. Multi-step curves are secondary: adding a shared fourth horizon has no direct one-step semantic increment, but it can still change optimization trajectories through local shared losses.

## Hypothesis Judgment

| Hypothesis | Judgment | Evidence |
| --- | --- | --- |
| K below the first informative horizon gives near-zero semantic velocity | supported | `single_h3` K2 = 0 |
| K at the first informative horizon turns on semantic velocity | supported | `single_h3` K3 = 0.003816, 5/5 positive over K2 |
| adding an uninformative horizon gives no direct semantic increment | supported for one-step velocity | `single_h3` K4-K3 = 0 |
| adding aligned informative horizon amplifies velocity | supported | `aligned_h3_h4` K4-K3 = 0.011447, 5/5 positive |
| adding low/conflict informative horizon weakens velocity | supported | `low_conflict_h3_h4` K4-K3 = -0.002862, 5/5 negative |

## Claim Boundary

Can claim:

```text
In controlled frozen-head decision-only dynamics, next-K semantic velocity obeys
an inclusion law: K matters when it newly covers an informative horizon, and
the increment depends on output-direction alignment.
```

Cannot claim:

```text
MTP is better in natural language.
All-position indirect transfer obeys the same rule.
Final Q must be monotonic in K.
Larger K is always worse or always better.
```

## Artifacts

```text
config: Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/a11_09_next_k_inclusion_law.json
result dir: Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/a11_09_next_k_inclusion_law/a11_09_next_k_inclusion_law_full_20260709_1228
main tables:
  tables/one_step_summary.csv
  tables/one_step_metrics.csv
  tables/condition_summary.csv
  tables/efficiency_by_seed.csv
  tables/output_geometry.csv
```
