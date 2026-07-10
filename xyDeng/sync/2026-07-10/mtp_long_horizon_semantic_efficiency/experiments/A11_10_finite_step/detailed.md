# A11_10 Finite-Step Semantic Efficiency Detailed Record

Status: completed.  
Run name: `a11_10_finite_step_semantic_efficiency_4gpu_20260709_full`.  
Job id: `pt-h43wss54`.  
Summary: `summary.md`.  
Protocol: `protocol.md`.

## 1. Purpose

A11_10 tests whether the existing first-order semantic velocity mechanism can be promoted to a finite-step margin hitting-time statement under explicit local optimization assumptions.

The tested conditional proposition is:

$$
M_K(t+1)\ge M_K(t)+\eta g_K-\eta^2B-\epsilon_t.
$$

If:

$$
\eta g_K-\eta^2B-\bar\epsilon>0,
$$

then:

$$
T_\gamma(K)
\le
\left\lceil
\frac{\gamma-M_K(0)}
{\eta g_K-\eta^2B-\bar\epsilon}
\right\rceil.
$$

The experiment audits the assumptions behind this proposition. It is not a broad K sweep and not a natural-language experiment.

## 2. Terminology / Definitions

| Term | Plain meaning | Concrete computation | Unit / formula | Why it matters | Cannot prove |
|---|---|---|---|---|---|
| reference semantic direction | fixed semantic output direction from initialization | centered informative output rows, summed over informative active horizons | `v_z^ref` | gives the theorem coordinate system | current-head utility alone |
| reference semantic margin | hidden-state progress in reference semantic directions | average projection of `h_z(t)` onto `v_z^ref` | $M_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}h_z(t)$ | target of finite-step bound | recovery without calibration |
| hidden semantic velocity | current loss gradient projected onto reference semantic direction | projected negative hidden gradient at decision position | $G_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_K)$ | source term for margin growth | multi-step convergence alone |
| direct active loss | the decision-prefix informative H3 term as it appears in the active objective | direct `CE(q_3(.|h_T),S_Z)`; in all-position active loss it is scaled by the per-position average | scalar loss | separates direct semantic supervision | absence of indirect learning |
| non-direct loss | active training loss minus direct active loss | `L_active - L_direct_active` | scalar loss | estimates perturbation from local/shared/indirect terms | causal path by itself |
| `Q_eval` | conservative branch recovery score | min of decoder accuracy, frozen probe accuracy, and branch-swap consistency | 0 to 1 | guards against one metric being misleading | direct supervision |
| native H3 accuracy | model-owned H3 prediction from `h_T` | accuracy of `q_3(.|h_T)=S_Z` | 0 to 1 | direct readout evidence | all hidden information |
| certified increment | empirical lower-bound step size | minimum early velocity minus curvature and perturbation corrections | $d_{cert}=\eta g_{min}-\eta^2\widehat B-\widehat\epsilon$ | checks whether first-order velocity certifies finite-step progress | tightness by itself |

## 3. Data Construction

The run uses the existing A11 controlled branch data.

Branch variable:

$$
Z\sim\operatorname{Unif}\{1,\ldots,m\}.
$$

Primary clean `tau=3` family:

$$
Y_1=A,\qquad Y_2=C,\qquad Y_3=S_Z.
$$

General-K geometry variants:

```text
shared_only_k4:     Y1=A, Y2=C, Y3=A,   Y4=C
single_h3:          Y1=A, Y2=C, Y3=S_z, Y4=A
aligned_h3_h4:      Y1=A, Y2=C, Y3=S_z, Y4=T_z
low_conflict_h3_h4: Y1=A, Y2=C, Y3=S_z, Y4=T_z
```

The `aligned_h3_h4` and `low_conflict_h3_h4` conditions use controlled output-row geometry inherited from A11_08 / A11_09.

## 4. Model / Training / Evaluation

Workspace:

```text
/data/250010109/Research_System/Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding
```

Runner:

```text
scripts/run_a11_03_04_semantic_efficiency.py
```

Config:

```text
configs/a11_10_finite_step_semantic_efficiency.json
```

Training settings:

| Field | Value |
|---|---:|
| seeds | `971, 972, 973, 974, 975` |
| max steps | `2000` |
| eval interval | `100` |
| learning rate | `0.003` |
| branches | `4` |
| model dimension | `64` |
| heads | `1` attention head; up to `4` prediction heads |
| train samples per branch | `16` |
| eval samples per branch | `64` |
| decoder/probe steps | `200` |

Submit command:

```bash
A11_ALLOW_REAL_SUBMIT=1 RUN_NAME=a11_10_finite_step_semantic_efficiency_4gpu_20260709_full bash scripts/submit_a11_10_finite_step_semantic_efficiency_4gpu.sh
```

ACP result:

```text
pt-h43wss54 SUCCEEDED
```

All four shards reached step `2000`, and aggregation completed.

## 5. Conditions

| Condition | Scope | Heads | Purpose |
|---|---|---|---|
| `shared_only_k4` + `K4_active` | decision-only | frozen | zero semantic velocity guard |
| `single_h3` + `K3_active` | decision-only | frozen | single informative H3 theorem window |
| `aligned_h3_h4` + `K4_active` | decision-only | frozen | positive vector-sum condition |
| `low_conflict_h3_h4` + `K4_active` | decision-only | frozen | low/conflicting vector-sum condition |
| `single_h3` + `K3_active` | decision-only | trainable | head drift audit |
| `low_conflict_h3_h4` + `K4_active` | decision-only | trainable | head drift audit under conflict |
| `single_h3` + `K3_active` | all-position | trainable | full all-position K3 |
| `single_h3` + `K3_mask_decision_prefix_L3` | all-position | trainable | remove direct decision H3 term |
| `single_h3` + `K2_plus_decision_prefix_L3` | all-position | trainable | add direct H3 to K2 background |
| `single_h3` + `K2_active` | all-position | trainable | non-covering all-position baseline |

## 6. Metric Formulas

Reference margin:

$$
M_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}h_z(t).
$$

Reference velocity:

$$
G_K^{ref}(t)=\frac1m\sum_z v_z^{ref\top}(-\nabla_{h_z}L_K(\theta_t)).
$$

Direct / non-direct decomposition:

$$
L_{active}=L_{direct}+L_{non-direct}.
$$

For all-position K3, the direct active term is the decision-position component of the all-position H3 average, so it is smaller than a separately added full decision-prefix H3 term.

Guarded recovery:

$$
Q(t)=\min\{A_{decoder}(t),A_{probe}(t),C_{swap}(t)\}.
$$

Margin hitting time:

$$
T_\gamma=\inf\{t:M_K^{ref}(t)\ge\gamma\}.
$$

Recovery time:

$$
T_{0.9}=\inf\{t:Q(t)\ge0.9\}.
$$

Certified total increment:

$$
d_{cert,total}=\eta g_{min,total}-\eta^2\widehat B.
$$

Direct corrected increment:

$$
d_{cert,direct}=\eta g_{min,direct}-\eta^2\widehat B-\widehat\epsilon.
$$

Head alignment:

$$
c_u(t)=
\frac{\langle v^{ref},v^{current}(t)\rangle}
{\|v^{ref}\|\|v^{current}(t)\|}.
$$

## 7. Primary Results

### 7.1 Finite-Step Assumption Summary

| Condition | Scope / heads | `G_total_min` | `G_direct_min` | `G_non_direct_min` | calibration pass | bound pass | support rate | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `shared_only_k4` + `K4_active` | decision / frozen | `0.000000` | `0.000000` | `0.000000` | `0.0` | `0.0` | `0.0` | zero guard passes |
| `single_h3` + `K3_active` | decision / frozen | `0.000954` | `0.000954` | `~0` | `0.4` | `0.2` | `0.2` | partial |
| `single_h3` + `K3_active` | decision / trainable | `0.000947` | `0.000947` | `~0` | `0.4` | `0.4` | `0.4` | partial |
| `aligned_h3_h4` + `K4_active` | decision / frozen | `0.003816` | `0.001908` | `0.001908` | `0.2` | `0.4` | `0.2` | partial |
| `low_conflict_h3_h4` + `K4_active` | decision / frozen | `0.000153` | `0.000404` | `-0.000252` | `0.0` | `0.0` | `0.0` | weak finite-step |
| `low_conflict_h3_h4` + `K4_active` | decision / trainable | `0.000153` | `0.000300` | `-0.000238` | `0.2` | `0.0` | `0.0` | partial but not certified |
| `K2_active` | all-position / trainable | `0.000000` | `0.000000` | `0.000000` | `0.0` | `0.0` | `0.0` | direct absent |
| `K2_plus_direct_H3` | all-position / trainable | `0.000951` | `0.000951` | `~0` | `1.0` | `0.6` | `0.6` | strongest finite-step support |
| `K3_active` | all-position / trainable | `0.000036` | `0.000036` | `~0` | `0.8` | `0.8` | `0.6` | support, direct term diluted |
| `K3_mask_direct_H3` | all-position / trainable | `~0` | `0.000000` | `~0` | `0.0` | `0.0` | `0.0` | direct absent |

### 7.2 Recovery Summary

| Condition | Scope / heads | `Q` reach rate | `AUC_Q` | early `AUC_Q` | final `Q` | final native H3 | final `M_Z_ref` |
|---|---|---:|---:|---:|---:|---:|---:|
| `shared_only_k4` + `K4_active` | decision / frozen | `0.0` | `0.250` | `0.250` | `0.250` | `nan` | `0.000` |
| `single_h3` + `K3_active` | decision / frozen | `0.4` | `0.513` | `0.475` | `0.550` | `0.550` | `2.771` |
| `aligned_h3_h4` + `K4_active` | decision / frozen | `0.2` | `0.489` | `0.478` | `0.500` | `0.500` | `5.189` |
| `low_conflict_h3_h4` + `K4_active` | decision / frozen | `0.2` | `0.464` | `0.428` | `0.500` | `0.500` | `0.180` |
| `K2_active` | all-position / trainable | `0.6` | `0.835` | `0.840` | `0.850` | `0.000` | `0.000` |
| `K3_mask_direct_H3` | all-position / trainable | `0.6` | `0.794` | `0.790` | `0.800` | `0.000` | `0.002` |
| `K2_plus_direct_H3` | all-position / trainable | `1.0` | `0.929` | `0.858` | `1.000` | `1.000` | `3.574` |
| `K3_active` | all-position / trainable | `1.0` | `0.974` | `0.948` | `1.000` | `1.000` | `2.065` |

### 7.3 Local Loss Guard

All conditions preserve local next-token learning:

```text
final Y1 accuracy = 1.0
final Y2 accuracy = 1.0
```

Final local losses are near zero in all trainable-head all-position conditions and also converge in frozen-head decision-only controls.

## 8. Hypothesis Evaluation

### H1: Velocity Persistence

Supported for direct informative conditions.

Evidence:

- `single_h3` decision-only frozen: `G_total_min=0.000954`.
- `aligned_h3_h4` decision-only frozen: `G_total_min=0.003816`.
- `low_conflict_h3_h4` decision-only frozen: `G_total_min=0.000153`, smaller but positive.
- all-position `K2_plus_direct_H3`: `G_total_min=0.000951`.
- all-position `K3_active`: `G_total_min=0.000036`, positive but diluted by all-position averaging.
- `shared_only_k4`: exactly zero.

Interpretation: A11_10 preserves the A11_08 / A11_09 direction logic over the early window, but the all-position scaling makes direct H3 velocity much smaller than a separate full direct H3 term.

### H2: Margin-To-Recovery Calibration

Partial.

Evidence:

- all-position `K2_plus_direct_H3`: calibration pass `1.0`.
- all-position `K3_active`: calibration pass `0.8`.
- decision-only `single_h3`: calibration pass `0.4`.
- low/conflict decision-only: calibration pass `0.0`.

Interpretation: margin is a useful recovery proxy in direct all-position conditions, but the fixed threshold `gamma=1.0` is not a universal calibration across geometry regimes and seeds.

### H3: Head Drift Bound

Supported as an audit condition, not as a no-drift claim.

Evidence:

- Frozen heads have zero drift.
- Trainable heads keep positive reference/current alignment: `single_h3` decision trainable `0.996`, all-position direct conditions `0.942` to `0.998`, low/conflict trainable `0.844`.
- Drift norms are nonzero, e.g. `single_h3` all-position K3 has drift about `1.015`, and K3 mask has drift about `1.305`.

Interpretation: reference geometry remains usable in these runs, but trainable-head curves must continue to report current geometry.

### H4: Shared / Indirect Perturbation Bound

Partially supported with an important boundary.

Evidence:

- all-position direct conditions keep nonnegative net direct semantic velocity and reach native H3 `1.0`.
- `K2_active` and `K3_mask_direct_H3` recover nontrivial `Q` (`0.85` and `0.80` final means) while native H3 stays `0.0` and `M_Z_ref` stays near zero.

Interpretation: all-position background can make the hidden state branch-readable without making the model's own H3 head predict `S_Z`. Therefore `Q` alone is not direct semantic supervision evidence.

## 9. Bound Tightness And Failure Mode

The finite-step proposition is mathematically valid conditional on a positive lower-bound increment. The run does not show that this bound is practically tight.

Observed issues:

- `d_cert_total_per_step` is often nonpositive after empirical curvature correction.
- When positive, `bound_steps_total` is very large, often hundreds of thousands to millions of steps.
- Several decision-only seeds have positive `G_total_min` but do not cross the margin or `Q` threshold within `2000` steps.

Interpretation:

```text
A11_10 supports the mechanism direction and several assumptions, but it does not close a strong finite-step efficiency theorem. The next proof must either use a sharper local smoothness/curvature argument or state a weaker sign-level finite-step proposition.
```

## 10. Artifact Map

Result root:

```text
Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/results/a11_10_finite_step_semantic_efficiency/a11_10_finite_step_semantic_efficiency_4gpu_20260709_full
```

Logs:

```text
Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/logs/acp/a11_10_finite_step_semantic_efficiency_4gpu_20260709_full
```

Key tables:

| File | Role |
|---|---|
| `tables/curve_metrics.csv` | all checkpoint metrics, 50 runs x 21 checkpoints |
| `tables/seed_level_hitting_time_bound.csv` | per-seed assumption audit and hitting-time bound |
| `tables/finite_step_assumption_summary.csv` | condition-level assumption pass rates |
| `tables/condition_summary.csv` | recovery and final metric summary |
| `tables/local_loss_guard_summary.csv` | Y1/Y2 local learning guard |
| `tables/one_step_summary.csv` | one-step geometry and gradient summary |
| `tables/output_geometry.csv` | initial output geometry audit |
| `run_manifest.json` | config and table manifest |

Implementation changes:

```text
scripts/run_a11_03_04_semantic_efficiency.py
configs/a11_10_finite_step_semantic_efficiency.json
scripts/submit_a11_10_finite_step_semantic_efficiency_4gpu.sh
```

## 11. Claim Boundary

Can claim:

```text
In the controlled A11 setting, direct informative-horizon supervision gives a persistent positive early semantic velocity for h_T, and direct all-position conditions recover native H3 and guarded Q more reliably than masked or non-covering controls.
```

Cannot claim:

```text
A11_10 proves a tight finite-step learning-efficiency theorem.
MTP is generally more efficient in natural language.
Larger K is generally better.
Q alone identifies direct semantic supervision.
Non-covering all-position training cannot make h_T branch-readable.
```

## 12. Next Decision

The next theoretical step should not be a broader experiment. It should refine the finite-step theorem:

1. either prove a sharper curvature / smoothness bound that makes `d_cert` positive and useful;
2. or state a weaker theorem: positive persistent semantic velocity plus empirical calibration predicts early recovery in this controlled family, without claiming a tight hitting-time certificate.
