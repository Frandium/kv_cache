# Protocol: A06_E01_2x2_expert_geometry_metric_audit

Primary anchor: `06_2x2_expert_geometry_gate_anchor.md`  
Anchor decision question: In the minimal $2\times2$ setting, which expert-initialization-induced compatibility metric can define a non-degenerate static oracle routing rule that supports both balanced load and metric-defined specialization under uniform input?  
Experiment role: First-stage static oracle metric audit for the 2D expert-geometry gate. No training is included.

## 1. Decision Question

Can two initialized $2\times2$ experts induce a compatibility metric $q_e(x)$ that produces a non-degenerate static oracle routing rule with both usable load behavior and metric-defined specialization under uniform 2D input?

This protocol compares metric definitions. It does not test a trainable router.

## 2. Tested Hypothesis

H1:
The most direct expert-matrix metric,

$$
q_e(x)=x^\top A_e^\top A_e x=\|A_ex\|^2,
$$

can produce non-degenerate assignments in anisotropic $2\times2$ expert cases.

H2:
Prototype metrics are useful baselines, but signed prototype similarity can be sign-sensitive and may discard matrix structure.

H3:
Projection-style metrics are meaningful only when the selected subspace is not the full $\mathbb{R}^2$ span. Full-rank projection distance should fail or become uninformative in the $2\times2$ full-span case.

H4:
Load calibration can be a secondary repair for imbalance, but it is not allowed to create the specialization claim. A metric remains valid only if positive margin and non-degeneracy survive calibration.

## 3. Rival Explanations

R1. Load balance alone may be mistaken for specialization.

Protection:
Always report mean margin and low-margin fraction beside natural and calibrated load.

R2. A metric may work only because of arbitrary singular-vector or prototype sign choices.

Protection:
Run sign-flip checks and measure assignment change rate.

R3. A metric may look valid only in hand-picked separated experts.

Protection:
Compare random unconstrained, isotropic, anisotropic, near-identical, and clearly separated expert conditions.

R4. Calibration may hide a bad metric.

Protection:
Report natural routing and calibrated routing separately. Calibration is secondary and cannot rescue a degenerate or zero-margin metric.

R5. Prototype success may be too thin to represent the advisor's intended $2\times2$ matrix geometry.

Protection:
Treat matrix response energy as the main candidate and prototype metrics as baselines.

## 4. Data / Model / Algorithm / Objective

Data:

- Primary input: $x\sim\mathrm{Unif}(S^1)$.
- Secondary input: uniform samples from the unit disk.

Experts:

Two experts are represented by $A_1,A_2\in\mathbb{R}^{2\times2}$. The convention is that $A_e$ maps the input $x$ to expert response $A_ex$. Use:

$$
A_e=U_e\Sigma_eV_e^\top .
$$

The input-side directions are the right singular vectors $v_{e,k}$, i.e. the columns of $V_e$. The singular values $\sigma_{e,k}$ measure input sensitivity. The matrix-response energy is represented by:

$$
G_e=A_e^\top A_e=V_e\Sigma_e^2V_e^\top .
$$

Metric candidates:

M1 signed prototype similarity:

$$
m_e=\mathrm{normalize}\left(\sum_k \sigma_{e,k}s_{e,k}v_{e,k}\right),\qquad s_{e,k}\in\{-1,+1\},
$$

$$
q_e(x)=x^\top m_e.
$$

M2 unsigned prototype similarity:

$$
q_e(x)=(x^\top m_e)^2.
$$

M3 projection energy:

$$
q_e(x)=\|V_{e,r}^\top x\|^2,
$$

where $V_{e,r}$ contains the selected top input-sensitive right singular direction or directions.

M4 matrix response energy:

$$
G_e=A_e^\top A_e,\qquad q_e(x)=x^\top G_e x=\|A_ex\|^2.
$$

Natural routing:

$$
g(x)=\arg\max_e q_e(x).
$$

Secondary calibrated routing:

$$
g_\tau(x)=\arg\max_e(q_e(x)-\tau_e),
$$

where $\tau_e$ is chosen to equalize sampled load.

Objective:
Identify whether any expert-initialization-induced metric should become the core mathematical model for the next router-construction or high-dimensional low-rank approximation stage.

Metric implementation contract:

| Metric | Concrete computation | What it tests | Main risk |
| --- | --- | --- | --- |
| M1 signed prototype | Compute SVD; choose oriented $v_{e,k}$; build $m_e=\mathrm{normalize}(\sum_k\sigma_{e,k}s_{e,k}v_{e,k})$; score $x^\top m_e$ | whether a linear prototype split is enough | arbitrary signs and discarded anisotropy |
| M2 unsigned prototype | Use the same $m_e$; score $(x^\top m_e)^2$ | whether removing global sign helps prototype routing | still depends on relative signs in $m_e$ |
| M3 top-1 projection | Use $V_{e,1}$; score $\|V_{e,1}^\top x\|^2$ | whether dominant input direction alone defines routing | ignores singular-value scale and secondary direction |
| M3 full-span control | Use $V_{e,1:2}$; score $\|V_{e,1:2}^\top x\|^2$ | known degenerate control | equals $\|x\|^2$ for both experts in 2D |
| M4 matrix response | Compute $G_e=A_e^\top A_e$; score $x^\top G_ex$ | full matrix input sensitivity | degenerates when $G_1=G_2$ or both are identity |

## 5. Conditions, Seeds, And Checkpoints

Conditions:

| Condition | Expert geometry | Expected behavior |
| --- | --- | --- |
| C1 random unconstrained | Random Gaussian $2\times2$ matrices | Stress test natural behavior and seed variability. |
| C2 orthogonal equal singular values | $A_e=U_eIV_e^\top$ | Matrix response energy degenerates because $G_e=I$. |
| C3 orthogonal unequal singular values | $A_e=U_e\mathrm{diag}(\sigma_1,\sigma_2)V_e^\top$, $\sigma_1>\sigma_2$ | Matrix energy should reveal sensitive directions. |
| C4 near-identical experts | $A_1\approx A_2$ | Low margin and unstable routing. |
| C5 clearly separated anisotropic experts | Different dominant right singular directions | Clear partition and positive margin. |

Seeds and samples:

- `expert_geometry_seeds = 100` for stochastic conditions.
- `input_samples_per_seed = 10000` on $S^1$.
- Optional `disk_samples_per_seed = 10000` for unit-disk robustness.

No checkpoints:
This is a static metric audit with no training.

## 6. Primary Metric

Primary metric:
`metric_validity`.

A metric-condition pair is valid only if it satisfies the relevant parts of this joint criterion:

- `non_degenerate = true`;
- `mean_margin > epsilon_margin`;
- natural load is acceptable, or calibrated load becomes acceptable while margin is preserved;
- predicted degenerate cases fail for the predicted reason;
- sign-sensitive metrics are correctly identified rather than treated as stable.

Use:

$$
\Delta_q(x)=q_{g(x)}(x)-q_{\mathrm{other}}(x).
$$

Why this metric decides the judgment:
The anchor question is not whether a partition exists visually. It asks whether an expert-geometry metric supports both uniformity and metric-defined specialization. `metric_validity` forces load, margin, degeneracy, and sign stability to be read together.

False-positive cost:
Accepting a balanced but arbitrary or sign-dependent split as expert specialization.

False-negative cost:
Rejecting a valid expert-geometry metric because raw natural load is imperfect even though calibration preserves margin.

## 7. Secondary Metrics

- `natural_load_error = |load_1 - 0.5|`
- `calibrated_load_error = |calibrated_load_1 - 0.5|`
- `mean_margin = E[q_selected(x) - q_other(x)]`
- `median_margin`
- `low_margin_fraction = P[margin < epsilon_margin]`
- `degeneracy_rate = P(|q_1(x)-q_2(x)| < epsilon_degenerate)`
- `sign_sensitivity = P[g_original(x) != g_sign_flipped(x)]`
- `prototype_angle`
- `dominant_direction_angle`
- `condition_failure_reason`

## 8. Known Good / Known Bad / Known Confusing Cases

Known good:
Clearly separated anisotropic experts.

Expected:
M4 matrix response energy should produce a clear partition, positive mean margin, and interpretable dominant-direction behavior.

Known bad:
Orthogonal equal-singular-value experts.

Expected:
For M4, $G_e=I$, so $q_1(x)=q_2(x)$ for all $x$ and the metric should be marked degenerate.

Known bad:
Near-identical experts.

Expected:
Margins should be small and routing should be unstable or low-confidence.

Known confusing:
Signed prototype similarity.

Expected:
It may partition the input but can change under arbitrary sign flips, so it should not be accepted as a stable matrix-geometry metric without the sign-sensitivity audit.

Known confusing:
Full-span projection energy in 2D.

Expected:
If both experts use the full two-dimensional span, projection energy to the span is uninformative because both spans equal $\mathbb{R}^2$.

Known confusing:
Calibrated load.

Expected:
Calibration can equalize load but should not be counted as specialization unless positive margin and non-degeneracy survive.

## 9. Stage-Level Profiling Plan

| Stage | Local question | Input evidence | Pass / fail / unclear rule | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S1 expert construction | Are matrices valid for the intended condition? | matrix norms, singular values, dominant directions | pass if condition labels match generated geometry | `tables/condition_summary.csv` | continue or mark invalid |
| S2 metric computation | Are all $q_e(x)$ scores finite and well-defined? | score ranges and missing values | fail if undefined prototypes or invalid scores are not flagged | `tables/aggregate_metric_summary.csv` | exclude invalid metric-condition rows |
| S3 natural assignment | Does the metric induce non-degenerate routing? | natural load, mean margin, degeneracy rate | pass if margin is positive and degeneracy low outside known-bad cases | `tables/aggregate_metric_summary.csv` | evaluate validity |
| S4 sign audit | Does the metric depend on arbitrary signs? | assignment change under sign flips | pass if sign sensitivity is low or explicitly marked as a weakness | `tables/sign_sensitivity.csv` | bound prototype claims |
| S5 calibration audit | Can load be balanced without destroying margin? | calibrated load and margin | pass if load improves and margin survives | `tables/calibration_summary.csv` | report as secondary |
| S6 visualization | Is the partition interpretable in 2D? | assignment maps and boundaries | pass if central cases have readable partition figures | `figures/partition_M*_C*.png` | include in summary after results |
| S7 metric selection | Which metric should move forward? | all prior tables and figures | pass if exactly one next metric decision is justified | `summary.md` after results | update anchor after result |

## 10. Algorithm Specification, If Nontrivial

input:

- number of expert seeds;
- number of input samples;
- expert geometry condition list;
- metric candidate list;
- load and margin thresholds.

parameters:

- `expert_geometry_seeds = 100`;
- `input_samples_per_seed = 10000`;
- `epsilon_margin = 1e-6`;
- `epsilon_degenerate = 1e-6`;
- `acceptable_natural_load_error = 0.05`;
- `acceptable_calibrated_load_error = 0.02`.

intermediate variables:

- $A_e$, $U_e$, $\Sigma_e$, $V_e$, $G_e$;
- oriented signs $s_{e,k}$ and prototype $m_e$ for M1/M2;
- $V_{e,1}$ for M3 top-1 projection and $V_{e,1:2}$ for M3 full-span control;
- $q_e(x)$ for M1--M4;
- $g(x)$ and $g_\tau(x)$;
- $\Delta_q(x)$;
- natural load, calibrated load, margin, degeneracy, sign sensitivity.

steps:

1. Generate $A_1,A_2$ for each expert condition and seed.
2. Compute SVD and verify the intended condition.
3. Sample $x$ uniformly from $S^1$.
4. Optionally sample from the unit disk for secondary robustness.
5. Compute metric scores exactly as follows:
   - M1: $m_e=\mathrm{normalize}(\sum_k\sigma_{e,k}s_{e,k}v_{e,k})$, then $q_e(x)=x^\top m_e$.
   - M2: reuse $m_e$, then $q_e(x)=(x^\top m_e)^2$.
   - M3-top1: $q_e(x)=\|V_{e,1}^\top x\|^2$.
   - M3-fullspan-control: $q_e(x)=\|V_{e,1:2}^\top x\|^2$.
   - M4: $G_e=A_e^\top A_e$, then $q_e(x)=x^\top G_ex$.
6. Compute natural assignment $g(x)$.
7. Compute natural load, margin, low-margin fraction, and degeneracy.
8. Apply sign-flip variants to M1/M2 by changing $s_{e,k}$ and report assignment change rate.
9. Fit calibration thresholds $\tau_e$ only on the sampled distribution.
10. Compute calibrated load and calibrated margin.
11. Generate partition figures for central metric-condition pairs.
12. Write metric and condition summary tables.
13. Decide which metric, if any, should become the core model for the next stage.

outputs:

- `tables/condition_summary.csv`
- `tables/sign_sensitivity.csv`
- `tables/calibration_summary.csv`
- `tables/aggregate_metric_summary.csv`
- `tables/metric_decision_summary.csv`
- `figures/partition_M*_C*.png`
- post-result `summary.md`
- post-result `detailed.md`

debug artifacts:

- singular value table;
- dominant-direction angle table;
- score histogram by metric and condition;
- partition visualizations.

pass conditions:

- At least one metric, preferably M4 matrix response energy, is non-degenerate in anisotropic expert cases.
- The selected metric has positive mean margin.
- The selected metric has interpretable 2D partitions.
- Degenerate cases fail for predicted reasons.
- Load can be natural or calibrated without destroying margin.

fail conditions:

- Only prototype metrics work while matrix response energy fails in the anisotropic cases.
- All useful partitions depend on arbitrary sign choices.
- Balanced load appears only after calibration and margin collapses.
- Expert geometry cannot distinguish inputs except in hand-picked cases.

failure reasons:

- physical prior false: initialized expert matrices do not provide usable input geometry;
- mathematical model false: the candidate $q_e(x)$ family cannot express useful compatibility;
- operationalization false: a candidate metric discards essential matrix structure or is sign-arbitrary;
- metric false: load and margin do not capture the intended judgment;
- implementation false: generated conditions or score computations do not match the protocol.

## 11. Success / Failure / Insufficient Evidence

Success:

- At least one metric, preferably matrix response energy, gives non-degenerate assignments in anisotropic expert cases.
- The metric gives positive mean margin and interpretable partitions.
- It fails predictably in equal-isotropic or near-identical expert cases.
- It separates natural load from calibrated load.
- Calibration, if used, preserves margin.

Failure:

- Matrix response energy fails in the cases where it should capture anisotropic matrix sensitivity.
- Prototype metrics are the only useful metrics, and their success depends on arbitrary signs.
- Calibration is required for all useful-looking results and destroys margin.
- Degenerate cases are not distinguishable from valid cases.

Insufficient evidence:

- Only plots are produced without margin and load tables.
- Only one expert condition is tested.
- Degenerate cases are omitted.
- Natural and calibrated load are mixed.
- Training is introduced before static metric validity is resolved.

## 12. What This Cannot Claim

This protocol cannot claim:

- real MoE routing collapse is solved;
- semantic feature specialization is achieved;
- training will preserve the partition;
- high-dimensional approximation works;
- common removal is sufficient;
- route assignment is causally bound to expert utility.

## 13. Approval Status

status: approved_and_executed  
approved_to_run: yes  
approval_source: user request on 2026-06-16  
execution_status: completed  
result_summary: `summary.md`  
detailed_record: `detailed.md`
