---
anchor_id: A06_2x2_expert_geometry_gate
parent_node: rq.expert_specialization
status: active_result_updated
created: 2026-06-16
canonical_language: en
---

# A06 2x2 Expert-Geometry Gate Anchor

## 0. Tiny Summary

父问题：Switch-style MoE 的 gate 如何和 expert 初始化结构发生联系，而不是作为与 expert 无关的随机分类器。

当前子问题：在两个 expert、二维输入、每个 expert 是一个 $2\times2$ 矩阵的最小情形下，哪一种由 expert 初始化诱导的度量可以定义不退化的 static oracle routing rule。

核心结论：A06_E01 支持 matrix response energy $q_e(x)=x^\top A_e^\top A_ex$ 作为当前最合适的 $2\times2$ expert 几何度量；M3 top-1 projection 是低秩近似候选；prototype 类度量只保留为 baseline。

最小证伪测试：比较 signed prototype、unsigned prototype、projection energy、matrix response energy，在随机、各向同性、各向异性、近同 expert、清晰分离 expert 条件下检查 load、margin、degeneracy 和 sign sensitivity。

当前边界：这只是 static oracle metric audit，不含训练，不证明真实 MoE、语义 specialization、高维 approximation 或 common removal。near-identical experts 必须读 margin 大小，不能只读二元 `metric_validity`。

下一步决策：测试一个 router-construction rule 能否近似 M4 oracle，同时保留 load 和 margin；M3 top-1 projection 作为低秩近似对照。

## 1. Problem Definition

Parent problem:
How can gating make MoE experts form stable, interpretable, reusable feature-level specialization?

Sharper subproblem:
Before studying training or high-dimensional approximation, define the minimal expert-aware routing metric in the two-expert, two-dimensional case where each expert is represented by a $2\times2$ initialization matrix.

Decision question:
In the minimal $2\times2$ setting, which expert-initialization-induced compatibility metric can define a non-degenerate routing rule that supports both balanced load and metric-defined specialization under uniform input?

Not in scope:
Training dynamics, transformer hidden states, real text, common component removal, high-dimensional approximation, semantic feature labels, and causal expert utility.

Operational definitions:

- Expert geometry means the input-side directions and sensitivities encoded by an initialized expert matrix $A_e\in\mathbb{R}^{2\times2}$.
- Compatibility metric means a scalar score $q_e(x)$ that measures how compatible input $x$ is with expert $e$'s initialization geometry.
- Static oracle routing rule means an assignment rule $g(x)=\arg\max_e q_e(x)$ computed directly from the candidate metric, before any trainable router is introduced.
- Metric-defined specialization means the selected expert has higher compatibility score than the other expert for the assigned input.
- Natural load means the load induced by $g(x)=\arg\max_e q_e(x)$ without thresholds.
- Calibrated load means the load induced by $g_\tau(x)=\arg\max_e(q_e(x)-\tau_e)$ after fitting thresholds only to equalize load on the sampled distribution.

## 2. Physical Priors

P1:
Expert-aware routing should be grounded in expert-side input geometry.

Meaning:
If experts are expected to specialize, a first-principles gate should measure how an input direction interacts with each expert's initialized matrix, not only assign tokens with an expert-agnostic random classifier.

Could be wrong if:
The initialized expert matrices provide no stable distinguishable geometry, or every geometry-derived metric becomes arbitrary or degenerate.

P2:
The full $2\times2$ matrix structure may matter.

Meaning:
A single prototype direction may discard the anisotropy and input sensitivity encoded by the matrix. A matrix response metric such as $x^\top A_e^\top A_e x$ is therefore a more direct candidate than a prototype alone.

Could be wrong if:
Prototype metrics and matrix response metrics make the same useful assignments, or the matrix response metric is always degenerate in the relevant conditions.

P3:
Uniform input requires separating load balance from specialization.

Meaning:
A useful metric should either naturally route close to 50/50 under uniform input or admit calibration without destroying positive compatibility margin. Load balance alone is not evidence of specialization.

Could be wrong if:
Every balanced rule has near-zero margin, or every high-margin rule is so imbalanced that it cannot support a uniform expert split.

## 3. Falsifiable Hypotheses

H1:
Matrix response energy is the main expert-matrix geometry candidate.

Supported if:
$q_e(x)=x^\top A_e^\top A_e x$ gives non-degenerate assignments, positive mean margin, interpretable partitions, and predictable failures in isotropic or near-identical expert cases.

Weakened if:
Matrix response energy cannot distinguish inputs except in hand-picked cases, or it only works after calibration that destroys the margin.

H2:
Prototype metrics are useful baselines but may be too thin.

Supported if:
Signed prototype metrics can produce partitions but are sign-sensitive, and the prototype family discards anisotropy or fails to match the behavior of matrix-sensitive expert conditions.

Weakened if:
Prototype metrics are robust, sign-stable, and match or exceed matrix response energy across the decisive expert conditions.

H3:
Calibration can repair load but cannot create specialization.

Supported if:
Threshold calibration improves load balance while preserving a positive margin only for metrics that were already non-degenerate.

Weakened if:
Calibration makes degenerate metrics look valid, or metric validity cannot be separated from threshold fitting.

## 4. Mathematical Model

Objects:

Two experts are matrices $A_1,A_2\in\mathbb{R}^{2\times2}$. The convention is that $A_e$ maps a 2D input $x$ to the expert response $A_ex$. Each expert has singular value decomposition:

$$
A_e=U_e\Sigma_eV_e^\top .
$$

The input-side directions are the right singular vectors $v_{e,k}$, i.e. the columns of $V_e$. The singular values $\sigma_{e,k}$ measure how strongly expert $e$ responds to those input directions. The matrix-response energy is represented by:

$$
G_e=A_e^\top A_e=V_e\Sigma_e^2V_e^\top .
$$

The primary input distribution is:

$$
x\sim\mathrm{Unif}(S^1).
$$

Uniform input over the unit disk is a secondary robustness check.

Core decomposition:

The core object is a family of compatibility metrics:

$$
q_e:\mathbb{R}^2\to\mathbb{R}.
$$

The natural oracle assignment is:

$$
g(x)=\arg\max_{e\in\{1,2\}} q_e(x).
$$

The calibrated assignment is secondary:

$$
g_\tau(x)=\arg\max_{e\in\{1,2\}} \left(q_e(x)-\tau_e\right),
$$

where $\tau_e$ is chosen only to equalize load on the sampled distribution.

Candidate metrics:

- M1 signed prototype similarity. Choose oriented signs $s_{e,k}\in\{-1,+1\}$ for the right singular vectors and define $m_e=\mathrm{normalize}(\sum_k \sigma_{e,k}s_{e,k}v_{e,k})$; then $q_e(x)=x^\top m_e$. This is the sign-sensitive linear baseline.
- M2 unsigned prototype similarity. Use the same $m_e$, but define $q_e(x)=(x^\top m_e)^2$. This removes global sign but still depends on the relative signs used to build $m_e$.
- M3 projection energy. Let $V_{e,r}$ contain the selected top $r$ right singular direction or directions. Define $q_e(x)=\|V_{e,r}^\top x\|^2$. The primary non-degenerate read is $r=1$; $r=2$ is an explicit full-span degenerate control in 2D.
- M4 matrix response energy. Define $G_e=A_e^\top A_e$ and $q_e(x)=x^\top G_e x=\|A_ex\|^2$.

Mechanism relation:
If expert initialization geometry contains distinguishable input sensitivity, then at least one $q_e$ should divide the uniform 2D input space into interpretable regions where the selected expert has a positive score margin:

$$
\Delta_q(x)=q_{g(x)}(x)-q_{\mathrm{other}}(x).
$$

Observable metrics:
Natural load error, calibrated load error, mean margin, low-margin fraction, degeneracy rate, sign sensitivity, partition interpretability, and predicted failure behavior in isotropic or near-identical expert cases.

Falsifier:
If all metrics either degenerate, depend on arbitrary sign choices, lose margin under load calibration, or fail to distinguish expert conditions, then the current expert-geometry metric model is insufficient.

## 5. Computational Realization

Input objects:

- Expert geometry conditions: random unconstrained matrices, orthogonal equal-singular-value matrices, orthogonal unequal-singular-value matrices, near-identical experts, and clearly separated anisotropic experts.
- Input samples from $S^1$, with optional unit-disk samples as secondary.
- Metric candidates M1, M2, M3, and M4.

Computed variables:

- $A_e$, $U_e$, $\Sigma_e$, $V_e$, $G_e$.
- For M1/M2: oriented right-singular-vector prototype $m_e$ and its sign-flip variants.
- For M3: $V_{e,1}$ for top-1 projection and $V_{e,1:2}$ for the full-span degenerate control.
- For M4: $G_e=A_e^\top A_e$.
- $q_e(x)$ for each metric and condition.
- Natural assignment $g(x)$ and calibrated assignment $g_\tau(x)$.
- Load, margin, degeneracy, and sign sensitivity statistics.

Algorithm stages:

1. Generate expert matrices for each condition.
2. Verify singular values, dominant input directions, and condition labels.
3. Sample uniform input directions.
4. Compute all metric scores and natural assignments.
5. Measure natural load, mean margin, low-margin fraction, and degeneracy.
6. Run sign-flip checks for M1/M2 prototype metrics before accepting any prototype result.
7. Apply optional load calibration and test whether margin survives.
8. Produce 2D partition visualizations.
9. Select the metric that should become the anchor's core model for the next stage.

Stage-local evidence:
Each stage must produce either a table row, a figure, or a failure reason that explains whether the metric is valid, degenerate, sign-unstable, load-imbalanced, or only calibration-dependent.

Expected artifacts:

- `tables/metric_summary.csv`
- `tables/condition_summary.csv`
- `tables/sign_sensitivity.csv`
- `tables/calibration_summary.csv`
- `figures/partition_M*_C*.png`
- post-result `summary.md`
- post-result `detailed.md`

## 6. Minimal Falsification Tests

Test:
Metric candidate audit.

Question:
Which candidate $q_e(x)$ gives non-degenerate routing with positive margin under meaningful $2\times2$ expert geometry?

Intervention / comparison:
Compare M1 signed prototype, M2 unsigned prototype, M3 projection energy, and M4 matrix response energy across random, anisotropic, and clearly separated expert conditions.

Primary metric:
`metric_validity`, defined by non-degeneracy, positive mean margin, acceptable natural load or calibrated load with margin preserved, and correct behavior in known degenerate cases.

Supports:
At least one metric, preferably M4, produces interpretable partitions with positive margin and usable load behavior.

Weakens:
Only arbitrary prototype rules work, all useful metrics require arbitrary sign choices, or matrix response energy fails in the anisotropic cases it should explain.

Insufficient evidence:
Only plots are produced, only one condition is tested, or load is reported without margin and degeneracy.

Test:
Degenerate and rival geometry checks.

Question:
Do the metrics fail for the predicted reasons when expert geometry contains no usable distinction?

Intervention / comparison:
Use orthogonal equal-singular-value experts, near-identical experts, full-rank subspace projection, and singular-vector sign flips.

Primary metric:
`failure_reason_accuracy`, meaning the observed failure matches the predicted degeneracy or sign-instability.

Supports:
Equal-isotropic or near-identical experts have low margin or degenerate assignment; sign-sensitive metrics change under sign flips; full-rank projection energy is uninformative when both experts span $\mathbb{R}^2$.

Weakens:
Degenerate cases look valid, or sign-sensitive metrics are treated as stable without explanation.

Insufficient evidence:
Degenerate controls are omitted.

Test:
Natural-vs-calibrated load split.

Question:
Is load balance a natural consequence of the metric, or only a threshold-fitting artifact?

Intervention / comparison:
Report $g(x)$ and $g_\tau(x)$ separately for every metric and condition.

Primary metric:
Natural and calibrated load error, always read together with mean margin.

Supports:
Calibration improves load without destroying margin for a metric that already has non-degenerate natural assignments.

Weakens:
Balanced load appears only after calibration while margin collapses or degeneracy remains high.

Insufficient evidence:
Natural and calibrated loads are mixed into one number.

## 7. Current Evidence

Observation:
The 0615 advisor discussion requested a return to the smallest mathematical case: two experts, $2\times2$ matrices, and a rigorous distance or compatibility rule that partitions a uniform 2D input space.

Interpretation:
The immediate research need is a metric audit, not a training run. The first result should tell us which expert-initialization-induced metric is mathematically meaningful enough to scale.

Boundary:
This is a mentor-driven modeling target, not evidence that any specific metric already works.

Evidence links:

- `daily_research_reports/0615/next/0615.md`

Observation:
A06_01 showed that expert-spectrum information can be computed and contains weak residual utility signal, but compressing the oracle into a linear router was a bottleneck.

Interpretation:
The 2D audit should evaluate the compatibility metric before asking whether a linear router can implement it.

Boundary:
A06_01 is high-dimensional real-text evidence and does not prove the 2D metric. It only motivates why metric choice should precede implementation choice.

Evidence links:

- `Projects/from-attention-to-search/main/problem_anchors/06_01_common_filtered_expert_spectrum_router_initialization_anchor.md`
- `Projects/from-attention-to-search/main/experiments/A06_01_common_filtered_expert_spectrum_router_initialization/summary.md`

Observation:
A06_E01 completed the static metric audit across 5 expert geometry conditions, 5 metric variants, and 100 seeds per stochastic condition.

Interpretation:
M4 matrix response energy is the best current core metric. It has valid rate 1.00 in the unequal-singular-value and separated-anisotropic conditions, natural load error 0.0000 in both, and mean margin 1.5444 / 2.3873. It also fails correctly in the equal-singular-value condition, where $G_e=I$ gives degenerate scores.

Boundary:
This supports M4 only as a static oracle compatibility metric. It does not show that a trainable router can approximate M4 or that high-dimensional low-rank approximation preserves the same margin.

Evidence links:

- `summary.md`
- `detailed.md`
- `tables/metric_decision_summary.csv`

Observation:
M3 top-1 projection also succeeds in anisotropic conditions but has smaller margins than M4. In C5, M3 top-1 mean margin is 0.6366 versus M4 mean margin 2.3873.

Interpretation:
M3 top-1 is a plausible low-rank approximation target, not the full core metric.

Boundary:
It ignores singular-value scale and secondary input directions.

Evidence links:

- `tables/aggregate_metric_summary.csv`

Observation:
M1/M2 prototype metrics route cleanly in many non-degenerate cases but have substantial sign sensitivity: mean sign sensitivity is 0.5333 for M1 and 0.2633 for M2.

Interpretation:
Prototype metrics are useful baselines but are too sign-dependent to be the core expert-matrix geometry metric without an additional sign convention.

Boundary:
This does not rule out a future prototype metric if a principled sign convention is added.

Evidence links:

- `tables/sign_sensitivity.csv`

Observation:
Near-identical experts expose a metric caveat. M4 can pass the binary `metric_validity` criterion after calibration, but its C4 natural mean margin is only 0.0226.

Interpretation:
The binary valid/invalid criterion is too permissive for near-identical experts. Future metric reads should treat margin scale as part of confidence.

Boundary:
This is a metric-read caveat, not a falsification of M4 in anisotropic cases.

Evidence links:

- `detailed.md`

## 8. Claim Boundary And Next Decision

Can claim:

- M4 matrix response energy is supported as the core $2\times2$ expert-geometry compatibility metric.
- M3 top-1 projection is a plausible low-rank approximation candidate.
- Prototype metrics are baselines, not the core metric, because they are sign-sensitive and discard matrix anisotropy.
- Full-span projection is invalid in the 2D setting.
- Load balance and metric-defined specialization must be read jointly with margin scale.

Cannot claim:

- This solves real MoE collapse.
- This proves semantic feature specialization.
- This proves training will preserve the partition.
- This proves high-dimensional scalability.
- This proves common component removal is sufficient.
- This proves route-utility binding or expert causal utility alignment.

Next decision:
Test whether a router-construction rule can approximate the M4 oracle while preserving load and margin, with M3 top-1 projection as the low-rank approximation baseline.
