---
experiment_id: A15_00_03_E01_gate_transferable_vs_local_residual_alignment
status: approved_for_full_execution
created: 2026-07-31
approval_date: 2026-07-31
primary_anchor: 15_00_03_gate_transferable_vs_local_residual_alignment
canonical_protocol: protocol.md
companion_protocol: protocol_cn.md
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: local-2xh100-remote-8x5090-fallback
---

# Protocol: Gate Preference for Pooled Versus Shard-Local Residual Modes

## 0. Approval Snapshot

The researcher approved this scientific contract and authorized implementation,
smoke, and full frozen execution on 2026-07-31. This authorization does not
include new training.

**Purpose:** After removing input-energy differences, determine whether the
decommon Gate gives an independently pooled centered-common candidate greater
expert-relative gain and native-route dependence than equal-dimensional
shard-local residuals.

**Primary anchor:** [A15_00_03](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_03_gate_transferable_vs_local_residual_alignment_anchor.md).

**Experiment role:** Frozen root-cause audit plus saved-checkpoint macro
trajectory. It is not stepwise training dynamics.

**Primary metrics:**

$$
B_{\ell,P:L}
=\log\frac{G_\ell(W,U_P)+\epsilon}
{\operatorname{median}_sG_\ell(W,U_{L_s})+\epsilon},
$$

and the pooled-minus-local native-winner margin support,
$\Delta D_{\ell,P:L}$.

**Registered minimum:**

- exactly the same LB/decommon 30k/40k/80k checkpoints as A15_00_02;
- exactly the same 128 pooled and 512 confirmation DCLM documents;
- the same actual-input cache, pooled basis, and local residual bases;
- 80k primary, 40k replication, 30k macro support;
- all 12 layers and a fair 64-versus-64-dimensional comparison.

**Pass:** At decommon 80k, the lower paired document/basis-bootstrap 95% bound
of the model-level median $B_{P:L}$ is above zero and the observed value
exceeds its singular-value-preserving orientation-null q95. The lower bound of
$\Delta D_{P:L}$ is also above zero. The 40k point estimates must have the same
sign. Top-1 flip is auxiliary only.

**Fail:** The registered measurement is valid and precise, but $B_{P:L}$ does
not prefer pooled or does not beat the null, or native-route support does not
prefer pooled.

**Insufficient:** Any hard guard on the shared object, basis cross-fit,
orientation null, equal-rank ablation, checkpoint crossing, route replay, or
precision fails.

**Authorization decision:** Implementation, smoke, and parallel full frozen
execution with A15_00_02 E01 are authorized. Local 2xH100 is primary; remote
8x5090 is a time/capacity fallback only. No new training is included.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete computation | Unit | Decision role | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| expert-relative Gate $\bar W$ | weights that change expert comparisons | $(I-\mathbf1\mathbf1^\top/E)W$ | weight | remove common logit shift | functional utility |
| pooled candidate $U_P$ | centered top-64 from independent pooled documents | shared A15_00_02 split | 64 directions | candidate common component | stability |
| local residual $U_{L_s}$ | shard-local top-64 after removing $U_P$ | cross-fitted residual PCA | 64 directions | local comparator | semantic specificity |
| equal-energy gain $G$ | expert-score difference caused by equal input energy | $\|\bar WU\|_F^2/64$ | logit²/activation² | remove eigenvalue amplification | native token use |
| $B_{P:L}$ | pooled gain relative to median local gain | log gain ratio | dimensionless | primary Gate preference | training benefit |
| margin support $D_A$ | support a subspace gives the native winner over the runner-up | native margin before minus after ablation | logit/token | primary native-use guard | winner correctness |
| route flip | whether removing a subspace changes the winner | top-1 identity change rate | token fraction | auxiliary intuition | benefit |
| orientation null | same Gate singular values, random input orientation | Haar rotation of right-singular space | null distribution | exclude Gate spectrum alone | all training rivals |
| fixed-basis Gate effect | change $W$ while holding $U$ fixed | $B(W_b,U_a)-B(W_a,U_a)$ | log-ratio change | macro weight effect | per-step gradient |

## 2. Anchor Alignment

- **Decision question:** Does the equal-energy Gate prefer the pooled candidate
  over local residuals?
- **Physical prior:** Repeated cross-data directions provide coherent
  cumulative updates, whereas rotating local directions may average out.
- **Core term:** $G(W,U_P)$ versus $G(W,U_{L_s})$.
- **Main falsifier:** $B_{P:L}$ does not beat zero/null, or route-use evidence
  does not prefer pooled.
- **Claim boundary:** Gate preference and native use only; no independent
  stability, semantic, functional, or training claim.

## 3. Tested Hypothesis

**H1:** At decommon 80k, the Gate gives the pooled candidate greater
expert-relative gain per input direction than local residuals; 40k has the
same sign. Removing the pooled candidate also removes more native-winner margin
support than removing an equal-rank local residual. Flip is auxiliary. A
same-direction LB result indicates cross-lineage replication, not a
center-versus-LB causal effect.

The checkpoint analysis asks how saved $W$ and $U$ jointly change
$B_{P:L}$. Monotonic strengthening is not required.

## 4. Rival Explanations

| Rival | Prediction | Separating test | Maximum conclusion |
| --- | --- | --- | --- |
| R0 input energy only | raw response high but $G/B$ not high | equal-energy $G$ | excludes eigenvalue amplification |
| R1 Gate anisotropy only | arbitrary orientations produce extreme ratios | singular-value-preserving null | direction specificity |
| R2 pooled-estimator privilege | Haar or wrong-layer pooled bases look equal | full/complement Haar and wrong layer | estimator control |
| R3 basis drift only | endpoint changes disappear with fixed $U$ | full $3\times3$ crossing | saved-state decomposition |
| R4 geometry but not route use | $B>0$ but margin support is not higher | equal-rank ablation | weight geometry versus native use |
| R5 pooled direction is useful | all metrics may pass | outside this Protocol | retained functional rival |

## 5. Data / Model / Algorithm / Objective

### 5.1 Shared data and basis contract

This experiment must not select new documents or fit an alternative basis. It
reads the pre-registered A15_00_02 objects:

- 128 pooled-basis documents;
- eight 64-document confirmation shards;
- each shard's 32 fit and 32 evaluation split;
- actual-input cache;
- every checkpoint/layer's $U_P$ and $U_{L_s}$;
- checkpoint, data, cache, and basis hashes.

If the shared cache or basis hard guard fails, this experiment stops. Its
matrix analysis may run before A15_00_02 has a scientific verdict.

### 5.2 Models and checkpoints

| Lineage | Checkpoints | Priority | Maximum conclusion |
| --- | --- | --- | --- |
| decommon | 30k/40k/80k | 80k primary, 40k replication, 30k support | running-center lineage preference |
| LB | 30k/40k/80k | same | descriptive cross-lineage boundary |

### 5.3 Actual input and DC

Every subspace contribution uses $x=r-\mu^{fit}$. The DC term
$C_EW\mu^{fit}$ is reported separately and never enters a 64-D band. For
decommon, capture both $g$ and $r=g-c$ and reuse A15_00_02's translation
invariance guard.

## 6. Conditions, Seeds, And Checkpoints

| Condition | Rival protected | Purpose | Evidence role | Passing observation | Artifact |
| --- | --- | --- | --- | --- | --- |
| pooled $G_P$ | R0 | candidate gain | primary | higher than local/null | gain table |
| local $G_{L_s}$ | R0/R2 | equal-rank comparator | primary | pooled gap positive | shard table |
| $B_{P:L}$ | R0/R1 | primary preference | primary | CI $>0$ and above q95 | layer heatmap |
| pooled ablation | R4 | native margin support | primary support | support is higher | route table |
| local ablation | R4 | fair rank-64 comparison | primary support | paired gap positive | paired plot |
| orientation null, 256 draws | R1 | preserve Gate singular values | hard control | observed exceeds q95 | null ledger |
| full/complement Haar | R2 | basis estimator control | control | real bases are higher | null table |
| wrong layer $+6$ | R2 | layer specificity | control | target layer is higher | layer table |
| 30k/40k/80k crossing | R3 | weight versus basis change | secondary | typed decomposition | crossing matrix |

Haar seeds, basis hashes, and checkpoint order are frozen before $B$ is read.
Layers and checkpoints are not independent seeds.

## 7. Primary Metric

For each layer and checkpoint:

$$
G_\ell(W,U)=\frac1{64}\|C_EW_\ell U\|_F^2,
$$

$$
B_{\ell,P:L}
=\log\frac{G_\ell(W,U_P)+\epsilon}
{\operatorname{median}_{s=1}^{8}G_\ell(W,U_{L_s})+\epsilon}.
$$

For $P_A=U_AU_A^\top$, remove only the centered contribution:

$$
z_{\ell}^{(-A)}
=z_\ell-C_EW_\ell P_Ax_\ell,
$$

$$
D_{\ell,A}
=\mathbb E\!\left[
m_{\mathrm{native}}(z_\ell)
-m_{\mathrm{native}}(z_\ell^{(-A)})
\right],
\qquad
\Delta D_{\ell,P:L}
=D_{\ell,P}-\operatorname{median}_sD_{\ell,L_s}.
$$

$D$ is in logit per token. $\Delta D>0$ means the pooled candidate gives the
native winner more margin support; it does not mean that winner is better for
language modeling.

The model-level summary is the median over 12 layers, with every layer
reported. Paired basis/document bootstrap resamples pooled and shard-fit
documents. Each orientation-null sample preserves the Gate's nonzero singular
values, rotates its input right-singular directions, and recomputes the full
$B_{P:L}$.

**Why it answers the question:** $G$ contains no covariance eigenvalue.
Therefore, $B_{P:L}>0$ means the Gate weights give the pooled candidate more
expert-relative squared gain per direction.

**False-positive cost:** Raw response alone confounds direction preference
with input magnitude. $B$ alone may describe a geometry that never affects
the winner. Both gain and margin-support gates are required.

## 8. Secondary Metrics

1. pooled/local raw response $V$, which includes real token energy;
2. equal-rank native-winner support and auxiliary top-1 flip;
3. DC expert-bias norm $\|C_EW\mu\|$;
4. full $W_{30/40/80}\times U_{30/40/80}$ crossing;
5. fixed-basis $\Delta_WB_{P:L}$ and fixed-Gate $\Delta_UB_{P:L}$;
6. fine gain profiles for localization only;
7. logical-batch-regrouped local-residual sensitivity;
8. descriptive LB/decommon differences, never a center-causal metric.

## 9. Known Good / Known Bad / Known Confusing Cases

- Replacing $U$ with the Gate's top right-singular directions must raise $G$
  relative to orthogonal directions.
- Orientation-null percentiles should be approximately uniform, and the SVD
  reconstruction error must satisfy numerical tolerance.
- $G_P>G_L$ need not imply $V_P>V_L$: the former is selectivity, the latter
  also contains input energy.
- Route effects across subspaces are not additive because the winner is a
  nonlinear argmax.
- An A15_00_02 scientific Fail does not make $B_{P:L}$ undefined; it only
  forbids naming pooled/local as stable/unstable.

## 10. Stage-Level Profiling Plan

| Stage | Local question | Pass/fail/unclear | Debug artifact | Handoff |
| --- | --- | --- | --- | --- |
| S0 | do shared objects match? | hashes match/stop | shared manifest | S1 |
| S1 | are $G/B$ implementations valid? | known cases and null valid/stop | unit ledger | S2 |
| S2 | does the endpoint prefer pooled? | typed H1/R0/R1 | gain tensors | S3 |
| S3 | does the native route use it? | typed H1/R4 | route ledger | S4 |
| S4 | is change due to $W$ or $U$? | weight/basis/mixed | crossing table | S5 |
| S5 | is joint wording licensed? | combined/separate | typed verdict | result record |

S2--S4 may run in parallel with A15_00_02's stability analysis. S5 alone reads
the other experiment's typed verdict.

**Resource plan:** This experiment does not repeat model forward passes. It
reads A15_00_02 S1's shared activation/Gate cache. Matrix, null, and route
ablation analyses may use otherwise idle RTX 5090 GPUs, but no eight-GPU
training job is required.

## 11. Algorithm Specification

**Inputs:** Shared frozen caches and bases, Gate matrices, and held-out
evaluation documents.

**Parameters:** 64 dimensions; 256 orientation-null draws; 2,000 paired
document/basis-bootstrap replicates; numerical $\epsilon$ frozen before
execution.

**Steps:**

1. verify shared manifests, basis orthogonality, and Gate shapes;
2. compute $G$ for pooled, eight local residual, Haar, and wrong-layer bases;
3. compute layerwise $B_{P:L}$ and orientation-null percentiles;
4. run centered equal-rank ablations on identical evaluation tokens;
5. compute margin support, flip, raw response, and DC;
6. complete all 30k/40k/80k $W\times U$ crossings;
7. bootstrap, aggregate, render, audit, and write typed verdicts.

**Outputs:** gain, route, null, and crossing tables; layer heatmap; checkpoint
trajectory; standalone typed verdict; conditional joint verdict.

**Failure reasons:** shared object, basis, SVD/null, route replay, precision,
checkpoint coordinates, or bootstrap precision.

### 11.1 Central figure contract

- **File:** `pooled_vs_local_gate_preference_and_use.png`
- **Question:** Does the equal-energy Gate prefer the pooled candidate, and
  does the native route rely on it more?
- **Metric/unit:** left, $B_{\ell,P:L}$ in log ratio; right,
  pooled-minus-local margin support in logit/token; flip difference in token
  fraction is auxiliary.
- **Data:** locked shared bases, Gate matrices, and confirmation evaluation
  documents.
- **Aggregation:** median over eight local bases; paired basis/document
  bootstrap 95% CI; layers are not seeds.
- **Axes:** x = layer; y = metric; color = checkpoint; facets = lineage and
  metric.
- **H1 pattern:** $B>0$ and margin-support difference $>0$.
- **Weakening pattern:** raw response only, $B\le0$, or $B>0$ without positive
  margin support.
- **Allowed conclusion:** equal-energy Gate geometry and native dependence.
- **Limitation:** no independent stability, semantics, function, or training
  utility.
- **Observed:** pending execution.

## 12. Success / Failure / Insufficient Evidence

- **Pass — pooled alignment with native use:** Both $B$ and margin-support
  gates pass.
- **Alignment-only:** $B$ passes but margin support does not; H1 as a whole
  fails and only weight geometry is retained.
- **Energy-only:** Raw response prefers pooled but $B$ fails.
- **Local-not-weaker:** Local residual gain or margin support is not weaker.
- **Lineage-conditioned:** LB and decommon disagree; report separately.
- **Insufficient:** Any hard guard or precision condition fails.

Only a Pass here plus a Pass in A15_00_02 licenses:

> The decommon Gate prefers a cross-document stable centered component over a
> less stable local residual.

## 13. What This Cannot Claim

This experiment cannot establish:

1. that the pooled candidate transfers across documents;
2. that pooled preference helps or hurts loss;
3. that local residuals contain no semantics;
4. that residual instability causes optimization failure;
5. that center alone causes LB/decommon differences;
6. that checkpoint crossing equals online gradient dynamics.

Even a joint Pass does not license a training-benefit claim. A matched
stability intervention would require a later, separately approved anchor and
Protocol.

## 14. Review Notes And Protocol Changes

The researcher approved on 2026-07-31:

1. a 64-versus-64 pooled/local comparison before broader residual coverage;
2. the two-gate $B_{P:L}$ plus equal-rank margin-support verdict;
3. a singular-value-preserving orientation null and separate Haar estimator
   controls;
4. 80k primary, 40k replication, and 30k macro support;
5. allowing parallel computation while A15_00_02 controls only joint naming;
6. a frozen audit with no pre-authorized eight-GPU training.

**Post-approval changes:** Generated the English canonical Protocol and marked
the Chinese companion approved. The researcher then authorized implementation,
smoke, and full frozen execution on 2026-07-31. No scientific condition changed.
