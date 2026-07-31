---
experiment_id: A15_00_02_E01_centered_common_subspace_stability
status: approved_for_full_execution
created: 2026-07-31
approval_date: 2026-07-31
primary_anchor: 15_00_02_centered_common_subspace_stability
canonical_protocol: protocol.md
companion_protocol: protocol_cn.md
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: local-2xh100-remote-8x5090-fallback
---

# Protocol: Cross-Document Stability of the Centered Common Subspace

## 0. Approval Snapshot

The researcher approved this scientific contract and authorized implementation,
smoke, and full frozen execution on 2026-07-31. This authorization does not
include new training.

**Purpose:** Determine whether the top-64 covariance subspace of the actual
Router input remains transferable across independent DCLM document groups
after de-meaning, and whether an equal-dimensional local residual subspace is
less transferable after an independently pooled top-64 is removed.

**Primary anchor:** [A15_00_02](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_02_centered_common_subspace_stability_anchor.md).

**Experiment role:** Frozen root-cause and metric audit. It does not train a
Router or an expert.

**Primary metric:** The held-out top-64 cross-capture gap above a
dimension-matched Haar q95, $\Gamma_{64}$, in activation-energy fraction.

**Registered minimum:**

- existing 12-layer, hidden-size-768, eight-expert, top-1 decommon and LB
  lineages;
- 80k primary endpoint, 40k replication, and 30k macro support;
- 128 independent pooled-basis documents;
- 512 independent confirmation documents in eight deterministic 64-document
  shards;
- the first 256 valid tokens per document;
- all documents disjoint from the Q1 and Q2 manifests;
- the actual Gate input at every layer.

**Pass:** At decommon 80k, the lower document-bootstrap 95% bound of the
model-level median $\Gamma_{64}$ is above zero, and the lower bound of
$\Gamma_{64}-\Gamma_{64}^{res}$ is above zero. The 40k point estimate must
have the same sign. LB is reported separately as a lineage replication.

**Fail:** The registered measurement is valid and precise, but the original
top-64 does not exceed the null, or the residual top-64 transfers equally well.

**Insufficient:** Any hard guard on document independence, actual-input
replay, center invariance, sample-size convergence, rank, numerical precision,
or bootstrap precision fails.

**Authorization decision:** Implementation, smoke, and parallel full frozen
execution with A15_00_03 E01 are authorized. Local 2xH100 is primary; remote
8x5090 is a time/capacity fallback only. No new training is included.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete computation | Unit | Decision role | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| upstream representation $g$ | Router reference before centering | hook the center input | activation | audit the transform | final Gate geometry |
| actual input $r$ | tensor received by the Gate | LB: $g$; decommon: $g-c$ | activation | only primary representation | expert-input geometry |
| shard mean $\mu_s$ | average Router state in a document group | mean over fit-document tokens | activation | isolate DC | centered stability |
| centered top-64 | strongest within-group variations after mean removal | top-64 covariance eigenspace | 64 directions | candidate centered-common space | semantic commonality |
| pooled top-64 $U_P$ | top-64 fitted on an independent document pool | pooled-basis split only | 64 directions | independently removable candidate | stability or utility |
| local residual-64 | strongest group-local modes after removing $U_P$ | residual covariance top-64 | 64 directions | candidate group-specific residual | semantic specificity |
| cross-capture $E_{s\to t}$ | target variation explained by a source basis | $\|X_tU_s\|_F^2/\|X_t\|_F^2$ | energy fraction | cross-document transfer | function |
| $\Gamma_{64}$ | transfer above a random 64-D direction | median $E_{s\to t}$ minus target Haar q95 | energy fraction | primary verdict | causal learning mechanism |
| projector overlap | overlap between two 64-D spaces | $\|U_s^\top U_t\|_F^2/64$ | $[0,1]$ | rotation diagnostic | target energy capture |
| mean dispersion | group-mean deviation from pooled mean | $\|\mu_s-\mu_P\|/\|\mu_P\|$ | ratio | protect against misleading cosine | centered stability |

## 2. Anchor Alignment

- **Decision question:** Does centered top-64 transfer across documents, and
  is residual top-64 less transferable?
- **Physical prior:** A fixed translation removes only DC. A shared Gate can
  accumulate reproducible directions more coherently than group-local ones.
- **Core term:** Held-out transfer contrast between $U_*$ and $\epsilon_s$ in
  $g=\mu+U_*a+\epsilon_s$.
- **Main falsifier:** Top and residual transfer are indistinguishable, or both
  fall inside the matched null.
- **Claim boundary:** Geometry only; no semantic, functional, or training
  claim.

## 3. Tested Hypothesis

**H1:** At decommon 80k, the centered top-64 transfers across document groups,
and its null-relative transfer exceeds that of local residual-64 after the
independently pooled top-64 is removed.

Expected pattern:

1. raw shard means may align, but this establishes DC only;
2. centered top-64 has $\Gamma_{64}>0$;
3. residual $\Gamma_{64}^{res}$ is lower or indistinguishable from zero;
4. 40k has the same direction; LB either replicates it descriptively or
   establishes a lineage-conditioned boundary.

## 4. Rival Explanations

| Rival | Prediction | Separating test | Maximum conclusion |
| --- | --- | --- | --- |
| R0 finite-sample PCA | apparent instability shrinks with more documents | 8/16/32 fit-document curve and half split | registered estimator convergence |
| R1 stable space is wider | top and residual both exceed null | equal-dimensional residual cross-capture | rank 64 is too narrow |
| R2 target spectrum alone | any 64-D direction captures high energy | target-specific Haar q95 and overlap | excludes random energy capture |
| R3 arbitrary shared layer | wrong-layer basis transfers equally | layer $+6$ basis | tests layer specificity |
| R4 shard topic accident | verdict changes across groupings | multiple ordered pairs, document bootstrap, logical-batch regrouping | limits grouping sensitivity |
| R5 stable but nonfunctional | geometry passes while functional admission fails | outside this Protocol | retained functional rival |

## 5. Data / Model / Algorithm / Objective

### 5.1 Models and checkpoints

| Lineage | Actual Gate input | Checkpoints | Role |
| --- | --- | --- | --- |
| decommon | $r=g-c$ with frozen saved running center | 30k/40k/80k | primary mechanism object |
| LB | $r=g$, center off, LB-trained | 30k/40k/80k | descriptive cross-lineage control |

Checkpoint roots, Gate shapes, and expert ordering must match A15_00 E01 and
be re-recorded with hashes before execution. LB and decommon are not a
single-variable causal comparison.

### 5.2 Document separation

Select 640 new DCLM held-out documents by document hash, each contributing its
first 256 valid tokens:

| Split | Documents | Use | Prohibited use |
| --- | ---: | --- | --- |
| pooled-basis | 128 | estimate $\mu_P,U_P$ | confirmation verdict |
| confirmation | 512 | eight shards of 64 | estimator tuning |

Each confirmation shard is deterministically split into 32 fit and 32
evaluation documents. All documents must be disjoint from Q1
calibration/evaluation and Q2 operationalization/fit/validation/final
manifests. If 640 eligible disjoint documents are unavailable, an amendment is
required before any activation metric is read.

### 5.3 Cross-fitting

- source bases read source-fit documents only;
- target capture reads target-evaluation documents only;
- target evaluation is centered by the target-fit mean;
- the pooled projector reads pooled-basis documents only;
- no token may fit a basis and evaluate its verdict.

## 6. Conditions, Seeds, And Checkpoints

| Condition | Rival protected | Purpose | Evidence role | Passing observation | Artifact |
| --- | --- | --- | --- | --- | --- |
| actual-input replay | wrong hook | validate object | hard guard | logits/top-1 replay | replay table |
| $g$ versus $g-c$ | center contamination | validate translation invariance | hard guard | centered covariance agrees within tolerance | invariance table |
| centered top-64 | R0/R2 | primary candidate | primary | $\Gamma>0$ | transfer heatmap |
| residual top-64 | R1 | equal-rank comparator | primary | paired gap positive | paired contrast |
| Haar-64, 256 orientations | R2 | random-direction null | hard control | real basis exceeds q95 | null ledger |
| wrong layer $+6$ | R3 | layer specificity | control | target-layer transfer is higher | layer table |
| 8/16/32 fit docs | R0 | estimator convergence | guard | direction stabilizes | sample curve |
| logical-batch regrouping | R4 | batch-composition sensitivity | secondary | boundary is typed if different | sensitivity table |
| 30k/40k/80k | checkpoint specificity | macro replication | support | 40k/80k same sign | checkpoint table |

Document hashes, PCA seeds, and Haar seeds must be frozen before the primary
result is read. Checkpoints and layers are not independent random seeds.

## 7. Primary Metric

For source shard $s$ and target shard $t$:

$$
E_{\ell,s\rightarrow t,64}
=\frac{\|X_{\ell,t}^{eval}U_{\ell,s,64}\|_F^2}
{\|X_{\ell,t}^{eval}\|_F^2}.
$$

Generate full-space Haar-64 bases for the original condition and Haar-64 bases
inside $U_P^\perp$ for the residual condition:

$$
\Gamma_{\ell,64}
=\operatorname{median}_{s\ne t}
\left[
E_{\ell,s\rightarrow t,64}
-q_{0.95}(E_{\ell,R\rightarrow t,64})
\right].
$$

The model-level summary is the median over all 12 layers, while every layer is
reported. Paired document-block bootstrap resamples source and target
documents and reports 95% intervals for $\Gamma_{64}$ and
$\Gamma_{64}-\Gamma_{64}^{res}$.

**Why it answers the question:** A direction is a transferable coordinate only
if a basis fitted on one document group captures centered variation in unseen
documents beyond a matched random direction.

**False-positive cost:** Mistaking finite-sample PCA for common structure would
incorrectly license a stability intervention. Independent pooling, q95, and
sample-size convergence are therefore mandatory.

## 8. Secondary Metrics

1. shard-mean cosine, centered mean-deviation cosine, and mean dispersion for
   both $g$ and $r=g-c$;
2. projector overlap $O_{s,t,64}$;
3. transfer profiles for $k\in\{16,32,128,256\}$;
4. pooled-basis energy capture in every shard;
5. deterministic logical-dataloader batch regrouping as sensitivity only;
6. fine eigenvalue spectra as interpretation, never as the verdict.

## 9. Known Good / Known Bad / Known Confusing Cases

- Within-shard fit/evaluation half splits should exceed Haar.
- Independent Haar source bases must not systematically exceed their null.
- Near-degenerate eigenvalues may rotate individual eigenvectors; the verdict
  is defined on projectors and captured energy, not vector signs or ordering.
- A large pooled mean can make raw mean cosine nearly one; mean dispersion and
  centered transfer remain mandatory.
- Residual capture is normalized by residual energy and is never mixed with
  raw total-energy response.

## 10. Stage-Level Profiling Plan

| Stage | Local question | Pass/fail/unclear | Debug artifact | Handoff |
| --- | --- | --- | --- | --- |
| S0 | are provenance and documents compatible? | compatible/amend/stop | manifests | S1 |
| S1 | are hooks and centering correct? | replay and invariance/stop | replay ledger | shared cache |
| S2 | is sample size adequate? | stable curve/unclear | convergence curve | S3 |
| S3 | do top and residual transfer? | typed H1/R0/R1 | transfer tensor | S4 |
| S4 | do real bases beat null/wrong-layer? | pass/fail/insufficient | null ledger | S5 |
| S5 | is one bounded verdict supported? | typed verdict | figures/tables | result record |

The read-only S1 activation cache is shared with A15_00_03. S2--S5 analyses
may execute in parallel.

**Resource plan:** If a later execution decision allocates eight RTX 5090
GPUs, the six lineage-by-checkpoint frozen forwards may use up to six GPUs.
The remaining GPUs are optional. Basis fitting, nulls, and bootstrap operate
on the shared cache. Fewer GPUs lower only parallelism, never conditions,
documents, layers, or controls. No optimizer, backward pass, or training is
part of this Protocol.

## 11. Algorithm Specification

**Inputs:** Frozen checkpoints, locked 640-document manifest, and actual-input
hooks.

**Parameters:** Primary $k=64$; 256 Haar orientations; fit-document counts
$\{8,16,32\}$; 2,000 document-bootstrap replicates.

**Steps:**

1. verify checkpoints, document disjointness, and actual-input no-op;
2. extract $g,r$ at every endpoint and layer with token/document mapping;
3. fit $\mu_P,U_P$ on pooled-basis documents;
4. fit $\mu_s,U_{s,64}$ on each shard's fit documents;
5. measure all ordered $s\ne t$ captures on target-evaluation documents;
6. repeat after projecting data into $(I-P_P)$;
7. compute full/complement Haar and wrong-layer controls;
8. run convergence, bootstrap, aggregation, and rendered-figure audit.

**Outputs:** checkpoint/data manifests, transfer tensors, null ledger,
mean/invariance tables, layer/checkpoint tables, central figure, and typed
verdict.

**Failure reasons:** object, leakage, finite-sample, rank, numerical, null,
precision, or checkpoint incompatibility.

### 11.1 Central figure contract

- **File:** `centered_common_vs_residual_transfer.png`
- **Question:** Does centered top-64 transfer across documents more strongly
  than equal-dimensional residual top-64?
- **Metric/unit:** $\Gamma_{\ell,64}$ and $\Gamma_{\ell,64}^{res}$,
  activation-energy fraction above Haar q95.
- **Data:** locked confirmation evaluation documents; bases from pooled or
  source-fit documents only.
- **Aggregation:** median over ordered shard pairs; document-block-bootstrap
  95% CI; layers are not seeds.
- **Axes:** x = layer 1--12; y = null-relative cross-capture; color =
  top/residual; facets = lineage and checkpoint.
- **H1 pattern:** top is above zero and above residual.
- **Weakening pattern:** both near zero or residual equally high.
- **Allowed conclusion:** cross-document geometric transfer only.
- **Limitation:** no semantic, Gate-use, function, or training conclusion.
- **Observed:** pending execution.

## 12. Success / Failure / Insufficient Evidence

- **Pass — stable-centered/common-local-residual split:** Both registered
  top-64 transfer and paired top-minus-residual gap pass.
- **Fail-R0 — no stable centered top:** Original top-64 does not exceed null.
- **Fail-R1 — stable structure broader than 64:** Original and residual spaces
  both transfer; the paired gap fails.
- **Lineage-conditioned:** LB and decommon disagree; report them separately.
- **Insufficient:** Any hard guard fails or intervals cannot separate the
  registered verdicts.

There is no 25%, 10%, or layer-count practical threshold. q95 and bootstrap
only test separability from the null or zero; every layer's effect size is
reported.

## 13. What This Cannot Claim

This experiment cannot establish that a stable direction:

1. is semantically or functionally common;
2. is used by the Gate;
3. benefits expert training;
4. causes decommon loss or load behavior;
5. should be removed;
6. generalizes to other models, scales, or datasets.

Gate use is adjudicated by A15_00_03. Training causality requires a later,
separately approved anchor and Protocol.

## 14. Review Notes And Protocol Changes

The researcher approved on 2026-07-31:

1. held-out cross-document capture, rather than batch-mean cosine, as primary;
2. top-64 as primary, with 16/32/128/256 sensitivity;
3. 128 pooled plus 512 confirmation documents in eight shards;
4. classifying equally stable residual structure as R1 rather than H1;
5. separate LB/decommon verdicts without a center/LB causal claim;
6. a fully frozen audit with no new eight-GPU training.

**Post-approval changes:** Generated the English canonical Protocol and marked
the Chinese companion approved. The researcher then authorized implementation,
smoke, and full frozen execution on 2026-07-31. No scientific condition changed.
