---
experiment_id: A15_02_01_E01_cross_update_compatibility_gate
status: completed_fail
created: 2026-07-30
updated: 2026-07-30
primary_anchor: 15_02_01_cross_update_compatibility_gate
chinese_companion: protocol_cn.md
approval_date: 2026-07-30
implementation_authorized: true
full_run_authorized: true
result: summary.md
---

# Protocol: One-Step Compatibility Admission For Middle And Long-Tail Bands

## 0. Approval Snapshot

**Execution outcome (2026-07-30):** the operational audit passed, but no M/T/N
candidate passed the preregistered 12-layer 80k Validation gate in both
lineages. The registered no-candidate stop rule fired; Final, 40k replication,
four-layer transfer, and E02 were not run. See [summary.md](summary.md).

The researcher approved implementation and the full E01 run on 2026-07-30.
This file is the canonical execution contract; the Chinese companion must
retain the same decision rule.

**Single decision question.** After controlling native Router logits, margin,
native expert, load, capacity, token count, difficulty, norms, document, and
batch, do middle, long-tail, or middle-plus-long-tail features of the actual
Router input add held-out prediction of whether two independent token groups
help each other when updating the same expert, and do they beat
equal-dimensional random and wrong-layer bases?

**Experiment role.** Frozen static-resolution diagnostic plus local functional
admission. It does not train the Router and cannot establish long-horizon
training benefit.

**Primary metric.** Held-out compatibility increment, $\Delta R^2$. It
measures how much extra one-step compatibility is predicted after adding a
band to the native controls. It grants admission to E02 but cannot prove that
E02 will improve training.

**Strong falsifier.** A band may create a different geometry while its
compatibility increment is no greater than zero, the equal-dimensional Haar
q95, or the wrong-layer result. This supports geometric novelty but rejects
functional resolution.

**Pass.** One preregistered candidate passes the 12-layer final-test increment,
random, and wrong-layer gates, has same-sign 40k replication, and transfers to
the locked four-layer branch checkpoint.

**Fail.** Measurement guards pass and precision is adequate, but no candidate
passes.

**Insufficient.** A critical route-replay, local-step, self-loss, independence,
basis, precision, or four-layer-transfer guard fails.

## 1. Objects And Bands

| Object | Exact definition | Purpose / what it answers | Cannot answer |
| --- | --- | --- | --- |
| Actual Router input $r_\ell$ | The tensor received directly by mlp.gate; deployed centering is already included | Keeps the spectral object identical to the deployed Gate input | Expert-input geometry |
| Head $H$ | Covariance eigen-ranks 1--64, 64 dimensions | Positive reference already known from Q1 | Non-head utility |
| Middle $M$ | Ranks 65--320, 256 dimensions | Tests medium-variance non-head information | Tail contribution |
| Long-tail $T$ | Ranks 321--768, 448 dimensions | Tests low-variance information | Rare-word or rare-token utility |
| Non-head $N$ | $M\cup T$, ranks 65--768, 704 dimensions | Tests the complete non-head block | Which sub-band contributes |
| Native controls $X_native$ | Full Gate logits, margin, expert, load/capacity, token NLL, hidden/band/gradient norms, position, document aggregates, and batch load | Removes registered Router, difficulty, and load explanations | All unobserved confounding |
| Functional compatibility | Change in one group's loss after another group updates the same expert | Defines local co-training value with actual loss | Long-horizon joint evolution |

Here long-tail denotes low covariance variance, not low data frequency.

## 2. Anchor Alignment

E01 adjudicates only the A15_02_01 admission clause:

1. Q2-A static resolution asks whether a band creates neighborhoods beyond
   native scores. It can establish only a different partition.
2. Q2-B local function asks whether that information predicts actual one-step
   cross-loss. This is the only E01 decision layer.
3. Q2-C joint training asks whether a selected treatment improves held-out NLL
   per FLOP. It belongs to the parent E02 and remains blocked until E01 Pass.

Q2-A cannot substitute for Q2-B, and Q2-B cannot substitute for Q2-C.

## 3. Hypothesis And Strongest Rivals

**H1.** At least one of $M$, $T$, or $N$ adds stable prediction of one-step
co-training compatibility beyond native controls, beats equal-dimensional
random and wrong-layer bases, and transfers to the four-layer checkpoint
planned for matched training.

| Rival | Prediction | Discriminator | Scope of answer |
| --- | --- | --- | --- |
| R0: geometry only | Neighborhoods differ but cross-loss is not predicted | Report novelty separately from $\Delta R^2$ | Rejects functional interpretation only |
| R1: dimension only | Any subspace of the same dimension works | 256 full-space Haar orientations | Rejects a dimension-only explanation |
| R2: arbitrary non-head | True middle/tail do not beat random directions within non-head | Non-head Haar controls for $M$ and $T$ | Tests whether covariance rank location is special |
| R3: arbitrary layer | A wrong layer basis works equally well | Preregistered wrong-layer mapping | Tests layer specificity, not causal direction |
| R4: norm/difficulty/outliers | Increment disappears after nuisance controls | Nuisance ablation and robust trimming | Rejects registered shortcuts only |
| R5: document/batch leakage | Effect disappears under document and batch resampling | Document splits, pair permutation, resampled ledger | Rejects registered leakage |
| R6: step-size artifact | Only the chosen large step yields compatibility | Self-loss and half-step guards | Validates the local operationalization |

## 4. Spectral Estimation Contract

The spectrum is estimated only on frozen calibration token IDs, never on Q2
fit, validation, or final-test documents. The 12-layer conditions reuse the Q1
verified actual-input bases from 32 training sequences of 256 tokens. The
four-layer checkpoint re-estimates its own per-layer bases on exactly those
calibration token IDs.

For each model, checkpoint, and layer:

$$
x_{i,\ell}=r_{i,\ell}-\mu_\ell,\qquad
\Sigma_\ell=\mathbb E[x_{i,\ell}x_{i,\ell}^{\top}]
=U_\ell\Lambda_\ell U_\ell^{\top}.
$$

Each layer and checkpoint gets its own basis because representations can drift
with depth and training. Reusing an unrelated basis would confound band effect
with representation drift.

Guards are actual-input replay, centered reconstruction, half-split projector
overlap, and eigen-rank order. Passing them establishes a reproducible spectral
object, not function.

## 5. Models, Layers, And Data Separation

### 5.1 Models and layers

| Condition | Checkpoint | Q2-A | Q2-B | Purpose / scope |
| --- | --- | --- | --- | --- |
| 12-layer LB | 80k primary; 40k replication | All 12 layers | Layers 1, 6, 12 | Shallow/middle/deep coverage without result-based layer selection |
| 12-layer decommon | 80k primary; 40k replication | All 12 layers | Layers 1, 6, 12 | Descriptive cross-lineage replication, not a center/LB causal ablation |
| Four-layer H768 branch | checkpoint 800, about 0.629B nominal tokens | All 4 layers | All 4 layers | Hard transfer gate to the exact planned 8-GPU configuration |

### 5.2 Documents

Freeze 512 new DCLM held-out documents of 1024 valid tokens, disjoint from Q1
calibration and evaluation:

| Split | Documents | Use | Cannot be used for |
| --- | ---: | --- | --- |
| Operationalization | 64 | Step size, known-good/bad smoke, pair feasibility | Candidate choice or final claim |
| Fit | 192 | Fit baseline and augmented ridge models | Final reporting after tuning |
| Validation | 128 | Select one band and ridge regularization | Pass claim |
| Final test | 128 | Single locked evaluation and document bootstrap | Reselecting band/layer/hyperparameters |

Whole documents define splits. If the read-only manifest cannot provide 512
eligible documents, execution stops for an amendment rather than silently
reducing the sample.

## 6. A/B Groups And Exact One-Step Procedure

Every model-checkpoint-layer-expert ledger is constructed using native
controls only. The identical ledger and compatibility target are reused for
all true, random, and wrong-layer features.

### 6.1 Pair contract

- Each group contains exactly 32 loss-bearing positions from one 1024-token
  document, all natively routed to the same target expert.
- A and B come from different documents, sequences, and dataloader batches.
- Within a model-layer-expert cell, each document forms at most one group and
  no token is reused across pairs.
- A document with fewer than 32 eligible positions for the expert is skipped.
- Pairs are matched on full logits, margin, token NLL, hidden norm, position,
  native load, and capacity headroom.
- The intended budget is at most 256 pairs per model-checkpoint-layer-split
  cell. The pre-outcome feasibility amendment below defines expert allocation.

This contract reduces known expert, difficulty, size, and document confounds.
It cannot guarantee semantic matching on every unobserved factor.

### 6.1.1 Pre-outcome feasibility amendment (2026-07-30)

Route-only S1 inspection, before computing any compatibility target or band
feature, showed that decommon assigns fewer than 64 total eligible tokens to
some layer-expert-split cells. Equal 32-pair-per-expert sampling is therefore
undefined and would turn native expert starvation into a false functional
verdict.

The repaired estimand preserves every group-level condition above and changes
only expert allocation:

- retain 32 tokens per group, one source document per group, disjoint A/B
  documents and logical batches, and no token reuse;
- construct all feasible matched pairs within each expert;
- allocate at most 256 pairs per model-layer-split in proportion to native
  route mass, subject to available no-reuse pairs, then redistribute unused
  quota across feasible experts;
- include expert identity and native load in controls and report achieved
  expert coverage;
- require at least 192 scientific pairs per cell; Operationalization is allowed
  fewer because it selects the measurement step but carries no claim.

This answers compatibility for the native-routed token population. It cannot
support a separate claim about an almost unused expert. Equal-per-expert
results remain a sensitivity analysis where feasible. The amendment does not
change documents, bands, outcomes, predictors, controls, selection, or Pass
rules and was registered before any outcome was observed.

### 6.2 Bidirectional one-step update

For target expert parameters $\theta_{\ell,e}$:

$$
\Delta_{A\rightarrow B}
=L_B(\theta_{\ell,e}-\eta\nabla_{\theta_{\ell,e}}L_A)
-L_B(\theta_{\ell,e}),
$$

$$
C_e(A,B)=-\frac12
(\Delta_{A\rightarrow B}+\Delta_{B\rightarrow A}).
$$

$C$ is measured in nat/token. Positive values mean mutual local help; negative
values mean conflict.

Execution order is:

1. freeze the model and cache native routes at every MoE layer;
2. start from the same target-expert snapshot;
3. compute A's masked loss and gradient for that expert only;
4. apply one explicit SGD probe step; all other parameters and routes remain
   fixed;
5. measure B's loss change;
6. restore the expert exactly and repeat B-to-A;
7. average the two directions.

The step size is selected only on Operationalization. A and B self-updates must
lower their own loss, and compatibility sign and pair ranking must be stable at
$\eta/2$. Optimizer moments are not used. This isolates a local functional
relation but is not AdamW training dynamics.

**Probe-precision guard.** Native bfloat16 routes, winners, and routing weights
are cached before the update. For the finite local loss only, the frozen
checkpoint parameter values are losslessly promoted to float32 and evaluated
without autocast. A route-only pre-outcome smoke showed that bfloat16 expert
output quantization was larger than the intended infinitesimal loss change and
could flip the self-loss sign. Float32 probing removes that measurement grid
without changing parameter values or routes. It answers the smooth local loss
geometry around the checkpoint, not a bfloat16 deployment effect.

## 7. Identical Tests For The Three Candidate Blocks

For every $S$ in $M$, $T$, and $N$:

1. **Static test:** residualize band coordinates against native controls and
   quantify whether residual-band neighbors differ from native-logit
   neighbors.
2. **Functional test:** use the same band pair features to predict the exact
   compatibility target and calculate held-out $\Delta R^2$.

Residualization is a ridge map trained on Fit, tuned on Validation, and applied
once to Final test. Within each native expert, residual-coordinate cosine kNN
with $k=32$ is compared with standardized-native-logit kNN.

For bounded static cost, each final document contributes 32 predictable
positions selected by SHA256(document hash, token position), yielding 4096
tokens. Selection is blind to band, expert, and outcome and does not change the
complete Q2-B ledger.

| Treatment | Dim. | Specific question | Required controls | Cannot mean |
| --- | ---: | --- | --- | --- |
| $M$ | 256 | Does medium variance add function? | full-space Haar-256, non-head Haar-256, wrong-layer $M$ | Tail effect |
| $T$ | 448 | Does low variance add function? | full-space Haar-448, non-head Haar-448, wrong-layer $T$ | Rare-token effect |
| $N=M+T$ | 704 | Does the full non-head block add function? | full-space Haar-704, wrong-layer $N$ | Which sub-band contributes |
| $H$ reference | 64 | Has native scoring already absorbed head? | full-space Haar-64 | A selectable Q2 treatment |

If $N$ passes while $M$ and $T$ fail, the only allowed statement is that the
joint non-head feature block adds information.

## 8. Random And Wrong-Layer Controls

For $k$ in 64, 256, 448, and 704, generate a 768-by-$k$ i.i.d. Gaussian matrix
from a fixed seed, QR-orthogonalize it, and retain the columns. Use 256
orientations per model-checkpoint-layer-dimension.

- Full-space Haar asks whether the real band beats an arbitrary subspace of the
  same dimension.
- Non-head Haar first samples inside the 704-dimensional $N$ span and maps back
  to model coordinates. It is defined for $M$ and $T$.
- $N$ has no meaningful within-$N$ Haar control because rotating the complete
  704-dimensional span leaves its projector unchanged.

Random bases change only band features; compatibility targets, ledgers, and
splits remain fixed.

Wrong-layer mappings are 1<-6, 6<-12, 12<-1 for 12-layer models and
1<-3, 2<-4, 3<-1, 4<-2 for the four-layer model. They test target-layer
specificity but cannot rule out genuinely shared cross-layer directions.

## 9. Metric Contract

| Metric | Computation / unit | Purpose and answer | Cannot prove |
| --- | --- | --- | --- |
| Residual neighborhood novelty $N_S$ | $1-|\mathrm{kNN}_S\cap\mathrm{kNN}_{native}|/32$, dimensionless | Whether the band makes a stable new partition | Whether the partition is useful |
| Compatibility $C_e(A,B)$ | Negative mean of two cross-loss changes, nat/token | Direct local co-training help/conflict | Long-term benefit |
| Gradient cosine | $\langle g_A,g_B\rangle/(\|g_A\|\|g_B\|)$ in $[-1,1]$ | First-order explanation for $C$ | Finite-step loss improvement |
| $\Delta R_S^2$ | final-test $R^2$ with native plus band minus native-only $R^2$ | Primary functional increment and E02 admission | Training improvement |
| Random gap | True $\Delta R^2$ minus equal-dimension random q95 | Rejects dimension-only geometry | Band causality |
| Wrong-layer gap | True minus wrong-layer $\Delta R^2$ | Rejects arbitrary-layer geometry | Absence of shared layer information |
| Nuisance ablation | Change after adding/removing registered nuisance controls | Whether a registered shortcut explains the signal | Removal of all confounding |
| Split/step stability | document bootstrap, 40k replication, half-step rank stability | Reproducibility across documents, state, and local step | General training dynamics |

## 10. Predictors And Primary Statistic

For each token, compute direction-normalized band coordinates:

$$
q_{i,S}=
\frac{U_{\ell,S}^{\top}(r_{i,\ell}-\mu_\ell)}
{\|U_{\ell,S}^{\top}(r_{i,\ell}-\mu_\ell)\|_2+\epsilon},
\qquad
\bar q_{A,S}=\frac1{|A|}\sum_{i\in A}q_{i,S}.
$$

Each band adds only two pair features:

$$
\phi_S=(\cos(\bar q_A,\bar q_B),\|\bar q_A-\bar q_B\|_2^2).
$$

Band energy is a separate nuisance control, so wider bands do not receive more
predictor parameters.

Baseline and augmented predictors are standardized ridge regressions. The
regularization grid from $10^{-4}$ through $10^4$ is selected only on
Validation. A failure rejects this preregistered low-capacity residual signal,
not every possible nonlinear signal.

Document and batch IDs are not memorized as one-hot predictors; they define
splits, matching, block bootstrap, and permutation nulls. Transferable
document aggregates and batch load/capacity statistics enter $X_native$.

$$
\Delta R_S^2
=R^2_{final}(C\mid X_{native},\phi_S)
-R^2_{final}(C\mid X_{native}).
$$

Use 2000 document-block bootstrap draws for confidence intervals and 256
orientation draws for empirical random q95. A/B pairs are not treated as
independent documents.

## 11. Selection And Verdict

### 11.1 One validation selection

1. Evaluate $M$, $T$, and $N$ on 12-layer 80k Validation.
2. A candidate must have same-direction model-level median $\Delta R^2$ in
   LB and decommon across registered layers 1/6/12 and exceed each relevant
   random q95 and wrong-layer result.
3. If several qualify, select the candidate with the largest paired
   document-bootstrap lower bound; ties prefer $M$, then $T$, then $N$.
4. Lock band, features, layers, step size, and ridge before Final test.
5. If none qualifies, verdict is Fail and four-layer transfer/E02 do not run.

### 11.2 Pass

All conditions must hold:

- On 12-layer 80k Final test, the selected candidate's document-bootstrap 95%
  lower bound is above zero and exceeds full-space random q95,
  non-head-random q95 where defined, and wrong-layer; LB and decommon agree in
  direction.
- The 40k replication has the same sign and no precise opposite result.
- At four-layer checkpoint 800, the locked candidate's pooled/median
  all-layer final-test lower bound is above zero and exceeds random and
  wrong-layer controls.
- Route replay, self-loss, half-step, pair independence, basis stability, and
  nuisance guards all pass.

No arbitrary practical effect threshold is imposed. Report continuous effect,
interval, and random gap. Pass means only that 8-GPU training is warranted.

### 11.3 Fail and Insufficient

Fail means valid and precise measurement but no selected candidate, a locked
candidate no better than zero/random/wrong-layer, or a clear four-layer
transfer failure.

Insufficient means intervals are too wide or a critical operational guard
fails. Insufficient must not be rewritten as absence of function.

## 12. Known-Good, Known-Bad, And Confusing Cases

| Check | Purpose | Expected result | Scope |
| --- | --- | --- | --- |
| Same-group self-update | Validate update sign and mask | Self-loss decreases | Implementation only |
| Two halves of one document, smoke only | Easy positive relation | Relatively high $C$ | Dynamic range only |
| Shuffled compatibility target | Known-bad null | $\Delta R^2$ near zero | Detects leakage |
| Pairwise band-feature permutation | Direction null | Increment disappears | Requires feature-target alignment |
| High-norm/high-NLL subset | Difficult nuisance case | Conclusion survives controls | Rejects obvious shortcut |
| Native no-op replay | Validate fixed routes | logits, winners, and loss match | Operational validity |

## 13. Execution Stages And Stop Rules

1. S0 provenance: freeze checkpoints, token IDs, bases, document splits, and
   seeds.
2. S1 measurement smoke: validate hooks, route replay, masked loss, restore,
   bidirectional update, and step size on Operationalization.
3. S2 Q2-A: run static novelty on all 12-layer and four-layer layers. This
   stage has no training-admission power.
4. S3 12-layer Q2-B Fit/Validation: generate compatibility once, compare all
   bands and controls, and lock one candidate. Stop if none qualifies.
5. S4 12-layer Final/40k replication: make the single final decision and
   checkpoint replication.
6. S5 four-layer transfer: test the locked candidate on all four layers. A
   failure keeps E02 blocked.
7. S6 evidence record: write summary and detailed records, then update the
   owning anchor. Do not submit training automatically unless the E01 verdict
   is Pass.

Failure of actual Router-input identity, document independence, exact parameter
restore, loss replay, or basis reconstruction stops all functional claims.

## 14. Figure Contract, Boundary, And Approval

Central outputs:

1. Static novelty versus functional $\Delta R^2$, separating different
   partitions from useful partitions.
2. Final-test $M/T/N$ increments against random and wrong-layer controls by
   model, layer, and checkpoint.
3. A compact four-layer transfer table for the already locked candidate only.

Even a Pass cannot establish lower long-run loss, better experts, better
training efficiency, semantic similarity, universal layer/scale validity, or
equivalence between the one-step SGD probe and AdamW dynamics.

The researcher approved all registered E01 conditions and execution on
2026-07-30. The only next decision is the E01 evidence verdict; only Pass
activates the conditional E02 authorization.
