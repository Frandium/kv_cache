---
anchor_id: 15_02_01_cross_update_compatibility_gate
parent_anchor: 15_02_middle_tail_functional_resolution
status: completed_fail
created: 2026-07-30
updated: 2026-07-30
canonical_language: en
---

# A15_02_01 Incremental Prediction Of Co-Training Compatibility

Parent anchor: [A15_02](../15_02_middle_tail_functional_resolution_anchor.md).

## 1. Problem Definition

This subanchor inherits the parent problem and adjudicates one admission
clause:

> After controlling native Router logits, margin, native expert, load,
> capacity, token count, representation norm, token loss, document, and batch,
> do middle, long-tail, or their union in the actual Router input add held-out
> prediction of cross-loss between two token groups updating the same expert,
> and do they beat equal-dimensional random and wrong-layer bases?

It separates three objects:

1. **Static novelty:** a band creates neighborhoods beyond native logits.
2. **Local functional compatibility:** one token group's expert update helps
   or harms another independent group.
3. **Long-horizon training benefit:** matched joint-training loss per FLOP.

Only the second object is decided here. The first is diagnostic, and the third
belongs to the parent's conditional E02.

### Core Metric Contract

| Metric | Plain meaning | Computation / unit | Why measure it / what it answers | Cannot answer |
| --- | --- | --- | --- | --- |
| Residual neighborhood novelty $N_S$ | New band neighbors after controlling native scores | held-out kNN new-neighbor fraction, dimensionless | Whether the band creates a different partition | Whether it is useful |
| Bidirectional cross-update compatibility $C_e(A,B)$ | Whether A's expert update helps B and vice versa | nat/token | Local same-expert co-training relation | Long-horizon benefit |
| Gradient cosine | Whether A/B expert gradients point together | $[-1,1]$ | Explains first-order origin of $C$ | Actual loss decrease |
| Held-out $\Delta R_S^2$ | Extra compatibility prediction on unseen documents | out-of-sample $R^2$ difference | **Subanchor primary metric; grants training admission** | Training improvement |
| Random gap | Increment over equal-dimensional random directions | $\Delta R^2$ difference | Rejects a dimension-only explanation | Layer specificity |
| Wrong-layer gap | Increment over the same ranks from another layer | $\Delta R^2$ difference | Rejects arbitrary-layer geometry | Long-horizon causality |

## 2. Physical Priors

1. A linear Gate retains only a few logit coordinates. If middle or long-tail
   structure relates to expert gradients, it should predict $C_e(A,B)$ within
   native-score strata.
2. Any high-dimensional subspace can create new neighbors. Static novelty
   cannot support a functional claim; held-out $\Delta R^2$ must beat random
   and wrong-layer controls.
3. Compatibility is local. Fixed routes and a small step isolate the target
   expert update, increasing internal validity while limiting transfer to full
   joint training.

## 3. Falsifiable Hypotheses

**H1 -- non-head compatibility signal.** At least one
$S\in\{M,T,N=M\cup T\}$ has positive held-out $\Delta R_S^2$ beyond native
controls, exceeds the q95 of 256 equal-dimensional random bases and the
wrong-layer control, and reproduces in both twelve-layer lineages and the
four-layer transfer checkpoint.

**Strongest rival R0 -- geometry only.** $N_S$ is high but
$\Delta R_S^2\le0$ or no better than random. Training remains blocked even if
the band changes routes or neighborhoods.

**R1 -- norm, outlier, or difficulty.** The increment is explained by hidden
norm, band energy, token NLL, gradient norm, or extreme samples and disappears
after controls.

**R2 -- document or batch leakage.** Shared document, context, or batch makes
compatibility appear positive and the signal disappears under document split
and batch resampling.

**R3 -- step-size artifact.** Only an oversized step produces the signal;
half-step sign or ranking is unstable, or self-update does not reduce its own
loss.

**Pass:** a preregistered candidate has final-document $\Delta R^2>0$ with a
document-bootstrap lower bound above zero, exceeds the equal-dimensional
random q95 and wrong-layer control, passes all operational guards, and
reproduces at the four-layer branch checkpoint.

**Fail:** the metric is valid and precise, but all of $M,T,N$ are nonpositive
or do not beat the required controls.

**Insufficient:** route replay, step-size, self-loss, basis stability, pair
independence, document-level precision, or four-layer transfer fails.

## 4. Mathematical Model

On an independent calibration set, define for actual Gate input

$$
\Sigma_\ell=U_\ell\Lambda_\ell U_\ell^\top,
\quad
M=U_{65:320},\quad T=U_{321:768},\quad N=U_{65:768}.
$$

For token $i$ and band $S$, use direction-normalized coordinates

$$
q_{i,S}=\frac{U_{\ell,S}^\top(r_{i,\ell}-\mu_\ell)}
{\|U_{\ell,S}^\top(r_{i,\ell}-\mu_\ell)\|_2+\epsilon}.
$$

Band energy remains a nuisance control, so pair similarity is not simply
input amplitude. For fixed-size group $A$,
$\bar q_{A,S}=|A|^{-1}\sum_{i\in A}q_{i,S}$. Registered pair features are
cosine and squared distance, identical across bands.

Let $\theta_{\ell,e}$ denote the target-layer expert parameters. A/B come from
different documents, natively route to the same expert, and match on native
controls. Replay native routes and update only that expert:

$$
\Delta_{A\rightarrow B}
=L_B(\theta_{\ell,e}-\eta\nabla_{\theta_{\ell,e}}L_A)
-L_B(\theta_{\ell,e}),
$$

$$
C_e(A,B)=-\frac12
\left(\Delta_{A\rightarrow B}+\Delta_{B\rightarrow A}\right).
$$

$C>0$ is mutual benefit and $C<0$ is conflict. The primary metric is

$$
\Delta R_S^2
=R^2_{test}(C\mid X_{native},\phi_S)
-R^2_{test}(C\mid X_{native}),
$$

where $X_{native}$ includes full logits, margin, expert stratum, load, token
NLL, hidden/band norm, gradient norm, position, document, and batch controls,
and $\phi_S$ contains the fixed band-pair features.

## 5. Computational Realization

**Spectrum:** reuse Q1's 32×256 calibration token IDs and validated
actual-input bases. Compatibility fit/test data cannot re-estimate
$U,\Lambda,\mu$. Q2 uses fresh DCLM holdout documents split by document into
fit, validation, and final test.

**Twelve-layer evidence:** LB and decommon 80k are primary checkpoints and 40k
provides replication. Static Q2-A covers all twelve layers; one-step Q2-B
preregisters layers 1/6/12 without effect-based selection.

**Four-layer transfer:** repeat the same gate on all four layers of the H768
four-layer checkpoint-800 (about 0.629B nominal tokens) that passed the 8×5090
fast-warmup and resume smoke. Parent E02 cannot unlock without replication
here.

**Pair ledger:** each group contains 32 routed tokens. A and B share no token,
document, or batch, route natively to the same expert, and match on full
logits, margin, loss, norm, and position. The ledger is built from native
controls only and reused for $M,T,N$ and every random basis.

**One-step update:** freeze all model parameters, cache and replay all native
MoE routes, start every direction from the same expert snapshot, update only
the target expert once with masked LM loss, measure the other group's loss,
restore, and reverse. Choose $\eta$ on calibration pairs only. Require lower
self-loss and stable compatibility sign/ranking at $\eta/2$.

**Random controls:** for $k=256,448,704$, use fixed-seed Gaussian QR to generate
256 full-space Haar subspaces. $M/T$ also receive equal-dimensional random
bases inside the 704-dimensional non-head space. Because $N$ fills non-head,
its controls are full-space Haar-704 and wrong-layer ranks 65--768. Every
random condition reuses the same $C$ target.

## 6. Minimal Falsification Tests

1. **Measurement smoke:** known-positive self/split pairs, shuffled-pair
   known-bad cases, and high-norm confusing cases validate $C$ but do not
   support a band claim.
2. **Static Q2-A:** residual neighborhood novelty and cross-document stability
   show whether a band creates a new partition; this has no admission power.
3. **Functional Q2-B:** compare baseline, +M, +T, and +N on one pair ledger to
   test compatibility increment beyond native scores.
4. **Direction controls:** equal-dimensional Haar, non-head random when
   definable, and wrong-layer bases reject dimension, arbitrary non-head, and
   arbitrary-layer explanations.
5. **Nuisance controls:** norm, outlier, difficulty, document, batch, and
   gradient norm reject nonfunctional shortcuts.
6. **Four-layer transfer:** grants access only to the predesigned E02 and does
   not establish twelve-layer training benefit.

## 7. Current Evidence

[A15_02_01_E01](../../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary.md)
completed with an operational Pass and a scientific Fail. Across registered
layers and lineages, true M/T/N residual-neighbor novelty was 0.732--0.902,
while a fixed equal-dimensional random reference already reached 0.714--0.877.
Thus non-head changes the static partition, but novelty is substantially a
generic high-dimensional effect.

The Validation model-level median compatibility increments were, for LB versus
decommon: M $-7.35\times10^{-5}$ versus $-4.30\times10^{-5}$; T
$+2.24\times10^{-4}$ versus $-5.20\times10^{-5}$; and N
$-5.90\times10^{-5}$ versus $-4.29\times10^{-5}$. T passed point gates only in
LB. No candidate was positive and above equal-dimensional random and wrong-
layer controls in both lineages.

All 3,072 Fit/Validation pairs completed; self-loss passed at rate 1.0, expert
restore was exact, and primary-versus-half-step Spearman was 0.87--1.00. The
empty candidate set therefore triggered the registered stop rule before Final,
40k replication, four-layer transfer, or E02.

## 8. Claim Boundary And Next Decision

This subanchor now establishes a bounded negative result: for the registered
80k models, layers, native-routed expert population, local step, DCLM documents,
two direction-only pair features, and low-capacity ridge, fixed covariance-rank
M/T/N geometry does not earn same-expert compatibility admission beyond native
controls, random directions, and wrong-layer bases.

It cannot establish long-horizon routing benefit, expert specialization,
semantic similarity, all-layer generality, or identity between one-step
gradient relations and training dynamics.

**Exactly one next decision:** close fixed covariance M/T/N bands as direct
dispatch coordinates, or open a new anchor that defines a function-aligned
subspace from expert gradients or cross-update residuals before reconsidering
matched training. The current E02 remains blocked.
