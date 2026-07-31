---
anchor_id: 15_00_03_gate_transferable_vs_local_residual_alignment
parent_anchor: 15_00_covariance_head_gate_alignment
status: full_execution_authorized
canonical_language: en
companion_language: zh
updated: 2026-07-31
---

# A15_00_03 Gate Preference For Pooled Common Versus Local Residual Modes


## 1. Problem Definition

A15_00 establishes strong equal-energy Gate alignment with a pooled covariance
head but does not determine whether that head is a cross-data common component
or an accidental calibration direction.

**Decision question:**

> At equal input energy, does the decommon Gate produce greater expert-relative
> logit gain and native route dependence for an independently pooled
> centered-common candidate than for equal-dimensional shard-local residuals,
> and how does this preference change across 30k/40k/80k?

This subanchor directly adjudicates pooled-versus-local preference. The joint
phrase “preference for stable common” is licensed only if A15_00_02
independently establishes the stability labels.

The primary metric is

$$
B_{\ell,P:L}
=\log\frac{G_{\ell,P}+\epsilon}
{\operatorname{median}_sG_{\ell,L_s}+\epsilon},
$$

a log equal-energy gain ratio. It measures Gate geometry, not utility.

## 2. Physical Priors

1. A fixed linear Gate can accumulate a coherent weight on directions that
   recur across data groups; rotating shard-local directions can cancel.
2. Pooled top directions have greater raw energy, so only eigenvalue-free
   equal-energy gain $G$ can adjudicate selection.
3. Endpoint alignment reflects both $W$ and representation-basis motion;
   checkpoint crossings are needed to separate them.

## 3. Falsifiable Hypotheses

**H1:** At decommon 80k, $B_{P:L}>0$ beyond a singular-value-preserving
orientation null, with directional replication at 40k. Removing the pooled
candidate also removes more native-winner margin support than removing an
equal-dimensional local residual; route flip is supporting evidence. LB
determines lineage sharing.

**Strongest rival R0:** Only raw input energy favors the pooled candidate;
$G$ and $B_{P:L}$ do not.

**R1:** Any pooled direction, wrong-layer basis, or matched orientation null
produces the same result.

**R2:** Endpoint change is representation drift; fixed-basis
$W_{30/40/80}$ comparisons do not reproduce it.

**R3:** Stable shared directions are functionally useful. This experiment
cannot label a real pooled preference as beneficial or harmful.

**Pass:** Both equal-energy gain and native-margin-support comparisons support
H1 at the decommon primary endpoint, replicate directionally at 40k, and pass
null and wrong-layer guards.

**Fail:** A valid precise result supports R0/R1, or native route dependence on
local residuals is not weaker.

**Insufficient:** A15_00_02 may independently be insufficient while this
anchor still decides pooled-versus-local preference, but the stability
interpretation remains unavailable. Failure of actual-input, basis, route
replay, checkpoint crossing, or precision guards makes this anchor itself
insufficient.

## 4. Mathematical Model

Let $\bar W_\ell=C_EW_\ell$, with
$C_E=I-\mathbf1\mathbf1^\top/E$. Fit a pooled 64-dimensional candidate
$U_{\ell,P}$. For shard $s$, fit a local residual basis after removing it:

$$
Y_{\ell,s}=X_{\ell,s}(I-U_{\ell,P}U_{\ell,P}^{\top}),
\qquad
U_{\ell,L_s}=\operatorname{TopSV}_{64}(Y_{\ell,s}).
$$

For any 64-dimensional basis $U$,

$$
G_\ell(W,U)=\frac1{64}\|\bar W_\ell U\|_F^2.
$$

$B_{\ell,P:L}$ compares pooled gain with the median local-residual gain.
Equal-rank ablation uses native-winner margin support as the route-use metric;
top-1 flip is supporting evidence.
Full $W_a\times U_b$ crossings for
$a,b\in\{30k,40k,80k\}$ separate saved-weight and basis effects but do not
identify per-step gradient causality.

## 5. Computational Realization

The [approved canonical E01 Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_03_E01_gate_transferable_vs_local_residual_alignment/protocol.md)
reuses A15_00_02 frozen actual-input caches, document splits, and 64-dimensional
bases while independently computing Gate gain, route effect, orientation null,
and checkpoint crossings.

Existing 12-layer H768 decommon and LB 30k/40k/80k checkpoints are used. The
80k endpoint is primary, 40k is replication, and 30k is macro-trajectory
support. All 12 layers are reported.

The two E01 analyses share S0 provenance and activation extraction and can
execute in parallel. Neither result may alter the other's estimator, data, or
metric post hoc.

## 6. Minimal Falsification Tests

1. Apply one implementation of $G$, flip, and margin to pooled, each local
   residual, and full/complement Haar-64 bases.
2. Use 256 singular-value-preserving Gate-orientation nulls and wrong-layer
   pooled bases.
3. Compare equal-rank ablations on identical held-out documents.
4. Use the full $3\times3$ checkpoint crossing to separate $W$ and basis
   motion.
5. Adjudicate LB and decommon separately; they are not a single-variable
   center/LB causal pair.

## 7. Current Evidence

A15_00 E01/E02 establish pooled covariance-head $G$ above middle/tail but do
not construct cross-shard pooled/local residual comparisons. A15_02_01 E01
establishes extra non-head partition novelty without a stable one-step
functional increment. That result is compatible with H1 but does not prove it
or show that pooled preference is harmful.

The E01 Protocol and full frozen execution were authorized on 2026-07-31.

## 8. Claim Boundary And Next Decision

A Pass supports only greater equal-energy Gate preference and native use for a
pooled centered-common candidate than for equal-dimensional shard-local
residuals in the registered checkpoints. Only a simultaneous A15_00_02 Pass
licenses the stable-common versus unstable-residual wording.

Even a joint Pass cannot establish training benefit, residual semantic
uselessness, or a causal explanation of decommon performance.

**Exactly one next decision:** complete the authorized frozen execution and
combine its typed verdict with A15_00_02. A joint Pass may open a later
decision about a matched stable-versus-local intervention; otherwise close or
narrow the residual-instability mechanism.
