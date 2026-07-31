---
anchor_id: 15_01_shallow_head_guided_deep_routing
status: subanchor_full_insufficient_stage_a_capability
canonical_language: en
companion_language: zh
depends_on: 15_00_covariance_head_gate_alignment
updated: 2026-07-30
---

# A15_01 Shallow Covariance-Head Guidance For Deep Routing


## 1. Problem Definition

This sibling parent asks a functional question that A15_00 explicitly does not
answer:

> After controlling the native deep Router score, load, capacity, parameters,
> token count, and compute, do token-specific shallow-head coordinates add
> held-out information about which tokens are compatible for training in the
> same deeper expert?

For layer $k<\ell$, shallow-head coordinates mean

$$
c_{k,H}(x)=U_{k,H}^\top(g_k(x)-\mu_k).
$$

They are carried from the same token. Applying $U_{k,H}$ directly to $g_\ell$
is invalid without a registered cross-layer transport map.

The parent primary metric is the held-out incremental prediction of one-step
cross-update compatibility beyond the native deep Router score and registered
nuisance controls. It decides functional admission, not end-to-end benefit.

## 2. Physical Priors

1. A shallow high-variance factor may provide a stable, low-noise coarse cohort
   before deeper representations and experts differentiate.
2. High variance is not equivalent to useful compatibility; common, position,
   length, or source-frequency factors can dominate the head.
3. Reusing one coarse partition at multiple depths can reduce churn but can
   also cause premature lock-in and repeated expert functions.

## 3. Falsifiable Hypotheses

**H1:** On independent tokens, $c_{k,H}$ predicts deeper one-step compatibility
beyond the native deep score and beats same-dimensional random and
token-shuffled controls. Conditional on that gate, shallow-head guidance
improves matched-compute early training.

**Strongest rival:** Any gain is caused by extra parameters, marginal feature
scale, altered load, or token identity rather than the spectral and
token-specific shallow signal.

**Pass:** The compatibility admission gate and matched-compute training test
both pass with load/capacity guards.

**Fail:** The model is capable, but shallow head does not beat the compatibility
controls, or its training gain does not beat the native and matched side-channel
controls after compute/load matching.

**Insufficient:** The registered variable is not stably present in the shallow
head, the base model is incapable, compatibility is underpowered, or routing
guards fail.

## 4. Mathematical Model

The proposed deep score is

$$
z_\ell(x)
=W_\ell g_\ell(x)+A_\ell c_{k,H}(x),
\qquad \ell>k.
$$

$A_\ell$ is a trainable auxiliary readout. The native path remains reachable
when $A_\ell=0$. Random and shuffled conditions use the same tensor shapes and
operations.

Let $Y_{ij}^{(\ell)}$ be an independently measured one-step cross-update
compatibility target for token groups $i,j$. The admission quantity is the
held-out residual gain

$$
\Delta_{\rm comp}
=\operatorname{Perf}(Y\mid s_{\rm native},c_{k,H},q)
-\operatorname{Perf}(Y\mid s_{\rm native},q),
$$

where $q$ contains registered load, norm, position, length, and outlier
controls. The exact performance score is frozen in the Protocol.

## 5. Computational Realization

The first subanchor uses a controlled four-layer MoE. Layers 1--2 first form a
stable representation; the layer-2 head basis is frozen. A genuine native
four-layer arm is compared with layers 3--4 receiving layer-2 head,
same-dimensional random, or token-shuffled head coordinates. The three
side-channel arms use identical auxiliary readouts. All arms use no
load-balance loss and share a non-gradient expert-score-bias rule. Informative
and nuisance tasks distinguish useful shallow structure from variance alone.

## 6. Minimal Falsification Tests

1. Verify that the registered controlled variable is captured by layer-2 head
   above random and is absent as a functional predictor in the nuisance task.
2. On independent token groups, require head compatibility gain beyond native,
   norm/outlier, random, and shuffled controls before joint training.
3. At matched cumulative FLOPs, compare head against a genuine native model
   and against random/shuffled arms with identical auxiliary parameters, load
   targets, capacity, data, and seeds.
4. Audit route margin, flips, load, expert-update conflict, and expert-function
   repetition to localize any loss difference.

## 7. Current Evidence

A15_00 establishes that trained Gates can strongly access their own-layer
covariance head. It does not establish shallow semantics or deep compatibility.
The old wrong-layer-basis control weakens direct basis transplantation but does
not test carrying the same token's shallow coordinates forward.

## 8. Claim Boundary And Next Decision

This parent can establish controlled incremental compatibility and a
matched-compute pilot effect. It cannot establish natural-language universality,
large-scale efficiency, or that covariance head represents domain semantics.
The current subanchor provides none of those effects because compatibility and
training were not reached. It supports only that the registered Stage-A probe
cannot distinguish head-specific access from generic 64-dimensional access in
this setup; H2 remains unresolved.

**Exactly one next decision:** decide whether to approve a new A15_01_01
Protocol with a non-saturated held-out Stage-A specificity criterion before
any B1 training is authorized.
