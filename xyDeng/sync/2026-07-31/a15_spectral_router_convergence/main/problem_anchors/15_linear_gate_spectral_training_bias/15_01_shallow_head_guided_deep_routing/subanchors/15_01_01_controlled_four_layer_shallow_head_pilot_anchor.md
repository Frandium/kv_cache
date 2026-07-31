---
anchor_id: 15_01_01_controlled_four_layer_shallow_head_pilot
parent_anchor: 15_01_shallow_head_guided_deep_routing
status: insufficient_stage_a_capability
canonical_language: en
companion_language: zh
updated: 2026-07-30
---

# A15_01_01 Controlled Four-Layer Shallow-Head Guidance Pilot


## 1. Problem Definition

This subanchor tests one controlled clause of A15_01:

> In a capable four-layer MoE, after a held-out compatibility gate and matched
> load/capacity/token/FLOPs, does supplying layer-2 head coordinates to the
> layer-3 and layer-4 Gates improve early held-out loss beyond a genuine native
> four-layer model and parameter-matched random/token-shuffled side channels?

The primary training metric is paired held-out NLL difference at a frozen
cumulative-FLOP budget,

$$
\Delta L_{H-C}=L_{H}-L_C,
\qquad C\in\{\text{native},\text{random},\text{shuffled}\}.
$$

It is measured in nat/token. Negative values favor shallow head. It does not
prove large-scale or natural-language efficiency.

## 2. Physical Priors

1. A stable layer-2 head can reduce deep routing search only when it predicts
   deep update compatibility.
2. Same architecture and compute are required because an auxiliary readout can
   help merely through extra capacity.
3. An informative task and a nuisance task are both required; otherwise the
   desired answer can be built into the generator.

## 3. Falsifiable Hypotheses

**H1:** Layer-2 head passes the held-out compatibility gate and yields
$\Delta L_{H-C}<0$ against the native model and both matched side-channel
controls on the informative task, with
no corresponding spurious advantage on the nuisance task.

**Strongest rival:** Extra parameters, feature scale, token identity, or load
redistribution explain the gain. The genuine native model or random/shuffled
side channels perform equally once cumulative compute is matched.

**Pass:** Capability and compatibility guards pass; head beats the native
model and both matched side-channel controls with paired uncertainty below
zero; load and capacity remain matched; the nuisance task does not show the
same effect.

**Fail:** All guards pass but head does not beat all three controls, or any
apparent gain is removed by compute/load matching.

**Insufficient:** The layer-2 registered variable is not stably head-captured,
the base task is not learned, the compatibility estimate is unresolved, or
routing/compute guards fail.

## 4. Mathematical Model

For $\ell\in\{3,4\}$,

$$
z_\ell
=W_\ell g_\ell+A_\ell c_{2,H},
\qquad
c_{2,H}=U_{2,H}^\top(g_2-\mu_2).
$$

$A_\ell$ is zero-initialized so every condition begins at the native score.
Random uses a frozen same-dimensional Haar subspace of $g_2$; shuffled uses
the correct $c_{2,H}$ from another token in the same batch under a registered
permutation. All three use identical $A_\ell$ shapes and operations.

Before training, an independent-token one-step cross-update target must show

$$
\Delta_{\rm comp}^{H}
>
\max(\Delta_{\rm comp}^{random},
\Delta_{\rm comp}^{shuffled})
$$

with registered uncertainty. This is a hard admission guard, not a substitute
for the NLL endpoint.

## 5. Computational Realization

A controlled generator contains a coarse variable that is high variance by
construction and either predicts deep expert compatibility (informative) or is
independent of the target operation (nuisance). A four-layer, eight-expert,
top-1 model first trains layers 1--2 to a capability and capture gate. The
layer-2 basis and lower layers are frozen. Layers 3--4 then receive a common
treatment-blind native calibration warmup; the compatibility audit and the
native/head/random/shuffled arms start from that cloned checkpoint. No
load-balance auxiliary loss is used; all arms share one non-gradient
expert-score-bias rule.

## 6. Minimal Falsification Tests

1. Capability and layer-2 head-capture gates on independent data;
2. independent-token compatibility admission versus native, norm/outlier,
   random, shuffled, and batch-resampling controls;
3. matched-compute native/head/random/shuffled training on informative and
   nuisance tasks with paired seeds and data order;
4. route margin, flip, load, capacity drop, expert-update conflict, and
   expert-function repetition as mechanism diagnostics.

## 7. Current Evidence

Experiment records: [Protocol](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/protocol.md),
[summary](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md),
and [detailed evidence](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/detailed.md).

**Observation:** the repaired smoke passed 11/11 engineering guards. The
authorized full run then completed Stage A for five seeds on both informative
and nuisance tasks and terminated with the registered status
`insufficient_stage_a_capability`. Coarse accuracy, content retention, absolute
head-probe accuracy, and split-half basis stability passed in all 10 task-seed
states. However, head-probe accuracy and the q95 over 256 same-dimensional
random-subspace probes both equaled 1.0 in every state. The strict specificity
gap was therefore zero, and head strictly exceeded random q95 in 0/10 states.
B0, Stage 0 compatibility, and B1 were not run; each has zero records.

**Interpretation:** the controlled coarse target is readable from the head, but
the current 64-dimensional full-accuracy probe is saturated and does not show
that this access is specific to the covariance head. This weakens the Stage-A
operationalization, not the shallow-head mechanism.

**Remaining uncertainty:** whether a non-saturated held-out specificity test
would isolate head-concentrated information, and whether that information
predicts compatibility or changes matched-FLOP training, remain unresolved.

## 8. Claim Boundary And Next Decision

**Supported:** the formal Stage-A execution is valid, the target and content
proxy were learned, the estimated head was stable, and the registered
head-versus-random specificity test was non-discriminating because both probes
saturated at perfect accuracy. The fail-closed stage boundary operated
correctly.

**Weakened:** the current Stage-A capture operationalization cannot qualify the
layer-2 head as a specific treatment variable in this setup.

**Unresolved:** H2 compatibility, comparison with random or shuffled features,
and any Router--Expert training-path or matched-FLOP effect were never tested.
The result must not be labeled H2 fail.

**Cannot claim:** absence of shallow-head information, compatibility failure,
random equivalence, training benefit or harm, from-initialization benefit,
online PCA, DCLM transfer, or large-scale efficiency.

**Exactly one next decision:** decide whether to approve a new Protocol with a
non-saturated held-out Stage-A specificity criterion before any B1 training is
authorized. The failed gate must not be bypassed or resumed.
