# A06_24_toy_a--A06_24_toy_g Common-Removed Differential Expert Proxy Pre-Anchors

## 0. Thinking Card

**Phenomenon:** MPI-style expert proxy can couple router rows to expert
parameters, but the raw principal expert direction may mostly represent common
or high-gain directions rather than rare / residual expert-distinctive
function.

**Mechanism guess:** A useful expert proxy should remove common directions and
use the expert's differential residual gain. If this proxy improves rare
routing margin but fails under early training, the problem is preservation,
not initialization alone.

**Key variables:** common projector $P_C$, residual projector $P_R=I-P_C$,
expert gain matrix $M_e$, raw proxy $u_e^{raw}$, residual proxy $u_e^{res}$,
differential proxy $u_e^{diff}$, rare margin, proxy-route NMI, sign-flip rate,
synthetic task loss.

**Causal relation:** If raw principal proxies are common-dominated, they should
show high overlap with $P_C$ and weak rare margin. If differential residual
proxies are closer to expert-distinctive function, they should improve rare
margin beyond hidden-centering and raw principal baselines. If early common
recapture reappears, residual input or residual row projection should reduce
sign flips.

**Observable metric:** The primary metric for A06_24_toy_a--A06_24_toy_d is rare margin at
step 0. The primary metric for A06_24_toy_e--A06_24_toy_g is rare margin at step 10 plus
sign-flip rate from step 0 to step 10.

**Rival Explanations:** Improvements may come only from load balancing, row
normalization, total norm reduction, projector artifacts, or an oracle synthetic
label that does not transfer to real DCLM.

**Decision:** Use the synthetic sequence only as a mechanism gate. Proceed
toward a real protocol only if differential residual proxies improve rare
margin beyond load-only controls and residual routing preserves that margin
under common recapture without a synthetic loss regression.

## 1. Problem Definition

**Parent problem:** In the active specialization mainline, routing
initialization can be valid at step 0 but early training can erase the intended
partition.

**Sharper subproblem:** Decide whether an expert proxy should be based on raw
dominant gain or on common-removed differential residual gain.

**Terminology / Definitions:**

| Term | Plain meaning | Concrete object or computation | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- |
| Common direction | Shared high-amplitude direction used by many samples | Synthetic vector $c$ and projectors estimated from hidden states | Tests whether raw proxy reads shared bias | Real language common semantics |
| Rare / residual feature | Group-specific direction that should determine the expert | Synthetic orthogonal vectors $r_g$ | Defines the intended route label | Natural-language feature identity |
| Rare margin | Target expert score minus strongest competitor | $z_{i,y_i}-\max_{e\ne y_i}z_{i,e}$ | Directly tests routing safety | Expert utility by itself |
| Proxy-route NMI | Route labels agree with synthetic feature labels | Normalized mutual information | Catches alignment and collapse | Functional specialization |
| Sign flip | A sample's rare margin crosses from positive to non-positive | Fraction of positive step-0 margins that are non-positive at step 10 | Tests preservation | Cause of the flip alone |

**Decision question:** Does common-removed differential expert proxy improve
rare-margin initialization and preservation more than raw principal proxy,
hidden-centering, or load-only controls on synthetic data?

**Not in scope:** Real DCLM claims, semantic expert claims, deployed router
claims, or proof that MPI is wrong in all settings.

## 2. Physical Priors

**P1:** Dominant expert gain can be common-dominated.

Meaning: The largest eigen-direction of $M_e$ may reflect common high-gain
computation shared by experts.

Could be wrong if: $u_e^{raw}$ has low common overlap and high rare margin.

**P2:** Hidden common removal can improve load without improving feature
specialization.

Meaning: Removing a shared hidden mean can reduce common logit bias, but this
does not identify expert-distinctive function.

Could be wrong if: hidden-centering alone improves rare margin and preservation
as much as differential proxies.

**P3:** Common and rare update directions can be orthogonal enough to support
differential step sizes.

Meaning: Reducing common update step size need not reduce rare update step
size; rare directions may use a larger step if common recapture is controlled.

Could be wrong if: larger rare step increases loss or sign flips.

## 3. Split Anchors And Decisions

### A06_24_toy_a Synthetic Sanity Split

**Decision question:** Are load improvements separable from rare-margin
improvements?

**Primary metric:** rare margin at step 0.

**Pass:** hidden-centering or row normalization improves load but not rare
margin, while differential proxy improves rare margin.

### A06_24_toy_b Synthetic Common Operator Robustness

**Decision question:** Does the common-removal conclusion survive multiple
synthetic common estimators?

**Primary metric:** projector-robust rare margin.

**Pass:** differential proxy stays better than raw proxy under mean, top-PC,
and oracle common projectors.

### A06_24_toy_c Synthetic Raw Principal Common-Dominance Audit

**Decision question:** Is raw principal expert proxy common-dominated?

**Primary metric:** mean common overlap of $u_e^{raw}$.

**Pass:** raw proxy has high common overlap and low rare margin.

### A06_24_toy_d Synthetic Residual Differential Proxy Initialization

**Decision question:** Does $u_e^{diff}$ improve step-0 rare margin beyond
$u_e^{raw}$ and $u_e^{res}$?

**Primary metric:** rare margin at step 0.

**Pass:** $u_e^{diff}$ has higher rare margin and proxy-route NMI than raw and
hidden-centering baselines.

### A06_24_toy_e Synthetic Early Preservation

**Decision question:** Does residual input or residual row projection preserve
the differential-proxy margin under early common recapture?

**Primary metric:** rare margin at step 10 and sign-flip rate.

**Pass:** residual input or residual row projection reduces sign flips without
synthetic loss regression.

### A06_24_toy_f Synthetic Functional Value

**Decision question:** If the route is preserved, does it bind to expert
utility in the synthetic ground-truth task?

**Primary metric:** selected-expert utility / route-function binding.

**Pass:** preserved differential routes select the group-matched expert better
than load-only controls.

### A06_24_toy_g Synthetic Differential Step-Size Feasibility

**Decision question:** Are the two update schemes feasible on synthetic data?

Scheme 1: common directions use a smaller learning rate while rare directions
keep the original learning rate.

Scheme 2: common directions use a smaller learning rate while rare directions
use a larger learning rate.

**Primary metric:** rare margin at step 10 with synthetic loss guard.

**Pass:** common-small / rare-large update increases rare margin and does not
increase sign flips or loss relative to ordinary update.

## 4. Claim Boundary And Next Decision

**Can claim if supported:** In this synthetic mechanism surface, raw principal
expert proxies can be common-dominated; common-removed differential proxies
better expose rare-margin initialization; preserving that margin requires
controlling common recapture; differential step sizes are feasible when common
and rare update directions are orthogonal by construction.

**Cannot claim:** Real DCLM transfer, semantic experts, MPI general failure,
or that a new router is validated.

**Next decision:** If A06_24_toy_a--A06_24_toy_g pass on synthetic data, write a separate
real-DCLM protocol that tests only the supported mechanism surface: raw
principal versus residual differential proxy, with projector robustness and
early preservation guards.
