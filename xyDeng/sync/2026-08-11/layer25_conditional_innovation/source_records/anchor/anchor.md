---
anchor_id: 15_08_target_conditioned_layer_innovation
status: e04_eligible_h2_pass_h1prime_fail
canonical_language: en
companion_language: zh
thinking_card: 15_08_target_conditioned_layer_innovation_thinking_card_cn.md
updated: 2026-08-11
---

# A15_08 Target-Conditioned Layer Innovation

Thinking Card source: [researcher-owned Chinese card](15_08_target_conditioned_layer_innovation_thinking_card_cn.md). Chinese companion: [anchor](15_08_target_conditioned_layer_innovation_anchor_cn.md).

## 1. Problem Definition

The A15 line ultimately asks whether layer-specific information can support layer-specific routing. Its immediate blocking question is narrower: when one named layer write changes a token representation, which part of that change is target-related information that was not linearly accessible from the pre-write token state?

The decision question is:

> At the answer token of a frozen Qwen3-8B controlled-composition task, does the part of the layer-25 normalized attention update not linearly predicted by the pre-write state provide held-out terminal-code prediction gain beyond that state and target-independent same-budget controls; if that gain exists, can two train-only target-conditioned innovation directions retain most of it beyond equal-rank random and label-permuted controls?

“Layer update” means the exact change between two states evaluated with the same layer-25 post-attention normalization. “Added information” means held-out predictive gain for the frozen eight-way terminal target under a registered linear readout family. It does not mean new Shannon information, factual knowledge stored in parameters, or native model use.

H2 Pass establishes only a bounded added-accessibility object. H1' Pass would
admit its registered two-dimensional compression for a later functional audit;
H1' Fail leaves the minimal sufficient rank unresolved. Router training and
expert utility are out of scope.

## 2. Physical Priors

1. Residual-stream states at matched hooks share one ambient hidden coordinate, so their exact difference is a valid computation update. This prior is false for mismatched tokens, hooks, or normalization maps.
2. Attention can make context-dependent distinctions newly accessible at one token even though the frozen network creates no external information. The exposing variable is held-out target prediction risk conditional on the pre-write state.
3. If target-related added accessibility is low-dimensional, a small subspace learned without confirmation labels should preserve the full-update gain and beat equal-rank null spaces.

## 3. Falsifiable Hypotheses

**Primary H2 — conditional added accessibility.** Let $X$ be the pre-write state
and $R_U$ the part of the exact update not linearly predicted from $X$. Adding
$R_U$ to a frozen linear readout on $X$ lowers untouched confirmation
cross-entropy for the terminal code, exceeds a target-independent balanced
mismatch bank, and agrees in sign with an equally dimensional $Z$-only versus
$X$-only comparison.

**Gated H1' — registered rank-two sufficiency.** Only if H2 passes, the top two
directions of a train-only conditional innovation matrix preserve at least
$80\%$ of the full-$R_U$ gain and exceed equal-rank random and
train-label-permuted directions under matched search budgets. Because an
eight-class linear target has discriminant rank at most seven, this is a local
rank-two retrieval test, not a test that the whole representation is low rank.

**Strongest rival.** The update may be nonzero and update-only readable while only rescaling or redundantly re-encoding distinctions already available in $X$. It predicts visible difference covariance and possibly strong update-only probes, but no reproducible conditional gain beyond $X$ and no advantage over matched null directions.

H2 Pass / Fail / Insufficient and gated H1' Pass / Fail / Insufficient are reported separately. H1' cannot rescue H2.

## 4. Mathematical Model

Let $h$ be the layer-25 residual before attention, $a$ the attention write, and $N$ the same post-attention RMSNorm used for both states:

$$
X=N(h),\qquad Z=N(h+a),\qquad U=Z-X.
$$

The identity $Z=X+U$ makes $U$ an exact update, not automatically new information. A label-free common basis $V_{pool}$ is the eigensystem of the TRAIN-only pooled covariance of episode-centered $X$ and $Z$; it provides one coordinate for locating old state, new state, and update without comparing unrelated layer-local ranks.

To isolate target-related innovation, fit TRAIN-only predictors of the update and centered one-hot target $Y_c$ from $X$ and form cross-fitted residuals

$$
R_U=U-\widehat{\mathbb E}_{lin}[U\mid X],\qquad
R_Y=Y_c-\widehat{\mathbb E}_{lin}[Y_c\mid X].
$$

The conditional innovation matrix is

$$
K_{new}=C_{UY}S_Y^{+}C_{UY}^{\top},\qquad
C_{UY}=\frac1nR_U^{\top}R_Y,\qquad
S_Y=\frac1nR_Y^{\top}R_Y.
$$

$K_{new}$ is positive semidefinite and has rank at most seven. Its generalized eigenvectors against $\Sigma_{R_U}+\rho I$ prioritize target-residual covariance relative to residual-update variance. They are candidate directions, not evidence until their confirmation gain is measured.

The primary held-out conditional gain is

$$
G_{true}=CE_{conf}(f_X(X))-CE_{conf}(f_X(X)+g_{full}(R_U)),
$$

in nats per example. It establishes only target- and readout-family-specific added accessibility.

## 5. Computational Realization

- **Data:** independent controlled episodes with two random bijections $U\to V$ and $V\to C$, eight balanced terminal codes, and paired one-hop and two-hop queries sharing the exact context and answer. TRAIN, DEVELOPMENT, and CONFIRMATION use disjoint episodes and wording families.
- **Representation:** frozen Qwen3-8B, answer token, block 25; store $X$, $Z$, $U$, and restricted-choice logits. The two-hop condition is primary; the paired one-hop condition is a secondary complexity control.
- **Common basis:** fit $V_{pool}$ from TRAIN only after within-episode centering; use it only to locate covariance and innovation mass.
- **Difference audit:** verify $Z=X+U$ and the covariance identity $\Sigma_Z-\Sigma_X=\Sigma_U+\operatorname{Cov}(X,U)+\operatorname{Cov}(U,X)$.
- **Conditional matrix:** cross-fit linear update and target predictors by episode on TRAIN, select regularization and the generalized-eigen ridge on DEVELOPMENT, freeze two directions, and open CONFIRMATION once.
- **Readout and controls:** fit an old-state ridge readout, a full-$R_U$
  correction, a same-dimensional $Z$ readout, 64 exactly balanced mismatch
  corrections, and matched-budget rank-two null families on TRAIN/DEVELOPMENT;
  CONFIRMATION supplies the registered cross-entropy differences and paired
  episode bootstrap only after the selection ledger is frozen.

Confirmation labels cannot construct the common basis, residualization models, innovation matrix, rank, regularization, threshold, or controls.

## 6. Minimal Falsification Tests

The decisive H2 comparison is old-only versus old-plus-full-$R_U$ on the same
confirmation examples. In every paired bootstrap draw, the true gain is also
compared with the higher-method 95th percentile of 64 independently regenerated,
exactly target-independent mismatch banks. H2 Pass requires positive lower 95%
bounds for $G_{true}$, the same-dimensional $Z$-versus-$X$ gain, and their
capacity contrast $T_{cap}$. H2 Fail requires the $G_{true}$ upper bound to be
non-positive under valid guards. Otherwise H2 is Insufficient.

H1' is evaluated only after H2 Pass. Let $G_2$ be the gain from the two frozen
conditional-innovation coordinates and $D_{80}=G_2-0.8G_{true}$. H1' Pass
requires positive lower 95% bounds for $D_{80}$ and the within-draw contrasts
against the q95 of both equal-rank random and train-label-permuted controls.
H1' Fail requires a non-positive upper bound for any of the three. Otherwise it
is Insufficient.

Capability, balance, hook identity, exact-difference reconstruction, synthetic known-good/bad/confusing cases, and artifact completeness are validity guards. They cannot rescue a failed primary comparison.

## 7. Current Evidence

The variance-interval audit showed that growth-selected subspaces did not beat
equal-dimensional random directions, closing variance growth as the definition
of new information. A15_08_E01 then observed positive update-readout gains, but
its fixed cyclic control was an invertible target relabeling and its full versus
rank-two objects were mismatched. Its scientific qualification is therefore
`PRIMARY_H2_INELIGIBLE_CONTROL_DESIGN`, not an H2 verdict.

[A15_08_E02](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E02_fresh_balanced_mismatch_repair/summary.md)
used wholly fresh episodes and repaired the E01 algebraic defects. Its numerical
arrays map to `H2_PASS_H1_FAIL`: on the complex confirmation condition,
$G_{true}=0.735082$ [0.715608, 0.756246], the same-dimensional state gain was
0.721760 [0.702151, 0.742898], $T_{cap}=0.734620$
[0.715064, 0.755466], and two dimensions retained 32.49% of the full point
gain. However, E02 materialized confirmation mismatch maps from confirmation
labels before its selection freeze, violating the frozen data-access rule.
Its controlling qualification is
`INELIGIBLE_PROTOCOL_CONFIRMATION_LABEL_PREUSE`; H2 and H1' remain
unadjudicated in E02.

[A15_08_E03](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E03_fresh_confirmation_freeze_repair/summary.md)
produced a diagnostic bundle, but a cold-read conformance audit found missing
namespaced `episode_id` and `map_id`, missing structured extraction receipts,
incomplete simple episode/bootstrap arrays, and a 13-entry analysis-only
manifest rather than the registered full-chain inventory. Its controlling
qualification is `INELIGIBLE_GUARD`; none of its numerical values can carry an
H2 or H1' verdict.

[A15_08_E04](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/summary.md)
is the fresh conformance-only repair. It leaves the scientific objects,
readouts, two frozen directions, control families, bootstrap, and thresholds
unchanged; adds the exact identity, receipt, secondary-array, and full-manifest
contracts; and uses 128 wholly new seed-8103 episodes with zero ID, context, or
text collision against E01/E02/E03. Its 59-artifact, 39-required-family record
audit is complete and eligible.

On E04 complex confirmation, $G_{true}=0.767207$
$[0.751296,0.785146]$, $G_{state}=0.754508$
$[0.738226,0.773069]$, and $T_{cap}=0.766839$
$[0.750643,0.784307]$. H2 therefore passes for the registered local object.
The two frozen innovation directions gave $G_2=0.255052$
$[0.244938,0.264427]$, retaining 33.24% of the full point gain, and
$D_{80}=-0.358713$ $[-0.369760,-0.348675]$. H1' therefore fails. Positive
contrasts against random and label-permutation rank-two controls show that the
directions are non-null; they do not make two dimensions sufficient.

The TRAIN-only spectrum separates two senses of “low rank.” Complex update
variance is head-concentrated in the common basis (98.41% in ranks 1--256), but
the two target-conditioned candidate directions place only 37.90% of their
squared mass there; their median mass rank is 374. The seven nonzero generalized
eigenvalues are also relatively flat, with only 31.17% of their mass in the
first two. Thus a low-rank-looking variance spectrum does not imply a
two-dimensional target-retrieval object. Because the two candidates fail
held-out sufficiency, their common-basis distribution is not a final location
claim for all added accessibility.

## 8. Claim Boundary And Next Decision

Established: matched residual coordinates permit an exact computation
difference, but difference and spectral variance do not define knowledge.
Under the registered target, data, layer transition, and linear readout family,
E04 establishes added accessibility beyond the old state and the balanced
same-budget rival. It also rejects the registered claim that two conditional-
innovation directions retain at least 80% of the full gain. This is one local
instance supporting H2 and one local rank-two H1' failure; it does not adjudicate
the meeting's global low-rank H1.

Unresolved: the smallest task-readout rank sufficient for this same $R_U$
object. The researcher has now parked that question rather than treating an
$r=3\ldots7$ ladder as the current next step. Transfer, native use, and routing
function remain downstream.

Cannot claim: information-theoretic creation, nonlinear novelty, factual knowledge storage, native MLP use, expert utility, Router gain, natural-language generality, or a universal layer law.

**Exactly one next decision:** obtain the registered eligible result from the
child [A15_08_01 approved Protocol](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/protocol.md).
The child direction, common basis, two figure roles, descriptive-rank boundary,
numerical contract, implementation, and one four-GPU run are approved. ACP job
The requested image/runtime is validated, but candidate data amendment A4 awaits Block B reconfirmation; no result or Router authorization follows from
submission alone.
