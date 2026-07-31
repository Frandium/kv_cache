---
anchor_id: 15_02_middle_tail_functional_resolution
status: blocked_by_compatibility_fail
created: 2026-07-30
updated: 2026-07-30
canonical_language: en
depends_on: 15_00_covariance_head_gate_alignment
---

# A15_02 Functional Resolution In Middle And Long-Tail Bands


## 1. Problem Definition

Q1 established that the trained linear Gate is aligned mainly with the
covariance head of its actual Router input, while middle and long-tail bands
remain accessible and affect some native routing decisions. It did not test
whether those weaker bands preserve functional relations omitted by the
linear logits.

**Parent problem:** Can spectral information complement the functional
resolution of a linear Gate and improve Router--Expert joint training at
matched compute?

**Decision question:** If a middle, long-tail, or middle+long-tail treatment
first passes an independent-token one-step compatibility gate, does routing a
four-layer DCLM MoE with that frozen band produce lower held-out next-token NLL
at the same cumulative FLOPs than both the native Router and an
equal-dimensional random subspace?

### Terminology And Metric Contract

| Term / metric | Plain meaning | Computation and unit | Why measure it / what it answers | Cannot answer |
| --- | --- | --- | --- | --- |
| Actual Router input $r_\ell$ | The representation received by the Gate | Direct `mlp.gate` pre-input hook | Keeps the spectrum deployment-aligned | Expert-input geometry |
| Middle $M$ | Medium-variance directions | eigen-ranks 65--320, 256 dimensions | Tests non-head medium-variance function | Semantics |
| Long-tail $T$ | Low-variance directions | eigen-ranks 321--768, 448 dimensions | Tests low-energy functional information | Rare words or data frequency |
| Non-head $N$ | Middle plus long-tail | ranks 65--768, 704 dimensions | Tests all non-head information | Separate middle and tail contributions |
| One-step compatibility gate | Whether two token groups help each other when updating one expert | Held-out $\Delta R^2$ in subanchor `15_02_01` | Decides whether training cost is warranted | Long-horizon benefit |
| Held-out NLL | Average next-token negative log likelihood on unseen documents | nat/token | Direct language-model quality | Mechanism or specialization |
| Matched-FLOP NLL difference | Treatment minus baseline held-out NLL at equal cumulative FLOPs | nat/token | **Parent primary metric; decides training efficiency** | Cross-scale or cross-data generality |

## 2. Physical Priors

1. **Linear-compression prior.** Eight Gate logits are a low-dimensional
   linear compression of a 768-dimensional Router input. Together with Q1's
   head alignment, this leaves open functional relations in middle or
   long-tail bands. No held-out compatibility increment weakens this prior for
   the registered task.
2. **Function-over-geometry prior.** A novel neighborhood, route flip, or more
   balanced load is not a benefit. A band must first predict same-expert
   cross-update loss and then be judged by joint-training NLL.
3. **Co-dynamics prior.** Even positive local compatibility may be canceled by
   Router, expert, load, and representation evolution. Compatibility is a
   necessary admission signal, not a sufficient endpoint claim.

## 3. Falsifiable Hypotheses

**H1 -- spectral functional resolution.** At least one
$S\in\{M,T,N\}$ adds held-out compatibility prediction beyond native controls
and beats equal-dimensional random and wrong-layer bases. After matched
training, that $S^*$ lowers held-out NLL per FLOP relative to native and random
projection.

**H1-M -- compatibility mechanism.** If the benefit comes from the co-training
relation measured in E01, the $S^*$ arm should reduce within-expert update
conflict. H1-M is an explanatory subhypothesis, not a replacement endpoint:
if NLL passes but conflict does not fall, retain the training-benefit result
and reject the compatibility-mechanism interpretation.

**Strongest rival R0 -- geometry only.** A band creates a new partition but
compatibility $\Delta R^2$ does not beat random or wrong-layer controls. Joint
training is then blocked.

**R1 -- local proxy does not transfer.** The compatibility gate passes, but
matched-FLOP NLL does not improve. This rejects one-step compatibility as a
sufficient selector for this treatment, not all possible spectral methods.

**R2 -- load or capacity confounding.** An apparent NLL difference is explained
by unmatched expert load, overflow, token dropping, parameter count, or actual
FLOPs rather than spectral orientation.

**Pass:** the subanchor passes first; then paired matched-FLOP NLL for $S^*$ is
stably below both native and random controls in the registered four-layer DCLM
setting, with load, capacity, token, parameter, data-order, and FLOP guards
passing.

**Fail:** the valid compatibility increment is precisely nonpositive; or the
gate passes and matched training precisely shows no advantage over native and
random controls.

**Insufficient:** compatibility operationalization, four-layer transfer,
basis stability, training stability, load/capacity/FLOP matching, or paired-seed
precision fails.

## 4. Mathematical Model

For actual Gate input $r_\ell$, define on an independent calibration set

$$
x_\ell=r_\ell-\mu_\ell,
\qquad
\Sigma_\ell=\mathbb E[x_\ell x_\ell^\top]
=U_\ell\Lambda_\ell U_\ell^\top.
$$

Let $P_{\ell,S}=U_{\ell,S}U_{\ell,S}^\top$. A training treatment uses the
basis and mean frozen at the branch checkpoint:

$$
r_{\ell,S}=\mu_{\ell,*}+P_{\ell,S}(r_\ell-\mu_{\ell,*}),
\qquad
z_{\ell,S}=W_\ell r_{\ell,S}+b_{\ell,S}.
$$

$b_{\ell,S}$ is a calibration-only frozen load-matching offset. It calibrates
branch-time expert shares and cannot be tuned on held-out loss. Every arm uses
the same projector kernel, Gate shape, and frozen-offset slot.

The subanchor selects only one passing $S^*$ from $M,T,N$ for training. An
equal-dimensional random projector $P_{\ell,R^*}$ represents the orientation
rival. At registered cumulative FLOPs $F^*$, the parent primary metric is

$$
\Delta L_{S^*:B}(F^*)
=L_{S^*}^{heldout}(F^*)-L_B^{heldout}(F^*),
\qquad B\in\{native,R^*\}.
$$

$\Delta L<0$ means lower validation loss at equal compute. It establishes the
benefit of this treatment but not its cause; secondary dynamics metrics supply
mechanistic evidence.

## 5. Computational Realization

### Stage 1: local functional admission

[Subanchor `15_02_01`](subanchors/15_02_01_cross_update_compatibility_gate_anchor.md)
compares middle, long-tail, and non-head features on existing twelve-layer LB
and decommon checkpoints and adds a transfer gate at the pretrained four-layer
branch checkpoint. It uses the same A/B token-group pairs and actual one-step
cross-update loss. Static neighborhood novelty is diagnostic only.

### Stage 2: conditional 8×5090 matched training

The [E02 Chinese review draft](../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_E02_matched_spectral_dispatch_training/protocol_cn.md)
is preregistered but blocked by Stage 1. The execution surface reuses the
validated H768, four-layer, eight-sparse-plus-one-shared-expert, top-1, DCLM,
8×5090 environment. After a common burn-in to about 0.63B tokens, each layer's
actual Router-input spectrum is estimated, and native, $S^*$, and $R^*$ arms
fork from identical model, optimizer, data-cursor, and RNG states.

Eight GPUs describe one arm's resource allocation, not independent seeds. A
single paired-seed pilot stops at 1B total tokens. Only after stability passes
does formal preliminary evidence use three paired seeds through 2B total
tokens.

## 6. Minimal Falsification Tests

1. **Compatibility admission:** $S^*$ held-out $\Delta R^2$ must exceed zero,
   the equal-dimensional random q95, and wrong-layer control. This separates
   functional prediction from new geometry but only grants training admission.
2. **Four-layer transfer guard:** the same $S^*$ must reproduce at the E02
   branch checkpoint, preventing direct transfer assumptions from the
   twelve-layer audit.
3. **Three-arm matched training:** native, $S^*$, and $R^*$ start from the same
   branch state, separating both the native baseline and random-orientation
   rival.
4. **Endpoint adjudication:** matched-FLOP NLL near 2B total tokens is primary.
   Loss--FLOP AUC, margin, flip, load, update conflict, and expert redundancy
   explain the path and cannot replace the primary metric.
5. **Identity guard:** if overlap between the frozen projector and the current
   covariance band falls to the random range, the result is only a fixed-
   subspace result, not persistent middle/tail routing.

## 7. Current Evidence

[A15_00 E01](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary.md)
and
[A15_00 E02](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary.md)
establish strong actual-input head alignment by 10k, weaker but nonzero
middle/tail access and native route effects, and relative broadening from 10k
to 30k. They did not measure compatibility, forced same-expert updates,
matched training, or held-out loss per FLOP.

[A15_02_01_E01](../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary.md)
now separates static from functional resolution. M/T/N changed residual
neighborhoods substantially (0.732--0.902 novelty), but equal-dimensional random
references were also high (0.714--0.877). No band produced a positive,
random-beating, wrong-layer-beating model-level compatibility increment in both
LB and decommon. The child verdict is Fail.

No $S^*$ exists, so the conditional E02 authorization did not activate. No
8x5090 job was submitted, and this parent has no matched-training evidence.

## 8. Claim Boundary And Next Decision

The registered parent route is blocked at its required local functional gate.
It establishes neither matched-FLOP benefit nor harm because the matched
training experiment was correctly not run.

It cannot establish natural semantic experts, all-layer or all-scale
generality, persistent spectral identity of a frozen band, a unique spectral
cause, harm from Q1 head alignment, or superiority to all Router designs.

**Exactly one next decision:** close fixed covariance bands as the dispatch
treatment for this parent, or replace the treatment only through a new approved
function-aligned-subspace anchor. E02 remains blocked and must not be resumed
under the current M/T/N definition.
