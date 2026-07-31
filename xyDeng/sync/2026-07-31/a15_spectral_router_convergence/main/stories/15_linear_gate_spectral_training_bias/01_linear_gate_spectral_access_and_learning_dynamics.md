---
story_id: A15_linear_gate_spectral_access_and_learning_dynamics
status: final_synthesis
updated: 2026-07-31
canonical_language: en
companion_cn: 01_linear_gate_spectral_access_and_learning_dynamics_cn.md
---

# From Spectral Access to Learning Dynamics in Linear MoE Gates

*What the Router sees, why high-variance directions can be learned first, and
why fixed bands are not yet routing coordinates.*

Status: final 2026-07-31 synthesis; controlled mechanism supported, real
joint-training transfer unresolved.

## 0. Terminology / Definitions

- **Actual Router input:** the representation passed directly to the linear
  Gate. It is not the representation subsequently sent to the experts.
- **Covariance spectrum:** the eigenvalues and eigenvectors of the centered
  actual Router input. Head, middle, and tail denote high-, intermediate-, and
  low-variance rank ranges; they do not denote token frequency or semantic
  importance.
- **Equal-energy Gate gain:** Gate sensitivity per input direction after
  removing covariance-eigenvalue amplification.
- **Current use:** native logit response, route flip, or margin dependence at a
  frozen checkpoint. Current use is not training benefit.
- **Finite-time speed bias:** one direction reaches the same fraction of its
  own target in fewer updates. It does not mean other directions are
  inexpressible or that the final solution is head-only.
- **Compatibility:** the bidirectional cross-loss effect when two independent
  token groups update the same expert once. It is a local functional admission
  target, not a long-horizon training result.
- **Pass / Fail / Insufficient:** Pass or Fail adjudicates a registered
  scientific hypothesis after validity guards pass. Insufficient means a
  prerequisite failed or the required evidence was unavailable.

## 1. Abstract

This report asks whether covariance information is a useful inductive bias for
a linear Mixture-of-Experts Gate. The evidence supports a narrower answer than
“the Router only sees the spectral head.” On the representation actually
consumed by the Gate, trained endpoints show much larger equal-energy gain in
the covariance head, while middle and tail access and route effects remain
nonzero. The alignment is already strongest at the earliest available 10k
checkpoint and becomes less exclusive through 30k, so saved checkpoints do
not support continual head sharpening.

The controlled E03-S experiment identifies one causal source of this pattern.
When the functional target is matched across directions, trace is held fixed,
and a linear Gate is trained by pure SGD, larger covariance eigenvalues reduce
the finite time needed to learn the corresponding modes. A flat spectrum
removes the order; larger spectral gaps increase it; whitening removes it; and
a tail-only target remains learnable. Thus covariance anisotropy can cause
head-first learning without implying head-only capacity.

Transfer to real joint training is unresolved. In E03-R, all three six-layer
DCLM runs violated the preregistered load guard before a valid persistent
formation time could be established. Post-collapse spectral changes are
descriptive but scientifically ineligible. The result is insufficient, not a
negative transfer result.

The functional boundary is also negative but narrower. Fixed middle, tail, and
non-head bands changed static neighborhoods yet failed to add stable held-out
prediction of one-step same-expert compatibility across both LB and decommon.
They therefore did not qualify for matched training. A separate shallow-head
pilot stopped even earlier because both head and random 64-dimensional probes
decoded the controlled coarse variable perfectly; that measurement was
saturated and did not test compatibility or training benefit.

The current knowledge update is therefore: covariance anisotropy is a
controlled finite-time learning-speed bias, but existing evidence neither
establishes its real Router--Expert formation path nor qualifies fixed
covariance bands as functional dispatch coordinates.

## 2. Research Question

The Q1 question is:

> After separating raw input energy from Gate selectivity, which covariance
> bands does a trained linear Gate access, and can covariance anisotropy itself
> cause the high-variance directions to be learned first?

Three downstream questions must remain separate:

1. does the same signature form in a load-stable real MoE trajectory;
2. do middle or tail features add functional information beyond native scores;
3. does any spectral dispatch rule improve held-out loss per matched FLOP?

Only Q1's controlled mechanism clause is now answered causally. Among the
downstream questions, the real-trajectory clause is insufficient, the
registered fixed-band functional gate failed, and matched training was not
run.

## 3. Why This Question Is Confusing

Four distinct effects can all look like “the Router follows the head”:

1. **Raw energy:** a fixed weight produces larger logits on a high-variance
   input direction.
2. **Gate orientation:** the trained expert-relative row space itself allocates
   more gain to high-variance directions.
3. **Learning speed:** the same functional target is acquired faster along a
   high-variance coordinate.
4. **Functional preference:** the expert-loss relation genuinely requires that
   coordinate.

Endpoint alignment establishes neither its origin nor its value. A net update
can itself be head-oriented while still diluting an even more head-oriented
existing Gate. A static band can create a novel partition while failing to
group tokens that train an expert compatibly. These distinctions determine the
experiment chain.

## 4. Data And Training Layout

| Evidence block | Model / data | Intervention or comparison | What it decides |
| --- | --- | --- | --- |
| E01 | Two 12-layer H768 top-1 DCLM lineages; 30k/40k/80k checkpoints | Actual Gate input, coarse/fine bands, full Gate-by-basis crossing | Endpoint access, current use, late saved-interval allocation |
| E02 | LB and batch-gradient 12-layer lineages; 10k/20k/30k | Same actual-input audit at earlier checkpoints | Earliest available alignment and 10k--30k broadening |
| E03-S | Linear 768-to-8 Gate; fixed Gaussian representation; eight seeds | Flat, 4:2:1, 16:4:1, whitened, and tail-only; pure SGD | Controlled covariance-speed causality |
| E03-R | Six-layer H768 top-1 DCLM; seeds 17/29/43 | From initialization, LM loss only, no LB auxiliary loss, dense diagnostics | Real formation signature subject to load validity |
| A15_02_01_E01 | LB/decommon 80k; layers 1/6/12; new held-out documents | One-step same-expert compatibility with random and wrong-layer controls | Fixed-band functional admission |
| Shallow pilot | Four-layer controlled top-1 MoE; two tasks, five seeds | Layer-2 head versus 256 random 64D probes before compatibility/training | Whether the proposed shallow feature is specifically head-concentrated |

All spectral checkpoint audits use the representation that the Gate actually
receives. E01/E02 use fixed calibration sequences to estimate bases and
separate held-out documents for response and routing diagnostics. E03-S uses
independent train, trajectory-evaluation, and held-out streams. E03-R saves
ordered calibration tensors at heavy steps so basis bootstrap, orientation
null, and $W_s\times U_t$ decomposition can be audited.

## 5. Physical Intuition

Covariance cannot create a routing objective. A Gate learns only when expert
losses differ across tokens. Once an expert-advantage signal exists, however,
larger input variance amplifies its cross-covariance with the Gate update. This
can make high-variance modes enter the Gate sooner.

The mechanism is conditional:

$$
\text{expert-advantage signal}
\times
\text{input covariance}
\times
\text{optimizer response}
\longrightarrow
\text{mode-learning speed}.
$$

If the expert advantage is absent, lies only in tail, is canceled by
preconditioning, or moves with a rapidly rotating representation, an
anisotropic spectrum does not guarantee head alignment.

## 6. Definitions And Metrics

Let centered actual Router input be $x$, with
$\Sigma=U\Lambda U^\top$, and let
$C_E=I-\mathbf1\mathbf1^\top/E$ remove the expert-common logit component.
For band $B$ with projector $P_B$ and basis $U_B$:

$$
V_B=\mathbb E\|C_EWP_Bx\|_2^2
$$

is the realized expert-relative logit response. It includes the input
eigenvalues. The response per observed band energy is

$$
S_B=\frac{V_B}{\mathbb E\|P_Bx\|_2^2}.
$$

$S_B$ controls total band energy but still weights directions within a wide
band by their observed variance. The exact equal-energy Gate gain is

$$
G_B=\frac1{d_B}\|C_EWU_B\|_F^2.
$$

The endpoint contrasts are

$$
B_{H:M}=\log(G_H/G_M),\qquad B_{H:T}=\log(G_H/G_T).
$$

They measure relative Gate orientation, not utility. E03-S instead measures
$T_B(0.5)$, the first persistent optimizer step at which band $B$ removes half
of its own initial target error, and compares log-time ratios. E03-R defines a
formation time only when both head contrasts beat matched orientation nulls,
remain stable under basis bootstrap, persist, and satisfy the load guard.

The functional admission metric is

$$
\Delta R_S^2
=R^2(C\mid X_{native},\phi_S)-R^2(C\mid X_{native}),
$$

where $C$ is one-step bidirectional compatibility, $X_{native}$ contains
native-score and nuisance controls, and $\phi_S$ contains two fixed band-pair
features. It can admit a band to training but cannot prove endpoint benefit.

## 7. Mathematical Model

Let frozen expert loss vector be $\ell(x)$ and define centered expert advantage

$$
a(x)=-C_E\ell(x).
$$

For soft routing loss $L=\mathbb E[p(Wx)^\top\ell(x)]$, the exact gradient is

$$
\nabla_WL
=\mathbb E[(\operatorname{Diag}(p)-pp^\top)\ell(x)x^\top].
$$

At balanced initialization $W=0$,

$$
\dot{\bar W}(0)=\frac1E\mathbb E[a(x)x^\top].
$$

If $a(x)=Ax+\varepsilon(x)$ with
$\mathbb E[\varepsilon(x)x^\top]=0$, then

$$
\dot{\bar W}(0)u_i=\frac{\lambda_i}{E}Au_i.
$$

Thus covariance scales an existing functional signal; it does not create one.
The locally solvable quadratic model gives

$$
\dot w_i=-(\kappa\lambda_i+\beta)w_i+\kappa\lambda_i a_i,
$$

with relative learning time

$$
T_i(\rho)=\frac{-\log(1-\rho)}{\kappa\lambda_i+\beta}.
$$

## 8. Theorem State

**Isotropic theorem.** If $\Sigma=\lambda I$, covariance gives every direction
the same time constant. Any direction selected by one finite run comes from
the target, initialization, or sampling, not from a covariance-defined head.

**Conditional anisotropic theorem.** If target expert advantage has comparable
norm across covariance directions, initialization is balanced, and the
optimizer does not cancel the scale, then larger $\lambda_i$ gives a shorter
finite learning time. With no regularization, slower directions can eventually
catch up; isotropic $L_2$ regularization can preserve a steady-state bias.

**Limits.** The theorem does not cover hard top-1 boundaries, adaptive AdamW
preconditioning, moving expert advantage, or moving representation bases. It
also does not imply that a head-aligned Gate has a more concentrated singular
value spectrum: right-subspace alignment and Gate singular-value concentration
are different objects.

## 9. Mechanism Decomposition

1. **Nearly homogeneous experts:** functional advantage is close to zero;
   covariance alone cannot start useful specialization.
2. **Small advantage forms:** if advantage is direction-balanced, high-
   variance modes enter the Gate sooner.
3. **Router--Expert feedback:** routing changes expert data and therefore the
   advantage spectrum. Feedback may reinforce head, broaden toward middle/tail,
   or destabilize load.
4. **Margin saturation or partition lock-in:** softmax sensitivity falls and
   capacity/load interventions may dominate the original speed effect.

E03-S tests stage 2 with a fixed target. E03-R was intended to observe stages
1--4 jointly, but its registered trajectory became invalid at stage 3 because
of load concentration.

## 10. Evidence From Anchors

| Question | Direct result | Verdict | Safest conclusion |
| --- | --- | --- | --- |
| Does trained Gate orientation remain head-heavy after energy control? | At 40k/80k, $G_H/G_M=4.03$--$6.36$ and $G_H/G_T=14.61$--$25.36$ across LB/decommon | Pass | Endpoint head alignment is not raw-energy-only |
| Can the Gate see middle/tail? | Gains, route flips, and margins are nonzero; relative access grows from 10k to 30k | Pass as measurement | Middle/tail are weaker, not invisible |
| Does saved training continually sharpen head? | Fixed-basis H:M effects are negative across all audited lineages and intervals | Fail for persistent sharpening | Positive net-update orientation does not imply increasing endpoint bias |
| Does covariance anisotropy cause a speed order under control? | Flat overlap, 4:2:1 gives about 1:2:4, 16:4:1 about 1:4:16, whitening restores overlap | Pass | Controlled finite-time speed causality |
| Does the homologous signature form in real DCLM? | All three seeds violate the 0.8 load guard before valid formation | Insufficient | Real transfer remains unresolved |
| Do fixed non-head bands predict compatibility? | No M/T/N candidate passes LB and decommon plus random/wrong-layer gates | Fail | Fixed bands are not admitted to matched training |
| Does shallow head guide deeper training? | Head and random 64D probes both achieve 1.0; later stages are not run | Insufficient | The Stage-A specificity test is saturated; H2 is untested |

## 11. Experiment Evidence

### Controlled learning times

| Condition | Median $T_H$ | Median $T_M$ | Median $T_T$ | Reading |
| --- | ---: | ---: | ---: | --- |
| Flat 1:1:1 | 140.82 | 140.83 | 140.80 | no covariance-defined order |
| Moderate 4:2:1 | 55.76 | 111.46 | 223.01 | approximately 1:2:4 |
| Strong 16:4:1 | 28.63 | 114.38 | 457.39 | approximately 1:4:16 |
| Strong-whitened | 140.78 | 140.82 | 140.88 | order removed |

![Controlled covariance anisotropy separates learning times](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/figures/e03_s_crossing_times.png)

The eight-seed medians exceed the 2,048-partition flat-rotation null in the
anisotropic conditions; the dose intervals are positive; and every tail-only
seed reduces independent held-out KL by more than the registered 0.5 gate.

### Real-run validity boundary

| Seed | First failing 20-step window | Share at first failure | Step-100 rolling maximum |
| ---: | --- | ---: | ---: |
| 17 | 56--75 | 0.80208 | 0.99045 |
| 29 | 53--72 | 0.80246 | 0.99110 |
| 43 | 60--79 | 0.81781 | 0.98916 |

![Load collapse precedes a valid formation verdict](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/figures/e03_r_load_collapse_and_contrasts.png)

No selected valid point through step 50 was a two-contrast orientation
candidate. Two seeds crossed both orientation nulls at step 120, but only
after near-single-expert concentration; those points cannot define formation.

## 12. Counterexamples And Rival Explanations

- **Energy-only:** rejected for audited endpoints because $G$ removes
  eigenvalue amplification and remains far above matched orientation nulls.
- **Arbitrary direction:** weakened in E03-S by the flat-rotation null and in
  checkpoint audits by singular-value-preserving random orientations.
- **Target already favors head:** controlled away in E03-S, but still possible
  in real joint training because expert advantage evolves.
- **Representation-only alignment:** insufficient as a complete endpoint
  explanation; fixed-basis Gate effects are measurable, but basis motion is
  material and sometimes opposes them.
- **Adaptive optimizer:** untested as a transfer mechanism; AdamW may flatten,
  preserve, or reshape covariance time constants.
- **Load-collapse artifact:** decisive for E03-R validity. It prevents causal
  interpretation of later spectral motion.
- **Different geometry equals function:** rejected by the compatibility gate;
  random high-dimensional views also create novel neighborhoods.
- **Shallow head is useless:** not tested. The shallow pilot lost specificity
  because both head and random probes reached the accuracy ceiling.

## 13. Claim Boundary

**Established:** actual-input Gate orientation is strongly head-dominant after
energy equalization in the audited checkpoint lineages; middle/tail access is
nonzero; continual post-10k head sharpening is not supported; covariance
anisotropy causally changes finite-time mode-learning speed in the registered
fixed-target pure-SGD construction; fixed M/T/N bands fail the registered
cross-lineage local functional-admission gate.

**Not established:** the exact origin of the existing DCLM endpoints; a valid
real Router--Expert head-formation trajectory; positive expert feedback;
semantic meaning of covariance ranks; middle/tail functional absence; shallow-
head training benefit; or validation-loss improvement per matched FLOP.

**Do not infer:** “linear Gates can only see head,” “all anisotropic spectra
force final head alignment,” “post-collapse contrast growth is formation,” or
“failure of fixed bands rules out every spectral or function-aligned method.”

## 14. Anchor Decomposition

- **A15_00:** endpoint access and saved-checkpoint allocation are recorded.
- **A15_00_01:** controlled covariance-speed clause passes; real transfer is
  `insufficient_load_guard`; trainable-expert S2 remains unexecuted.
- **A15_01 / A15_01_01:** shallow-to-deep mechanism remains open; the current
  Stage-A specificity operationalization is non-discriminating.
- **A15_02 / A15_02_01:** fixed M/T/N bands fail local compatibility admission,
  so conditional matched training remains blocked.

These branches answer different causal levels. None may substitute static
alignment for compatibility or compatibility for matched-training benefit.

## 15. Relation To The Mainline

A15 is a mechanism and admission line supporting, but not replacing, the A06
functional-specialization mainline. A06 requires held-out expert utility under
controlled routing. A15 explains why a Gate may become spectrally biased and
tests whether fixed spectral coordinates deserve training compute. Current A15
evidence does not establish useful expert specialization or a deployable
Router design.

## 16. Exactly One Next Decision

For the Q1 dynamics scope, decide whether to approve a new E03-R Protocol with
a separately validated non-gradient load-stability mechanism, a frozen
attribution boundary, and a small pre-full stability gate. Completion requires
all three seeds to pass beyond the previous failure window without maximum
20-step expert share above 0.8 or dead-expert failure before any 2B-token run
is authorized.

The current E03-R trajectories must not be resumed under a changed rule, E03-S
S2 must not launch automatically, and A15_02 matched training remains blocked.

## Source Map

- [A15 line index](../../problem_anchors/15_linear_gate_spectral_training_bias/README.md)
- [A15_00 anchor](../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor.md)
- [E03 dynamics subanchor](../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md)
- [E01 result](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary.md)
- [E02 result](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary.md)
- [E03-S result](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
- [E03-R result](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
- [Shallow pilot result](../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md)
- [Compatibility-gate result](../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary.md)
- [Controlled theory article](../../../daily_research_reports/0731/router_spectral_learning_dynamics_theory_package/01_理论论文_线性MoE_Router的频谱学习动力学.md)
