---
anchor_id: 15_00_01_spectral_learning_dynamics
parent_anchor: 15_00_covariance_head_gate_alignment
status: controlled_pass_real_insufficient_load_guard
canonical_language: en
companion_language: zh
updated: 2026-07-30
---

# A15_00_01 Conditional Spectral Learning Dynamics Of A Linear Router


## 1. Problem Definition

The parent establishes equal-energy covariance-head alignment at trained
endpoints but does not identify its pre-10k cause. This subanchor asks one
question:

> After matching the spectrum of expert-advantage signal and separating
> optimizer and representation motion, does covariance anisotropy causally
> shorten the learning time of high-variance Gate modes, and is that signature
> present during the first 2B tokens of a real MoE run?

“Spectral acceleration” means a shorter time to reach the same registered
fraction of a direction's own routing target. It is not raw-logit amplification,
functional utility, or a claim that every anisotropic problem ends head-aligned.

The primary controlled metric is

$$
R_{M:H}(\rho)=\frac{T_M(\rho)}{T_H(\rho)},
\qquad
R_{T:H}(\rho)=\frac{T_T(\rho)}{T_H(\rho)},
$$

where $T_B(\rho)$ is optimization time or tokens for band $B$ to reach fraction
$\rho$ of its known target fit. The ratios are dimensionless. They decide
learning speed, not downstream expert value.

## 2. Physical Priors

1. **Conditional covariance multiplier.** Near a balanced unsaturated Gate,
   the update is proportional to expert-advantage/input cross-covariance. If
   advantage coefficients are matched, larger input eigenvalues shorten the
   mode time constant.
2. **Isotropic symmetry.** A flat spectrum supplies no covariance-defined
   direction; systematic alignment with a preselected basis must come from the
   task, optimizer, or finite-sample symmetry breaking.
3. **Expert feedback is not automatic.** Trainable experts can amplify,
   compensate, or reverse the initial ordering because their relative losses
   define the routing target.

## 3. Falsifiable Hypotheses

**H1:** With spectrally balanced fixed expert advantage, anisotropy yields
$R_{M:H}>1$ and $R_{T:H}>1$ with an ordered dose response; flat or whitened
inputs remove the ordering. A real run forms positive equal-energy head
contrast early, with raw-gradient, applied-update, and representation
contributions separately identifiable.

**Strongest rival R1:** The expert-advantage target is itself head-oriented;
covariance is only correlated with the target. Rotating or moving the target to
tail changes the learned direction independently of eigenvalues.

**R2:** Adaptive optimization or representation-basis drift creates the
observed endpoint alignment. Fixed representations pass while real
$W_t\times U_t$ decomposition does not attribute it to Gate updates.

**Pass:** The fixed-advantage causal test supports H1, whitening removes the
speed ordering, the tail-only control remains learnable, and the real run shows
the registered early signature.

**Fail:** Under valid matched signal, anisotropy does not change learning time,
or the apparent effect survives whitening and is explained by target
orientation.

**Insufficient:** Capability, expert-advantage, numerical, checkpoint-density,
or real-training stability guards fail; or controlled causality passes but the
real signature is unresolved.

## 4. Mathematical Model

For actual Router input $x$ with $\Sigma=U\Lambda U^\top$, let
$\bar W=C_EW$ and let centered expert advantage be $a(x)=-C_E\ell(x)$. At a
balanced softmax Gate,

$$
\dot{\bar W}(0)=\frac1E\mathbb E[a(x)x^\top].
$$

If $a(x)=Ax+\varepsilon$ with
$\mathbb E[\varepsilon x^\top]=0$, then the initial mode update is

$$
\dot{\bar W}(0)u_i=\frac{\lambda_i}{E}Au_i.
$$

The registered local quadratic model is

$$
\dot w_i=-(\kappa\lambda_i+\beta)w_i
+\kappa\lambda_i a_i,
$$

whose mode time is

$$
T_i(\rho)=\frac{-\log(1-\rho)}{\kappa\lambda_i+\beta}.
$$

The proof and counterexamples are in the
[self-contained theory note](../../../../../daily_research_reports/0731/router_spectral_learning_dynamics_theory_package/01_理论论文_线性MoE_Router的频谱学习动力学.md).
The model cannot determine the time-varying expert advantage $A_t$ in a jointly
trained MoE.

## 5. Computational Realization

**E03-S** uses a known covariance basis, an eight-output linear Gate, and a
registered expert-advantage target. Flat, anisotropic, whitened, and tail-only
conditions isolate covariance, task orientation, and preconditioning. A
trainable-expert stage is conditional on the fixed-advantage gate.

**E03-R** trains a small top-1 DCLM MoE from initialization through at most 2B
tokens. It records $W_t$, raw Gate gradients, optimizer-applied updates, a
fixed-probe $U_t$, $W_s\times U_t$ crossings, signed band cross terms, margin,
flip, load, and the singular spectrum of $C_EW_t$. It uses no load-balance
auxiliary loss. A checkpointed, non-gradient expert-score bias is shared across
layers as a load-stability guard and is reported separately from $W_t$.

## 6. Minimal Falsification Tests

1. Under fixed advantage, compare flat against at least one anisotropic
   spectrum using identical target, initialization distribution, optimizer,
   samples, and update budget.
2. Whiten the anisotropic input and require the registered speed ordering to
   disappear; move all target advantage to tail and require tail learnability.
3. Only after the fixed-advantage test is valid, compare frozen and trainable
   experts to detect additional feedback amplification.
4. In real training, define the first persistent head-alignment time from a
   matched orientation null and decompose it into Gate-weight and basis motion.

## 7. Current Evidence

[E01](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary.md)
and
[E02](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary.md)
establish strong equal-energy head alignment on the actual Router input,
nonzero middle/tail access, and post-10k broadening. They do not observe the
formation event or separate expert advantage, optimizer, and representation
causes.

E03-S records are the
[Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/protocol.md),
[Summary](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md),
and
[Detailed evidence ledger](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/detailed.md).
The full run produced a scientific **PASS for the registered controlled S0/S1
clause only**.

The audited target definition is

$$
A_{gate}=\tau A_{raw},\qquad \tau=0.25,
$$

where $A_{raw}$ is the direction-matched expert-score coefficient and
$A_{gate}$ is the reachable Gate-logit target. Consequently, $F_B$ and $T_B$
compare $W_t$ with $A_{gate}$; in the whitened condition the learned weight is
first mapped back to the original spectral coordinates.

Across eight seeds, moderate 4:2:1 anisotropy gave median
$(D_{M:H},D_{T:H})=(0.69268,1.38588)$ and strong 16:4:1 anisotropy gave
$(1.38477,2.77145)$, both above the matched flat-rotation q95 values
$(0.003277,0.003019)$. Strong whitening returned the medians to
$(0.000053,0.000637)$ inside the flat 95% envelopes, and the tail-only target
reduced held-out KL by at least 0.9999939 in every seed. Thus, in this
fixed-basis Gaussian, pure-SGD construction, covariance anisotropy causally
changes finite-time Gate-mode learning speed; the slower tail is not explained
by inability to express or learn a tail target.

This result does not adjudicate why an actual DCLM Router becomes head-aligned.
E03-R's formal records are the
[Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/protocol.md),
[Summary](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md),
and
[Detailed evidence ledger](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/detailed.md).
Its three frozen-source runs ended with scientific
**INSUFFICIENT (`insufficient_load_guard`)**. A single expert exceeded 80% of a
20-step layer load in every seed, with first violations ending at steps 72, 75,
and 79 and rolling maxima near 0.99 by step 100. The jobs were stopped because
this historical validity guard could not recover.

No selected valid point through step 50 was a two-contrast orientation-null
candidate. Seeds 29 and 43 crossed both orientation-null q95 values at step 120,
but only after near-single-expert load concentration, so those observations
were not eligible for basis bootstrap or $T_{form}$. Actual-input replay,
basis, raw/applied identity, capacity, source-freeze, and analysis-closure
guards passed. Thus E03-R leaves the real formation question unresolved; it
does not provide a negative result for the controlled E03-S mechanism. S2
remains eligible because S1 passed but was not run; eligibility is not evidence
for positive expert feedback and does not trigger S2 automatically.

## 8. Claim Boundary And Next Decision

**Supported:** with a matched reachable Gate target, fixed Gaussian
representation, trace-normalized spectra, and pure SGD, larger covariance
eigenvalues shorten the finite time needed to learn the corresponding linear
Gate modes. Flat-spectrum, rotation-null, whitening, dose-response, and
tail-capability controls support this controlled causal statement.

**Weakened within that controlled system:** target orientation, finite-sample
directional asymmetry, raw-logit energy amplification, and tail
inexpressibility do not explain the registered speed ordering.

**Unresolved:** whether the same signature forms in a load-stable real DCLM
trajectory, whether it is produced by raw Gate gradients, AdamW-applied
updates, or representation-basis motion, and whether trainable experts amplify
it. The registered E03-R score-bias condition failed as a stable carrier for
that question. S2 remains a separate, eligible but unexecuted experiment rather
than an automatic continuation.

This subanchor cannot claim that covariance caused the existing DCLM endpoint,
that every trained Router must align with the covariance head, that the
post-collapse E03-R contrasts are head formation, that middle or tail
directions lack functional value, that expert feedback is positive, or that
spectral routing improves validation loss per FLOP.

**Exactly one next decision:** decide whether to approve a new E03-R Protocol
with a separately validated load-stability mechanism, a frozen attribution
boundary, and a small pre-full stability gate before any new 2B-token run.
