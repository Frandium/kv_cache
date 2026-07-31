---
experiment_id: A15_00_E03_S_controlled_spectral_learning_dynamics
status: approved_for_full_execution
canonical_language: en
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol: E03-S Controlled Spectral Learning Dynamics

## 0. Approval Snapshot

The researcher approved the snapshot and authorized protocol completion,
implementation, and smoke execution on 2026-07-30. After the registered smoke
passed every engineering guard, the researcher explicitly authorized full
execution on 2026-07-30.

- **One question:** with direction-matched expert-advantage signal, does input
  covariance anisotropy causally shorten the learning time of high-variance
  linear-Gate modes?
- **Role:** controlled root-cause audit, not a Router-method test.
- **Primary metric:** middle/head and tail/head time ratios to 50% target fit.
- **Resource:** ACP, one node, idle 8x5090, profile `5090-8-spot`.

Primary anchor: [A15_00_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md).

## 1. Terminology And Definitions

Let $A_{raw}$ be the expert-centered, direction-matched score coefficient,
$\tau=0.25$, and $A_{gate}=\tau A_{raw}$ be the coefficient in the Gate-logit
parameterization. For $C_E=I_E-\mathbf1\mathbf1^\top/E$ and $\bar W=C_EW$,
define

$$
F_B(t)=1-
\frac{\|(\bar W_t-A_{gate})U_B\|_F^2}
{\|(\bar W_0-A_{gate})U_B\|_F^2}.
$$

Head, middle, and tail are covariance ranks 1--64, 65--320, and 321--768;
the fine resolution is twelve consecutive 64-direction bands. $T_B(0.5)$ is
the linearly interpolated first optimizer step at which $F_B$ reaches 0.5 and
remains there for two registered evaluations. Non-crossing observations are
right-censored. The metric measures learning speed, not functional utility.

## 2. Anchor Alignment And Decision Question

The theory establishes a conditional covariance multiplier for a fixed basis,
locally linear expert advantage, and unwhitened gradient flow. E03-S separately
controls the expert target, covariance spectrum, and optimizer. It decides only
whether the controlled causal root exists; E03-R separately checks a real-workload
signature.

## 3. Hypotheses And Rivals

H1 predicts

$$
R_{M:H}=T_M(0.5)/T_H(0.5)>1,\qquad
R_{T:H}=T_T(0.5)/T_H(0.5)>1,
$$

for moderate and strong spectra, a positive strong-minus-moderate dose effect,
and return to the flat null after whitening.

The strongest rivals are target-direction imbalance, total-energy/effective-LR
change, tail inexpressibility, and trainable-expert feedback. Equal-norm columns
of $A_{raw}$, trace-normalized spectra, tail-only, whitening, and frozen-versus-joint
stages isolate these rivals.

## 4. Data And Splits

- $d=768$, $E=8$, one fixed Haar basis $U$ per seed.
- $s\sim N(0,I)$ and $x=U\Lambda^{1/2}s$.
- Raw H:M:T eigenvalue ratios are flat 1:1:1, moderate 4:2:1, and strong
  16:4:1; each is rescaled so $\operatorname{tr}(\Sigma)/d=1$.
- $A_{raw}\in\mathbb R^{8\times768}$ is expert-centered and every column has
  equal norm; $A_{gate}=0.25A_{raw}$. Conditions share both, initialization,
  and latent sample stream.
- Independent train, trajectory-evaluation, and final-held-out streams.
- Full seeds: 20260730--20260737.

For S1, $q(x)=\operatorname{softmax}(A_{gate}x)
=\operatorname{softmax}(0.25A_{raw}x)$ and the loss is
$-\mathbb E[q^\top\log p_W]$. A held-out target entropy below
$0.35\log E$ invalidates the seed. Strong-whitened preserves the same target
while presenting $\Lambda^{-1/2}U^\top x$ to the Gate.

## 5. Model And Optimizer

S0 integrates the exact quadratic dynamics with $\kappa=1$, $\beta=0$, and
zero initialization. S1 uses `Linear(768, 8, bias=False)`, pure SGD with LR
0.02, no momentum or weight decay, batch size 4096, and at most 8,000 steps.
Evaluate every 10 steps through 400 and every 50 thereafter. S0 and decision
statistics use float64; S1 may train in float32.

S2 runs only after full S1 passes. Eight trainable two-layer experts learn a
fixed teacher, with matched teacher energy across spectral directions. Report
only paired joint-minus-frozen effects and the evolving expert-advantage
spectrum.

## 6. Conditions

S0: flat/moderate/strong. S1: flat/moderate/strong, strong-whitened, and
strong tail-only. S2: frozen versus trainable experts. Tail-only has exactly
zero head/middle target columns and is judged by tail fit and held-out KL.

## 7. Matching And Guards

Hold fixed $A_{raw}$, $A_{gate}$, $U$, initialization, samples, optimizer, batch, budget,
evaluation grid, precision, and analysis. Require trace relative error
$\le10^{-6}$, column-norm relative spread $\le10^{-6}$,
$C_EA_{gate}=A_{gate}$, S0 maximum
relative error $\le10^{-5}$, finite losses/gradients, and at least 50% held-out
KL reduction for tail-only.

## 8. Primary Metric

For every seed,

$$
D_{M:H}=\log T_M-\log T_H,\qquad
D_{T:H}=\log T_T-\log T_H.
$$

Use paired seed medians; there is no arbitrary practical-effect threshold.
The flat-rotation null uses 256 Haar partitions of matching dimensions per
flat seed. Both anisotropic contrasts must exceed matched null q95. The dose
effect uses an exact paired sign/permutation 95% interval for strong minus
moderate.

## 9. Secondary Metrics

Report fine-band learning curves and times, equal-energy Gate gain
$G_B=\|C_EWU_B\|_F^2/d_B$, held-out KL, gradient norm, maximum logit, target
and predicted entropy, and S2 expert-advantage/update/conflict/load diagnostics.
Wall time and memory are engineering metrics only.

## 10. Known Cases

S0 is the positive control; flat rotation is the null; strong-whitened should
return to the flat envelope; tail-only must learn its nonzero target. A debug
head-biased $A_{raw}$ must be rejected by the target-energy guard.

## 11. Profiling And Figure Contract

Persist config, code hashes, seed, environment, all evaluation metrics,
crossing/censoring state, and guards. The central plot shows $F_B(t)$ versus
optimizer step for H/M/T across spectrum panels. It may establish only the
ordering of matched target-fit times. The central table reports all $T$, ratios,
null q95, censoring, and guards; S2 remains separate.

## 12. Execution Contract

The authorized smoke is one 8x5090 ACP job with one process per GPU and one
fixed seed per process. It runs all S0 conditions, 128 S1 steps per condition,
and 16 S2 wiring steps without changing $d/E$/bands/spectra/target definitions.
Smoke passes when all ranks finish, S0 error passes, all metrics are finite,
tail-only has nonzero gradients, whitening covariance passes, and the manifest
is complete. Smoke never receives the scientific H1 verdict.

Full execution is authorized. It uses eight seeds, the 8,000-step cap, and the
256-rotation null. S2 remains gated on the frozen full-S1 verdict and must not
start merely because the infrastructure run succeeds.

## 13. Pass / Fail / Insufficient

Pass requires valid S0/guards, both moderate and strong contrasts above flat
q95, a positive dose effect, whitened return to flat, and tail capability.
Fail requires valid controls but absent anisotropic timing effects or a
non-removed whitening effect attributable to target direction. Any failed
positive control, matching/capability/numerical guard, censoring problem, or
missing registered seed is insufficient.

## 14. Claim Boundary And Next Decision

At most, this experiment can establish a causal finite-time covariance-speed
bias for the registered fixed-basis linear Gate with matched expert advantage
and SGD. It cannot establish AdamW equivalence, the DCLM cause, universal head
alignment, middle/tail uselessness, expert positive feedback, or loss/FLOP
improvement.

The only next decision is the registered S1 scientific pass/fail/insufficient
verdict after the full trajectories, rotation null, whitening return, and tail
capability guards are complete.

Protocol clarification before full execution: the smoke implementation and
target distribution always used temperature $\tau=0.25$, but the original
written $F_B$ expression omitted that scale. Because cross-entropy is optimized
at $W=A_{gate}=\tau A_{raw}$, all fit fractions and crossing times compare to
$A_{gate}$. For the whitened condition, the learned weight is first mapped back
to the original spectral coordinate before this comparison. No data, target
distribution, threshold, condition, seed, or training hyperparameter changed.
