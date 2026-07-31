---
experiment_id: A15_01_01_E01_controlled_four_layer_shallow_head_pilot
status: approved_for_full_execution
canonical_language: en
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol: Controlled Four-Layer Shallow-Head Pilot

## 0. Approval Snapshot

The researcher authorized protocol completion, implementation, and smoke and
required a genuine normal four-layer control. The formal comparison therefore
contains native N4 plus parameter-matched head, random, and shuffled side-channel
arms. No load-balance auxiliary loss is used; every arm shares the same
non-gradient auxiliary-loss-free expert bias. After the repaired smoke passed
all eleven engineering guards, the researcher explicitly authorized full
execution on 2026-07-30.

The one question is whether layer-2 head coefficients, after passing an
independent co-training-compatibility gate, reduce held-out NLL per matched
training FLOP in layers 3--4 relative to normal training and matched controls.
Resource: ACP, one node, idle 8x5090, profile `5090-8-spot`.

Primary anchor: [A15_01_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_01_shallow_head_guided_deep_routing/subanchors/15_01_01_controlled_four_layer_shallow_head_pilot_anchor.md).

## 1. Terminology And Definitions

N4 is a genuine four-block top-1 MoE whose deep Gates read only their current
layer inputs. H2 adds same-token coefficients from covariance ranks 1--64 of
the actual layer-2 Gate input. R2 uses a frozen same-dimensional Haar subspace;
SH2 uses another token's true head coefficients. One-step cross-update
compatibility is the symmetric held-out loss benefit after updating one expert
on an independent token group. Incremental held-out $R^2$ decides admission;
held-out NLL at matched cumulative training FLOPs decides the training claim.

## 2. Anchor Alignment And Decision Question

This pilot tests the distinct mechanism in which an already formed shallow
head guides deeper routing. It does not test local middle/tail routing and is
not implied by A15_00 head alignment. Compatibility must pass before any
training-effect comparison.

## 3. Hypotheses And Rivals

H1 requires layer-2 head features to provide held-out residual compatibility
prediction beyond native-score/load/norm/outlier/position controls and beyond
random, shuffled, wrong-layer, and batch-resampling controls. H2 then requires
H2 NLL below N4, R2, and SH2 at matched FLOPs on the informative task, without
the same ordering on a nuisance task. Rivals are generic side capacity, extra
compute, load redistribution, and a generator shortcut.

## 4. Controlled Data And Splits

Each sample is one token with coarse identity $c\in[8]$, transformation family
$r\in[8]$, content $v\sim N(0,I_{32})$, and nuisance position $p\in[32]$.
A fixed orthogonal encoder creates a 256-dimensional input with a 64-dimensional
high-variance coarse code, 128-dimensional content code, and 64-dimensional
low-variance family/position code. Marginals, trace, labels, and noise are
matched across tasks.

Informative sets $r=c$; nuisance samples $r$ independently while encoding it in
the low-variance block. Labels are
$y=\arg\max_{k\le16}(M_rv)_k$ for frozen registered matrices $M_r$.
Training, Stage-A validation, compatibility fit/validation/test, and B1 held-out
evaluation use independent RNG streams. Full paired seeds are 3101--3105.

## 5. Four-Layer Model And Router

Use four residual top-1 MoE blocks, width 256, eight experts per block, expert
MLP width 512 with GELU, no shared expert, and no hard capacity limit. Native
Gates are bias-free `Linear(256,8)`. For layers 3--4,

$$
z_\ell=W_\ell g_\ell+A_\ell s_2,
$$

where $s_2$ is head, random, or shuffled and each $A_\ell\in\mathbb R^{8\times64}$
is zero-initialized. N4 has no adapter. Every arm uses the same zero-initialized
non-gradient load bias with update speed $10^{-3}$, clip [-0.1,0.1], and expert
centering; `lambda_lb=0`.

## 6. Training Stages And Algorithm

Stage A trains layers 1--2 with
$L_A=\operatorname{CE}(\hat c,c)+0.1\|\hat v-v\|^2$, then fits and freezes
$\mu_2,U_{2,H}$ on an independent actual-input calibration set. Full guards are
coarse accuracy at least 0.90, content explained variance at least 0.80,
head-only probe accuracy at least 0.85 and above 256 random-subspace q95, and
split-half projector overlap at least 0.80.

The frozen full Stage-A optimizer is AdamW for 500 steps, batch 512, constant
LR $3\times10^{-4}$, betas (0.9,0.95), and weight decay 0.01. Capability
validation has 4,096 samples; the two projector-calibration halves have 2,048
samples each; the head probe uses 2,048 fit and 4,096 test samples. Exactly 256
registered 64-dimensional Haar-random probes define q95.

Stage B0 attaches layers 3--4 and the 16-class head and performs 300 native-only
steps. Require held-out accuracy 0.25--0.75, NLL at least 0.10 nat below uniform,
and max expert share at most 0.60. Freeze one shared B0 checkpoint per seed/task.
Its frozen optimizer is AdamW, batch 512, constant LR $10^{-4}$, betas
(0.9,0.95), weight decay 0.01, with 4,096 held-out samples.

For Stage 0, form disjoint token-group pairs $(A,B)$ and update only the current
native-routed deep expert by one B1-sized SGD step. Define

$$
K(A,B)=-\tfrac12[(L_B(\theta-\eta\nabla L_A)-L_B(\theta))
+(L_A(\theta-\eta\nabla L_B)-L_A(\theta))].
$$

The base ridge model contains native deep-score similarity, route equality,
load stratum, representation norms, Mahalanobis outlier scores, and position.
Candidate features are layer-2 head, same-dimensional random, token-shuffled
head, and layer-1 wrong-basis similarities. Pair splits are 60/20/20 for fit,
hyperparameter selection, and untouched test.
Each seed/task uses 256 disjoint pairs of group size four from a 32,768-token
pool. Ridge features are standardized on the fit split only; alpha is selected
by validation MSE from
$\{10^{-4},10^{-3},10^{-2},10^{-1},1,10,100\}$. Aggregate uncertainty uses
2,000 paired seed-group bootstrap repetitions; the registered null retains
1,000 batch-label resamples.

Only after Stage 0 passes, Stage B1 clones N4/H2/R2/SH2 from B0. Run at most
2,000 steps with batch 512 and AdamW LR $3\times10^{-4}$, betas (0.9,0.95),
weight decay 0.01, and cosine decay. Paired arms share data order and all
common initialization.
Evaluate NLL/FLOPs/margin/route flips/load/bias at step 0, every 50 steps, and
step 2,000. Record expert conflict and redundancy at every 200 steps and final.

## 7. Conditions And Matching

N4 is the genuine native control. H2/R2/SH2 have identical adapter parameters
and FLOPs. N4 has fewer parameters, so it is compared at matched cumulative
FLOPs. All arms share B0, optimizer, tokens, batch, data order, load-bias rule,
zero capacity drop, evaluation sets, and no early stopping.

## 8. Primary Metrics

Admission uses

$$
\Delta R_X^2=R^2(\text{base}+X)-R^2(\text{base})
$$

on untouched compatibility-test pairs. H2 needs a paired group-bootstrap lower
bound above zero, positive lower bounds over R2/SH2/wrong-layer, and a value
above 1,000 batch-label-resampling null q95.

For training, freeze

$$
F_\star=F_{N4}(2{,}000\ \text{B1 steps})
$$

using the pre-run analytic forward+backward+update MAC counter. Interpolate all
curves at $F_\star$ and compute
$\Delta L_{H-C}=L_H-L_C$ for $C\in\{N4,R2,SH2\}$. Units are nat/token. All
three exact paired-permutation 95% intervals across five seeds must be below
zero.

## 9. Secondary Metrics

Report FLOPs/tokens to registered NLL levels; margin, route flips, load, dead
experts, bias norm/saturation; expert update norm and within-expert group
gradient conflict; expert-output functional redundancy; basis capture/stability;
parameter count, analytic FLOPs, measured step time, and peak memory. These
explain paths but cannot replace matched-FLOP NLL.

## 10. Known Cases

All arms must have identical step-0 logits/routes/outputs. SH2 has no fixed
points and inverse replay recovers H2. Random is orthogonal and 64-dimensional;
wrong-layer must not be refit to layer 2. Oracle-$c$ grouping should be compatible
only in informative, while oracle-$r$ dominates nuisance. Zero side coefficients
must exactly reproduce N4.

## 11. Logging And Figure Contract

Persist config/code/data hashes, seed/task/arm, Stage A/B0 guards,
compatibility-split hashes, per-step NLL/FLOPs/load/bias/margin, evaluations,
and manifests. The central NLL-versus-FLOPs plot has informative and nuisance
panels and may support only the controlled matched-compute claim. The admission
plot compares incremental $R^2$ with controls and resampling q95. The central
table reports all three NLL contrasts, paired intervals, and matching guards.

## 12. Execution Contract

The authorized smoke is one 8x5090 ACP job whose eight ranks cover two tasks by
four arms. Each rank deterministically reconstructs the task-shared Stage-A/B0
state and verifies matching hashes. Engineering reductions are 32 Stage-A
steps, 16 B0 steps, eight compatibility pairs, and eight B1 steps. Geometry,
side dimension, N4, no-LB setting, and bias rule remain unchanged. Smoke passes
when all arms finish, B0 hashes match, step-0 equivalence and split isolation
hold, cross-update values are finite, the FLOP counter distinguishes N4 from
side arms, bias cadence/no-gradient holds, and manifests are complete. Smoke
does not apply scientific capability, compatibility, or NLL gates.

Full execution is authorized as 5 paired seeds x 2 tasks x 4 arms. B1 remains
strictly blocked whenever the registered Stage-0 admission test fails.
All five seeds complete Stage A, B0, and Stage 0 before the global informative
admission verdict. Any Stage-A/B0/data/gradient guard failure is insufficient
and blocks B1; a valid informative Stage-0 non-admission is scientific fail and
also blocks B1.

## 13. Pass / Fail / Insufficient

Pass requires valid Stage A/B0, informative Stage-0 admission, H2 below all
three controls at $F_\star$, matching guards, and no identical three-control
advantage in nuisance. Valid admission failure, or post-admission failure to
beat all controls, is scientific fail. Any capture/capability/precision/seed/
split/load/bias/FLOP/reproducibility guard failure is insufficient.

## 14. Claim Boundary And Next Decision

At most, passing establishes a controlled benefit after freezing a two-layer
trunk and layer-2 basis. It cannot establish from-scratch end-to-end benefit,
online PCA, DCLM or natural-language gains, middle/tail uselessness, the E03
covariance theorem, or large-scale efficiency.

The only next decision is the registered scientific verdict after Stage A/B0,
Stage-0 admission, and—only when admitted—the matched-FLOP B1 comparison are
complete across all five paired seeds.

Protocol freeze before full execution: Stage-A and B0 budgets, optimizer
details, held-out/calibration sizes, compatibility pair construction, ridge
grid, and evaluation cadence above were fixed using capability-only preflight.
The B0 LR was set to $10^{-4}$ because seed-3101 native calibration stayed
inside the pre-registered B0 capability window; no H2/R2/SH2 B1 result was run
or inspected. A preflight warning that random probes may match the head probe
does not alter the generator, random-control count, threshold, or fail-closed
Stage-A rule.
