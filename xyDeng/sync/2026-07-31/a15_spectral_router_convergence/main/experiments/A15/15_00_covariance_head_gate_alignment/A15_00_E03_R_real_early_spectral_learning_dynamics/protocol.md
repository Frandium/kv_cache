---
experiment_id: A15_00_E03_R_real_early_spectral_learning_dynamics
status: approved_for_full_execution
canonical_language: en
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol: E03-R Real Early Router Spectral Dynamics

## 0. Approval Snapshot

The researcher authorized protocol completion, implementation, and smoke. The
approved revision removes all load-balance auxiliary loss and uses a shared
auxiliary-loss-free expert-score bias as an engineering anti-dead-expert guard.
The bias is non-differentiable, logged separately, and excluded from spectral
Gate weights. The first smoke exposed a mixed-precision replay mismatch; after
the corrected same-batch GPU-fp32 replay smoke passed every registered guard,
the researcher explicitly authorized full execution on 2026-07-30.

The one question is whether a six-layer DCLM MoE develops, before 2B tokens, an
early head-formation signature homologous to the controlled E03-S dynamics.
The primary metric is formation time $T_{form}$ in training tokens. Resource:
ACP, one node, idle 8x5090, profile `5090-8-spot`.

Primary anchor: [A15_00_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md).

## 1. Terminology And Definitions

The actual Router input $g_{\ell,t}$ is the post-attention RMS-normalized tensor
passed directly to `mlp.gate`. H/M/T are covariance ranks 1--64/65--320/321--768;
fine resolution is twelve blocks of 64. Equal-energy gain is
$G_{\ell,B}=\|C_EW_\ell U_{\ell,B}\|_F^2/d_B$ and contrasts are log H:M and H:T
gain ratios. Raw Gate gradient is the accumulated LM gradient before optimizer
preconditioning; applied update is the measured post-minus-pre AdamW weight.

## 2. Anchor Alignment And Decision Question

E01/E02 begin at roughly 7.86B nominal tokens. E03-R starts at initialization
and densely separates Gate-weight motion, representation-basis motion, and
optimizer transformation. It tests a real-workload signature only; E03-S owns
the covariance causal claim.

## 3. Hypotheses And Rivals

H1 predicts finite $T_{form}$ in at least two of three seeds, positive fixed-basis
Gate contribution during formation, a head-prioritizing raw gradient or applied
update signature, and possible later middle/tail catch-up. Rivals are an
intrinsically head-biased expert advantage, basis rotation, AdamW rewriting, and
the non-gradient load-bias intervention. The first remains unresolved here;
crossings, raw/applied logging, and bias counterfactuals audit the others.

## 4. Data And Splits

Train on the DCLM binary stream at
`/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10`, sequence
length 1024, global batch 768, and 786,432 nominal tokens per optimizer step.
Step 2,544 is 2.000683008B nominal tokens. A disjoint fixed 32x256 calibration
buffer fits each basis; 64x256 held-out documents compute response and routing
metrics. Token hashes are shared across seeds. Full seeds are 17, 29, and 43.

## 5. Model, Router, And Optimizer

Use six decoder layers, hidden width 768, six attention heads, three KV heads,
eight sparse plus one shared expert per layer, top-1 routing, and expert
intermediate width 1536. The linear Gate is uncentered. Optimize LM loss only
with `lambda_lb=0`. AdamW uses LR $10^{-4}$, betas (0.9,0.95), epsilon
$10^{-8}$, weight decay 0.01, 1,000-step linear warmup, and a 100B-token cosine
horizon. Forward/backward is bf16; diagnostics use float32/float64. Exact resume
includes model, optimizer, scheduler, sampler, RNG, score bias, and diagnostics.

Scores are $z_{\ell,e}=W_{\ell,e}g_\ell+b_{\ell,e}$. After globally aggregating
one optimizer step of top-1 counts,

$$
b_{\ell,e}\leftarrow\operatorname{clip}
\left[b_{\ell,e}+10^{-3}
\frac{\bar c_\ell-c_{\ell,e}}{\bar c_\ell+10^{-6}},-0.1,0.1\right],
$$

then center $b_\ell$ across experts. Biases are zero-initialized buffers with no
gradient and exactly one update per optimizer step.

## 6. Conditions

The scientific run has one registered condition: native linear Gate plus the
specified non-gradient bias. Every probe snapshot replays actual $Wg+b$,
weight-only $Wg$, and bias-only routing. No whitening, centering, or alternative
Router method is introduced.

## 7. Matching And Guards

Lock geometry, initialization algorithm, data-order rule, optimizer, scheduler,
bias rule, probe tokens, save grid, and analysis. Require actual-input logit
replay relative error $\le10^{-5}$ and top-1 agreement 1; leak-free hashed splits;
basis orthogonality $\le10^{-4}$; diagnostic forward/gradient invariance; bias
outside the optimizer with one update per step; exact-resume replay; finite
values and all six layers; and no 20-step window with max load share above 0.8
or at least four dead experts.

## 8. Primary Metric

At every heavy snapshot, aggregate contrasts by the median across six eligible
layers. A 256-sample Haar-Stiefel null preserves each $C_EW$ singular spectrum
while randomizing its right subspace. $T_{form}$ is the earliest token count at
which both contrasts exceed matched null q95, both paired calibration-bootstrap
lower bounds exceed zero, at least four layers have both positive contrasts,
and all conditions persist for two following heavy snapshots.

## 9. Dynamics Decomposition And Secondary Metrics

Compute the full $B(W_s,U_t)$ crossing and fixed-basis $\Delta_WB$ versus
fixed-Gate $\Delta_UB$. At every optimizer step, project raw gradients and
applied updates into the most recent basis and record

$$
C_B=2\langle C_EWU_B,C_E\Delta WU_B\rangle_F/d_B,
\quad Q_B=\|C_E\Delta WU_B\|_F^2/d_B,
$$

checking $\Delta G_B=C_B+Q_B$. Also report fine bands, $V/S$, flips, margins,
actual/weight-only loads, dead experts, zero capacity drop, bias norm/update,
expert update/conflict diagnostics, and Gate singular values/stable rank.

## 10. Known Cases

Random initialization must be covered by the orientation null. Randomized Gate
right subspaces must erase the original head result. Frozen-$U$ replay sets
$\Delta_UB=0$; frozen-$W$ replay sets $\Delta_WB=0$; bias-only changes may not
alter $G_B(W,U)$.

## 11. Logging And Figure Contract

Log tokens/LR/loss, raw gradient, applied update, signed cross terms, clipping,
margin, load, dead experts, bias, and step time every optimizer step. Heavy
snapshots occur at 0; every step 1--100; every ten steps 101--1000; then
1100/1250/1500/1750/2000/2250/2544. The central trajectory plot shows both
contrasts, null q95, seeds, and $T_{form}$; it establishes timing, not cause.
The decomposition figure separates raw gradient, applied update, Gate-weight,
and basis effects; it cannot exclude expert-advantage spectrum.

## 12. Execution Contract

The authorized smoke is one eight-GPU DDP ACP job, seed 17, with the same
six-layer/768/8E/top-1/no-LB/bias implementation. Engineering reductions are
sequence length 256, global batch 16, fresh steps 0--24, checkpoint at 24, and
exact resume to 26; heavy snapshots are 0/1/2/4/8/16/24/26 with 8 calibration
and 8 probe sequences. It passes only if GPU/NCCL, DCLM loading, fresh/resume,
actual-input replay, raw/applied identity, basis, non-gradient bias cadence, and
manifest guards pass. No scientific $T_{form}$ verdict is allowed.

Full execution is authorized. Each of the three registered seeds runs to step
2,544. Any training, bias, or save-grid change requires protocol revision.

## 13. Pass / Fail / Insufficient

Pass requires finite $T_{form}$ in at least two of three valid seeds, positive
fixed-basis Gate contribution not explained by basis drift alone, and complete
raw/applied reporting. Fail requires no finite formation in at least two valid
seeds or basis-drift-only formation. Fewer than three valid seeds or any key
training/load/probe/input/gradient/update/save/resume guard failure is
insufficient. E03-R failure does not refute E03-S.

## 14. Claim Boundary And Next Decision

At most, passing establishes an early formation signature under the registered
six-layer DCLM, AdamW, and auxiliary-loss-free-bias condition. It cannot prove
covariance is the unique cause, no-bias equivalence, middle/tail uselessness,
Gate singular concentration as head alignment, or training-efficiency gains.

The only next decision is the registered full-run pass/fail/insufficient
verdict after all three seed trajectories and decomposition guards complete.
