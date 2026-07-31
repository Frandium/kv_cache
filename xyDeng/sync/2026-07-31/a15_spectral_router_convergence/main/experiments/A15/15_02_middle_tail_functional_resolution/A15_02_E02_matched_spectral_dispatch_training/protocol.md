---
experiment_id: A15_02_E02_matched_spectral_dispatch_training
status: blocked_by_E01_fail_not_run
created: 2026-07-30
updated: 2026-07-30
primary_anchor: 15_02_middle_tail_functional_resolution
depends_on: A15_02_01_E01_cross_update_compatibility_gate_pass
chinese_companion: protocol_cn.md
approval_date: 2026-07-30
implementation_authorized: conditional_on_E01_pass
pilot_authorized: conditional_on_E01_pass
full_run_authorized: conditional_on_E01_pass_and_pilot_guards
blocking_result: ../A15_02_01_E01_cross_update_compatibility_gate/summary.md
---

# Conditional Protocol: Matched Spectral-Dispatch Training On 8x5090

## 0. Approval Snapshot

**Dependency outcome (2026-07-30):** E01 completed with no admissible
candidate. Conditional authorization did not activate; this experiment was not
implemented or submitted and remains blocked. See the
[E01 result](../A15_02_01_E01_cross_update_compatibility_gate/summary.md).

The researcher approved this conditional contract on 2026-07-30. It remains
hard-blocked until A15_02_01_E01 has a formal Pass. A Pass authorizes
implementation and the registered 1B pilot; passing the pilot guards authorizes
the registered three-seed 2B run. E01 Fail or Insufficient cancels submission.

**Single decision question.** When the E01-locked band $S^*$ is used as the
band-only Gate input of a four-layer DCLM MoE, does it achieve lower held-out
next-token NLL than the native Router and an equal-dimensional random subspace
at the same cumulative actual FLOPs?

**Experiment role.** Matched joint training. This is the first evidence layer
that can answer training benefit.

**Primary metric.** At approximately 2B total tokens and a common cumulative
actual-FLOP point, paired held-out NLL differences from $S^*$ to native and
random, in nat/token. Negative means better.

**Strong falsifier.** E01 local compatibility passes, but $S^*$ fails to beat
native or random at matched FLOPs. This rejects the selected treatment as a
long-horizon improvement, not all possible spectral Routers.

## 1. Training Object And Claim Boundary

At a common 0.629B-token branch checkpoint, estimate and freeze each layer's
spectral basis, restrict every sparse Gate input to the selected $M$, $T$, or
$N$ band, and continue joint training of Router, experts, and upstream model.

This is a band-only intervention. It is not an additive side channel, online
PCA, or a complex Router.

| Term | Plain meaning | Purpose / answer | Cannot answer |
| --- | --- | --- | --- |
| Band-only Gate input | The Gate receives token variation only inside a frozen band while branch-time mean is preserved | Directly tests training with that fixed subspace | Whether adding the band to native is better |
| Native arm | Gate receives the complete actual input | Existing-design baseline | Direction specificity |
| Random arm | Gate receives an equal-dimensional Haar subspace | Distinguishes true band from generic dimensionality reduction | Every possible random orientation |
| Matched FLOPs | Arms are compared at equal cumulative measured computation | Training efficiency | Full cluster cost or wall-time efficiency |
| Held-out NLL | Mean next-token negative log likelihood on unseen documents | Language-model quality | Why experts changed |

## 2. Dependency And Unlock Conditions

Implementation and submission require all of:

1. E01 selects exactly one $S^*\in\{M,T,N\}$ on 12-layer final test.
2. The locked band passes the all-four-layer pooled/median transfer gate at the
   four-layer checkpoint.
3. Band ranks/dimension, pair features, step-size result, and E01 evidence
   record are frozen.
4. The pilot may not switch to a runner-up band.
5. Static novelty without a compatibility Pass cancels E02.

These rules limit selection bias; they do not guarantee long-horizon transfer.

## 3. Hypothesis And Rivals

**H1.** The selected band retains co-training relations compressed by the
native Gate. Band-only dispatch lowers within-expert update conflict and
reduces held-out NLL faster than native and random at matched FLOPs.

| Rival | Prediction | Discriminator | Scope |
| --- | --- | --- | --- |
| R0: local proxy does not transfer | E01 passes but NLL does not improve | Matched-FLOP final NLL | Rejects this treatment's training benefit |
| R1: dimensionality reduction only | $S^*$ and random improve similarly | $S^*$ versus random | Tests covariance-band specificity |
| R2: load artifact | Loss difference follows overflow/drop/load mismatch | Load, capacity, drop guards | Rejects obvious load confounding |
| R3: extra computation | Treatment has more actual FLOPs/effective tokens | Profiler FLOPs, tokens, parameters, kernel hashes | Establishes compute comparability |
| R4: frozen band loses spectral identity | Current band overlap falls to Haar level | Frozen/current overlap | Determines whether “spectral” remains an allowed label |
| R5: compatibility mechanism is wrong | NLL improves without lower conflict | Dynamic conflict audit | Keeps benefit claim but rejects mechanism |

## 4. Frozen Band Intervention

For seed $s$ and layer $\ell$, estimate the branch-checkpoint actual-input mean
and basis on independent calibration tokens and freeze:

$$
P_{\ell,S^*,s}=U_{\ell,S^*,s}U_{\ell,S^*,s}^{\top}.
$$

For arm $a\in\{\mathrm{native},S^*,R^*\}$:

$$
r'_{\ell,a}
=\mu_{\ell,s}+P_{\ell,a}(r_\ell-\mu_{\ell,s}),
\qquad
z_{\ell,a}=W_\ell r'_{\ell,a}+b_{\ell,a}.
$$

$P_{\mathrm{native}}=I$; $P_{R^*}$ is an equal-dimensional Haar projector.
$P$, $\mu$, and $b$ are frozen buffers. $W$, upstream representations, and
experts train normally.

The same selected rank block is applied to all four sparse Gates, with a
layer-specific branch basis. No result-based layer selection is allowed. If
E01 does not pass the pooled/median four-layer gate, a separate layer-selective
Protocol would be required.

### Mean preservation

Projection changes token-to-token directions but preserves the branch-time
mean. This prevents deletion of a fixed expert-score offset from being
confounded with band restriction. It does not prevent later mean drift.

### Frozen load-matching offset

On calibration tokens only, solve a sum-to-zero expert offset:

$$
b_{\ell,a}
=\arg\min_{\mathbf 1^\top b=0}
\|p_{\ell,a}(b)-p_{\ell,\mathrm{native}}\|_2^2
+10^{-4}\|b\|_2^2,
$$

where $p$ is branch-time top-1 expert fraction. This aligns aggregate initial
load but neither fixes token-level routes nor forces later loads to match.

### Compute-path matching

Every arm calls the same frozen dense 768-by-768 projector kernel, same Gate
shape, and same offset slot. Native uses identity and zero offset. This matches
the algorithmic path and extra FLOPs; wall time remains a separately reported
system metric.

## 5. Three Arms

| Arm | Gate input | Question | Must match | Cannot prove alone |
| --- | --- | --- | --- | --- |
| Native | Full $r_\ell$ | Does restriction beat the existing Router? | init, optimizer, data, kernel, tokens, FLOPs | Direction specificity |
| $S^*$ | E01-locked $M$, $T$, or $N$ | Does the true band treatment work? | Dimension and offset procedure with random | Universal spectral mechanism |
| $R^*$ | Equal-dimensional Haar subspace | Is any benefit generic regularization? | Dimension and computation with $S^*$ | Full random-orientation distribution |

Random is never selected by E01 or training loss. Each paired seed maps to one
fixed Haar draw; the three full seeds therefore cover three draws without
tuning orientation.

## 6. 8x5090 Resource And Fixed Training Configuration

Execution surface:

- worker package:
  /data/250010109/MoE_Router/experiments/20260730_h768_4layer_switch_5090_tuning
- workspace: share-space
- AEC2 cluster: computing-cluster-5090-01g
- worker spec: n12lp.nn.i10a.8
- one node and eight RTX 5090 GPUs per arm, spot quota, normal priority
- image:
  registry.cn-sh-01.sensecore.cn/lepton-trainingjob/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04

| Item | Frozen value | Purpose / scope |
| --- | --- | --- |
| Model | H768, 4 layers, 8 sparse plus 1 shared expert, top-1 switch | The validated small-model configuration only |
| Router | Linear, running center, load-balance weight 0.01 | Preserves branch lineage |
| Sequence/batch | 1024; local 12/GPU; accumulation 8; global 768 | Fixed token/FLOP axis |
| Optimizer | LR $10^{-4}$, weight decay 0.01, warmup 636 steps | Warmup ends near 0.500B tokens |
| Activation checkpointing | Off | Matches validated memory path |
| Nominal tokens/step | 786,432 | Common token axis; effective tokens are still audited |

Eight GPUs are data parallelism for one arm, not eight seeds. Parallel arms
need three 8-GPU nodes; sequential execution is allowed if paired seeds retain
identical data order and branch snapshot.

## 7. Branch, Seeds, And Token Budget

### 7.1 Common branch

For every seed, train a common native model from initialization to step 800
(629,145,600 nominal tokens), saving model, optimizer, scheduler, dataloader
cursor, RNG, and running center. Estimate the seed-specific spectrum, then
clone the three arms. The existing checkpoint-800 may serve as the pilot seed.
The other full-evidence seeds need their own common burn-ins; arms may not
independently recreate burn-in.

### 7.2 Registered observations

| Optimizer step | Nominal total tokens | Purpose | Answer scope |
| ---: | ---: | --- | --- |
| 800 | 629,145,600 | Branch/no-op start | Initial equivalence |
| 954 | 750,256,128 | Earliest divergence | First route/load response |
| 1272 | 1,000,341,504 | Pilot endpoint | Feasibility and direction only |
| 1908 | 1,500,512,256 | Mid trajectory | Persistence or reversal |
| 2544 | 2,000,683,008 | Full primary endpoint | Initial 2B training benefit |

### 7.3 Pilot and full evidence

- Pilot: one paired seed by three arms, step 800 to 1272. It checks
  implementation, load, memory, loss direction, and metric availability. It
  cannot Pass the parent anchor.
- Full: three paired seeds by three arms through step 2544, nine branch jobs
  plus seed-specific common burn-ins.
- The pilot seed may count as one full seed only if code/config/data/basis
  hashes remain unchanged and neither band nor metric is changed after seeing
  pilot results.

## 8. Primary Metric: Matched-FLOP Held-Out NLL

Use 1024 DCLM held-out documents of 1024 valid tokens, independent of training,
basis calibration, and E01. Freeze token hashes before execution.

For baseline $B\in\{\mathrm{native},R^*\}$:

$$
\Delta L_{S^*:B}(F^*)
=L_{S^*}^{\mathrm{heldout}}(F^*)
-L_B^{\mathrm{heldout}}(F^*).
$$

$F^*$ is the largest cumulative actual-FLOP point reached by all arms. If
step-2544 FLOPs differ by less than 1%, use the common endpoint; otherwise
interpolate only on the registered 1908--2544 curve.

This metric directly asks whether the selected arm delivers better unseen-text
modeling at equal computation. It cannot establish mechanism or cross-scale
generality.

Estimate intervals with 2000 paired seed-by-document hierarchical bootstrap
draws. With three seeds, a Pass remains preliminary matched-training evidence,
not a scaling law.

## 9. Training-Process Metrics

These metrics are computed on fixed audit documents at registered checkpoints.
They explain the path but never replace the NLL decision or change treatment.

| Metric | Computation / unit | Purpose / answer | Cannot prove |
| --- | --- | --- | --- |
| Loss-FLOP AUC | Integral of held-out NLL over measured FLOPs from branch to $F^*$ | Earlier learning versus endpoint-only lead | Final superiority |
| Router margin | Top-1 minus top-2 logit, logit/token | Decision confidence and saturation | Correct routing |
| Route flip | Fraction of fixed probe tokens changing expert from step 800 | Whether learning paths diverge | Improvement |
| Load/overflow/drop | Expert shares, capacity overflow, dropped-token rate | Effective-compute and data fairness | Expert functional quality |
| Expert update norm | Actual expert optimizer-update Frobenius norm per routed token | Learning-pressure allocation | Useful updates |
| Within-expert conflict | Median gradient cosine and symmetric compatibility under current routes | Whether the E01 mechanism appears during training | Final loss reduction |
| Functional redundancy | Mean pairwise correlation of forced-expert token-loss-change profiles | Functional similarity/divergence of experts | Beneficial specialization |
| Frozen/current band overlap | $\|U_{*,S}^{\top}U_{t,S}\|_F^2/d_S$ | Whether the frozen projector remains the current rank band | Functional value |
| System metrics | Measured FLOPs, valid tokens, wall time, memory, hashes | Compute fairness and reproducibility | Algorithmic mechanism |

If NLL passes but conflict does not fall, report training benefit without the
compatibility mechanism.

## 10. Matching And Validity Guards

| Guard | Requirement | Purpose | Failure verdict |
| --- | --- | --- | --- |
| Initial state | Model/optimizer/data cursor/RNG/center hashes match within a paired seed | Isolates projector treatment | Insufficient |
| Parameters | Identical trainable names/counts; $P,\mu,b$ frozen | Excludes capacity difference | Insufficient |
| Data | Identical document/token order; valid-token difference below 0.1% | Excludes data amount | Insufficient |
| Compute | Same kernel; cumulative measured FLOP difference below 1% | Excludes extra compute | Use registered interpolation or Insufficient |
| Branch load | Expert-share total variation at most 0.02 after offset | Prevents initial collapse | Do not start |
| Capacity | Same capacity; drop-rate difference at most 0.05 percentage point and absolute drop below 0.1% | Excludes effective-token confound | Insufficient |
| Native no-op | Identity-projector logits/loss/winners match unmodified code | Validates intervention kernel | Stop |
| Spectral identity | Frozen/current overlap remains above equal-dimensional Haar null | Preserves spectral label | Below null, call it fixed-subspace only |
| Resume | 800-to-802 and later resume replay the same next batch/loss | Prevents checkpoint-induced trajectory change | Stop |

These guards validate comparison but do not force routes and later loads to
remain equal; those are legitimate treatment-mediated paths and are recorded.

## 11. Execution Flow And Stop Rules

1. S0 contract freeze: freeze E01 result, code, resource, data, seeds, bases,
   and projector hashes.
2. S1 CPU/unit smoke: verify idempotence, rank, mean preservation, offset, and
   identity-kernel no-op.
3. S2 8-GPU short smoke: run two optimizer steps per arm from the same snapshot
   and verify SM120/NCCL, memory, gradients, load, checkpoint, and resume.
4. S3 1B pilot: run one paired seed through step 1272; this can reveal collapse,
   mismatch, or clear degradation but cannot establish benefit.
5. S4 full: after pilot guards pass, run three paired seeds through step 2544.
6. S5 evidence record: adjudicate primary NLL first, then use process metrics
   for explanation.

Stop if any GPU peak allocated memory exceeds 29.5 GiB; if two consecutive
50-step loss windows rise by more than 2%; on non-finite loss/gradient,
unrecoverable checkpoint, no-op/resume failure, or capacity-guard violation.

At 1B, if $S^*$ is clearly worse than both native and random, record pilot Fail
and do not continue. A wide interval is pilot Insufficient. Once full begins,
ordinary intermediate NLL ordering cannot trigger selective stopping.

## 12. Pass, Fail, And Insufficient

### Pass

After every matching guard passes:

1. At step 2544/common $F^*$, paired point estimates for both
   $\Delta L_{S^*:\mathrm{native}}$ and $\Delta L_{S^*:R^*}$ are negative.
2. Both paired hierarchical-bootstrap 95% upper bounds are below zero.
3. Token drop, FLOPs, parameters, and branch load do not explain the result.

This supports only a preliminary matched-compute benefit for the registered
four-layer H768 DCLM configuration over the 1--2B-token range.

### Fail

Fail if valid and precise measurement shows $S^*$ does not beat either native
or random. If $S^*$ and random both improve but cannot be separated, direction
specificity fails while generic dimensionality reduction remains an
observation. If neither dynamic conflict nor NLL improves, the local-proxy
transfer hypothesis fails.

### Insufficient

Insufficient if three-seed intervals remain wide; any projector, load,
capacity, FLOP, resume, data, or spectral-identity guard fails; a pilot failure
cannot be distinguished from implementation; or only the 1B pilot exists.

No arbitrary minimum NLL effect is imposed. Report nat/token differences,
relative perplexity changes, and loss-FLOP AUC continuously.

## 13. Figure And Evidence Contract

1. Primary: held-out NLL versus cumulative actual FLOPs, showing three arms,
   three seeds, the branch, and 0.75/1/1.5/2B markers. The primary reading is
   the 2B common-$F^*$ paired difference.
2. Mechanism: within-expert conflict plus route/load trajectories. It cannot
   replace the primary plot.
3. Expert path: update norm and functional redundancy. It cannot establish
   beneficial specialization.
4. Validity table: parameters, tokens, FLOPs, drop, memory, resume, and band
   overlap, with every decision guard visible.

The summary records the conclusion, primary comparison, boundary, and next
decision. Detailed records own seeds, job IDs, commands, logs, failures,
tables, and artifact map. Large logs and checkpoints remain on the worker
surface.

## 14. Final Boundary And Authorization

Even a Pass cannot establish universality across MoEs, scales, or domains;
unique semantics of middle/tail; causal production of better experts; harm
from Q1 head alignment; an online/adaptive spectral Router; or a scaling law.

The only current decision is the E01 verdict. The conditional approval cannot
bypass the E01 gate. E01 Pass activates the registered pilot; pilot-guard Pass
activates the registered full run.
