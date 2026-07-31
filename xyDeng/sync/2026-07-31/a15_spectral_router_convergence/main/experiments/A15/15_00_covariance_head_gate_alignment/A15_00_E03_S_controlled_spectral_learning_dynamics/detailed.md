# Detailed: A15_00_E03_S Controlled Spectral Learning Dynamics

Primary anchor: [A15_00_01 spectral learning dynamics](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md)  
Protocol: [approved E03-S protocol](protocol.md)  
Summary: [summary.md](summary.md)

## 0. Quick Recap

- **Purpose:** isolate whether input covariance anisotropy causally changes the
  finite-time learning speed of a direction-matched linear Gate.
- **Hypothesis:** anisotropic spectra make head directions reach the same target
  fit fraction before middle/tail; the gap grows with spectral anisotropy and
  disappears under whitening.
- **Experiment logic:** hold the Gate-space target, trace, basis, seed streams,
  optimizer, and evaluation fixed; intervene only on spectrum, whitening, or
  tail-only target position; compare against 2,048 flat-spectrum rotations.
- **Conclusion:** scientific S0/S1 **PASS**. Moderate learning times were about
  1:2:4, strong times about 1:4:16, whitening returned all bands to the flat
  range, and tail-only capability passed.
- **Evidence:** ACP job `om-zn7r7i23` succeeded; all eight seeds, 40 condition
  trajectories, 2,048 null partitions, decision gates, source hashes, and the
  independent audit passed.
- **Stage boundary:** `s2_eligible=true`; `s2_launched=false`.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit or formula | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Head / middle / tail | High-, medium-, and low-variance registered directions | covariance ranks 1--64, 65--320, and 321--768 | 64, 256, 448 dimensions | Defines the speed comparison | Semantic importance |
| $A_{raw}$ | Direction-matched expert-score coefficient | expert-centered matrix with equal column norm in the spectral basis | score / activation | Removes target-energy preference | Real expert advantage |
| $A_{gate}$ | Reachable Gate target | $A_{gate}=\tau A_{raw}$, $\tau=0.25$ | logit / activation | Aligns the metric with the soft target | Real-workload target |
| Fit fraction $F_B(t)$ | Relative target error removed in band $B$ | $1-\|(W_t-A_{gate})U_B\|_F^2/\|(W_0-A_{gate})U_B\|_F^2$ | fraction | Equal-progress comparison across bands | Task loss benefit |
| $T_B(0.5)$ | Persistent half-fit time | first interpolated crossing of 0.5 followed by two registered evaluations at or above 0.5 | optimizer step | Primary finite-time speed quantity | Final convergence quality |
| $D_{M:H}$ / $D_{T:H}$ | Relative learning-time contrasts | $\log T_M-\log T_H$ / $\log T_T-\log T_H$ | dimensionless | Positive means middle/tail learn more slowly | Functional usefulness |
| Flat-rotation null | Directional timing differences with no covariance-preferred direction | 256 Haar partitions per flat seed | distribution of $D$ | Controls finite-sample asymmetry | DCLM transfer |
| Strong-whitened | Same strong latent sample and target, but identity-covariance Gate input | map learned weight back to original spectral coordinates before $F_B$ | transformation | Isolates covariance from task direction | Optimizer universality |
| Tail-only | Target has zero head/middle coefficients | final held-out tail KL reduction | fraction | Tests whether tail is learnable at all | Equal tail speed |
| S2 eligibility | Protocol gate after S1 | true only if S1 verdict is pass | Boolean | Permits a later expert-feedback decision | Evidence that feedback is positive |

## 2. Anchor Link And Decision Point

Earlier checkpoint audits established a trained endpoint that was strongly
aligned with covariance head after equalizing input energy, but they did not
identify why that state formed. E03-S tests one conditional root cause under
full control: the covariance multiplier predicted by the local linear-Gate
dynamics.

This experiment decides the controlled clause only. E03-R separately asks
whether the signature appears during real DCLM training. S2 separately asks
whether trainable experts add feedback beyond the fixed-target result.

## 3. Protocol Compliance Audit

| Protocol item | Actual execution | Verdict |
| --- | --- | --- |
| Approval | full execution explicitly authorized after smoke | MATCH |
| Gate-space target | $A_{gate}=0.25A_{raw}$ used for both objective and fit | MATCH |
| Dimensions | $d=768$, $E=8$; 64/256/448 coarse bands; 12×64 fine bands | MATCH |
| Spectra | flat 1:1:1, moderate 4:2:1, strong 16:4:1; mean eigenvalue one | MATCH |
| S1 conditions | flat, moderate, strong, strong-whitened, strong-tail-only | MATCH |
| Seeds | `20260730`--`20260737` | MATCH |
| Budget | 8,000 steps and 193 evaluations per seed/condition | MATCH |
| Optimizer | SGD 0.02, batch 4096, no momentum, no weight decay | MATCH |
| Rotation null | 256 partitions per seed, 2,048 total | MATCH |
| Crossing rule | 50% crossing plus two persistent evaluations, linear interpolation | MATCH |
| Tail capability | independent final-held-out KL reduction at least 0.5 per seed | APPLIED; PASS |
| Scientific decision | anisotropy, dose, whitening, capability, and guards jointly applied | APPLIED; PASS |
| S2 | hard-disabled in config and independently audited | CORRECTLY NOT RUN |

The smoke artifacts were retained unchanged. The full implementation imported
the smoked numerical/model helpers, added a frozen full config, exact-resume
state, crossing/null analysis, and fail-closed source-hash audit.

## 4. Setup

### Research question

With expert-score signal matched across spectral directions, does
trace-normalized covariance anisotropy itself shorten high-variance linear-Gate
mode learning time?

### Data construction and splits

For each seed, draw one fixed Haar basis $U$ and generate

$$
s\sim\mathcal N(0,I),\qquad x=U\Lambda^{1/2}s.
$$

Raw head:middle:tail eigenvalues are 1:1:1, 4:2:1, or 16:4:1 and are rescaled
so $\operatorname{tr}(\Sigma)/768=1$. Training, trajectory evaluation, and
final-held-out use independent deterministic streams. Conditions within one
seed share the basis, target, initialization, and latent stream.

### Target clarification and objective

$A_{raw}\in\mathbb R^{8\times768}$ is expert-centered and has equal spectral
column norms. The actual target coefficient is

$$
A_{gate}=\tau A_{raw},\qquad \tau=0.25,
$$

so

$$
q(x)=\operatorname{softmax}(A_{gate}x)
=\operatorname{softmax}(0.25A_{raw}x).
$$

The Gate minimizes soft-label cross entropy. This clarification is necessary:
the reachable optimum is $W=A_{gate}$, so the fit metric and crossing time must
also compare with $A_{gate}$. A pre-submit unit test requires the fit fraction
to equal one at that optimum.

### Model and update rule

- bias-free linear Gate `Linear(768, 8)`;
- zero initialization in the expert-centered equivalence class;
- pure SGD, learning rate 0.02, no momentum or weight decay;
- batch size 4096;
- 8,000 optimizer steps;
- evaluation every 10 steps through 400 and every 50 thereafter.

S0 separately integrates the registered quadratic dynamics using float64 RK4
and compares with the closed form. S1 training is float32; metric accumulation
and decision outputs are retained with explicit numeric guards.

### Conditions

1. **Flat:** covariance does not designate a direction; this is the symmetry
   control and source of the rotation null.
2. **Moderate:** 4:2:1 tests the first anisotropy dose.
3. **Strong:** 16:4:1 tests the larger dose.
4. **Strong-whitened:** preserves strong latent samples and targets while
   removing covariance anisotropy at the Gate input; weights are mapped back
   before fit computation.
5. **Strong-tail-only:** zeros head/middle target coefficients and tests whether
   tail can be learned within the common budget.

### Changed and held-fixed variables

Only the registered spectrum, whitening transform, or target support changes.
Held fixed are $A_{raw}$, $A_{gate}$, $U$, zero initialization, latent streams,
trace, optimizer, batch, update budget, evaluation grid, precision, and
analysis code.

### Execution

- ACP job: `om-zn7r7i23`
- run name: `a15-e03s-5090x8-full-20260730T175200Z`
- resource: one `n12lp.nn.i10a.8` worker, eight RTX 5090 GPUs, spot quota
- start: 2026-07-30 17:48:53 UTC
- completion: 2026-07-30 17:50:12 UTC
- terminal state: `SUCCEEDED`
- retries: 0
- local preflight: fail-closed contract, compile/syntax, dry-run, and 10/10 tests passed

### Known setup limitations

The representation basis is fixed, the target is linear and known, inputs are
Gaussian, and SGD has no adaptive preconditioner. There is no sparse top-1
expert loss or changing expert advantage in S1. These choices establish the
causal root cleanly but limit transfer.

## 5. Metrics And Decision Rules

### Primary fit and timing metrics

For band $B$,

$$
F_B(t)=1-
\frac{\|(W_t-A_{gate})U_B\|_F^2}
{\|(W_0-A_{gate})U_B\|_F^2}.
$$

$T_B(0.5)$ is the first linearly interpolated 50% crossing that remains at or
above 50% for two further evaluations. A non-crossing observation is
right-censored, never replaced by 8,000.

For each seed,

$$
D_{M:H}=\log T_M-\log T_H,\qquad
D_{T:H}=\log T_T-\log T_H.
$$

Seed medians are the primary summaries.

### Rotation null and decision thresholds

Each flat seed receives 256 independently generated Haar partitions with the
same 64/256/448 dimensions. The analysis recomputes $F_B,T_B,D$ for each
partition, pools all 2,048 null contrasts, and uses the conservative empirical
`higher` q95.

- middle/head q95: 0.0032765;
- tail/head q95: 0.0030189.

Moderate and strong medians must exceed their corresponding q95. The
strong-minus-moderate paired differences use a conservative exact sign
interval; with eight seeds its coverage is 0.9921875. Both lower bounds must be
positive.

Strong-whitened medians must lie inside the pooled flat 95% envelopes:

- middle/head: $[-0.0040138,0.0039158]$;
- tail/head: $[-0.0038179,0.0036756]$.

Tail-only must reduce independent held-out KL by at least 0.5 for every seed.

### Secondary guards

Trace error, expert-centering, target norm matching, S0 closed-form error,
finite metrics, target entropy, tail gradient connectivity, whitening
covariance, seed/rank completeness, source hashes, and artifact completeness
must all pass. Failure of these prerequisites yields insufficient evidence,
not a scientific fail.

## 6. Main Results

### Decision evidence

| Condition / gate | Median $T_H$ | Median $T_M$ | Median $T_T$ | Median $D_{M:H}$ | Median $D_{T:H}$ | Registered decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Flat | 140.8217 | 140.8275 | 140.7955 | -0.00054 | -0.00019 | symmetry control |
| Moderate 4:2:1 | 55.7565 | 111.4594 | 223.0071 | 0.69268 | 1.38588 | above both null q95 |
| Strong 16:4:1 | 28.6295 | 114.3797 | 457.3943 | 1.38477 | 2.77145 | above both null q95 |
| Strong-whitened | 140.7792 | 140.8232 | 140.8833 | 0.000053 | 0.000637 | inside both flat envelopes |
| Strong tail-only | undefined | undefined | 451.6602 | undefined | undefined | capability passed |

Full machine-readable table:
[full_primary_results.csv](tables/full_primary_results.csv).

The dose intervals were:

- middle/head: $[0.69058,0.69485]$;
- tail/head: $[1.38299,1.38737]$.

All eight differences were positive for both contrasts. The minimum tail-only
held-out KL reduction was 0.9999939. No main crossing or rotation-null crossing
was censored.

### Stage-level profiling evidence

- **S0:** maximum relative closed-form error remained below the registered
  $10^{-5}$ threshold.
- **S1 trajectory:** every condition emitted 193 evaluation points, coarse and
  fine fit, Gate gain, KL, gradients, logits, and entropies.
- **Flat null:** all 2,048 partitions produced valid contrasts.
- **Artifact audit:** eight ranks, eight seeds, 40 full condition records,
  source hashes, full horizon, decision recomputation, and S2 non-launch all
  passed.

### Debug-only evidence

The earlier smoke and the full preflight validated source wiring, the reachable
$A_{gate}$ optimum, crossing persistence, exact sign interval, resumable
condition state, and a miniature rotation-null path. These protect validity but
do not contribute observations to the scientific effect.

### Failed or ambiguous evidence

No registered S0/S1 condition or guard failed. The ACP environment did not
expose its job id inside the worker, so generated manifests show
`job_id="unknown"`; the immutable run name, terminal ACP record, and worker
record bind all artifacts to `om-zn7r7i23`.

S2 remains scientifically unknown rather than failed: it was deliberately not
launched.

## 7. Visualization Results

### Covariance anisotropy separates learning times; whitening removes it

![Covariance anisotropy separates learning times; whitening removes it](figures/e03_s_crossing_times.png)

**Purpose:** test whether the covariance intervention orders band learning
times and whether whitening removes that order.

**Setup:** eight paired seeds per condition; the same matched Gate-space target,
trace, initialization, latent streams, SGD, update budget, and crossing rule.

**Metric definition:** $T_B(0.5)$ is the first interpolated optimizer step at
which band $B$ reaches 50% fit and stays there for two further registered
evaluations.

**Metric unit:** optimizer steps; log-scaled vertical axis.

**Data source:** the 40 completed condition records in the full run directory.

**Aggregation:** open points are individual seeds, filled symbols are medians,
and vertical bars show the observed seed range. Color and marker jointly encode
head, middle, and tail.

**Axes / legend:** horizontal axis is the trace-matched covariance condition;
vertical axis is $T_B(0.5)$; circle/square/triangle denote head/middle/tail.

**Expected if supported:** flat bands overlap; moderate and strong separate in
inverse-eigenvalue order; strong separates more; whitening restores overlap.

**Expected if weakened or incomplete:** no ordered separation, no dose effect,
or persistent separation after whitening.

**Observed result:** flat and whitened bands overlap near 141 steps; moderate is
approximately 1:2:4; strong is approximately 1:4:16.

**Take-home:** covariance anisotropy is sufficient to create the predicted
finite-time spectral learning-speed order in this controlled Gate.

**Remaining uncertainty:** whether adaptive optimization, moving
representations, and changing experts preserve, weaken, or reverse the effect.

**What this figure does not prove:** real DCLM formation, expert specialization,
band functionality, or loss/FLOP improvement.

**Anchor update implication:** the controlled covariance-speed clause passes;
E03-R and S2 remain separate tests.

Plot generator:
[plot_full_results.py](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/scripts/plot_full_results.py).

## 8. Stage Evidence And Failure Decomposition

| Stage | Evidence | Passed / failed / unclear | Failure reason | What this rules out |
| --- | --- | --- | --- | --- |
| Full contract preflight | Protocol status, source hash, target definition, schedule, resource tuple, 10/10 tests | Passed | none | stale smoke source, unreachable fit target, wrong full contract |
| S0 | exact quadratic comparison | Passed | none | numerical or spectrum-construction failure |
| Flat S1 | equal times and 2,048-rotation null | Passed | none | pre-registered directional bias and finite-sample rival |
| Moderate S1 | both contrasts far above q95 | Passed | none | absence of first covariance dose |
| Strong S1 | both contrasts far above q95 | Passed | none | absence of dose scaling |
| Whitening | both contrasts inside flat envelopes | Passed | none | target direction alone causing the timing order |
| Tail-only | every held-out KL reduction above 0.999993 | Passed | none | tail inexpressibility within the common budget |
| Independent audit | all evidence, decision, source, and S2 guards | Passed | none | incomplete or inconsistent artifact chain |
| S2 joint experts | not launched | Unclear by design | separate authorization required | nothing about expert feedback |
| Real DCLM signature | outside E03-S | Unclear by design | E03-R owns this question | nothing about real formation cause |

- **Falsified physical prior:** none; the registered conditional prior is supported.
- **Falsified mathematical model:** none within this controlled finite-time regime.
- **Falsified operationalization / proxy:** none after the $A_{gate}$ clarification.
- **Falsified implementation:** none within the full audit.
- **Falsified metric:** none; all crossings and null contrasts were observed.
- **Rivals weakened within scope:** unmatched target energy, total trace,
  finite-sample direction choice, task-direction-only ordering, and tail
  inexpressibility.
- **Remaining rivals outside scope:** adaptive optimizer effects,
  representation-basis motion, time-varying expert advantage, sparse feedback,
  and real-data non-Gaussianity.

## 9. Full Experiment Record

The user first authorized an engineering smoke. That job passed all registered
S0/S1/S2-wiring guards without a scientific verdict. Before full execution,
the objective/metric semantics were audited: because the soft target uses
$\tau A_{raw}$, the reachable fit target was explicitly named
$A_{gate}=\tau A_{raw}$ in the bilingual Protocol. The full validator blocks
execution unless that clarification and the full authorization are present.

The full package then froze `configs/full.json`, retained the exact smoked
source hash, added condition-level exact-resume state for the spot worker, and
submitted one job. Eight ranks independently ran one registered seed, each
completed five S1 conditions and 256 flat rotations, and rank zero recomputed
the registered verdict. ACP completed in 79 seconds with zero retries. S2 was
hard-disabled throughout.

## 10. Interpretation

The result is stronger than observing larger raw head logits. Each band was
asked to fit an equal-norm target, and the measured quantity was the fraction
of that target learned. Moderate and strong learning-time ratios closely track
the inverse covariance eigenvalue ratios predicted by the local dynamics. The
flat control shows that the code and target do not prefer the named head, while
whitening shows that task direction alone does not retain the order. Tail-only
shows that slower learning is not inability to represent the tail target.

Thus, under the registered assumptions, the covariance spectrum changes which
Gate modes are learned first. This is a finite-time optimization statement,
not a statement that head modes are more useful.

## 11. Claim Boundary

### Supported

For the registered fixed-basis, Gaussian, direction-matched linear Gate with
pure SGD, covariance anisotropy causally changes finite-time mode learning
speed. The effect scales with the spectrum, disappears under whitening, exceeds
the flat rotation null, and does not arise from tail inexpressibility.

### Not supported

- that the existing DCLM Router endpoints formed for this reason;
- that AdamW or moving Transformer representations follow the same constants;
- that trainable experts amplify rather than compensate for the Gate bias;
- that head contains greater functional utility;
- that middle/tail should be removed or separately routed;
- that any spectral Router improves validation loss, FLOPs, or wall time.

## 12. Next Decision

Decide whether to authorize the registered S2 frozen-versus-trainable expert
stage. It may start only from this passing S1 gate and must remain separately
reported. Current audited state: `s2_eligible=true`, `s2_launched=false`.

## 13. Links And Artifact Map

- **anchor:** [A15_00_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md)
- **protocol:** [protocol.md](protocol.md)
- **summary:** [summary.md](summary.md)
- **central figure:** [e03_s_crossing_times.png](figures/e03_s_crossing_times.png)
- **main table:** [full_primary_results.csv](tables/full_primary_results.csv)
- **code workspace:** [a15_e03_s_controlled_dynamics](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/)
- **runner:** [run_full.sh](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/scripts/run_full.sh)
- **submitter:** [submit_full_acp.sh](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/scripts/submit_full_acp.sh)
- **config:** [full.json](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/configs/full.json)
- **key code:** [e03_s_full.py](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/src/e03_s_full.py)
- **plot generator:** [plot_full_results.py](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/scripts/plot_full_results.py)
- **worker record:** [full_run_record.md](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/full_run_record.md)
- **result directory:** [a15-e03s-5090x8-full-20260730T175200Z](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/runs/a15-e03s-5090x8-full-20260730T175200Z/)
- **scientific decision:** [scientific_summary.json](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/runs/a15-e03s-5090x8-full-20260730T175200Z/scientific_summary.json)
- **independent audit:** [full_audit.json](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/runs/a15-e03s-5090x8-full-20260730T175200Z/full_audit.json)
- **manifest:** [manifest.json](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/runs/a15-e03s-5090x8-full-20260730T175200Z/manifest.json)
- **logs:** [full torchrun log](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_s_controlled_dynamics/runs/a15-e03s-5090x8-full-20260730T175200Z/logs/full_torchrun_20260730T174857Z.log)
- **recoverability state:** rank-local condition and rotation-null checkpoints under the result directory
- **reproduction command:** `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESOURCE_PROFILE=5090-8-spot RUN_NAME=<new-name> RESUME_FULL=1 bash scripts/run_full.sh full`
- **ACP job:** `om-zn7r7i23`

## 14. Two-Hour Monitoring Closure

The operator monitoring window ran from 2026-07-30 17:39:55 UTC through
19:40:19 UTC. The final live ACP read kept `om-zn7r7i23` in `SUCCEEDED` with
zero retries. No post-completion platform-state or artifact change altered the
registered scientific Pass or the boundary that S2 was eligible but unlaunched.
