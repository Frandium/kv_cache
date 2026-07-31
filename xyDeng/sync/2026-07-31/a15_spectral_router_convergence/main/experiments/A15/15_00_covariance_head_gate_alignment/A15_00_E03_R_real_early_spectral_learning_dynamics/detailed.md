# Detailed: A15_00_E03_R Real Early Spectral Learning Dynamics

Primary anchor: [A15_00_01 spectral learning dynamics](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md)  
Protocol: [approved E03-R protocol](protocol.md)  
Summary: [summary.md](summary.md)

## 0. Quick Recap

- **Purpose:** test whether a real six-layer DCLM MoE develops a stable early
  equal-energy covariance-head alignment signature and separate Gate-weight
  motion from representation-basis motion.
- **Hypothesis:** both head/middle and head/tail contrasts cross a matched
  orientation null early, persist, and retain a positive fixed-basis Gate
  contribution while routing remains load-stable.
- **Experiment logic:** train three registered seeds from initialization with
  LM loss only and a non-gradient score bias; save actual Gate inputs, complete
  spectral bases, raw gradients, applied updates, routing counterfactuals, and
  full $W_s\times U_t$ ingredients on a dense early grid.
- **Conclusion:** scientific **INSUFFICIENT — `insufficient_load_guard`**. A
  single expert exceeded 80% of a 20-step layer load in every seed by steps
  72--79. Jobs were stopped because this historical guard cannot recover.
- **Evidence:** all three frozen-source jobs, per-step records through steps
  130--132, complete heavy artifacts through step 120, terminal load audit,
  selected 256-sample orientation nulls, and the independent analysis closure.
- **Boundary:** this is a failure of the registered real-training stability
  condition, not a scientific failure of E03-S or of covariance acceleration.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit or formula | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Actual Router input $g_{\ell,t}$ | Representation directly consumed by the Gate | post-attention RMSNorm output passed to `mlp.gate` | 768-vector per token | Audits deployed Gate geometry | Expert-input geometry |
| Head / middle / tail | Covariance-ranked frequency bands | ranks 1--64 / 65--320 / 321--768 of the centered calibration covariance | 64 / 256 / 448 dimensions | Defines coarse spectral comparison | Semantic content |
| Equal-energy Gate gain $G_{\ell,B}$ | Gate weight in a band after dividing by band width | $\|C_EW_\ell U_{\ell,B}\|_F^2/d_B$ | squared Gate weight per direction | Removes raw input-energy amplification | Functional utility |
| Head contrast | Relative equal-energy head preference | $\log(G_H/G_M)$ and $\log(G_H/G_T)$, median across six layers | dimensionless | Primary formation signal | Unique formation cause |
| Orientation null | Same centered-Gate singular spectrum with randomized right subspace | 256 Haar-Stiefel draws per seed/step | contrast distribution and q95 | Controls random alignment and Gate singular scale | Load stability |
| Paired basis bootstrap | Refit the covariance basis after resampling whole calibration sequences | 200 resamples of the same ordered 32 sequences | contrast distribution; 2.5% lower bound | Tests basis-sampling stability | Validity after a failed load guard |
| 20-step maximum load share | Most concentrated expert in one layer over 20 updates | sum global counts across a window, normalize within layer, maximize over expert and layer | fraction | Registered anti-collapse guard | Expert function |
| Dead experts | Experts with zero aggregated tokens in the same 20-step layer window | count of zero entries | count | Second load guard | Under-trained but nonzero experts |
| $T_{form}$ | First valid persistent head-formation point | both contrasts above null q95, both bootstrap lower bounds positive, at least four layers positive, persistent for three heavy snapshots | nominal tokens | Primary real-run verdict | Covariance as unique cause |
| Fixed-basis $\Delta_W$ | Contrast change attributable to Gate-weight motion with basis held across adjacent snapshots | symmetric $W_s\times U_t$ crossing difference | change in log ratio | Separates Gate from basis motion | Expert-advantage spectrum |
| Fixed-Gate $\Delta_U$ | Contrast change attributable to basis motion with Gate held across adjacent snapshots | complementary symmetric crossing difference | change in log ratio | Audits representation drift rival | Causal basis effect by itself |
| Spectral-state snapshot | Heavy analysis artifact | Gate state, basis, ordered fp32 calibration inputs, routing replay, hashes | file per heavy step | Enables null/bootstrap/crossing analysis | Model/optimizer resume |

## 2. Anchor Link And Decision Point

E03-S established that covariance anisotropy causally changes finite-time mode
learning speed in a controlled, fixed-basis, matched-target linear Gate. E03-R
tests whether an analogous signature can be measured in a jointly trained,
sparse top-1 DCLM MoE. It does not retest the controlled causal clause.

The decision point was whether a valid real trajectory yields finite
$T_{form}$ in at least two of three seeds with a positive Gate-weight
contribution. The registered load guard is a prerequisite: a near-single-expert
trajectory changes the Router--Expert system whose formation is being studied.

## 3. Protocol Compliance Audit

| Protocol item | Actual execution | Verdict |
| --- | --- | --- |
| Approval | smoke and full execution explicitly authorized | MATCH |
| Resource | independent one-node 8×RTX-5090 SPOT job per seed | MATCH |
| Seeds | 17, 29, 43 | MATCH |
| Frozen source | one verified 43-file snapshot, identical SHA in all manifests | MATCH |
| Actual Gate input | post-attention RMSNorm tensor captured at `mlp.gate` | MATCH |
| Model | 6L, H768, 8 sparse + 1 shared expert, top-1 | MATCH |
| Data/batch | DCLM, sequence 1024, global batch 768 | MATCH |
| Objective | LM only, `lambda_lb=0` | MATCH |
| Bias | registered non-gradient, once-per-step count update | MATCH |
| Capacity | uncapped dense expert dispatch, zero token drop | MATCH |
| Diagnostics | step records and registered heavy grid through termination | MATCH through step 120 |
| Actual-input replay | relative logit error 0 and top-1 agreement 1 at audited snapshots | PASS |
| Basis | orthogonality errors near $10^{-15}$ | PASS |
| Raw/applied identity | below registered $2\times10^{-6}$ limit | PASS |
| Analysis closure | exact resume, null, bootstrap, crossing, and artifact paths | PASS, non-scientific closure |
| Load stability | every seed produced a 20-step maximum share above 0.8 | **FAILED** |
| Registered 2B endpoint | jobs stopped near step 130 after terminal guard failure | NOT COMPLETED BY DESIGN |
| Three-seed $T_{form}$ verdict | prerequisite invalid; full scientific analyzer not eligible | INSUFFICIENT |

The central figure and selected-point table exist and were audited. The
Protocol's insufficient-evidence rule, rather than its scientific pass/fail
rule, governs this result.

## 4. Setup

### Research question

Does a real, load-stable DCLM MoE develop a persistent early equal-energy head
alignment signature, and if so is its local change attributable to Gate weights
rather than only to representation-basis motion?

### Data and leak-free split

Training uses
`/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10` at sequence
length 1024. Each optimizer step processes 768 sequences, or 786,432 nominal
tokens. Training indices are deterministic by seed and step. The fixed
calibration buffer uses 32 independent length-256 sequences and the held-out
probe uses 64 further sequences beyond the training cutoff. Their token hashes
are identical across seeds and recorded in every manifest and heavy artifact.

### Model, Router, and representation

- six decoder layers, hidden width 768;
- six attention heads and three KV heads;
- eight sparse experts plus one shared expert per layer;
- expert intermediate width 1536;
- top-1 sparse routing;
- actual Router input is the post-attention RMS-normalized representation;
- covariance basis is computed independently at each layer and heavy step from
  the ordered calibration inputs.

### Objective, optimizer, and score bias

The only differentiable objective is LM cross entropy. AdamW uses learning rate
$10^{-4}$, betas $(0.9,0.95)$, epsilon $10^{-8}$, weight decay 0.01, 1,000-step
linear warmup, and a 127,156-step cosine horizon. Forward/backward uses bf16;
diagnostics use fp32/fp64. DDP uses eight processes and re-entrant activation
checkpointing.

For expert counts $c_{\ell,e}$ and their within-layer mean $\bar c_\ell$, the
non-gradient score bias follows the registered rule

$$
b_{\ell,e}\leftarrow\operatorname{clip}\left[
b_{\ell,e}+10^{-3}
\frac{\bar c_\ell-c_{\ell,e}}{\bar c_\ell+10^{-6}},-0.1,0.1
\right],
$$

then centers the eight bias values. It is outside AdamW and updated once after
each optimizer step. Because centering follows clipping, a final centered value
may differ slightly from $\pm0.1$; the observed values reconstruct the approved
rule to below $8\times10^{-9}$.

### Registered schedules

- step 0 and every step 1--100 are heavy snapshots;
- steps 110, 120, ..., 1000 are heavy snapshots;
- later heavy points are 1100, 1250, 1500, 1750, 2000, 2250, and 2544;
- training checkpoints are registered first at step 250 and then at sparse
  later points;
- endpoint step 2544 is 2.000683008B nominal tokens.

### Execution identity

| Seed | Job | Run | Final observed state | Retries |
| ---: | --- | --- | --- | ---: |
| 17 | `om-9cp56m56` | `e03-r-full-seed17-frozen-20260730T183100Z` | `SUSPENDED` | 0 |
| 29 | `om-t86sk3ee` | `e03-r-full-seed29-frozen-20260730T183100Z` | `SUSPENDED` | 0 |
| 43 | `om-ryaekljn` | `e03-r-full-seed43-frozen-20260730T183100Z` | `SUSPENDED` | 0 |

The shared source SHA is
`02400ef6d5ba30d89d736ba0e1c23b18fb3228e46d5822d7ee9a7be3b330fe13`.
All three manifests record the snapshot path and `verified_prelaunch=true`.

### Known limitations after termination

The stop completed before the first registered model/optimizer checkpoint at
step 250. Complete heavy spectral states through step 120 are retained, but
they cannot resume training. No statement in this record treats an analysis
snapshot as a checkpoint.

## 5. Metrics And Decision Rules

### Load prerequisite

For window start $s$, layer $\ell$, and expert count $c_{t,\ell,e}$,

$$
L^{20}_{s,\ell}=
\max_e\frac{\sum_{t=s}^{s+19}c_{t,\ell,e}}
{\sum_{e'}\sum_{t=s}^{s+19}c_{t,\ell,e'}}.
$$

The trajectory is invalid if any $L^{20}_{s,\ell}>0.8$ or at least four
experts have zero aggregated counts in that layer/window. This is a hard
validity guard, not a score to optimize and not a scientific effect threshold.

### Equal-energy contrasts and orientation null

With $C_E=I-\mathbf1\mathbf1^\top/8$ and band basis $U_{\ell,B}$,

$$
G_{\ell,B}=\frac{\|C_EW_\ell U_{\ell,B}\|_F^2}{d_B}.
$$

The observed quantities are the six-layer medians of
$\log(G_H/G_M)$ and $\log(G_H/G_T)$. The matched null retains every numerical
nonzero singular value of $C_EW_\ell$ and uses 256 random Haar-Stiefel right
subspaces. Each observed aggregate is compared with its seed/step-specific
null q95.

### Formation rule

$T_{form}$ is the earliest heavy point for which:

1. both observed aggregate contrasts exceed null q95;
2. at least four layers have both contrasts positive;
3. both 200-resample paired-sequence bootstrap lower bounds exceed zero;
4. all conditions persist for the next two heavy snapshots;
5. every engineering guard, especially load stability, remains valid.

After a historical load failure, later contrast crossings are descriptive only
and cannot enter this rule.

### Dynamics decomposition

The saved Gate weights $W_s$ and bases $U_t$ define a full crossing tensor
$B(W_s,U_t)$. Adjacent symmetric differences separate fixed-basis
$\Delta_WB$ from fixed-Gate $\Delta_UB$. Per-step raw gradients and measured
post-minus-pre AdamW updates also satisfy the registered
$\Delta G_B=C_B+Q_B$ identity within $2\times10^{-6}$.

## 6. Main Results

### Decision evidence: terminal load failure

| Seed | First failing window | Layer | Dominant expert | Share | Step-100 rolling maximum | Step-120 rolling maximum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 56--75 | 5 | 0 | 0.80208 | 0.99045 | 0.99526 |
| 29 | 53--72 | 2 | 7 | 0.80246 | 0.99110 | 0.99699 |
| 43 | 60--79 | 4 | 1 | 0.81781 | 0.98916 | 0.99562 |

The failure was reproducible across seeds and preceded any eligible persistent
two-contrast formation verdict. Exactly zero dead experts were observed in the
reported failing windows; concentration, not the four-dead-expert clause,
triggered termination.

### Spectral observations before and after invalidation

| Seed | Step | H:M observed / q95 | H:T observed / q95 | Both-positive layers | Orientation candidate | Load-valid at step |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 17 | 50 | 0.00282 / 0.05625 | 0.00905 / 0.04692 | 2 | no | yes |
| 29 | 50 | 0.01326 / 0.06037 | 0.00405 / 0.05278 | 2 | no | yes |
| 43 | 50 | -0.04373 / 0.06536 | 0.02100 / 0.05632 | 2 | no | yes |
| 17 | 120 | 0.04966 / 0.05623 | 0.07102 / 0.04895 | 4 | no | **no** |
| 29 | 120 | 0.07071 / 0.05763 | 0.08649 / 0.04802 | 5 | yes | **no** |
| 43 | 120 | 0.06119 / 0.04595 | 0.07938 / 0.04854 | 6 | yes | **no** |

The apparent late crossing in two seeds is ambiguous evidence, not formation.
It occurs while roughly 99% of the rolling load is assigned to one expert in a
layer. Full-run basis bootstrap was not used to rescue those post-failure
points. The 200-resample implementation was already exercised successfully in
the non-scientific analysis closure.

### Load-bias and dynamics observations

At step 100, 7--8 of the 48 layer/expert bias coordinates reached the clip in
the immediately reconstructed update. The post-centering maximum absolute bias
was about 0.101, exactly consistent with the approved clip-then-center rule.
This shows that the intervention saturated while load concentrated; it does
not by itself prove that the bias caused the collapse.

For the adjacent step 99-to-100 crossing, the median fixed-basis Gate-weight
contribution was positive for both contrasts in every seed:

| Seed | $\Delta_W$ H:M | $\Delta_W$ H:T | $\Delta_U$ H:M | $\Delta_U$ H:T |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.000231 | 0.000235 | 0.003176 | 0.001546 |
| 29 | 0.000238 | 0.000239 | 0.001852 | 0.001131 |
| 43 | 0.000196 | 0.000207 | -0.000074 | 0.001291 |

The per-layer crossing identity residual was zero at reported precision. These
are post-failure profiling observations and cannot support H1.

### Engineering evidence that passed

- source snapshot verified before and inside all jobs;
- world size 8 and all registered 5090 resource fields matched;
- fixed token hashes and calibration sequence order matched across seeds;
- every selected calibration tensor had shape `[6,32,256,768]`, fp32 dtype,
  150,994,944 bytes, and matching SHA256;
- actual-input logit replay relative error was zero and top-1 agreement one;
- basis orthogonality errors were about $3$--$4\times10^{-15}$;
- diagnostic state and pending-counter invariants passed;
- no token was dropped by capacity;
- score-bias update count equaled optimizer step;
- raw/applied update identities stayed below tolerance;
- float64 orientation-null Stiefel and singular-Gram preservation errors stayed
  near $10^{-15}$ in closure and selected terminal analysis.

### Debug-only and invalid development runs

The first smoke `om-nw3err6p` exposed mixed-dtype actual-input replay and was
retained as a failed engineering run. Repaired smoke `om-83od0hh5` passed.
Early full attempts `om-kz4e4hxd`, `om-7zyerkix`, `om-quiu73be`, and
`om-a7l6kbsa` were stopped after exposing DDP/static-graph, exact top-k tie, or
missing calibration-input closure issues; none is scientific evidence. The
final jobs in this record used a new common frozen snapshot after the complete
Protocol-to-artifact-to-analyzer closure passed.

## 7. Visualization Results

### Load collapse precedes any interpretable head-formation verdict

![Load collapse precedes any interpretable head-formation verdict](figures/e03_r_load_collapse_and_contrasts.png)

**Purpose:** determine whether spectral contrast motion occurs within the
registered load-stable training regime.

**Setup:** three independent seeds; same frozen source, DCLM setup, score-bias
rule, heavy grid, actual Gate input, and analysis. Selected contrast points are
steps 0, 25, 50, 75, 100, 110, and 120.

**Metric definition:** panel A plots the maximum $L^{20}_{s,\ell}$ ending at
each step. Panels B/C plot six-layer median log head/middle and head/tail gain
ratios and their matched 256-sample orientation-null q95.

**Metric unit:** load fraction and dimensionless log ratio.

**Data source:** final frozen run per-step JSON records, heavy snapshots, Gate
states, covariance bases, and selected terminal-evidence JSON records.

**Aggregation:** load maximizes over layers and experts after 20-step count
aggregation. Contrasts take the median over six layers. Null q95 is computed
independently by seed and heavy step.

**Axes / legend:** horizontal axis is optimizer step; the red dashed line is
the 0.8 load limit; solid colored contrast lines are observations; dashed
colored lines are null q95; hollow observations occur after that seed's first
load failure.

**Expected if supported:** all load curves remain at or below 0.8 while both
contrast curves rise above their q95 and persist. **Expected if weakened or
incomplete:** the load guard fails first or contrast crossing appears only in
the invalid region.

**Observed result:** all three load curves exceeded 0.8 by step 79 and approached
one by step 100. No selected pre-failure point was an orientation candidate.
Two step-120 candidates are hollow because their trajectories were already
invalid.

**Take-home:** E03-R is insufficient; post-collapse spectral motion cannot be
used as a stable Router-learning signature.

**Remaining uncertainty:** whether a separately validated load-stability
mechanism permits a valid real trajectory and whether its spectral motion is
Gate- or basis-driven.

**What this figure does not prove:** absence of stable real head formation,
failure of E03-S, functional inferiority of middle/tail, or training-efficiency
benefit.

**Anchor update implication:** controlled covariance acceleration remains
supported; real-workload transfer remains unresolved.

Plot generator:
[build_terminal_evidence.py](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/scripts/build_terminal_evidence.py).

## 8. Stage Evidence And Failure Decomposition

| Stage | Evidence | Passed / failed / unclear | Failure reason | What this rules out |
| --- | --- | --- | --- | --- |
| Static contract and smoke | config, actual input, replay, basis, update identity, bias cadence | Passed | none after repair | basic wiring and metric implementation failure |
| Analysis closure | exact restart, calibration artifact, 200 bootstrap, 256 null, crossing | Passed, non-scientific | none | analyzer unable to consume registered artifacts |
| Source freeze | one content-addressed snapshot and three matching manifests | Passed | none | cross-seed source drift |
| Initial heavy states | step 0/1 replay, basis, artifact SHA, capacity, no dead experts | Passed | none | corrupt initial diagnostic chain |
| Load stability | first failures ending at 72, 75, and 79 | **Failed** | one expert exceeded 0.8 of a 20-step layer load | treating these jobs as normal multi-expert trajectories |
| Pre-failure spectral candidate | selected points through step 50 | Not observed | at least one contrast below null and too few positive layers | early selected-point formation before collapse |
| Post-failure spectral candidate | seeds 29/43 at step 120 | Ambiguous and ineligible | trajectory already invalid | nothing about stable formation |
| Paired bootstrap for full candidate | not executed for terminal claim | Ineligible | load prerequisite failed first | nothing beyond the load-boundary decision |
| Three-seed $T_{form}$ | no eligible full analysis | Insufficient | terminal load guard | neither H1 nor the stable-real-run rival |

- **Falsified physical prior:** none. The controlled covariance multiplier was
  not tested under a valid real trajectory.
- **Falsified mathematical model:** none. The local model does not guarantee
  load stability in a jointly trained sparse MoE.
- **Falsified operationalization / proxy:** the registered score-bias condition
  failed as a load-stable carrier for the E03-R question.
- **Falsified implementation:** no diagnostic implementation failure remains;
  the executed bias matched the approved formula. Its ability to stabilize load
  was insufficient.
- **Falsified metric:** none. The metric was deliberately not applied after its
  prerequisite failed.
- **Remaining rivals:** evolving expert advantage, top-1 positive feedback,
  bias strength and saturation, AdamW, representation drift, and non-Gaussian
  data may drive both load concentration and spectral motion.

## 9. Full Experiment Record

The experiment began with a registered engineering smoke. After repairing
actual-input dtype replay, exact top-k tie semantics, and a PyTorch 2.8 DDP
static-graph incompatibility, small runs passed the numerical guards. A later
audit found that the scientific analyzer required ordered fp32 calibration
Router inputs at every heavy snapshot for paired sequence bootstrap. The
initial full jobs were stopped, and the missing artifact, sidecars, and
fail-closed analyzer guards were added.

The full-shape analysis closure then compared continuous step 2 with a fresh
step-1 checkpoint followed by process-restart step 2. Loss, batch indices,
bias, scheduler, model, optimizer, boundary activation hashes, token hashes,
and artifact hashes passed. The actual 200-resample bootstrap, 256-sample null,
and full crossing executed. The final float64 null preservation errors were
$2.66\times10^{-15}$ for Stiefel orthogonality and
$5.57\times10^{-15}$ for singular-Gram relative error. Typed verdict was
`analysis_closure_pass_not_scientific`.

The source was then frozen into one 43-file content-addressed snapshot. All
three independent jobs verified it in-worker, started with zero retries, and
passed their first heavy/step guards. Dense monitoring at steps 25 and 50 found
no orientation candidate and no load failure. At step 100, a retrospective
registered 20-step audit found maximum shares near 0.99 in all seeds. Exact
first failing windows ended at steps 72, 75, and 79. Because this condition is
historical and irreversible, the three jobs were stopped at 18:44:10 UTC.

The stop raced only with in-progress computation: final complete per-step
records are 131/130/132, and complete heavy states end at 120. No step-250
training checkpoint exists. All material artifacts were preserved; no invalid
run was deleted or relabeled.

## 10. Interpretation

The most important result is not that the head contrast was absent or present.
It is that the registered real-training carrier failed before that question
could be adjudicated. At early valid selected points the Router did not exceed
its orientation null. Later, two seeds did exceed both nulls, but only while a
single expert received nearly all of a layer's traffic. Such a state can alter
expert advantage, Gate gradients, and the representation basis together, so it
cannot be interpreted as ordinary Router--Expert formation.

The diagnostic pipeline nevertheless changed our operational understanding:
an auxiliary-loss-free score bias with the approved update and clip was not a
sufficient load-stability mechanism for this six-layer top-1 DCLM run. A new
real-run design must establish stability without silently changing the
scientific object or attributing a load intervention to spectral causality.

## 11. Claim Boundary

### Supported

- the three registered E03-R full trajectories violated the load guard;
- the violation was reproducible and preceded an eligible persistent
  head-formation decision;
- actual-input, basis, update, capacity, source, and analyzer paths were
  operational and auditable;
- post-collapse contrast crossings are ineligible evidence under the Protocol.

### Not supported

- that a load-stable real DCLM MoE does or does not form head alignment;
- that covariance, expert feedback, AdamW, or basis motion caused the observed
  post-collapse spectral changes;
- that the E03-S controlled covariance-speed result is weakened;
- that middle/tail information is unavailable or functionally useless;
- that the registered score bias is worse than every alternative;
- that spectral routing improves validation loss, FLOPs, or wall-clock time.

## 12. Next Decision

Decide whether to approve a new E03-R Protocol with a separately validated
load-stability mechanism. Approval must freeze the intervention, load guard,
attribution boundary, and a small pre-full stability gate before new 2B-token
runs are submitted.

## 13. Links And Artifact Map

- **anchor:** [A15_00_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor.md)
- **protocol:** [protocol.md](protocol.md)
- **summary:** [summary.md](summary.md)
- **central figure:** [e03_r_load_collapse_and_contrasts.png](figures/e03_r_load_collapse_and_contrasts.png)
- **selected table:** [e03_r_terminal_selected_points.csv](tables/e03_r_terminal_selected_points.csv)
- **code workspace:** [a15_e03_r_real_dynamics](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/)
- **worker record:** [full_run_record.md](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/full_run_record.md)
- **runner:** [run_full.sh](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/source_snapshots/02400ef6d5ba30d89d736ba0e1c23b18fb3228e46d5822d7ee9a7be3b330fe13/package/scripts/run_full.sh)
- **submitter:** [submit_full_acp.sh](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/source_snapshots/02400ef6d5ba30d89d736ba0e1c23b18fb3228e46d5822d7ee9a7be3b330fe13/package/scripts/submit_full_acp.sh)
- **frozen source manifest:** [source_manifest.json](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/source_snapshots/02400ef6d5ba30d89d736ba0e1c23b18fb3228e46d5822d7ee9a7be3b330fe13/source_manifest.json)
- **terminal analyzer:** [build_terminal_evidence.py](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/scripts/build_terminal_evidence.py)
- **seed-17 run:** [e03-r-full-seed17-frozen-20260730T183100Z](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/runs/e03-r-full-seed17-frozen-20260730T183100Z/)
- **seed-29 run:** [e03-r-full-seed29-frozen-20260730T183100Z](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/runs/e03-r-full-seed29-frozen-20260730T183100Z/)
- **seed-43 run:** [e03-r-full-seed43-frozen-20260730T183100Z](../../../../../XingyuD/MoE_Routing_Experiments/active/a15_e03_r_real_dynamics/runs/e03-r-full-seed43-frozen-20260730T183100Z/)
- **terminal selected evidence:** `runs/_monitoring/terminal_seed{17,29,43}_evidence.json`
- **closure run:** `runs/e03-r-analysis-closure-20260730T180916Z/`
- **ACP jobs:** `om-9cp56m56`, `om-t86sk3ee`, `om-ryaekljn`; closure
  `om-dhfmxf34`
- **reproduction boundary:** the frozen command remains in each ACP job record,
  but rerunning this invalid condition is not authorized. A new Protocol and
  source snapshot are required.

## 14. Two-Hour Monitoring Closure

The operator monitoring window ran from 2026-07-30 17:39:55 UTC through
19:40:19 UTC. The final live ACP read kept `om-9cp56m56`, `om-t86sk3ee`, and
`om-ryaekljn` in the requested `SUSPENDED` state with zero retries. The retained
step and heavy-snapshot boundaries did not change. This terminal state is a
successful fail-closed execution of the Protocol, not a scientific Pass or
Fail for real head formation.
