---
experiment_id: A15_00_E01_actual_router_input_band_response
status: approved_for_implementation_smoke_and_full_run
execution_status: completed_strict_h1_fail_typed_result
result_summary: summary.md
created: 2026-07-30
updated: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
companion_cn: protocol_cn.md
execution_scope: implementation_smoke_and_full_run
---

# Protocol: A15_00_E01 Actual-Router-Input Band Access And Training Allocation

## 0. Approval Snapshot

**Approval status:** APPROVED. On 2026-07-30 the researcher approved
implementation, smoke testing, and the full frozen audit. This approval does
not authorize new training or band-only dispatch.

**Purpose:** Determine how strongly the trained linear Gate can access and
currently uses the covariance head, middle, and tail of the representation it
actually receives, and whether two saved training intervals allocate net Gate
change preferentially to the head.

**Primary anchor:** [A15_00 actual-input band access and training allocation](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor.md).

**Decision question:** Relative to the head, are middle/tail equal-energy
access and native route use weaker, and do both 30k--40k and 40k--80k net Gate
changes favor the head after input energy and representation-basis drift are
separated?

**Physical prior:** Large covariance eigenvalues mechanically amplify realized
logit response and may also provide stronger optimization signal.

**Core model term:**

$$
G_{\ell,A}(W,U)=\frac{1}{d_A}\|C_EW_\ell U_{\ell,A}\|_F^2.
$$

**Primary metric:**

$$
\mathbf B_\ell^{\rm coarse}
=\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right).
$$

The complete twelve-band $G_{\ell,b}$ profile is mandatory.

**Main falsifier:** Realized head response is high but equal-energy head gain
is not; or either saved interval lacks head-directed net Gate change and the
endpoint trend is explained by basis drift.

**Experiment role:** Frozen root-cause and metric audit.

**Minimal setup:** Two existing lineages, 12 layers, width 768, 8 experts,
top-1 routing; checkpoints 30k/40k/80k; 32 training calibration sequences and
64 held-out evaluation documents, each 256 tokens; coarse and fine bands.

**Conditions:** Six endpoints, two intervals, full
$W_{30/40/80}\times U_{30/40/80}$ crossing, all coarse/fine metrics,
orientation nulls, wrong-layer bases, calibration half-splits, center/DC, and
native no-op replay.

**Pass -- persistent head allocation:** Within each lineage, both 40k and 80k
coarse head contrasts exceed the matched orientation null with paired 95%
intervals above zero; both interval update directions favor the head and
exceed update nulls; both fixed-basis $\Delta_W\mathbf B$ intervals are above
zero; all hard guards pass.

**Fail:** With guards passed and uncertainty narrow, evidence supports an
energy-only account, middle/tail equal-energy gain is not weaker, or neither
interval contains head-directed Gate-weight change.

**Typed outcomes:** Early-only, late-only, reversal/non-monotonic,
representation-drift-only, or lineage-conditioned outcomes are reported
instead of being forced into pass/fail.

**Insufficient:** The actual-input replay, checkpoint compatibility, basis
stability, null, or uncertainty guard fails, or an interval cannot distinguish
positive, zero, and negative directions.

**Can claim:** Equal-energy access, realized response, native route dependence,
and saved-interval net Gate allocation for the two audited lineages.

**Cannot claim:** Functional middle/tail utility, band-only-dispatch loss,
per-step gradient dynamics, a causal mechanism from initialization, universal
model behavior, or improved loss/FLOP.

**Approval decision:** APPROVED FOR IMPLEMENTATION, SMOKE, AND FULL FROZEN
AUDIT.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit / formula | Decision role | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Router reference $g_\ell$ | Upstream Gate-branch representation | MoE input before the Gate's own centering | activation | Validate the deployed transform | Actual Gate covariance by itself |
| Gate-effective input $r_\ell$ | Exact tensor consumed by the linear Gate | Direct `mlp.gate` pre-input hook; LB $r=g$, decommon $r=g-c$ | activation | Primary covariance and replay object | Semantic content |
| Expert input $h_\ell$ | Tensor consumed by sparse experts | First MoE argument | activation | Known-bad object control | Router geometry |
| Coarse head $H$ | Highest-variance directions | ranks 1--64 | 64 directions | Head comparison | Semantic commonness |
| Coarse middle $M$ | Middle-variance directions | ranks 65--320 | 256 directions | Middle comparison | Functional utility |
| Coarse tail $T$ | Low-variance directions | ranks 321--768 | 448 directions | Tail comparison | Functional utility |
| Fine band $F_j$ | Consecutive eigen-rank block | $[64(j-1)+1,64j]$ | 64 directions | Localize non-monotonic structure | Semantic frequency |
| $G_A$ | Equal-energy Gate gain | $\|C_EWU_A\|_F^2/d_A$ | logit$^2$/activation$^2$ | Primary access strength | Token use or utility |
| $\Psi_A$ | Gate row-space mass in a group | $\|C_EWU_A\|_F^2/\|C_EW\|_F^2$ | fraction | Total weight allocation | Input energy or route effect |
| $V_A^\perp$ | Realized expert-contrast logit response | $\mathbb E\|C_EWP_Ax\|^2$ | logit$^2$/token | Native response | Learned preference |
| $v_A^\perp$ | Realized response per direction | $V_A^\perp/d_A$ | logit$^2$/token/direction | Guard unequal group width | Full eigenvalue removal |
| $S_A^\perp$ | Response per unit observed band energy | $V_A^\perp/\mathbb E\|P_Ax\|^2$ | logit$^2$/activation$^2$ | Partial energy control | Pure Gate orientation in a wide group |
| Route flip $F_A$ | Whether removing a band changes top-1 | Section 8 | token fraction | Current native use | Loss utility |
| Margin change $D_A$ | Native margin lost after removal | Section 8 | logit | Current decision dependence | Dispatch utility |
| $\mathbf B^{\rm update}$ | Orientation of net Gate displacement | Section 7 | two log ratios | Saved-interval allocation | Per-step gradients |
| $\Delta_W\mathbf B$ | Endpoint contrast change from Gate weights at fixed bases | Section 7 | log-ratio change | Gate-weight effect | Earlier trajectory |
| $\Delta_U\mathbf B$ | Endpoint contrast change from basis drift at fixed Gates | Section 7 | log-ratio change | Drift rival | Cause of drift |

## 2. Anchor Alignment

**Decision question:** Are middle/tail access and current native use weaker than
the head, and do both saved intervals allocate more Gate gain to the head?

**Prior tested:** Covariance energy creates a mechanical response advantage
and may create an optimization advantage.

**Core terms:** $G_A$ and $\mathbf B^{\rm coarse}$.

**Falsifier:** $V_H^\perp$ is high without higher $G_H$, or the two Gate
displacements do not favor the head after the $W/U$ crossing.

**Boundary:** Access, current native use, and saved-interval allocation only;
no functional dispatch utility.

## 3. Tested Hypothesis

### H1 -- Persistent head-directed allocation

1. At 40k and 80k, $G_H>G_M$ and $G_H>G_T$.
2. The complete fine profile does not reveal a contradictory hidden peak.
3. Net 30k--40k and 40k--80k Gate displacements both favor the head.
4. Fixed-basis Gate effects are positive rather than basis-only effects.
5. Middle/tail may remain accessible and used, but at weaker measured strength.

### H0-energy -- Energy only

$V_H^\perp$ is high, but $G_H$ has no stable advantage over middle/tail.

### H0-drift -- Representation only

Endpoint changes are dominated by $\Delta_U\mathbf B$ while
$\Delta_W\mathbf B$ is near zero.

### H0-stage -- Stage-specific or non-monotonic

The two saved intervals have different or opposite directions.

## 4. Rival Explanations

1. Input energy: report $V\rightarrow S\rightarrow G$; decide preference with $G$.
2. Unequal coarse dimensions: direction-normalize $G$ and $v$; also report total $V$ and $\Psi$.
3. Expert-common logit shift: use $C_E$; raw $W$ is debug only.
4. Center/DC: report $C_EW\mu^{(r)}$ separately; decommon's $c$ is already in $r=g-c$.
5. Gate norm/singular-value growth: preserve nonzero singular values in orientation nulls.
6. Representation-basis drift: compute the complete $3\times3$ crossing.
7. Finite calibration: half-split bases, dimension-matched overlap null, independent half verdicts.
8. Fine-band selection: simultaneous bootstrap/Haar envelope; no post-hoc peak selection.
9. Unequal interval length: do not compare raw 10k-step and 40k-step magnitudes as rates.
10. Model specificity: adjudicate LB and decommon separately.

## 5. Data / Model / Algorithm / Objective

### 5.1 Frozen models and checkpoints

| Lineage | Router mode | Checkpoint root | Steps |
| --- | --- | --- | --- |
| decommon | running center; Gate receives $g-c$ | `/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768--linear_running_center_8gpu_gbs768-center_running-gate_off-acp_off-lb_0-linear/checkpoints` | 30000, 40000, 80000 |
| LB | no center; load-balancing training | `/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-linear_nocenter_lb001_8gpu-center_off-gate_off-acp_off-lb_0.01-linear/checkpoints` | 30000, 40000, 80000 |

Use `checkpoint-STEP/mp_rank_00_model_states.pt`. Before analysis record size,
SHA-256, model config, Gate shape, layer count, expert order, center state,
tokenizer, and code version. If either 30k endpoint fails before any primary
metric is read, use the latest common early endpoint: 20k, then 10k. The two
lineages may not use different early steps.

### 5.2 Data

- Calibration source: `/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10`.
- Calibration sample: 32 fixed, non-overlapping 256-token uint32 sequences.
- The binary training stream has no source-document boundary; calibration
  units must be called sequences, not documents.
- Evaluation source: held-out
  `/mnt/bucket/109_cache_dir/hf_data/dclm_eval_holdout/global-shard_02_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst`.
- Evaluation sample: 64 deterministic documents, 256 tokens each.
- The same token IDs, masks, and ordering are used for all six endpoints.
- Evaluation uncertainty resamples documents; calibration half-splits and
  basis bootstraps resample training sequences, never individual tokens.

### 5.3 Actual Router input and native no-op

Capture at each layer:

1. upstream Router reference $g_\ell$;
2. direct `mlp.gate` pre-input $r_\ell$;
3. native Gate logits;
4. expert input $h_\ell$ as a known-bad control.

Require LB $r=g$ and decommon $r=g-c$. Offline logits must replay the native
Gate with relative Frobenius error $\le10^{-5}$ and top-1 agreement 1.0.

### 5.4 Covariance and two resolutions

For every model, checkpoint, and layer, fit the covariance eigensystem on
calibration $x_\ell=r_\ell-\mu_\ell^{(r)}$. A primary basis fitted on $g$ or
$h$ violates the Protocol.

$$
F_j=[64(j-1)+1,64j],\quad j=1,\ldots,12,
$$

$$
H=F_1,\qquad M=F_2\cup F_3\cup F_4\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

### 5.5 Statistics

- Primary model aggregation: median across 12 layers; retain every layer.
- Layers are not independent seeds; do not run pseudo-significance tests over layers.
- Pair checkpoint/basis resampling with identical sequence/document indices.
- Freeze 200 calibration-basis bootstraps, 2000 evaluation bootstraps, and 256 orientation-null samples.
- Use simultaneous max-deviation envelopes for twelve fine bands.
- Mark unstable layers ineligible and report the eligible set.

## 6. Conditions, Seeds, And Checkpoints

| Item | Clause / rival | Role | Pass | Fail | Insufficient | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Six-checkpoint provenance | Coordinate compatibility | hard guard | compatible | incompatible | unresolved | checkpoint manifest |
| Actual Gate input/no-op | Correct object | hard guard | exact replay | mismatch | unresolved | no-op audit |
| 30k/40k/80k endpoints | Access/current use | primary | all metrics | typed rival | wide interval | endpoint tables |
| Coarse H/M/T | Registered comparison | primary | resolved contrasts | non-head pattern | uncertain | coarse table |
| Fine 12x64 | Hidden within-group structure | mandatory | all bands | contradiction | unstable basis | fine heatmap |
| 30k--40k and 40k--80k | Macro trajectory | primary | typed verdict | typed rival | uncertain | trajectory table |
| Full $3\times3$ crossing | Weight versus basis | primary | all cells | basis-only | incompatible | crossing table |
| Endpoint/update orientation nulls | Scale and random direction | guard | above q95 | inside null | invalid/tiny | null table |
| Wrong-layer basis | Layer specificity | secondary | same layer stronger | wrong layer equal/stronger | mismatch | control table |
| Half-split basis | Finite calibration | hard guard | profile/verdict repeats | stable contradiction | unstable | stability table |
| Center/DC | Non-band mean response | hard guard | reconstructed separately | leakage | unresolved | DC table |
| Expert-input replay | Known-bad $h$ object | negative control | differs as expected | surprising identity | unavailable | debug table |

## 7. Primary Metric

Let

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad \bar W=C_EW.
$$

For any coarse or fine set $A$,

$$
G_{\ell,A}(W,U)=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2,
$$

$$
\mathbf B_\ell^{\rm coarse}(W,U)
=\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right),\quad \epsilon=10^{-12}.
$$

Do not draw 10% or 25% practical decision lines. Report zero and the matched
orientation-null envelope. Also report

$$
\Psi_{\ell,A}=\frac{\|\bar W_\ell U_{\ell,A}\|_F^2}{\|\bar W_\ell\|_F^2}.
$$

### 7.1 Orientation null

For $\bar W=L\Sigma R^\top$, sample a 768-dimensional Haar-Stiefel right
factor $R_j^{\rm null}$ and form

$$
\bar W_j^{\rm null}=L\Sigma R_j^{{\rm null}\top}.
$$

This preserves every nonzero singular value and randomizes only orientation
relative to the covariance basis. Repeat independently for $C_E\Delta W$.

### 7.2 Saved-interval allocation and $W/U$ crossing

For $\mathcal T=\{30,40,80\}$ compute every

$$
\mathbf B_{\ell;s,t}=\mathbf B_\ell^{\rm coarse}(W_{\ell,s},U_{\ell,t}).
$$

For $a\to b\in\{30\to40,40\to80\}$,

$$
\Delta W_\ell^{a\to b}=W_{\ell,b}-W_{\ell,a},
$$

$$
\mathbf B_\ell^{{\rm update},a\to b}
=\frac12\left[
\mathbf B_\ell^{\rm coarse}(\Delta W_\ell,U_{\ell,a})
+\mathbf B_\ell^{\rm coarse}(\Delta W_\ell,U_{\ell,b})
\right],
$$

$$
\Delta_W\mathbf B_\ell^{a\to b}
=\frac12\left[(\mathbf B_{\ell;b,a}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;a,b})\right],
$$

$$
\Delta_U\mathbf B_\ell^{a\to b}
=\frac12\left[(\mathbf B_{\ell;a,b}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;b,a})\right].
$$

Also report $\|C_E\Delta W\|_F$ and fine-band $G_{F_j}(\Delta W)$. A
numerically tiny displacement makes direction metrics uninterpretable.

### 7.3 Evidence rule

A coarse contrast is supported when the observed model-level statistic exceeds
the matched Haar q95 and its paired 95% interval is above zero. An interval
containing zero is insufficient, not evidence of no effect. A precise interval
at or below zero or a stable statistic inside the null fails the corresponding
head-advantage contrast. These are inferential guards, not practical
effect-size thresholds.

## 8. Secondary Metrics

### 8.1 Realized response

$$
V^\perp_{\ell,A}=\mathbb E\|C_EW_\ell P_{\ell,A}x_\ell\|^2,
\qquad v^\perp_{\ell,A}=V^\perp_{\ell,A}/d_A,
$$

$$
S^\perp_{\ell,A}=\frac{V^\perp_{\ell,A}}
{\mathbb E\|P_{\ell,A}x_\ell\|^2}.
$$

$S$ remains eigenvalue-weighted inside a wide group and is not a pure Gate
preference measure.

### 8.2 Current native route use

With native expert-relative logits $z$,

$$
z_\ell^{(-A)}=z_\ell-C_EW_\ell P_{\ell,A}x_\ell,
$$

fix the native winner $e^*=\arg\max_e z_{\ell,e}$ for each token and define

$$
m_{\rm native}(q)=q_{e^*}-\max_{e\ne e^*}q_e.
$$

The signed margin continues to use the original winner even after a flip; it
therefore measures support for the native decision rather than confidence in
the intervened winner.

$$
F_{\ell,A}=\Pr[\arg\max z_\ell\ne\arg\max z_\ell^{(-A)}],
$$

$$
D_{\ell,A}=\mathbb E[m_{\rm native}(z_\ell)-m_{\rm native}(z_\ell^{(-A)})].
$$

Coarse $F/D$ are total-group effects because dimensions differ; equal-width
fine bands localize the dependence.

### 8.3 Center/DC and reconstruction

Report $C_EW\mu^{(r)}$ separately. The twelve projectors must reconstruct
centered $r$. Report evaluation cross terms if band responses are not additive.

### 8.4 Basis stability

Fit independent bases on fixed 16/16 calibration-sequence halves. Report
coarse/fine projector overlap, a dimension-matched random-overlap null, full
$G$ profiles, and independent coarse verdicts. No fixed 0.75 threshold is used.

### 8.5 Registered figures

1. **Endpoint full-band access and use:** fine $G,V,v,S,F,D$ plus coarse summaries.
2. **Coarse endpoint macro trajectory:** $B_{H:M}$ and $B_{H:T}$ at 30k/40k/80k with intervals and nulls.
3. **Two-interval Gate-by-basis decomposition:** full crossing, $\mathbf B^{\rm update}$, $\Delta_W\mathbf B$, $\Delta_U\mathbf B$, and fine $G(\Delta W)$.

Each figure may support only access/use/saved-interval claims, not utility or
per-step dynamics.

## 9. Known Good / Known Bad / Known Confusing Cases

**Known good:** Direct Gate input replays native logits/top-1; fine projectors
reconstruct centered $r$; coarse projectors equal registered fine-band unions;
nulls preserve singular values; synthetic aligned weights recover the correct
band and a synthetic reversal recovers opposite interval labels.

**Known bad:** Using $h$ as Router input; showing only coarse or a selected fine
peak; claiming preference from $V$ or $S$ alone; interpreting endpoint
difference as Gate-weight change without crossing; comparing raw interval
magnitudes as rates; running frozen middle/tail-only redispatch as utility.

**Known confusing:** High $V_H$ without high $G_H$ is energy-only; nonzero but
weaker middle/tail $G$ means accessible, not invisible; head-biased $G$ without
head-biased $F/D$ separates access from current use; opposite intervals are a
reversal; opposite lineages are model-conditioned.

## 10. Stage-Level Profiling Plan

| Stage | Local question | Pass / fail / unclear | Artifact |
| --- | --- | --- | --- |
| S0 provenance | Are six endpoints comparable? | compatible / pre-metric fallback / stop | checkpoint manifest |
| S1 object/no-op | Is $r$ the exact Gate input? | exact replay / stop | no-op audit |
| S2 basis | Are both resolutions stable and reconstructive? | reproduce / layer insufficient | basis stability |
| S3 endpoints | How much access exists? | quantified typed profile | endpoint gain |
| S4 current use | How much native response/dependence exists? | quantified typed profile | band use |
| S5 macro trajectory | Where do net Gate displacements point? | interval label | update direction |
| S6 decomposition | Is endpoint change from $W$, $U$, or both? | W/U/mixed/unclear | crossing |
| S7 aggregate | What conclusion is supported? | pass/fail/typed/insufficient | verdict |

### 10.1 Approved 24-hour execution envelope

No optimizer step or new model training occurs. Up to the available GPUs may
parallelize the six frozen replays without removing registered conditions.
Plan: 0--3 h provenance/no-op; 3--9 h capture and bases; 9--15 h endpoint and
route-use metrics; 15--20 h crossing/nulls; 20--23 h uncertainty/figures;
23--24 h audit and evidence records. A failed hard guard yields an insufficient
debug record, not post-hoc condition deletion.

## 11. Algorithm Specification

**Inputs:** Six endpoints; fixed calibration/evaluation token IDs; Gate weights,
centers, upstream $g$, actual $r$, expert $h$, and native logits.

**Frozen parameters:** width 768; twelve 64-direction fine bands; coarse
1--64/65--320/321--768; $\epsilon=10^{-12}$; null seed 20260730; bootstrap
seed 20260731; 256/200/2000 null/basis/evaluation replicates.

**Steps:** Validate provenance; apply pre-metric fallback if required; capture
$g/r/h$/logits and no-op replay; fit per-endpoint bases; construct both
resolutions; run reconstruction, half-split, and wrong-layer controls; compute
$G,\Psi,V,v,S,F,D$; build the full crossing; compute both interval update
directions and $W/U$ decomposition; apply paired uncertainty and simultaneous
fine-band envelopes; emit a typed verdict. Do not redispatch tokens.

**Required outputs:**

- `checkpoint_manifest.json`
- `data_manifest.json`
- `noop_audit.json`
- `basis_stability.csv`
- `band_metrics_fine.csv`
- `band_metrics_coarse.csv`
- `endpoint_contrasts.csv`
- `route_ablation.csv`
- `gate_basis_crossing_3x3.csv`
- `update_direction_fine.csv`
- `orientation_null.csv`
- `verdict.json`
- three registered central figures

## 12. Success / Failure / Insufficient Evidence

Every model/layer reports three independent axes: access
($G,\Psi$), current use ($V,v,S,F,D$), and saved-interval allocation
($G(\Delta W),\mathbf B^{\rm update},\Delta_W\mathbf B,\Delta_U\mathbf B$).

**Full pass:** Both lineages separately satisfy all four approval-snapshot pass
conditions. The allowed conclusion is persistent head-directed equal-energy
gain and net Gate allocation in the two audited lineages, while reporting the
measured middle/tail access and use.

**Energy-only:** High realized head response without equal-energy head
advantage. The head response advantage is explained by input energy.

**Middle/tail accessible but weaker:** Access is a continuous $G$ measurement,
not a thresholded label. If middle/tail profiles reproduce across calibration
halves while head contrasts are positive, report their measured weaker access.
Only reproducible $F/D$ permits saying they are currently used.

**Middle/tail not weaker:** A precise coarse contrast at or below zero or a
stable equal-width fine band at least as strong as the head weakens the claim.

**Trajectory labels:** persistent; early-only/saturation; late-only;
non-monotonic/reversal; representation-drift-only; lineage-conditioned.

**Insufficient:** Any provenance, input/no-op, reconstruction, basis, null,
tiny-$\Delta W$, uncertainty, or mandatory-output guard fails.

## 13. What This Cannot Claim

E01 cannot establish that a linear Gate is expressively unable to read
middle/tail information; that middle/tail lacks functional value; that
band-only dispatch changes held-out loss; that covariance causally creates
Gate gradients; that net displacement equals every optimizer update; that
experts formed because of a band; that spectral dispatch improves training or
loss/FLOP; or that two lineages represent all models.

A later functional Protocol must control native linear score, load, capacity,
token count, and batch and use held-out loss or independent-token cross-update
compatibility. A later online-dynamics Protocol may use an approved 8x5090
small-model run to record per-step gradients, margins, flips, loads, and
expert-update interactions.

## 14. Review Notes And Protocol Changes

**Researcher-approved 2026-07-30 revisions:** Use the actual Gate input; use
30k/40k/80k with 20k then 10k pre-metric fallback; register coarse
1--64/65--320/321--768 and fine 12x64; remove practical 25%/10%/8-of-12
thresholds; separate access, current use, saved-interval allocation, and
functional utility; authorize implementation, smoke, and the full frozen audit
but no new training.

**Pre-metric execution amendment:** Training logs confirm the registered
calibration source, but that source is a binary uint32 token stream without
source-document boundaries. The calibration unit is therefore clarified as a
fixed non-overlapping 256-token training sequence. Counts, source separation,
checkpoints, bands, metrics, and held-out document bootstrap are unchanged.
