---
experiment_id: A15_00_E02_early_head_alignment_onset
status: approved_for_implementation_smoke_and_full_frozen_audit
execution_status: completed_early_onset_pass_progressive_strengthening_fail
result_summary: summary.md
created: 2026-07-30
updated: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
execution_scope: existing_checkpoint_frozen_audit_only
---

# Protocol: A15_00_E02 Early Onset of Head-Aligned Linear-Gate Training

## 0. Approval Snapshot

**Approval:** APPROVED on 2026-07-30 for implementation, smoke, and the full
frozen audit. No new training is authorized or required.

**One decision question:** In the LB and batch-gradient training lineages, is
strong equal-energy head alignment already present at 10k, and do the saved
10k--20k and 20k--30k Gate changes strengthen, maintain, or dilute it?

This experiment extends E01 earlier in training. It does not revisit the
decommon lineage and does not test middle/tail-only dispatch.

## 1. Definitions

| Term | Concrete meaning | Decision role | Cannot establish |
| --- | --- | --- | --- |
| actual Gate input $r_\ell$ | tensor directly entering `mlp.gate` at layer $\ell$ | only representation used to fit the covariance basis | expert-input geometry or semantics |
| covariance head $H$ | eigen-ranks 1--64 of centered $r_\ell$ | largest-variance group | semantic commonness |
| middle $M$ | ranks 65--320 | comparison group | functional usefulness |
| tail $T$ | ranks 321--768 | comparison group | functional uselessness |
| fine band $F_j$ | consecutive block of 64 eigenvectors | resolves within-group structure | a task-defined frequency |
| equal-energy Gate gain $G_A$ | squared expert-relative logit gain per unit input energy in subspace $A$ | removes covariance-amplitude advantage | token frequency or loss benefit |
| endpoint contrast $B$ | log gain ratio between head and another group | answers what the trained Gate is aligned with | when that alignment formed |
| update orientation $B^{update}$ | spectral orientation of net saved-checkpoint Gate displacement | says where the net weight displacement points | per-step gradients or endpoint strengthening |
| fixed-basis Gate effect $\Delta_W B$ | change in endpoint contrast caused by replacing $W_a$ with $W_b$ while holding each basis fixed | says whether Gate-weight change strengthens or dilutes head alignment | causal optimizer mechanism |
| basis effect $\Delta_U B$ | change caused by replacing $U_a$ with $U_b$ while holding Gate weights fixed | separates representation drift | why representations drifted |

## 2. Anchor Alignment

The parent Q1 asks whether a trained linear Router merely produces larger raw
responses in high-variance directions, or is itself oriented toward those
directions after equalizing input energy. E01 established head-aligned
endpoints at 30k/40k/80k but did not locate the onset. E02 tests the earliest
available common checkpoints.

**Physical prior:** larger covariance directions generate larger training
signals unless the objective or optimization counteracts them. Therefore a
head-aligned Gate may form early and then remain approximately fixed.

**Strongest rival:** endpoint head alignment may be present at 30k only because
of later Gate updates or representation-basis drift; alternatively the early
net Gate updates may not consistently prefer the head at all.

**Main falsifier:** at 10k the endpoint contrasts are not distinguishable from
zero/random orientation, or the two early fixed-basis Gate effects do not show
a consistent strengthening pattern.

## 3. Hypotheses and Typed Outcomes

### H1-early: head alignment is already established by 10k

For each lineage, the 10k model-level medians of both $B_{H:M}$ and $B_{H:T}$
are positive under paired basis bootstrap and exceed the matched orientation
null. This bounds onset only as **before 10k**.

### H1-progressive: saved early Gate changes keep strengthening head alignment

For each interval, both fixed-basis effects $\Delta_WB_{H:M}$ and
$\Delta_WB_{H:T}$ are positive with bootstrap intervals above zero. The net
displacement $B^{update}$ must be reported, but it is not a substitute for
$\Delta_WB$.

### Rival R-maintain/dilute

The Gate is already head-aligned by 10k, but one or both subsequent
$\Delta_WB$ effects are near zero, mixed across contrasts, or negative. The
training tendency formed earlier than observed and is then maintained or
diluted rather than progressively strengthened.

### Rival R-drift

Endpoint changes are predominantly explained by $\Delta_UB$, while
$\Delta_WB$ is unresolved or opposite.

### Lineage-conditioned

LB and batch-gradient receive separate verdicts. If their typed outcomes
differ, report the difference; do not average it into a universal law.

## 4. Models, Checkpoints, and Scope

| Lineage | Training configuration | Frozen checkpoints |
| --- | --- | --- |
| LB | no centering; linear Gate; $\lambda_{LB}=0.01$ | 10k, 20k, 30k |
| batch-gradient | running input center; `batch_only` center gradient; no LB | 10k, 20k, 30k |

Checkpoint roots:

- LB: `/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-linear_nocenter_lb001_8gpu-center_off-gate_off-acp_off-lb_0.01-linear/checkpoints`
- batch-gradient: `/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-moe_linear_running_center_batchgrad_8gpu-center_running-gate_off-acp_off-lb_0-linear/checkpoints`

Both have 12 MoE layers, hidden width 768, eight experts, and top-1 routing.
The batch-gradient run is **not** a pure gradient-switch causal control: during
training its differentiable batch component also changes the forward center.
At frozen evaluation, the direct Gate pre-input is captured, so the metric is
still computed on the deployed Gate input.

## 5. Data and Registered Bands

Reuse the exact E01 deterministic data contract:

- 32 non-overlapping DCLM training sequences $\times$ 256 tokens for fitting
  the covariance basis;
- 64 held-out DCLM documents $\times$ 256 tokens for realized-response and
  route-ablation metrics;
- identical token IDs, ordering, and masks across all six endpoints.

For the covariance basis $U_\ell=[u_{\ell,1},\ldots,u_{\ell,768}]$ sorted by
decreasing eigenvalue,

$$
F_j=[64(j-1)+1,64j],\quad j=1,\ldots,12,
$$

$$
H=F_1,\qquad M=F_2\cup\cdots\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

No band may be selected after reading the result.

## 6. Primary Metrics

Let $W_\ell\in\mathbb R^{E\times d}$ and remove expert-common logit shifts by

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad \bar W_\ell=C_EW_\ell.
$$

For band $A$ of dimension $d_A$,

$$
G_{\ell,A}(W,U)=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2,
$$

with unit `logit² / activation² / direction`. It measures Gate selectivity
after equalizing direction energy; it does not measure functional utility.

The coarse endpoint contrasts are

$$
B_{\ell,H:M}=\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},\qquad
B_{\ell,H:T}=\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon},
\quad \epsilon=10^{-12}.
$$

$B>0$ means greater equal-energy head gain; $\exp(B)$ is the gain ratio.
Report all fine-band $G_{\ell,F_j}$ values as well as the coarse comparison.

For interval $a\to b$, define $\Delta W=W_b-W_a$ and

$$
B^{update}_{a\to b}
=\frac12\left[B(\Delta W,U_a)+B(\Delta W,U_b)\right].
$$

This reports the orientation of the net displacement, not whether the endpoint
became more head-aligned. The fixed-basis Gate effect is

$$
\Delta_WB_{a\to b}=\frac12\{[B(W_b,U_a)-B(W_a,U_a)]
+[B(W_b,U_b)-B(W_a,U_b)]\},
$$

and the representation-basis effect is

$$
\Delta_UB_{a\to b}=\frac12\{[B(W_a,U_b)-B(W_a,U_a)]
+[B(W_b,U_b)-B(W_b,U_a)]\}.
$$

Compute the complete $3\times3$ crossing $B(W_s,U_t)$ for
$s,t\in\{10k,20k,30k\}$; the formulas above may not be approximated from only
diagonal cells.

## 7. Secondary Metrics

On held-out tokens, also report

$$
V_A^\perp=\mathbb E\|C_EW P_Ax\|_2^2,
\qquad
S_A^\perp=\frac{V_A^\perp}{\mathbb E\|P_Ax\|_2^2}.
$$

$V$ is raw realized expert-relative logit response; $S$ removes total band
energy but remains eigenvalue-weighted within a multi-direction band. Neither
replaces $G$ for the equal-energy question.

Removing each registered band from the centered token representation yields:

- route flip: fraction of tokens whose top-1 expert changes;
- margin support: decrease in the native top-1 versus runner-up logit margin.

These describe current route dependence, not loss benefit or training
compatibility.

## 8. Guards and Uncertainty

1. **Provenance:** record path, size, SHA-256, Gate shape, model config, and
   checkpoint identity before metrics.
2. **Actual input/no-op:** hook direct `mlp.gate` pre-input and replay native
   logits with relative Frobenius error $\le10^{-5}$ and top-1 agreement 1.0.
3. **Basis validity:** orthogonality error $\le10^{-4}$, decreasing
   eigenvalues, and complete 768-rank coverage.
4. **Basis stability:** calibration half-split projector overlap must exceed a
   random same-dimension overlap null; unstable layer/contrast cells are
   ineligible rather than silently pooled.
5. **Orientation null:** preserve every nonzero singular value of $C_EW$ (or
   $C_E\Delta W$) and randomize only its right orientation using 256
   Haar-Stiefel samples.
6. **Bootstrap:** 200 paired calibration-sequence basis bootstraps; 2,000
   paired held-out-document bootstraps for token metrics.
7. **Aggregation:** median across eligible layers, while retaining every layer.
   Layers are not independent seeds.
8. **No practical hard cutoff:** report continuous ratios, zero, confidence
   intervals, and the matched null envelope. Do not add 10%/25% effect lines or
   a required layer count after seeing results.

## 9. Decision Rules

Guards are evaluated first. With valid guards, assign each lineage:

- **early-present:** both 10k endpoint contrasts have lower bootstrap bounds
  above zero and observed medians above their matched orientation-null q95;
- **progressive-strengthening:** early-present and both contrasts of
  $\Delta_WB$ are positive in both 10k--20k and 20k--30k intervals;
- **early-present-maintained/mixed:** early-present, with later fixed-basis
  effects unresolved or mixed rather than jointly positive;
- **early-present-diluted:** early-present, with precise negative fixed-basis
  effects on both contrasts in at least one interval;
- **late-emerging:** 10k unsupported but a later endpoint is supported and a
  preceding fixed-basis effect is positive;
- **no-head-specificity:** endpoint contrasts are precisely non-positive or
  inside the matched random-orientation envelope;
- **insufficient:** a hard guard fails, no eligible layers remain, or the
  intervals cannot distinguish positive, zero, and negative directions.

The cross-lineage result is **lineage-conditioned** whenever the two typed
outcomes differ materially. These are statistical evidence labels, not
practical-effect thresholds.

## 10. Figure and Table Contracts

1. **Decisive figure:** two rows (LB, batch-gradient), columns for endpoint
   $B_{H:M}/B_{H:T}$ and interval $\Delta_WB$; show layer traces, model median,
   paired 95% interval, zero, and matched null q95. It answers whether head
   alignment predates 10k and whether saved Gate changes strengthen it.
2. **Fine-band heatmap:** all 12 bands for $G$, $V$, $S$, route flip, and
   margin support at all six endpoints. It may reveal non-monotonic detail but
   cannot establish functional benefit.
3. **Crossing/decomposition figure:** complete $W_s\times U_t$ matrices and
   $B^{update}/\Delta_WB/\Delta_UB$ by interval.
4. **Compact decision table:** endpoint gain ratios and fixed-basis effects
   with uncertainty, one row per lineage/contrast/step or interval.

Every rendered figure must be visually audited before curation.

## 11. Execution Stages

1. Freeze Protocol and checkpoint/data manifests.
2. Parameterize the validated E01 worker without changing E01 default outputs.
3. Run unit tests, data-hash comparison, preflight, and one smoke endpoint per
   lineage.
4. Extract all six endpoints, one lineage per GPU when possible.
5. Run per-lineage bootstrap/orientation-null analysis.
6. Build combined tables, figures, and the typed verdict.
7. Write `summary.md` and `detailed.md`; curate only the compact evidence and
   figures beside them.

Raw artifacts remain in the worker run directory and are not promoted into the
research mainline.

## 12. Claim Boundary

E02 may establish:

- whether equal-energy head alignment is already visible by 10k;
- whether 10k--20k and 20k--30k net Gate changes are head-oriented;
- whether those Gate-weight changes strengthen, maintain, or dilute endpoint
  head alignment after holding the representation basis fixed;
- whether this description agrees between LB and batch-gradient.

E02 cannot establish:

- the exact onset before 10k;
- per-step gradient dynamics or a causal optimizer mechanism;
- that `batch_only` alone caused a lineage difference;
- that middle/tail routing is functionally better or worse;
- joint-training efficiency, expert formation, or universality across models.

## 13. Approval and Execution Contract

- Approved scope: implementation, smoke, and full frozen E02 audit.
- Excluded scope: decommon re-analysis, new training, middle/tail-only Router,
  Q2/Q3, graph publication, root sync, commit, and push.
- If an unregistered model mismatch or invalid checkpoint is found before
  primary metrics, stop and record an amendment. Do not substitute a different
  lineage or step after reading results.
