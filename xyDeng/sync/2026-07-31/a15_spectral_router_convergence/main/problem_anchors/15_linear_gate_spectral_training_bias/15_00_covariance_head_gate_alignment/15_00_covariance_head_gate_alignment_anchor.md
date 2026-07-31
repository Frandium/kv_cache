---
anchor_id: A15_00_covariance_head_gate_alignment
parent_node: rq.expert_specialization
status: controlled_pass_real_insufficient_load_guard
created: 2026-07-30
updated: 2026-07-31
canonical_language: en
companion_cn: 15_00_covariance_head_gate_alignment_anchor_cn.md
source_docs:
  - ../../../../../../daily_research_reports/0730/focus.md
  - ../../../../../../daily_research_reports/0730/problems.md
  - "../../../../../../daily_research_reports/0729/meeting_recordings/0729 组会录音_笔记.md"
  - ../../../../../../daily_research_reports/0728/router_layer_band_probe/results/summary.md
  - ../../../../../../daily_research_reports/0729/router_output_counterfactual/summary_cn.md
---

# A15_00 Band Access And Training Allocation On The Actual Router Input


The researcher approved and executed E01--E03 on 2026-07-30. E01 supports
head-dominant equal-energy access from 30k onward but rejects persistent late
fixed-basis strengthening. E02 shows that the alignment is already much
stronger at 10k and then weakens relatively through 30k. E03-S passes the
controlled covariance-speed clause, while E03-R is insufficient because its
real-training load guard fails before valid formation. See the
[E01 result](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary.md),
[E02 early-onset result](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary.md), and
[E03 dynamics subanchor](subanchors/15_00_01_spectral_learning_dynamics_anchor.md).

## 1. Problem Definition

**Parent problem:** Can a linear MoE Router use middle- and low-variance
information in its actual input, or does it mainly create expert-relative
logit differences along the covariance head?

"The Router sees a band" conflates four propositions:

1. **Access:** equal-energy expert-contrast Gate gain in the subspace;
2. **Current use:** realized token-level logit response and native route
   dependence on the band;
3. **Training allocation:** whether checkpoint-to-checkpoint net Gate change
   favors the band and strengthens endpoint preference;
4. **Functional utility:** whether dispatching with the band improves held-out
   loss or joint-training compatibility.

E01 and E02 audit the first three on frozen checkpoints. They do not answer
the fourth: redispatching experts formed under the native Router would confound
expert mismatch, load, and capacity.

**Single decision question:** Across the audited linear-Gate lineages, is the
trained Gate head-dominant after equalizing input-direction energy, by when is
that alignment already present among available checkpoints, and do saved
Gate-weight changes strengthen or dilute it after separating representation-
basis drift?

**Primary metric:** the coarse equal-energy contrast vector

$$
\mathbf B_\ell^{coarse}(W,U)
=
\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right).
$$

The complete fine 12-by-64 $G_{\ell,b}$ profile is mandatory localization
evidence.

**Main falsifier:** Head $V_H^\perp$ is high, but equal-energy head gain is not
above middle or tail; or the two net Gate intervals do not consistently favor
the head and endpoint change is explained by basis drift.

### Terminology And Definitions

| Term | Concrete object / computation | Unit | Decision role | Cannot prove |
| --- | --- | --- | --- | --- |
| Actual Router representation | Directly hooked mlp.gate input and upstream Router reference | activation | Only allowed covariance object | Expert-input geometry |
| Coarse head | covariance eigen-ranks 1--64 | 64 directions | Highest-variance group | Semantic commonness |
| Coarse middle | eigen-ranks 65--320 | 256 directions | Middle-variance group | Functional value |
| Coarse tail | eigen-ranks 321--768 | 448 directions | Low-variance group | Functional value |
| Fine band | Each consecutive block of 64 eigen-ranks, giving 12 bands | 64 directions | Localizes peaks and non-monotonicity | Semantic frequency |
| Equal-energy gain $G_A$ | $\|C_EWU_A\|_F^2/d_A$ | logit squared per activation squared | Gate access per direction | Realized use or utility |
| Realized response $V_A^\perp$ | $\mathbb E\|C_EWP_Ax\|^2$ | logit squared per token | Total current logit response | Learned preference |
| Current routing use | Route flip and native-margin change after removing a group | token fraction / logit | Native decision dependence | Loss improvement |
| Training allocation | Band gain of net Gate displacement and fixed-basis endpoint change | log ratio | Interval weight-change direction | Per-step gradients or utility |
| Functional utility | Held-out loss or compatibility effect of band-based dispatch | loss / compatibility | Later qualification decision | Not measured by E01/E02 |

## 2. Physical Priors

**P1 -- input energy mechanically amplifies realized response.** Even with
equal Gate gain in every direction, larger $\lambda_i$ makes head
$V^\perp$ larger. Raw response cannot establish learned preference; $G$
removes covariance-eigenvalue amplification.

**P2 -- optimization may allocate more Gate gain to high-variance directions.**
For comparable error correlation, a high-variance coordinate can produce a
larger and more stable Gate gradient, while a low-variance coordinate usually
needs a larger weight for equal logit variance. This predicts head-biased
$\mathbf B^{coarse}$ and net updates. The prior is weakened if middle/tail
equal-energy gain or net updates are not weaker.

**P3 -- Router weights and Router representations co-evolve.** Endpoint
$W_t$--$U_t$ alignment can change through $W$, $U$, or their interaction.
The full $W_{30/40/80}\times U_{30/40/80}$ crossing is required.

## 3. Falsifiable Hypotheses

**H1 -- persistent head-directed training allocation.**

1. At 40k and 80k, both components of $\mathbf B^{coarse}$ are stably positive
   and above a singular-value-preserving orientation null;
2. middle/tail may have nonzero access or route effect, but are weaker than
   the head;
3. both 30k--40k and 40k--80k net Gate changes favor the head and increase
   fixed-basis head-versus-middle and head-versus-tail endpoint contrasts.

**E02 follow-up H1-early -- alignment is established by 10k.** At the earliest
common checkpoint, both head contrasts exceed zero and the matched
singular-value-preserving orientation null.

**E02 follow-up H1-progressive -- 10k--30k continues to sharpen it.** Both
10k--20k and 20k--30k fixed-basis Gate effects are positive for H:M and H:T.
The registered rival is early formation followed by maintenance or spectral
broadening.

**Strongest rival R0 -- energy-only dominance.** $V_H^\perp$ is high because
$\lambda_H$ is high; equal-energy gain and net Gate changes do not favor head.

**Rival R1 -- representation-only drift.** Endpoint contrasts change because
$U_t$ moves, while the fixed-basis Gate effect is near zero.

**Rival R2 -- stage-specific or non-monotonic training.** Only one interval
favors head, or the two intervals have opposite signs.

| Evidence | H1 | R0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| Head $V_H^\perp$ | high | high | may be high | unrestricted |
| $\mathbf B^{coarse}$ | both positive | flat/nonpositive | endpoint may be positive | stage-dependent |
| $\mathbf B^{update}$ | head-biased in both intervals | not head-biased | unstable/apparent | intervals disagree |
| $\Delta_W\mathbf B$ | positive in both intervals | near zero | near zero | intervals disagree |
| $\Delta_U\mathbf B$ | unrestricted | unrestricted | explains most change | unrestricted |

**Pass:** Both lineages support H1.

**Fail:** Valid measurements precisely support energy-only dominance,
middle/tail not being weaker, or no head-directed Gate change in either
interval.

**Insufficient:** Object/no-op, checkpoint, basis-stability, or uncertainty
guards fail, or interval effects cannot distinguish positive, zero, and
negative directions.

There is no 25%, 10%, or 8-of-12 practical hard margin. Haar q95 and bootstrap
intervals only determine distinguishability from random orientation or zero.
All effect sizes and layerwise results remain mandatory.

## 4. Mathematical Model

### 4.1 Actual Router Input And Logit Decomposition

Let $r_\ell$ be the directly hooked Gate-effective input. For LB,
$r_\ell=g_\ell$; for decommon, $r_\ell=g_\ell-c_\ell$. Define the covariance
basis on $r_\ell$, the representation actually consumed by the linear Gate.
The upstream $g_\ell$ is captured only to validate that transformation:

$$
x_\ell=r_\ell-\mu_\ell^{(r)},\qquad
\Sigma_\ell=\mathbb E[x_\ell x_\ell^\top]
=U_\ell\Lambda_\ell U_\ell^\top.
$$

If calibration singular values are $s_i$, then $\lambda_i=s_i^2/N$. Router
logits decompose as

$$
z_\ell=W_\ell r_\ell
=W_\ell\mu_\ell^{(r)}+\sum_AW_\ell P_{\ell,A}x_\ell.
$$

Top-1 routing depends only on expert contrasts:

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad
\bar W_\ell=C_EW_\ell.
$$

### 4.2 Two Band Resolutions

$$
F_j=\{64(j-1)+1,\ldots,64j\},\qquad j=1,\ldots,12,
$$

$$
H=F_1,\qquad
M=F_2\cup F_3\cup F_4\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

Thus $d_H=64$, $d_M=256$, and $d_T=448$. Cross-group gain comparisons are
direction-normalized; total response is reported separately.

### 4.3 Access And Realized Response

For any coarse group or fine band $A$,

$$
G_{\ell,A}(W,U)
=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2,
$$

$$
V^\perp_{\ell,A}
=\mathbb E\|\bar W_\ell P_{\ell,A}x_\ell\|_2^2,
\qquad
v^\perp_{\ell,A}=\frac{V^\perp_{\ell,A}}{d_A},
$$

$$
S^\perp_{\ell,A}
=\frac{V^\perp_{\ell,A}}
{\mathbb E\|P_{\ell,A}x_\ell\|_2^2}.
$$

$G$ removes all $\lambda_i$ factors. $S$ remains $\lambda_i$-weighted inside
a wide coarse group. $V$ is total realized response and $v$ is per-direction
realized response.

### 4.4 Current Routing Use

For $z_\ell^{(-A)}=z_\ell-\bar W_\ell P_{\ell,A}x_\ell$, define

$$
F_{\ell,A}
=\Pr[\arg\max z_\ell\ne\arg\max z_\ell^{(-A)}],
$$

$$
D_{\ell,A}
=\mathbb E[m_{\rm native}(z_\ell)-m_{\rm native}(z_\ell^{(-A)})].
$$

These measure native decision dependence, not redispatch utility.

### 4.5 Two Training Intervals And The $W/U$ Decomposition

For a registered three-checkpoint set $\mathcal T$ and every
$s,t\in\mathcal T$, compute

$$
\mathbf B_{\ell;s,t}
=\mathbf B_\ell^{coarse}(W_{\ell,s},U_{\ell,t}).
$$

For each adjacent registered interval $a\to b$, let
$\Delta W_\ell^{a\to b}=W_{\ell,b}-W_{\ell,a}$ and define

$$
\mathbf B_{\ell}^{update,a\to b}
=\frac12\left[
\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,a})
+\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,b})
\right],
$$

$$
\Delta_W\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;b,a}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;a,b})
\right],
$$

$$
\Delta_U\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;a,b}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;b,a})
\right].
$$

$\mathbf B^{update}$ is directional and must be read with
$\|\bar{\Delta W}\|_F$ and $\Delta_W\mathbf B$. The intervals have unequal
lengths, so raw magnitudes are not compared as rates.

### 4.6 Adjudication Flow

~~~mermaid
flowchart LR
  R["Directly hooked Router representation"] --> C["Coarse 3 bands and fine 12 by 64"]
  C --> V["Realized response V and per-direction response v"]
  C --> G["Equal-energy access G and H:M/H:T contrasts"]
  C --> U["Band-removal flips and margins"]
  G --> J1{"Is head still stronger than middle/tail?"}
  V --> J1
  J1 -- "Only V is head-heavy" --> E["Energy-only"]
  J1 -- "G is also head-heavy" --> Q["Saved-checkpoint W by U decomposition"]
  Q --> J2{"When is alignment present, and do later W changes sharpen it?"}
  J2 -- "No" --> S["Stage-specific, non-monotonic, or basis drift"]
  J2 -- "Yes" --> H["Persistent late head-directed allocation"]
  U --> B["Current use only, not functional utility"]
~~~

## 5. Computational Realization

**Models and checkpoints:** E01 uses LB and decommon at 30k/40k/80k. E02 uses
LB and batch-gradient at 10k/20k/30k. Every lineage is a 12-layer, width-768,
eight-expert, top-1 linear-Gate model. All primary state files were verified
read-only with hashes, configs, Gate shape, expert order, center state,
tokenizer, and coordinate signatures.

**Router representation:** Capture the mlp.gate pre-input, native logits, and
upstream Router reference directly. $h_\ell$ is a known-bad control. Offline
recomputation must replay native logits and top-1.

**Data:** DCLM holdout separated from the training top-level shard. Thirty-two
fixed 256-token training sequences fit $\mu,U$; 64 evaluation source documents
measure held-out response, flips, and margins. All checkpoints use identical
tokens. The training binary stream has no source-document boundaries.

**Output granularity:** Every model x checkpoint x layer emits all three
coarse groups and all 12 fine bands. A head/rest-only summary is invalid.

**Necessary controls:** Singular-value-preserving random input orientations for
$\bar W$ and $\bar{\Delta W}$; wrong-layer bases; calibration half-splits;
separate center/DC; band reconstruction; native no-op.

**Excluded from E01/E02:** Band-only dispatch, middle/tail-only dispatch,
forced-expert loss, one-step compatibility, expert training, and loss/FLOP.

## 6. Minimal Falsification Tests

**Decisive tests:** [E01 actual-input multiresolution Protocol](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/protocol.md)
and [E02 early-onset Protocol](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/protocol.md).

**Endpoint comparison:** At 30k, 40k, and 80k, report complete coarse/fine
$V^\perp/v^\perp/S^\perp/G/F/D$ and compare
$(B_{H:M},B_{H:T})$ with zero and the orientation null.

**Training comparison:** In both intervals, compute fine $G(\Delta W)$ and
coarse $\mathbf B^{update}$, $\Delta_W\mathbf B$, and
$\Delta_U\mathbf B$.

**Evidence rule:** No practical effect-size hard margin. Coarse access and
trajectory comparisons use model-level Haar q95 and paired
calibration-sequence basis-bootstrap intervals. Held-out use metrics resample
evaluation documents. The fine profile uses a simultaneous envelope to
prevent post hoc peak selection among 12 bands.

**Pass:** In both lineages, coarse head gain is above middle and tail at 40k
and 80k, and both net Gate intervals favor head and strengthen fixed-basis
endpoint contrasts.

**Fail:** With all guards valid, results precisely support energy-only
dominance, middle/tail equal-energy gain not being weaker, or no head-directed
Gate change in either interval.

**Insufficient / typed alternatives:** One positive interval is early-only or
late-only, not persistent. Opposite intervals are non-monotonic. A dominant
$U$ effect is representation-drift-only. Failed validity or precision guards
produce insufficient evidence.

**Allowed claim:** Equal-energy access, realized response, native decision
dependence, and net weight-allocation direction across two saved intervals for
the audited Gates.

**Cannot claim:** Middle/tail functional utility, loss effects of their
dispatch, per-step gradients, a from-initialization cause, or universal model
behavior.

## 7. Current Evidence

**Direct result -- actual-input access:** The [E01 evidence record](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary.md)
passed checkpoint, native-input replay, basis reconstruction/stability, and
singular-value-preserving orientation-null guards. At 40k/80k, median
$G_H/G_M$ is 5.41/6.36 for LB and 4.03/4.27 for decommon; median $G_H/G_T$ is
19.98/25.36 and 14.61/17.15. All log contrasts have paired basis intervals
above zero and exceed matched Haar q95 near 0.04. F1 is the strongest
model-median fine band at all six endpoints.

**Direct result -- current use:** Middle/tail $G$, response, flip, and margin
effects are nonzero but weaker. At 80k, median head/middle/tail route-flip
fractions are 0.741/0.126/0.018 for LB and 0.645/0.089/0.013 for decommon.

**Direct result -- saved intervals:** Every net displacement has positive
$\mathbf B^{update}$ above its matched null. Nevertheless,
$\Delta_WB_{H:M}$ is precisely negative in both intervals and both lineages.
$\Delta_WB_{H:T}$ is positive in both LB intervals, negative in decommon
30k--40k, and crosses zero in decommon 40k--80k. Late positive endpoint H:M
movement includes a positive $U$ contribution while the fixed-basis $W$
contribution is negative.

**Interpretation:** The energy-only endpoint rival is rejected. The strict H1
of persistent fixed-basis strengthening across both contrasts is rejected.
The supported type is head-aligned endpoints and head-oriented net
displacements without persistent relative head strengthening. A positive
$\mathbf B^{update}$ cannot substitute for $\Delta_W\mathbf B$ because adding
an update to an already more head-biased $W$ includes signed cross terms.

**Direct result -- earliest available onset:** The
[E02 evidence record](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary.md)
passes every validity guard. At 10k, LB has median $G_H/G_M=10.42$ and
$G_H/G_T=37.11$; batch-gradient has 9.19 and 42.73. The corresponding log
contrasts are far above matched orientation-null q95 values of 0.034--0.048.
Thus the trained Router--representation system is strongly and non-randomly
head-aligned by 10k.

**Direct result -- 10k--30k broadening:** At 30k those ratios fall to
5.38/19.60 for LB and 4.99/24.80 for batch-gradient. Both lineages have
negative fixed-basis $\Delta_WB_{H:M}$ in both intervals. Batch-gradient also
has negative $\Delta_WB_{H:T}$ in both intervals; LB has small positive H:T
effects, but negative $\Delta_UB_{H:T}$ more than offsets them at the endpoint.
Middle/tail gain and route dependence remain nonzero and increase relatively.

**Updated interpretation:** The strong endpoint alignment is formed before
the first available 10k checkpoint, not progressively sharpened from 10k to
30k. Because 10k is about 7.86B nominal tokens and initialization was not
saved, E02 does not identify the exact onset or isolate Gate-gradient cause
from pre-10k joint representation co-adaptation.

**Controlled causal result:** The
[E03-S result](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
passes the registered fixed-target test. With matched Gate-space targets,
trace-normalized Gaussian inputs, and pure SGD, 4:2:1 covariance produced
approximately 1:2:4 head/middle/tail learning times and 16:4:1 produced
approximately 1:4:16. Flat and whitened conditions returned the three learning
times to the same range, and the tail-only target remained learnable. Thus
covariance anisotropy is a causal finite-time speed factor in the registered
controlled system; it is not a functional preference or a real-DCLM result.

**Real-trajectory result:** The
[E03-R result](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
is scientific `insufficient_load_guard`. In all three seeds, a single expert
exceeded 80% of one layer's 20-step load before a valid persistent formation
time could be established; rolling concentration approached 0.99 by step 100.
The diagnostic chain passed, but post-collapse spectral movement is ineligible
as evidence of ordinary Router--Expert formation.

## 8. Claim Boundary And Next Decision

**Supported now:** Across E01/E02, the trained Gate has reproducible,
layer-specific head-dominant equal-energy access on its actual input.
Middle/tail are accessible and currently used, but at weaker measured
strength. The alignment is already strongest at the earliest available 10k
checkpoint; 10k--30k endpoint ratios then decline while remaining strongly
positive.

**Weakened / rejected now:** High head $V$ is not merely an input-energy
artifact. Neither 10k--30k nor 30k--80k shows a universal persistent
fixed-basis strengthening of both head contrasts. Reading
$\mathbf B^{update}$ alone as “training makes the Gate increasingly
head-biased” is invalid.

**Unresolved within Q1:** Whether a load-stable real DCLM trajectory exhibits
the controlled head-first signature; raw Gate gradient versus optimizer-
applied update; signed $W$--update band interactions; representation-basis
co-adaptation; and whether trainable experts amplify or compensate the bias.
The new local mechanism question is whether centered-common structure remains
stable across documents after de-meaning and whether the Gate prefers it over
shard-local residuals.

**Cannot claim:** Expressive inability to read middle/tail, a causal covariance
account of the existing DCLM endpoints, benefit or harm of head alignment, loss effects of
middle/tail dispatch, functional expert specialization, or loss/FLOP
improvement.

**Exactly one next decision:** Complete the authorized parallel frozen
execution of the two approved E01 Protocols under
[A15_00_02 centered-common stability](subanchors/15_00_02_centered_common_subspace_stability_anchor.md)
and
[A15_00_03 pooled-versus-local Gate preference](subanchors/15_00_03_gate_transferable_vs_local_residual_alignment_anchor.md).
The analyses share one activation extraction and then execute independently.
A matched stability intervention is considered only if both pass. This
execution remains inside Q1 and does not test functional utility or authorize
new training.
