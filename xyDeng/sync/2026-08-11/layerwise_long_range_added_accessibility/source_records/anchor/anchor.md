---
anchor_id: 15_08_01_layerwise_long_range_compositional_innovation
status: AWAITING_HUMAN_BLOCK_B_RECONFIRMATION
canonical_language: en
chinese_companion: 15_08_01_layerwise_long_range_compositional_innovation_anchor_cn.md
thinking_card: 15_08_01_layerwise_long_range_compositional_innovation_thinking_card_cn.md
parent_anchor: ../15_08_target_conditioned_layer_innovation_anchor.md
parent_line: 15_spectral_representation_and_functional_routing
execution_authority: human_approved_2026_08_11
created: 2026-08-11
updated: 2026-08-11
---

# A15_08_01 Layerwise Long-Range Added Accessibility And Representation Rank

Researcher judgment: [Chinese Thinking Card](15_08_01_layerwise_long_range_compositional_innovation_thinking_card_cn.md). Parent definition: [A15_08 Target-Conditioned Layer Innovation](../15_08_target_conditioned_layer_innovation_anchor.md). Chinese companion: [Anchor](15_08_01_layerwise_long_range_compositional_innovation_anchor_cn.md). Execution contract: [approved Protocol](../../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/protocol.md).

The researcher confirmed the research direction, attention-only representation site, complete layerwise gain curves, representation rank rather than task rank, exclusion of Router work, and the TRAIN-only equal-layer trace-normalized common basis. The requested Reserved image and current-container runtime now pass validation. Exact-tokenizer preflight showed that the original long entity strings do not fit before the registered far slot. Candidate amendment A4 replaces only their surface encoding with compact globally disjoint identifiers and passes the full 320-world preflight, but Block B awaits human reconfirmation. All prior jobs are non-evidentiary; no scientific result exists.

## 1. Problem Definition

A15_08_E04 established one local fact at block 25 on a controlled two-hop task: the normalized attention update not linearly predicted from the old state added held-out target accessibility beyond target-independent same-budget controls. E04 did not determine how that gain changes with depth or whether the update representation associated with a remote necessary fact has low representation rank.

The one primary decision question is:

> In a frozen Qwen3-8B and a matched one-hop/two-hop by near/far relational task, do deep attention writes add more held-out target accessibility specific to a remote necessary bridge fact than shallow writes do, after removing generic relocation and generic two-hop difficulty?

The same frozen representations also support a registered characterization that does not alter the primary verdict: how do the full covariance spectrum, effective rank, and common-data-basis distribution of the matched long-range interaction update change across all 36 blocks? “Representation rank” means the covariance rank of the update activations themselves, not the task-readout rank bounded by seven for an eight-class target.

Terms are defined as follows:

- A **source entity** is the query starting point; a **bridge entity** is the intermediate entity connecting that source to the final answer.
- A **bridge fact** maps the source to the bridge entity. It is necessary for two-hop queries and unnecessary for one-hop queries.
- A **terminal fact** maps the bridge entity to the final answer.
- A **fictional relational world (world/episode)** contains one random relation system, every target, and all four matched conditions. It is the independent resampling unit.
- The **decision token** is the final query token before the model emits an answer.
- **Added accessibility** is the held-out cross-entropy reduction from adding the current-layer residual update to a frozen old-state readout. It is not Shannon-information creation.
- **Effective rank** summarizes how update-covariance energy is distributed over its full eigenspectrum. It does not prove target function for those dimensions.

All 36 blocks receive an individual measurement and appear in the depth curves. The earlier ten adjacent pairs were a local-slope sampling device under a limited measurement budget; once a complete layerwise curve became required, they no longer define the formal sample. Adjacent-layer differences remain descriptive and are neither independent replicates nor cross-layer vector definitions of new knowledge.

## 2. Physical Priors

1. **A matched within-layer MLP-input coordinate permits exact subtraction.** At the same token, block, and post-attention RMSNorm, the no-write counterfactual and actual MLP inputs share one hidden coordinate. This prior fails for mismatched tokens, hooks, or normalization maps.
2. **Added accessibility and spectral energy are separate evidence layers.** Held-out ridge gain asks how much readable target information the layer adds; covariance spectrum asks how the corresponding update energy is distributed. Spectral concentration cannot replace gain, and positive gain does not establish low representation rank.
3. **Necessity by distance isolates the named long-range mechanism.** Moving the bridge changes physical position in both one-hop and two-hop inputs, but only two-hop requires the bridge. The difference-in-differences removes additive generic position and generic hop effects.

## 3. Falsifiable Hypotheses

**Primary H2-LC — depth-selective long-range added accessibility.** When the frozen model is capable and the bridge dependency passes counterfactual checks, the residual-update-gain interaction associated with a remote necessary bridge is positive in deep blocks and larger overall in deep than in shallow blocks. The absolute remote two-hop gain must also increase, preventing deterioration of control cells from manufacturing a positive interaction.

**Strongest rival R1 — generic retrieval position plus generic two-hop difficulty.** Distance affects one-hop and two-hop similarly, while two-hop adds only distance-independent difficulty. R1 permits distance and hop main effects but predicts no necessity-by-distance interaction that strengthens with depth.

Other named rivals are redundant re-encoding of old information, greater old-state cross-entropy headroom in the hard condition, early availability, a middle-only peak promoted into a deep law, template or code shortcuts, and insufficient frozen-model capability on remote two-hop inputs.

**Secondary H1-REP — spectrally compact representation of the long-range interaction update.** Every layer must report its complete spectrum, entropy effective rank, 80%-variance rank, and energy curve in one frozen common data basis. The researcher has not approved an absolute effective-rank threshold for a “low-rank Pass”; this experiment therefore characterizes and compares H1-REP without inventing a Pass/Fail boundary. Target-conditioned ranks $2\ldots7$, generalized eigen-directions, and Router are out of scope.

The main falsifier is a valid-guard 95% upper bound of the registered deep-minus-shallow interaction $T_{depth}$ at or below zero. That rejects the current deep-emergence statement; no spectrum can rescue it.

## 4. Mathematical Model

### 4.1 Matched relational worlds

For world $w$, sample two type-matched bijections:

$$
\phi_w:\mathcal S\rightarrow\mathcal B,
\qquad
\psi_w:\mathcal B\rightarrow\mathcal Y.
$$

For source $S_i$, define the bridge and final answer:

$$
B_i=\phi_w(S_i),
\qquad
Y_i=\psi_w(B_i).
$$

A one-hop query exposes $B_i$ and needs only the terminal fact $\psi(B_i)=Y_i$. A two-hop query exposes only $S_i$ and must first use the bridge fact $\phi(S_i)=B_i$. The terminal fact stays near the query. The bridge fact alone swaps with a same-relation, same-token-length matched distractor between near and far positions.

The four cells are one-hop near $1N$, one-hop far $1F$, two-hop near $2N$, and two-hop far $2F$. The four inputs for one $(w,i)$ keep the world, fact multiset, answer, template family, and total token length fixed.

### 4.2 Layer state, raw attention write, and normalized update

At the pre-answer decision token, let:

- $h_{\ell,w,i,c}\in\mathbb R^d$ be the actual input residual of block $\ell$; for $\ell>1$ it is the previous block's actual output;
- $a_{\ell,w,i,c}\in\mathbb R^d$ be the raw attention write after output projection and before residual addition;
- $H_{\ell,w,i,c}\in\mathbb R^d$ be the actual full output of block $\ell$;
- $N_\ell$ be block $\ell$'s post-attention RMSNorm.

Define the no-write counterfactual and actual MLP inputs:

$$
X_{\ell,c}=N_\ell(h_{\ell,c}),
\qquad
Z_{\ell,c}=N_\ell(h_{\ell,c}+a_{\ell,c}),
$$

and the exact normalized attention update:

$$
U_{\ell,c}=Z_{\ell,c}-X_{\ell,c}.
$$

$a_\ell$ is already the additive write in the residual coordinate, so $a_\ell-h_\ell$ is not defined as an increment: the two tensors have different computational roles. The raw $a_\ell$ is retained only as a write-spectrum diagnostic; $U_\ell$ is the primary functional object. A layerwise readout from $H_\ell$ describes cumulative state accessibility and cannot replace the within-layer increment.

### 4.3 Conditional residual update and layerwise gain

Using TRAIN worlds only and complete-world grouped cross-fitting, fit:

$$
\widehat m^U_{\ell,c}(X)
\approx\mathbb E_{lin}[U_{\ell,c}\mid X_{\ell,c}],
$$

and define:

$$
R_{U,\ell,c}
=U_{\ell,c}-\widehat m^U_{\ell,c}(X_{\ell,c}).
$$

$R_U$ is the computation update not recovered from the current layer's old state by the registered linear predictor. It is not statistically independent and is not a knowledge matrix.

Fit a base ridge readout $b_{\ell,c}$ from $X_{\ell,c}$ and a same-budget additive correction $q_{\ell,c}$ from $R_{U,\ell,c}$ on TRAIN; use DEVELOPMENT only for regularization and CONFIRMATION only for frozen evaluation. For one confirmation example:

$$
g_{\ell,c}
=CE\!\left(b_{\ell,c}(X_{\ell,c})\right)
-CE\!\left(b_{\ell,c}(X_{\ell,c})
+q_{\ell,c}(R_{U,\ell,c})\right).
$$

The equal-world mean is:

$$
G_{\ell,c}=\mathbb E_{conf}[g_{\ell,c}]
\quad\text{nats/example}.
$$

The layerwise necessity-by-distance interaction is:

$$
I_\ell
=(G_{\ell,2F}-G_{\ell,2N})
-(G_{\ell,1F}-G_{\ell,1N}).
$$

AI_PROPOSAL: partition all 36 layers equally into early $\mathcal L_E=\{1,\ldots,12\}$, middle $\mathcal L_M=\{13,\ldots,24\}$, and deep $\mathcal L_D=\{25,\ldots,36\}$. The one primary metric is:

$$
T_{depth}
=\operatorname{median}_{\ell\in\mathcal L_D}I_\ell
-\operatorname{median}_{\ell\in\mathcal L_E}I_\ell.
$$

The complete $G_{\ell,c}$ and $I_\ell$ curves carry the trend interpretation. Adjacent differences $I_{\ell+1}-I_\ell$ describe local slope only and are not independent tests.

### 4.4 Representation rank and a common cross-layer data basis

Within the same world and target, define the long-range necessary-interaction update:

$$
D_{\ell,w,i}
=(R_{U,\ell,2F}-R_{U,\ell,2N})
-(R_{U,\ell,1F}-R_{U,\ell,1N}).
$$

Center the eight target rows inside each world to obtain $\widetilde D_\ell$, then compute the TRAIN covariance:

$$
\Sigma_{D,\ell}^{tr}
=\frac{1}{n-1}\widetilde D_{\ell,tr}^{\top}\widetilde D_{\ell,tr}.
$$

Let its nonnegative eigenvalues be $\mu_{\ell,1}\ge\mu_{\ell,2}\ge\cdots$ and let $p_{\ell,j}=\mu_{\ell,j}/\sum_k\mu_{\ell,k}$. Report representation rank as:

$$
r_{eff,\ell}
=\exp\!\left(-\sum_jp_{\ell,j}\log p_{\ell,j}\right),
$$

$$
r_{80,\ell}^{var}
=\min\left\{r:
\frac{\sum_{j=1}^{r}\mu_{\ell,j}}
{\sum_j\mu_{\ell,j}}\ge0.8
\right\}.
$$

These quantities measure representation rank of update energy without target labels and are not bounded by the eight-class task-rank definition. The report must also give normalization by the sample-identifiable maximum rank and TRAIN/DEVELOPMENT/CONFIRMATION replication, so a finite-sample rank cap is not mislabeled as model low rank.

To compare all 36 layers in one directional coordinate, AI_PROPOSAL constructs an equal-layer, total-variance-normalized TRAIN pooled covariance:

$$
\Sigma_{common}
=\frac1{36}\sum_{\ell=1}^{36}
\frac{\Sigma_{D,\ell}^{tr}}
{\operatorname{tr}(\Sigma_{D,\ell}^{tr})},
$$

$$
\Sigma_{common}
=V_{common}\Lambda_{common}V_{common}^{\top}.
$$

This basis comes from the cross-layer candidate long-range interaction updates, not parameter matrices or confirmation data. Trace normalization prevents a high-energy layer from defining the coordinate by itself; equal layer weight treats shallow and deep blocks symmetrically. A numerically negligible layer trace makes that layer's spectrum invalid rather than forcibly normalized.

In the frozen common basis, normalized layer-$\ell$ energy along common direction $k$ is:

$$
e_{\ell,k}
=\frac{v_k^{\top}\Sigma_{D,\ell}v_k}
{\operatorname{tr}(\Sigma_{D,\ell})},
\qquad
F_\ell(r)=\sum_{k=1}^{r}e_{\ell,k}.
$$

$F_\ell(r)$ is cumulative candidate-update energy in the common basis. It is not directionwise target accessibility; accessibility remains owned by $G$, $I$, and $T_{depth}$.

## 5. Computational Realization

### 5.1 Data, task, and supervision boundary

AI_PROPOSAL: freeze Qwen3-8B and its tokenizer; use eight balanced terminal codes; generate 128/64/128 completely disjoint TRAIN/DEVELOPMENT/CONFIRMATION worlds. Each world contains eight targets and four matched cells. Source and bridge entities, mappings, template instances, and complete texts are split-disjoint; the shared answer alphabet is an intentional overlap.

Input length, terminal-fact position, and matched distractors are controlled. The model must pass remote two-hop restricted-choice capability and bridge-swap counterfactual guards or the representation verdict is Insufficient. The answer code cannot become a pre-answer input shortcut.

The ridge learning task maps $X_{\ell,c}$ to base eight-class logits and $R_{U,\ell,c}$ to additive logit corrections. TRAIN alone fits models, DEVELOPMENT alone selects regularization, and CONFIRMATION labels cannot select layers, plot ranges, ridge values, the common basis, or any spectral threshold.

### 5.2 Extraction identity and complete depth coverage

One frozen-model extraction must save $h_\ell$, $a_\ell$, $X_\ell$, $Z_\ell$, $U_\ell$, and full-block output $H_\ell$ at the same decision token for all 36 blocks. Every block verifies $Z_\ell-X_\ell-U_\ell=0$, attention output-projection identity, token identity, and replay identity.

The design never computes $a_\ell-h_\ell$, $Z_b-X_a$, or subtraction between two layers' separately ranked PCAs. The raw-$a_\ell$ spectrum, full-state $H_\ell$ readout CE, and adjacent-layer slope are auxiliary trajectories. Only conditional $R_U$ gain enters H2-LC.

### 5.3 Frozen common basis and representation spectra

$D_\ell$, layer covariances, $r_{eff,\ell}$, $r_{80,\ell}^{var}$, and $V_{common}$ are frozen from TRAIN representations alone. DEVELOPMENT and CONFIRMATION only project into that basis and report the same metrics. Confirmation curves cannot reorder common ranks, delete layers, change normalization, or select a more favorable rank definition.

The current design constructs no target-conditioned matrix, generalized eigen-direction, or task-readout rank; projects onto no layer-local MLP parameter eigensystem; and trains no Router.

### 5.4 Two required central figures

| Figure | Required question | Axes and aggregation | Allowed conclusion | Limitation |
| --- | --- | --- | --- | --- |
| Layerwise added-gain figure | How do $G_{\ell,c}$ and the necessity-by-distance interaction change with depth? | x: blocks 1--36; y: nats/example; four conditions and $I_\ell$ in separate panels; equal-world paired intervals | exposes the complete depth trend and carries direct H2-LC evidence | cannot locate vector directions or prove low rank |
| Common-basis representation-spectrum figure | How do representation rank and common-direction energy of the long-range interaction update change with depth? | one panel: block versus $r_{eff}$/$r_{80}^{var}$; one panel: common rank versus $F_\ell(r)$; all 36 curves colored from light shallow to dark deep | describes representation-rank trajectory and spectral redistribution in one data basis | cannot rename spectral energy as target accessibility or Router function |

Adjacent-layer differences may appear as faint overlays or a supporting table, but create no third primary verdict. The Protocol must freeze axes, units, curve identities, intervals, and allowed readings for both figures.

## 6. Minimal Falsification Tests

### 6.1 Formal H2-LC verdict

Under valid capability, bridge-dependency, identity, data, and record guards:

**Pass** requires strictly positive paired 95% lower bounds for $T_{depth}$, the deep interaction median, the absolute deep-minus-shallow remote-two-hop gain, the headroom-matched $T_{depth}$, and the contrast against the q95 of target-independent same-budget mismatch banks.

**Fail** requires valid guards and a 95% upper bound of $T_{depth}$ at or below zero, or a precise negative absolute/headroom/capacity clause that supports a named rival. Fail must map to generic distance, generic two-hop, early availability, control deterioration, old-state headroom, or same-budget capacity.

**Insufficient** applies when a decisive interval crosses zero, a capability or bridge-swap guard fails, confirmation leaks, complete world-level arrays are missing, or any layer's hook identity is inconsistent.

A middle-only peak, a compact spectrum, or one large adjacent-layer jump cannot rescue a global deep-versus-early Fail.

### 6.2 Representation-rank reporting rules

Representation rank must be reported for all 36 layers; best layers or splits cannot be selected. These observations permit only bounded descriptions:

| Observation | Allowed update | Forbidden claim |
| --- | --- | --- |
| $r_{eff}$ and $r_{80}^{var}$ are stable across splits and small relative to identifiable rank | update energy for this long-range interaction object is spectrally concentrated | target information is low rank, natively used, or Router-readable |
| effective rank decreases with depth | deep candidate-update geometry becomes more compact | deep added accessibility is larger; that still requires $T_{depth}$ |
| effective rank increases or broadens | deep candidate updates use more distributed representation energy | H2-LC must fail or information is absent |
| TRAIN is compact but DEV/CONF does not reproduce | the apparent low-rank geometry is unstable | no positive H1-REP statement |
| $D_\ell$ trace is negligible | no interpretable interaction-update energy exists at that layer | normalized noise cannot be called low rank |

Because no absolute low-rank threshold has been human-approved, this Anchor registers no H1-REP Pass/Fail. The Protocol must report the quantities without choosing a threshold after inspecting the curves.

## 7. Current Evidence

1. The [advisor variance-interval report](../../../../../../../daily_research_reports/0810/docs/DAILY_SUMMARY_ADVISOR_VARIANCE_INTERVAL_20260810.md) found that variance-growth-selected 160-dimensional intervals did not consistently beat equal-dimensional random directions. This closed parameter-direction variance growth as a sufficient definition of added information and motivates held-out gain plus an activation-derived common basis.
2. [A15_02_07 TAX](../../../../experiments/A15/15_02_layerwise_representation_spectral_atlas/A15_02_07_E01_matched_taxonomy_full/summary.md) found update-only readability without conditional novelty beyond the old state. It demonstrates why a nonzero update spectrum or update-only probe cannot replace conditional gain; TAX had no necessary bridge or distance intervention.
3. [A15_08_E04](../../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/summary.md) obtained eligible full-$R_U$ added accessibility at block 25: $G_{true}=0.767207$ nats/example with 95% interval $[0.751296,0.785146]$. It validates the local ridge metric, not a complete depth trend, long-range specificity, or representation-rank law.
4. The current implementation has 37 passing model-independent tests; the candidate compact encoding also passes exact-tokenizer construction for all 320 worlds and 10,240 records. No eligible replacement job may be submitted until A4 is confirmed, and no 36-layer scientific array or verdict exists.

## 8. Claim Boundary And Next Decision

If H2-LC passes, the strongest allowed claim is:

> For one frozen Qwen3-8B, one matched synthetic relation family, one pre-answer decision token, and one registered linear readout family, deep attention writes add more held-out target accessibility specific to a remote necessary bridge fact than shallow writes do.

The representation spectra may simultaneously establish the effective-rank and common-direction-energy trajectory of the matched long-range interaction update across 36 layers. Without a human-approved absolute rank threshold, they do not produce a global low-rank verdict.

Even joint positive gain and compact spectra cannot establish that deeper states retain all shallower information, Shannon information is created, factual knowledge is stored, task information or a Router occupies the same low-rank space, the MLP natively uses it, experts benefit, natural language shares the pattern, or a Router improves NLL or load balance.

**Exactly one next decision:** approve or reject candidate data amendment A4, which replaces the original long entity strings with six-character globally disjoint source/bridge identifiers while preserving the registered task structure.

**Completion criterion:** the four rank-local identity receipts, data/extraction/selection freeze, untouched CONFIRMATION evaluation, all five decision intervals, guards, rank tables, common-basis curves, and both registered figure families are complete and lineage-audited.

**Resume action:** after A4 approval, submit one Reserved job on the requested image with the fresh `_r2` root; after eligible evidence exists, write canonical Summary and Detailed records without strengthening the registered verdict.
