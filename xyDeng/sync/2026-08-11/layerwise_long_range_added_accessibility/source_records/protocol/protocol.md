---
experiment_id: A15_08_01_E01_layerwise_long_range_gain_and_representation_rank
anchor_id: 15_08_01_layerwise_long_range_compositional_innovation
status: APPROVED_FOR_REMOTE_4XH100_FULL_RUN
canonical_language: en
created: 2026-08-11
updated: 2026-08-11
approved_scope: implementation_smoke_and_remote_full_run
execution_authority: human_approved_2026-08-11
design_frozen_at: 2026-08-11T12:03:11Z
scientific_body_sha256: 7197183c95b4cbfefeaf431f13da0d06313100c7c10649ea3272ac053430a08a
---

# Protocol: A15_08_01_E01 Layerwise Long-Range Gain And Representation Rank

## 0. Approval Snapshot

- **Document status:** APPROVED_FOR_REMOTE_4XH100_FULL_RUN.
- **Approval status:** the researcher confirmed Blocks A--G, including the numerical data/readout/control contracts and the common-basis per-direction activation curve, on 2026-08-11.
- **Block audit status:** A--G CONFIRMED.
- **Cross-block consistency:** PASS.
- **Execution scope approved:** implementation, model-independent preflight, four-GPU identity/smoke, and one formal single-node 4xH100 ACP full run. Publication, external sync, Git commit, and Git push remain unauthorized.
- **Purpose:** determine whether deep attention writes add more held-out terminal-target accessibility specific to a remote necessary bridge fact than shallow writes, and separately describe the representation rank and common-basis energy of the matched interaction update across all 36 blocks.
- **Primary Anchor:** [A15_08_01 Layerwise Long-Range Added Accessibility And Representation Rank](../../../../problem_anchors/15_spectral_representation_and_functional_routing/15_08_target_conditioned_layer_innovation/subanchors/15_08_01_layerwise_long_range_compositional_innovation_anchor.md).
- **Anchor decision question:** after removing generic relocation and generic two-hop difficulty, is the necessity-by-distance added-accessibility interaction larger in deep blocks than in shallow blocks?
- **Anchor physical prior tested:** at one block and token, the counterfactual no-write and actual post-attention MLP inputs share the same normalization and hidden coordinate, so their difference is an exact computational update.
- **Anchor core model term tested:** the deep-minus-early contrast of the layerwise necessity-by-distance gain interaction, $T_{depth}$.
- **Anchor falsifier:** with all validity guards passing, the paired 95% upper bound of $T_{depth}$ is at or below zero.
- **Experiment role:** operationalization and proxy test. It extends the eligible local E04 accessibility metric across depth and characterizes, but does not functionally admit, the corresponding representation geometry.
- **Primary metric:** $T_{depth}$ in nats/example.
- **Claim boundary:** a Pass supports one model-, task-, token-, and linear-readout-family-specific deep-versus-shallow accessibility statement. Representation rank remains descriptive and has no H1 Pass threshold.
- **Minimal approved setup:** frozen Qwen3-8B; one answer-preceding decision token; all 36 blocks; matched one-hop/two-hop by near/far worlds; TRAIN/DEVELOPMENT/CONFIRMATION split by whole world; one `h100-4-spot` ACP worker node with four independent GPU shards.
- **Treatment:** the distance effect when the moved bridge fact is necessary, $G_{\ell,2F}-G_{\ell,2N}$.
- **Control:** the same distance effect when the bridge fact is unnecessary, $G_{\ell,1F}-G_{\ell,1N}$.
- **Changed variable:** whether the relocated bridge fact is necessary to answer the query.
- **Held fixed:** model and checkpoint, tokenizer, world and answer, fact multiset, terminal-fact position, near/far slots, distractor type and token length, template family within a matched quartet, total token length, decision token, hook identity, ridge family and budget, split roles, and analysis rule.
- **Conditions:** one-hop near (1N), one-hop far (1F), two-hop near (2N), and two-hop far (2F).
- **Pass:** every registered H2-LC lower-bound clause in Section 15 is strictly positive.
- **Fail:** a valid primary upper bound at or below zero, or a registered typed guard that precisely supports control degradation, headroom, or generic-capacity explanation.
- **Insufficient:** capability, bridge-dependency, identity, data, confirmation-freeze, artifact, or precision guard failure; a decisive interval crossing zero is also Insufficient.
- **Cannot claim:** Shannon-information creation, factual storage, whole-state low rank, task-sufficient rank, generalized-eigen direction identity, native MLP use, expert utility, natural-language generality, or Router gain.
- **Approval decision:** confirmed. Execute only the approved P0--P7 chain and stop on any registered guard or resource failure.

## 1. Terminology And Definitions

- A **world** is one independently sampled pair of bijections, its eight targets, and all four matched conditions. It is the independent resampling unit.
- A **source entity** $S$ starts a two-hop query. A **bridge entity** $B$ connects that source to the answer. A **terminal code** $Y$ is one of eight balanced answer codes.
- A **bridge fact** is $\phi(S)=B$. A two-hop query needs it; a one-hop query is directly given $B$ and does not need it.
- A **terminal fact** is $\psi(B)=Y$. It stays near the query in every condition.
- **Near** and **far** name the post-tokenization distance between the end of the moved bridge-fact span and the decision token.
- The **decision token** is the final query token before the model emits the answer.
- $h_{\ell,w,i,c}$ is the actual input residual to block $\ell$.
- $a_{\ell,w,i,c}$ is the raw attention output after output projection and before residual addition.
- $H_{\ell,w,i,c}$ is the complete actual block output.
- $N_\ell$ is the block-$\ell$ post-attention RMSNorm.
- $X_{\ell,c}=N_\ell(h_{\ell,c})$ is the no-attention-write counterfactual MLP input.
- $Z_{\ell,c}=N_\ell(h_{\ell,c}+a_{\ell,c})$ is the actual MLP input.
- $U_{\ell,c}=Z_{\ell,c}-X_{\ell,c}$ is the exact normalized attention effect. The experiment never defines $a_\ell-h_\ell$ as an increment because $a_\ell$ is already the additive write.
- $R_{U,\ell,c}=U_{\ell,c}-\widehat m^U_{\ell,c}(X_{\ell,c})$ is the update not reconstructed by the frozen registered linear residualizer. It is not statistically independent and is not a knowledge matrix.
- For eight logits $s$ and target code $y$, $CE(s,y)=-\log\operatorname{softmax}(s)_y$ uses the natural logarithm, so every reported gain has unit nats/example.
- $G_{\ell,c}$ is the held-out cross-entropy reduction obtained by adding the frozen $R_U$ correction to the frozen old-state readout, in nats/example.
- $I_\ell=(G_{\ell,2F}-G_{\ell,2N})-(G_{\ell,1F}-G_{\ell,1N})$ is the necessity-by-distance interaction.
- $T_{depth}$ is the median $I_\ell$ in blocks 25--36 minus the median in blocks 1--12.
- $D_{\ell,w,i}$ is the matched necessity-by-distance interaction of $R_U$ vectors for one world and target.
- **Representation effective rank** means entropy effective rank or 80%-variance rank of $\operatorname{Cov}(D_\ell)$; it is not task-readout rank.
- The **common activation basis** is the eigensystem of the equal-layer average of TRAIN $D_\ell$ covariances after per-layer trace normalization.
- A **mismatch bank** is a target-independent, balanced within-world re-pairing of confirmation $R_U$ rows. It preserves dimensionality and correction budget while breaking target alignment.
- **Headroom** is old-state cross-entropy before the $R_U$ correction. A harder cell can otherwise show a larger gain merely because it has more reducible loss.

## 2. Anchor Alignment

This Protocol tests only the child Anchor's H2-LC clause and its descriptive H1-REP characterization. It does not reopen the E04 target-conditioned rank ladder, construct generalized eigen-directions, select a parameter band, or train a Router.

The direct scientific chain is:

$$
\text{matched attention effect}
\rightarrow
\text{old-state-residualized update}
\rightarrow
\text{fresh held-out target gain}
\rightarrow
\text{necessity-by-distance interaction}
\rightarrow
\text{deep-versus-early contrast}.
$$

The separate geometric chain is:

$$
D_\ell
\rightarrow
\Sigma_{D,\ell}
\rightarrow
(r_{eff,\ell},r_{80,\ell}^{var})
\rightarrow
V_{common}
\rightarrow
F_\ell(r).
$$

The second chain cannot rescue a failure in the first chain. Low representation rank and positive target accessibility are logically separate observations.

## 3. Tested Hypothesis

**Primary H2-LC.** If the frozen model can use the bridge fact and all extraction/data guards pass, deep attention writes add more terminal-target accessibility specific to a remote necessary bridge fact than shallow attention writes.

The registered primary estimand is:

$$
T_{depth}
=
\operatorname{median}_{\ell=25}^{36} I_\ell
-
\operatorname{median}_{\ell=1}^{12} I_\ell.
$$

Blocks 13--24 are a pre-registered middle descriptive region. They cannot be reassigned to early or deep after confirmation is opened. All 36 layer values must be retained and plotted. Adjacent differences $I_{\ell+1}-I_\ell$ are descriptive local slopes, not independent samples and not a second verdict.

**Secondary H1-REP characterization.** Report the complete nonnegative covariance spectrum, entropy effective rank, 80%-variance rank, trace, and common-basis cumulative energy for $D_\ell$ in every block and split. No absolute threshold or Pass/Fail rule is registered for H1-REP.

## 4. Rival Explanations

| Rival | Separating evidence | What remains if it passes |
| --- | --- | --- |
| Generic distance retrieval plus generic two-hop difficulty | the 2-by-2 difference-in-differences $I_\ell$ | a necessity-specific distance interaction |
| Positive interaction caused only by degradation of a control cell | absolute deep-minus-early change in $G_{\ell,2F}$ | actual growth of remote two-hop added gain |
| Harder cells have more old-state loss headroom | DEVELOPMENT-frozen headroom adjustment applied to CONFIRMATION | gain not explained by a linear headroom relation |
| Any same-dimensional correction can appear useful | 64 target-independent balanced mismatch banks and q95 contrast | target alignment beyond matched capacity |
| Bridge fact is not actually used | bridge-swap counterfactual guard | a necessary bridge mechanism rather than prompt correlation |
| Model cannot solve the far two-hop task | restricted-choice capability guard | an interpretable representation comparison |
| A low effective rank is only one nuisance direction | target gain remains primary; trace and split replication are reported | bounded geometric concentration only |
| A positive gain is caused by task-rank compression | no target-conditioned rank search or generalized eigensystem is constructed | full-update accessibility without a task-rank claim |

## 5. Data Construction And Provenance

### 5.1 Approved population and split

The following numerical contract is **HUMAN_CONFIRMED**:

| Split | Worlds | Targets per world | Conditions per target | Records |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 128 | 8 | 4 | 4,096 |
| DEVELOPMENT | 64 | 8 | 4 | 2,048 |
| CONFIRMATION | 128 | 8 | 4 | 4,096 |

Proposed generation seeds are 2026081101, 2026081102, and 2026081103. Splits are disjoint by complete world: source and bridge entity strings, both bijections, template instance, record text, and identifiers cannot cross splits. The answer alphabet A--H is intentionally shared and must pass the existing Qwen3-8B single-answer-token audit.

TRAIN uses a symbolic-table wording family, DEVELOPMENT a lookup-table wording family, and CONFIRMATION a prose-table wording family. Wording family is split-fixed and cannot be selected after capability inspection. This preserves the E04 anti-memorization principle while making wording shift a known limitation.

### 5.2 Matched world construction

For world $w$, sample two independent type-matched bijections:

$$
\phi_w:\mathcal S_w\rightarrow\mathcal B_w,
\qquad
\psi_w:\mathcal B_w\rightarrow\mathcal Y.
$$

For each of eight sources:

$$
B_i=\phi_w(S_i),
\qquad
Y_i=\psi_w(B_i).
$$

Each prompt contains all eight bridge facts and all eight terminal facts, so every answer code appears exactly once. The target cannot be identified from frequency. The one-hop query begins at $B_i$; the two-hop query begins at $S_i$. The same bridge fact is present in all four cells but is necessary only in two-hop.

Near/far is constructed only by swapping the target bridge fact with a same-relation, same-token-length distractor fact occupying the other registered slot. No fact is inserted, deleted, or rewritten between near and far. The terminal fact remains in its fixed near slot.

The following distance contract is **HUMAN_CONFIRMED**:

- total input length: exactly 1,024 tokenizer tokens before answer generation;
- terminal-fact end to decision token: 8--24 tokens;
- near bridge-fact end to decision token: 32--64 tokens;
- far bridge-fact end to decision token: 512--768 tokens;
- a record outside its range is resampled before any model forward;
- the four records for one world/target have identical total token length.

The generator must create padding through semantically irrelevant, type-matched relation facts rather than repeated answer tokens. It must audit that the moved target and distractor spans have equal token length and that every fact multiset is identical within each near/far pair.

### 5.3 Required identifiers and leakage guards

Every row records split, world_id, record_id, quartet_id, target_index, condition, source entity, bridge entity, target code, template family, token length, decision-token index, bridge-span indices, terminal-span indices, near/far gap, and hashes of context and complete text.

Model-independent preflight must establish:

1. exact record counts, balance, and identifier uniqueness;
2. eight target codes exactly once per world and condition;
3. complete quartet membership for every world and target;
4. exact within-pair fact-multiset and token-length identity;
5. distance-window compliance after tokenization;
6. zero entity, mapping, exact text, or complete-context collision across splits;
7. no target shortcut outside the balanced terminal mapping table;
8. no CONFIRMATION field enters fitting, hyperparameter selection, basis construction, curve selection, or threshold selection.

Any violation stops before extraction. Repairing the generator before any model output exists reopens Block B; repairing data after confirmation representations exist requires a new experiment ID.

## 6. Learning Task And Permitted Supervision

The base model is frozen. There is no language-model training.

For each block $\ell$ and condition $c$, the permitted offline tasks are:

1. reconstruct $U_{\ell,c}$ linearly from $X_{\ell,c}$ to obtain $R_{U,\ell,c}$;
2. predict the centered eight-way terminal target from $X_{\ell,c}$ with a base ridge readout;
3. predict the base readout's target residual from $R_{U,\ell,c}$ with an additive ridge correction;
4. evaluate frozen base and correction readouts on untouched CONFIRMATION worlds.

The eight-way regression target for code $y$ is the centered vector
$t_y=e_y-\frac18\mathbf 1$. Base and correction outputs are treated as logits;
cross-entropy is always computed after softmax over the same eight registered
single-token codes.

TRAIN target labels may fit readouts. DEVELOPMENT labels may select ridge penalties and the frozen headroom slope. CONFIRMATION labels may only compute registered losses, gains, guards, intervals, and final figures after every selection artifact is frozen.

Target labels, target residuals, readout coefficients, and confirmation outcomes are forbidden from $D_\ell$, $\Sigma_{common}$, $V_{common}$, $r_{eff}$, $r_{80}^{var}$, and common-rank ordering. Condition identities and quartet matching are permitted because they define the registered necessity-by-distance representation object.

The prediction unit is one record. The independent unit for uncertainty is one complete world. Invalid rows are never silently dropped; a missing member makes the whole world incomplete and the affected split invalid until regenerated before unsealing.

## 7. Model, Architecture, And Frozen Variants

Use the frozen local Qwen3-8B checkpoint at /data/share/Qwen3-8B with its exact tokenizer, bfloat16 model weights, SDPA attention, and use_cache=False. The registered model has 36 blocks and hidden width 4,096.

At the same answer-preceding decision token in every block, extract in one forward:

$$
h_\ell,\quad
a_\ell,\quad
X_\ell=N_\ell(h_\ell),\quad
Z_\ell=N_\ell(h_\ell+a_\ell),\quad
U_\ell=Z_\ell-X_\ell,\quad
H_\ell.
$$

Store analysis tensors in float32. Raw $a_\ell$ and full output $H_\ell$ are auxiliary diagnostics. The primary functional object is $R_{U,\ell,c}$ derived from $U_\ell$. No cross-layer state subtraction, no $a_\ell-h_\ell$, and no subtraction of separately ranked PCA coordinates is permitted.

For every block and smoke/full extraction:

- replay maximum absolute logit error must be at most $10^{-5}$;
- direct attention output-projection relative error must be at most $10^{-5}$;
- stored $Z_\ell-X_\ell-U_\ell$ relative error must be at most $10^{-6}$;
- decision-token, record, condition, world, layer, dtype, and hidden-index identities must agree across all saved tensors;
- no per-layer rotation, whitening, or parameter eigensystem may precede common-basis construction.

Any failed identity makes the affected extraction Insufficient and blocks analysis.

## 8. Training Objective And Optimization

All fitted objects use deterministic linear ridge regression. For feature matrix $A$, target matrix $B$, and $n$ rows:

$$
\widehat W_\lambda
=
\arg\min_W
\frac1n\lVert B-AW\rVert_F^2
+\lambda\lVert W\rVert_F^2.
$$

The proposed penalty grid is the E04 verified grid:

$$
\lambda\in
\{10^{-4},10^{-3},10^{-2},10^{-1},1,10,100,1000\}.
$$

Ties on DEVELOPMENT cross-entropy or reconstruction MSE choose the larger $\lambda$.

### 8.1 Frozen standardization

For each layer, pool all four TRAIN cells. For object $A\in\{X,U\}$ with $n$
rows and width $d$, freeze:

$$
\mu_{A,\ell}=\frac1n\sum_{j=1}^{n}A_{\ell,j},
\qquad
s_{A,\ell}
=
\sqrt{
\frac1{nd}
\sum_{j=1}^{n}
\lVert A_{\ell,j}-\mu_{A,\ell}\rVert_2^2
}.
$$

Apply $(A-\mu_{A,\ell})/s_{A,\ell}$ unchanged to every cell and split. A
nonfinite scale or a scale at most $10^{-12}$ makes that layer invalid.
Condition-specific, feature-wise-variance, or split-specific rescaling is
forbidden. Per-layer trace normalization occurs only later in common-basis
construction.

### 8.2 World-grouped cross-fitting

The following is **HUMAN_CONFIRMED**:

1. deterministically permute TRAIN world IDs with seed 2026081111 and assign them round-robin to five folds, keeping all targets and conditions of one world together and fold sizes within one world;
2. select each layer/cell residualizer penalty by DEVELOPMENT reconstruction MSE;
3. produce out-of-fold TRAIN $R_U$ with the selected residualizer and refit that residualizer on all TRAIN worlds for DEVELOPMENT and CONFIRMATION;
4. select the base-readout penalty by DEVELOPMENT cross-entropy;
5. produce out-of-fold TRAIN base logits with that penalty and refit the base readout on all TRAIN worlds for DEVELOPMENT and CONFIRMATION;
6. fit the correction on out-of-fold $R_U$ against centered one-hot target minus out-of-fold base logits;
7. select the correction penalty by DEVELOPMENT cross-entropy of frozen base plus candidate correction;
8. freeze all standardizers, folds, penalties, residualizers, base readouts, and corrections before opening CONFIRMATION.

No optimizer iteration, early stopping, checkpoint choice, layer selection, rank selection, or confirmation-dependent refit is allowed.

## 9. Conditions, Seeds, And Checkpoints

There is one frozen model checkpoint and four data conditions. All 36 blocks are measured; none is selected or excluded after extraction.

Proposed deterministic randomness registry:

| Purpose | Value | State |
| --- | --- | --- |
| TRAIN / DEV / CONF data | 2026081101 / 2026081102 / 2026081103 | CONFIRMED |
| five-fold TRAIN grouping | 2026081111 | CONFIRMED |
| 64 mismatch-bank recipes | 2026081200--2026081263 | CONFIRMED |
| 2,000 paired world bootstrap draws | 2026081299 | CONFIRMED |
| synthetic known-case tests | 2026081191 | CONFIRMED |

Every seed and generated mapping must be serialized before CONFIRMATION labels or losses are opened. Deterministic reruns must reproduce hashes of data, fold assignments, mismatch maps, bootstrap indices, selected penalties, and common-basis eigenvalues within the registered numerical tolerance.

## 10. Primary Metric And Frozen Analysis Plan

For one confirmation record:

$$
g_{\ell,c}
=
CE\!\left(b_{\ell,c}(X_{\ell,c})\right)
-
CE\!\left(
b_{\ell,c}(X_{\ell,c})
+q_{\ell,c}(R_{U,\ell,c})
\right).
$$

Average the eight targets inside each world, then average worlds equally:

$$
G_{\ell,c}
=
\mathbb E_{w\in CONF}
\left[
\frac18\sum_{i=1}^8 g_{\ell,w,i,c}
\right]
\quad\text{nats/example}.
$$

Compute:

$$
I_\ell
=(G_{\ell,2F}-G_{\ell,2N})
-(G_{\ell,1F}-G_{\ell,1N}),
$$

$$
T_{depth}
=
\operatorname{median}_{\ell=25}^{36}I_\ell
-
\operatorname{median}_{\ell=1}^{12}I_\ell.
$$

Resample complete CONFIRMATION worlds with replacement 2,000 times. The same sampled world indices are used for all layers, conditions, targets, guards, and mismatch banks in one draw. Report the point estimate and two-sided percentile 95% interval. The 2.5th and 97.5th percentiles use `numpy.quantile(..., method="linear")`; the exact NumPy version is recorded. No normal approximation, layer bootstrap, or seed averaging may replace this rule.

No world or layer is excluded as an outlier. A missing or nonfinite row invalidates its entire world; if the world cannot be deterministically reconstructed from the frozen data before analysis unsealing, the result is Insufficient. No multiplicity correction is applied to descriptive layerwise pointwise bands; they cannot create a layer-specific Pass. The conjunctive primary gate is conservative and has no rescue rule.

### 10.1 Absolute remote-two-hop guard

$$
A_{2F}
=
\operatorname{median}_{\ell=25}^{36}G_{\ell,2F}
-
\operatorname{median}_{\ell=1}^{12}G_{\ell,2F}.
$$

This must have a strictly positive paired lower bound for H2-LC Pass.

### 10.2 Deep interaction guard

$$
M_D=\operatorname{median}_{\ell=25}^{36}I_\ell.
$$

This must have a strictly positive paired lower bound for H2-LC Pass.

### 10.3 DEVELOPMENT-frozen headroom adjustment

For each layer, fit on DEVELOPMENT records:

$$
g_{\ell,c}
=
\alpha_{\ell,c}
+\beta_\ell L^{base}_{\ell,c}
+\epsilon,
$$

where condition-specific intercepts protect the slope from absorbing the design-cell means. Let $\bar L_{\ell}^{DEV}$ be the pooled DEVELOPMENT base loss. Freeze:

$$
g_{\ell,c}^{head}
=
g_{\ell,c}
-\widehat\beta_\ell
\left(
L^{base}_{\ell,c}-\bar L_{\ell}^{DEV}
\right).
$$

Apply that formula unchanged to CONFIRMATION, then recompute $G^{head}$, $I^{head}$, and $T_{depth}^{head}$. Its paired lower bound must be positive for Pass. A rank-deficient or nonfinite DEVELOPMENT fit makes this guard Insufficient.

### 10.4 Target-independent capacity contrast

Each of 64 mismatch banks applies one balanced within-world target permutation to CONFIRMATION $R_U$, using the same map across the four conditions of one world. It preserves layer, condition, vector dimension, q-readout, and sample count while breaking target alignment. Every bank must have an exactly uniform 8-by-8 target-source contingency and empirical mutual information at most $10^{-12}$.

For each bootstrap draw and bank $j$, compute its complete $T_{depth,j}^{mis}$. Define:

$$
T_{cap}
=
T_{depth}
-
Q_{0.95}^{higher}
\left(T_{depth,1:64}^{mis}\right)
$$

inside each draw. $Q_{0.95}^{higher}$ is the ascending order statistic at
index $\lceil0.95m\rceil$ for $m=64$ values, hence the 61st value; this is
equivalent to `numpy.quantile(..., method="higher")`. The paired lower bound of
$T_{cap}$ must be positive for Pass.

Analysis is unsealed only after the data, extraction, readout, mismatch, bootstrap, common-basis, and expected-artifact manifests are frozen and hashed.

## 11. Secondary Metrics And Guards

### 11.1 Capability and bridge necessity

Before representation interpretation, evaluate the frozen model's restricted-choice probability over the eight terminal codes:

- confirmed capability threshold: point accuracy at least 0.80 in 1N, 1F, and 2N;
- confirmed far two-hop threshold: point accuracy at least 0.60 in 2F and its world-bootstrap 95% lower bound above chance 0.125;
- no condition may have nonfinite target logits or fewer than all registered worlds.

For the bridge-swap guard, choose one pre-hashed same-type partner source in each world and swap the two sources' bridge assignments. This preserves a bijection, all entity and answer-code marginals, table positions, token lengths, and the terminal mapping while changing the target source's two-hop answer from $Y_{orig}$ to $Y_{cf}$. For hop condition $k\in\{1,2\}$ define the paired log-odds response:

$$
\Delta^{bridge}_{k}
=
\left[
\log\frac{p(Y_{cf}\mid P_{swap,k})}{p(Y_{orig}\mid P_{swap,k})}
-
\log\frac{p(Y_{cf}\mid P_{orig,k})}{p(Y_{orig}\mid P_{orig,k})}
\right].
$$

The world-bootstrap 95% lower bounds of both $\Delta^{bridge}_{2}$ and $\Delta^{bridge}_{2}-\Delta^{bridge}_{1}$ must be positive. This guard establishes sensitivity to the bridge mapping beyond a generic prompt edit; it does not establish overall task capability.

Failure of capability or bridge dependence yields Insufficient, not H2-LC Fail.

### 11.2 Cumulative-state and raw-write diagnostics

Report base-readout cross-entropy from full block output $H_\ell$ as accumulated state accessibility. Report the raw-$a_\ell$ covariance trace and effective rank as a sensitivity to the post-attention normalization site. Neither diagnostic enters H2-LC, changes the common basis, or creates another verdict.

### 11.3 Representation rank and common basis

For each split and layer:

$$
D_{\ell,w,i}
=(R_{U,\ell,2F}-R_{U,\ell,2N})
-(R_{U,\ell,1F}-R_{U,\ell,1N}).
$$

Center the eight $D_{\ell,w,i}$ rows within each world and compute:

$$
\Sigma_{D,\ell}^{split}
=
\frac{1}{n-1}
\widetilde D_{\ell,split}^{\top}
\widetilde D_{\ell,split}.
$$

Let $\mu_{\ell,j}$ be the nonnegative eigenvalues and $p_{\ell,j}=\mu_{\ell,j}/\sum_k\mu_{\ell,k}$. Report:

$$
r_{eff,\ell}
=
\exp\!\left(-\sum_jp_{\ell,j}\log p_{\ell,j}\right),
$$

$$
r_{80,\ell}^{var}
=
\min\left\{
r:
\frac{\sum_{j=1}^{r}\mu_{\ell,j}}
{\sum_j\mu_{\ell,j}}
\ge0.8
\right\}.
$$

Also report each value divided by the split-identifiable maximum
$r_{max}^{split}=\min(4096,7W_{split})$. For sample-size-matched stability, use the 64 lowest pre-hashed world IDs from each split and report the same rank curves without selecting a favorable subset.

The human-approved common basis is:

$$
\Sigma_{common}
=
\frac1{36}\sum_{\ell=1}^{36}
\frac{\Sigma_{D,\ell}^{TRAIN}}
{\operatorname{tr}(\Sigma_{D,\ell}^{TRAIN})},
$$

$$
\Sigma_{common}
=V_{common}\Lambda_{common}V_{common}^{\top}.
$$

“TRAIN-only” here means that only out-of-fold TRAIN rows enter
$\Sigma_{D,\ell}^{TRAIN}$ and its eigendecomposition. The residualizer penalty
may be frozen by the registered DEVELOPMENT reconstruction rule, but no
DEVELOPMENT row and no target label enters a covariance or basis fit.

If a layer trace is nonfinite, at most $10^{-12}$ after frozen scaling, or below $10^{-6}$ times the median positive TRAIN trace, its normalized spectrum is invalid and must be shown as missing rather than normalized noise. This affects only H1-REP characterization unless the same extraction/data defect also affects H2-LC.

Freeze $V_{common}$ and its descending eigenvalue order from TRAIN only. For every layer and split:

$$
e_{\ell,k}^{split}
=
\frac{
v_k^\top\Sigma_{D,\ell}^{split}v_k
}{
\operatorname{tr}(\Sigma_{D,\ell}^{split})
},
\qquad
F_\ell^{split}(r)=\sum_{k=1}^{r}e_{\ell,k}^{split}.
$$

$F_\ell(r)$ is cumulative representation energy in a common activation coordinate. It is not accessibility by direction and cannot be compared with $G_{\ell,c}$ as if the two had the same unit. The coordinate preserves the model's shared hidden-channel index but is not invariant to the different learned RMSNorm scales or MLP functions at different blocks; raw-$a_\ell$ sensitivity remains required.

## 12. Known Good, Known Bad, And Known Confusing Cases

| Type | Constructed case | Required behavior |
| --- | --- | --- |
| Known good | inject a target-aligned update with a fixed deep-only necessity-by-distance effect | recover positive $T_{depth}$ and the injected representation rank within tolerance |
| Known bad | make near and far prompts identical | every $I_\ell$ and $T_{depth}$ is zero within numerical tolerance |
| Known bad | permute TRAIN terminal labels within world | no stable true correction gain or bridge-specific interaction |
| Known bad | use $U_\ell=0$ | residualizer, gain, trace, and rank guards refuse a positive result |
| Known confusing | decrease $G_{\ell,1F}$ while holding $G_{\ell,2F}$ fixed | positive $I_\ell$ but failed $A_{2F}$; typed control-degradation outcome |
| Known confusing | make base loss larger in 2F and gain proportional to base loss | raw interaction positive but headroom-adjusted contrast nonpositive |
| Known confusing | inject one high-variance target-independent direction | low effective rank without positive accessibility |
| Known confusing | inject target gain across many orthogonal directions | positive H2-LC with broad representation spectrum; H2 can Pass without low-rank evidence |

All synthetic cases are implementation tests only. They cannot be mixed with scientific data or reported as evidence for Qwen3-8B.

## 13. Scientific Procedure And Stage Contracts

| Stage | Goal | Input and assumptions | How and why | Expected output | Pass / fail / insufficient | Named failure | Artifact | Handoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 protocol freeze | make every selection auditable | confirmed Blocks A--G, no execution data | hash the scientific body, choices, seeds, paths, and artifact inventory | freeze ledger | all blocks confirmed and consistency PASS; otherwise stop | design-not-frozen | protocol freeze receipt | permits a later scope request only |
| P1 data preflight | prove the four cells isolate necessity by distance | generator, tokenizer, proposed world contract | generate schema-only data and run all balance, distance, matching, and leakage audits | JSON/manifest/audit | all checks pass; any defect stops | data-construction failure | data preflight bundle | permits implementation smoke request |
| P2 identity implementation | prove exact layer objects exist | frozen model and a few known records | extract all 36 sites and verify token/hook/replay/projection identities | identity report and tensors | every block passes tolerances | representation-identity failure | identity smoke bundle | permits scientific smoke request |
| P3 end-to-end smoke | prove the complete algorithm and artifacts | two worlds per split with all targets/cells | run residualizers, readouts, mismatch banks, bootstrap, basis, rank, and plots on reduced data | schema-complete smoke record | all expected arrays and known cases pass; no scientific verdict | pipeline or artifact failure | smoke manifest | permits separate full-run review |
| P4 full extraction | obtain frozen representations | approved full split and identity-tested code | one model pass per record saves all six tensors for 36 blocks | split tensors/logits/manifests | counts, hashes, identities, capability inputs complete | extraction/capability-input failure | raw extraction roots | hands frozen tensors to P5 |
| P5 TRAIN/DEV freeze | freeze every learned and geometric object | TRAIN/DEV tensors only | fit standardizers, cross-fitted residualizers/readouts, headroom slopes, and common basis | selection ledger and hashes | no confirmation access and all objects complete | selection leakage or fit failure | preconfirmation freeze | permits confirmation unseal |
| P6 confirmation analysis | compute registered evidence once | frozen P5 objects and untouched CONFIRMATION | evaluate gains, guards, mismatch banks, bootstrap, ranks, and two figures | arrays, tables, plot contracts, typed verdict | Section 15 mapping only | scientific Fail or Insufficient | result bundle | hands evidence to P7 |
| P7 record audit | create canonical evidence record | complete eligible P0--P6 bundle | audit lineage and conformance before writing Summary/Detailed | eligibility report and curated records | all hashes and decision arrays trace; otherwise ineligible | evidence-lineage failure | manifest, summary, detailed | returns to researcher judgment |

No later stage may continue after an upstream hard stop. Scientific underperformance is not an engineering stop once P6 begins; it must be recorded under the frozen rule.

## 14. Algorithm, Outputs, Figures, And Provenance Contract

### 14.1 Approved worker and artifact roots

The following paths are approved for the registered execution scope:

- worker source: `Projects/from-attention-to-search/XingyuD/MoE_Routing_Experiments/active/a15_08_01_e01_layerwise_long_range_gain_and_representation_rank/`
- raw run root: `/data/250010109/MoE_Router/experiments/20260811_a15_08_01_layerwise_long_range_gain_rank/`
- canonical record root: `Projects/from-attention-to-search/main/experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/`

Raw model outputs, full tensors, caches, logs, and checkpoints remain outside Research_System. Only audited tables, figures, manifests, Summary, and Detailed records may enter the canonical record root after execution.

### 14.2 Required central figures

Exactly two central figure families are registered. Debug plots cannot enter Summary.

**Figure 1 — Layerwise long-range added-accessibility evidence.**

| Contract item | Frozen requirement |
| --- | --- |
| Question | Does the necessity-by-distance gain interaction strengthen with depth, and do the registered guards support the same reading? |
| Panel A | x: blocks 1--36; y: $G_{\ell,c}$ in nats/example; 1N blue solid (`#0072B2`), 1F light-blue dashed (`#56B4E9`), 2N vermillion solid (`#D55E00`), and 2F purple dashed (`#CC79A7`); equal-world point estimates with pointwise paired 95% bands |
| Panel B | x: blocks 1--36; y: $I_\ell$ in nats/example; black curve, zero line, fixed early/middle/deep background bands, integer block ticks, and no smoothing |
| Panel C | forest display of $T_{depth}$, $M_D$, $A_{2F}$, $T_{depth}^{head}$, and $T_{cap}$ with paired 95% intervals and zero line |
| Data source | complete CONFIRMATION world-level gain arrays and frozen bootstrap indices |
| Allowed conclusion | the registered deep-versus-early accessibility verdict and which guard supports or blocks it |
| Limitation | cannot identify a vector subspace, prove representation low rank, or establish native use |

**Figure 2 — Representation-rank and common-basis spectrum.**

| Contract item | Frozen requirement |
| --- | --- |
| Question | How compact and how commonly oriented is the matched long-range interaction update across depth? |
| Panel A | x: blocks 1--36; y: $r_{eff}/r_{max}$ and $r_{80}^{var}/r_{max}$; metric is encoded by solid versus dashed and split by marker (`TRAIN` circle, `DEV` triangle, `CONF` square); invalid-trace layers shown explicitly; an aligned thin log10-trace strip protects interpretation |
| Panel B | x: common rank 1--4,096 on a fixed log2 axis with labeled ticks 1, 8, 32, 128, 512, 2,048, 4,096; y: CONFIRMATION $F_\ell(r)$ from 0 to 1; all 36 unsmoothed curves; block depth uses fixed `cividis_r((\ell-1)/35)` so shallow layers are light and deep layers dark |
| Panel C | x: common rank 1--4,096 on the same log2 axis; y: log-scaled CONFIRMATION per-direction energy $\bar e_{\ell,k}$; all 36 curves use the same depth colors as Panel B; only this display density receives the frozen nine-rank smoothing below |
| Data source | TRAIN-frozen $V_{common}$ and split-specific complete $D_\ell$ covariance records |
| Allowed conclusion | numerical representation-rank trajectory and redistribution of update energy in one common activation basis |
| Limitation | $F_\ell(r)$ is not target accessibility, task-sufficient rank, a generalized-eigen result, or Router function; cross-layer orientation remains sensitive to layer-specific RMSNorm channel scaling |

For Panel C only, reflect-pad the raw nonnegative vector $e_{\ell,1:4096}$ by four entries on each side, convolve with the fixed kernel $\frac19\mathbf 1_9$, and renormalize the resulting 4,096 values to sum to one:

$$
\widetilde e_{\ell}
=
\operatorname{conv}_{valid}
\left(
\operatorname{reflectpad}_{4}(e_{\ell}),
\frac19\mathbf 1_9
\right),
\qquad
\bar e_{\ell,k}
=
\frac{\widetilde e_{\ell,k}}
{\sum_j\widetilde e_{\ell,j}}.
$$

The plotted ordinate is $\max(\bar e_{\ell,k},10^{-12})$ on a logarithmic y-axis. Smoothing is display-only: all ranks, $F_\ell(r)$, tables, guards, and verdicts use raw $e_{\ell,k}$.

Required tables are: layer-by-condition gains and intervals; registered contrast table; capability/bridge guard table; per-layer trace/rank table; common-basis eigenvalue and cumulative-energy table; split-stability table; and protocol-conformance ledger.

### 14.3 Required provenance

The full evidence manifest must enumerate and hash:

- Protocol revision and scientific-body hash;
- code, tests, configuration, environment, model, tokenizer, and dirty-state receipts;
- data JSON, manifests, IDs, tokenizer/distance/matching/leakage audits;
- extraction commands, environment, tensor/logit manifests, identities, and failures;
- standardizers, folds, ridge grids, selected penalties, residualizers, readouts, headroom fits, and access ledger;
- mismatch recipes/maps/audits, bootstrap indices, all per-record and per-world arrays;
- $D_\ell$ tensors or reproducible sufficient statistics, layer covariances, spectra, $V_{common}$, eigenvalues, rank and energy tables;
- plot contracts, raw and display-smoothed common-basis energy tables, plotting source hash, PNG and PDF figures;
- result eligibility audit, Summary, and Detailed record.

An artifact path without producer, transformation, hash, aggregation, and registered decision role is not evidence lineage.

### 14.4 Approved remote resource and sharding contract

Use the user-selected `h100-4-spot` profile: ACP, workspace `share-space`, AEC2 cluster `share-cluster`, worker spec `n6ls.iu.i40.4`, one worker node, four H100 GPUs, four launcher processes, `spot` quota, and normal priority. Deterministically assign complete worlds to GPU ranks by the frozen sorted-record round-robin index; no world may cross shards.

Before full extraction, each rank independently runs the P2 identity suite on its first two complete worlds, writes a rank-local receipt, and enters a four-rank barrier. Any missing receipt or failed token/hook/replay/projection/tensor identity terminates the job before full extraction. Each rank then writes only its own shard root; the merge verifies disjoint identifiers, complete registered counts, and four successful receipts before P5.

The ceiling remains at most 16 aggregate GPU-hours, implemented as at most four hours of four-GPU occupied wall time after model execution begins; raw storage remains at most 150 GiB and ridge/spectral/bootstrap analysis at most 32 aggregate CPU-hours. Exceeding a ceiling stops and returns an amendment request; it cannot silently reduce worlds, layers, context length, saved tensors, controls, or bootstrap draws.

## 15. Success, Failure, And Insufficient Evidence

After all capability, bridge, identity, data, freeze, and record guards pass:

**H2_LC_PASS** requires strictly positive paired 95% lower bounds for all five:

1. $T_{depth}$;
2. $M_D$;
3. $A_{2F}$;
4. $T_{depth}^{head}$;
5. $T_{cap}$.

**H2_LC_FAIL_PRIMARY** requires the paired 95% upper bound of $T_{depth}$ to be at or below zero.

If the primary point and interval support a positive interaction but exactly one named guard has a nonpositive upper bound, return its typed scientific failure:

- **H2_LC_FAIL_CONTROL_DEGRADATION** for $A_{2F}$;
- **H2_LC_FAIL_HEADROOM** for $T_{depth}^{head}$;
- **H2_LC_FAIL_GENERIC_CAPACITY** for $T_{cap}$;
- **H2_LC_FAIL_NO_DEEP_INTERACTION** for $M_D$.

If two or more named guards have nonpositive upper bounds, return **H2_LC_FAIL_MULTIPLE_GUARDS** and list every failed guard; do not select a preferred rival after seeing the result. `H2_LC_FAIL_PRIMARY` takes precedence whenever its own rule is met.

If no Pass or typed Fail rule is met because any decisive interval crosses zero, return **H2_LC_INSUFFICIENT_PRECISION**.

If a capability, bridge-dependency, identity, leakage, freeze, completeness, or numerical guard fails, return the corresponding **H2_LC_INSUFFICIENT_<GUARD>**. A guard failure is never repaired by a favorable curve.

H1-REP receives no Pass/Fail token. Its record is a complete numerical characterization with explicit invalid layers and split stability. No visual elbow, post hoc threshold, or comparison with the eight-class task rank may convert it into a low-rank verdict.

## 16. What This Cannot Claim

Even H2_LC_PASS cannot establish:

- that a deterministic layer creates new Shannon information;
- that the deeper state retains every shallower distinction;
- that the update stores factual knowledge;
- that nonlinear novelty has been exhausted;
- that the full representation or task information is low rank;
- that any common-basis prefix is sufficient for target retrieval;
- that a generalized-eigen, MLP-parameter, or Router coordinate has been identified;
- that the common activation basis is invariant to layer-specific RMSNorm scaling or aligns the functions of different layer MLPs;
- that the MLP or model natively uses the linearly readable signal;
- that experts benefit, routes specialize, NLL improves, or load balances;
- that the result transfers to natural language, another model, another token, or another context length.

A broad spectrum cannot falsify H2-LC. A compact spectrum cannot rescue H2-LC or prove Router-readability.

## 17. Block Confirmation, Freeze, And Amendment Ledger

| Block | State | Current basis | Decision still needed |
| --- | --- | --- | --- |
| A Question and claim boundary | CONFIRMED | human-approved A15_08_01 Anchor and common basis | none |
| B Data construction and provenance | CONFIRMED | 128/64/128 worlds and frozen 1,024-token distance/matching contract | none |
| C Learning task and supervision | CONFIRMED | cell-specific ridge readouts, complete-world cross-fitting, and untouched-confirmation boundary | none |
| D Model and architecture | CONFIRMED | Qwen3-8B, all-36-block $U_\ell$ object, and E04-verified identity tolerances | none |
| E Objective and optimization | CONFIRMED | frozen ridge grid, world-grouped folds, pooled scaling, and headroom fit | none |
| F Comparison and decision evidence | CONFIRMED | five-clause Pass, typed Fail mapping, 64 banks, 2,000 draws, two figure families, and nine-rank display smoothing | none |
| G Execution and reproducibility | CONFIRMED | worker/raw roots, P0--P7 stops, `h100-4-spot`, four-rank checks, 16 GPU-hour / 150 GiB ceiling | none |

Cross-block consistency audit:

| Check | Draft result |
| --- | --- |
| data exposes necessity by distance | PASS |
| supervision obeys split boundary | PASS under Sections 5--8 |
| model hooks realize the named update | PASS by design; P2 remains a mandatory execution guard |
| objective measures held-out added accessibility | PASS under frozen base-plus-correction CE |
| treatment and controls change only registered variables | PASS by design; tokenizer matching remains a mandatory execution guard |
| primary metric separates the strongest rival | PASS through interaction plus absolute/headroom/capacity guards |
| confirmation cannot select objects | PASS under P5 freeze and access ledger |
| resources and artifacts cover all conditions | PASS under the approved four-rank sharding and ceiling |
| verdict maps to one bounded claim | PASS |

Overall cross-block consistency is **PASS**. The researcher approved implementation, smoke, and one formal single-node 4xH100 ACP full run on 2026-08-11. Publication, sync, commit, and push remain unauthorized.

The scientific body and decision-bearing configuration freeze at `2026-08-11T12:03:11Z`. After execution starts, any material change must be append-only and record before/after values, reason, observed-data status, affected blocks, human approval, and evidence-eligibility impact. Observed results and interpretation belong only in summary.md and detailed.md.

### Execution ledger

- `2026-08-11T12:33:46Z`: the approved single-node four-GPU full run was submitted as ACP job `pt-3qikirn4`, display name `a15-08-01-layerwise-new-info-20260811T123800Z`, on `share-cluster / n6ls.iu.i40.4 / spot`. The bound raw root is `/data/250010109/MoE_Router/experiments/20260811_a15_08_01_layerwise_long_range_gain_rank/`. This operational entry changes no scientific-body field and is outside the frozen Sections 1--16 hash.
- `2026-08-11T12:46:12Z`: job `pt-3qikirn4` was stopped at `STARTING` and reached `SUSPENDED` at `2026-08-11T12:46:15Z`. It had zero allocated replicas, and the bound raw root did not exist; therefore it produced no model execution or experiment evidence.

### Execution Amendment A1: Reserved quota replaces Spot quota

- **Approved at:** `2026-08-11T12:46:40Z`.
- **Before:** resource label `h100-4-spot` and ACP `quota_type=spot`, represented only by the non-executed job `pt-3qikirn4`.
- **After:** resource label `h100-4-reserved` and ACP `quota_type=reserved` on the unchanged workspace `share-space`, cluster `share-cluster`, worker spec `n6ls.iu.i40.4`, one worker, four H100 GPUs, four launcher processes, image, mount, priority, sharding, and ceilings.
- **Reason:** the researcher explicitly required a Reserved node and prohibited Spot execution.
- **Observed-data status:** no worker was allocated and no raw root, model output, metric, or figure existed before this amendment.
- **Affected block:** Block G execution resource only. Sections 1--16, the estimand, data, model, extraction objects, analyses, figures, and decision rules are unchanged.
- **Human approval:** explicit user instruction, `2026-08-11`.
- **Evidence eligibility:** job `pt-3qikirn4` is ineligible and non-evidentiary. Exactly one replacement Reserved job may execute under this Protocol; only its complete guarded outputs can enter `summary.md` or `detailed.md`.
- `2026-08-11T12:47:12Z`: the sole eligible replacement was submitted as ACP job `pt-luxlf19m`, display name `a15-08-01-layerwise-new-info-20260811T124800Z`, on `share-cluster / n6ls.iu.i40.4 / reserved`. The ACP record independently reports `quota_type=RESERVED`; all other execution fields and the bound raw root are unchanged.
- `2026-08-11T12:48:36Z`: Reserved job `pt-luxlf19m` ended `FAILED` during environment preflight. It verified the frozen Protocol hash and all four H100 devices, then stopped before completing the runtime receipt, data construction, model loading, or any registered analysis. Its raw root contains only preflight artifacts and is permanently ineligible for scientific evidence.

### Execution Repair A2: Activate the frozen Qwen3 Transformers dependency

- **Approved at:** `2026-08-11T12:51:13Z` as an in-scope repair required to execute the already approved design.
- **Observed failure:** the runtime receipt file was opened but remained empty; no data or model artifact was produced. Reproduction in the shared base environment showed that bare `python` could not import `transformers`.
- **Implementation repair:** prepend the existing frozen dependency root `/data/250010109/dependency_packages/hf_transformers_qwen3` to `PYTHONPATH`, require its `transformers/` directory before creating a run root, and preserve runtime-import stderr as a separate artifact.
- **Validation:** the repaired environment imports Transformers `4.53.3` with the existing NumPy and Torch stack, and all 34 model-independent tests pass.
- **Affected block:** Block G runtime activation and failure observability only. Sections 1--16, data, estimands, model weights, tensor definitions, analyses, figures, and decision rules are unchanged.
- **Evidence eligibility:** `pt-luxlf19m` and its raw root are ineligible. One Reserved replacement may use a fresh append-only raw root ending `_r1`; only that replacement can produce eligible evidence.
- `2026-08-11T12:51:42Z`: the repaired Reserved replacement was submitted as ACP job `pt-z5m90trk`, display name `a15-08-01-layerwise-new-info-20260811T125200Z`, with verified `quota_type=RESERVED`. Its fresh bound raw root is `/data/250010109/MoE_Router/experiments/20260811_a15_08_01_layerwise_long_range_gain_rank_r1/`.
- `2026-08-11T12:52:47Z`: Reserved job `pt-z5m90trk` ended `FAILED` after runtime activation and all 34 then-current tests passed, but before data construction. The source-provenance command treated the project-local non-Git directory as fatal. No model was loaded and no scientific output was produced; the job and `_r1` root are ineligible.

### Execution Amendment A3: Requested image and current-container runtime

- **Approved at:** `2026-08-11T12:58:34Z` by explicit researcher instruction.
- **Before:** `registry.cn-sh-01.sensecore.cn/ccr-zhicheng-03/shphd-private-lite:container1-20260523142552` plus an inline dependency-path activation.
- **After:** `registry.cn-sh-01g.sensecore.cn/ccr-zhicheng-03/shphd-lite-0710:container1-20260710154849`, Reserved quota only, and a no-install activation that uses the image's Python 3.10 / Torch 2.3 / CUDA 12.4 / NumPy 1.24.4 environment plus the existing frozen Transformers 4.53.3 directory. The activation also freezes SciPy 1.12.0, scikit-learn 1.2.0, and Matplotlib 3.8.4.
- **Provenance repair:** a non-Git project-local source surface now writes an explicit `NOT_A_GIT_WORKTREE` receipt instead of aborting; source hashes remain mandatory.
- **Affected block:** Block G image, runtime activation, and provenance observability only. No scientific field in Sections 1--16 changes.
- **Validation:** requested-image dry run, exact runtime activation, frozen Protocol hash, and 37 model-independent tests pass.

### Data Amendment A4: Compact disjoint entity surface forms

**State: HUMAN_CONFIRMED_AND_EXECUTABLE at `2026-08-11T13:02:50Z`.**

- **Observed failure:** exact Qwen3 tokenizer preflight with the original long entity strings stopped before model loading because the complete mandatory fact table exceeded the prefix available before the registered far slot.
- **Before:** source and bridge strings embed the full split name, seed, world index, and target index.
- **Candidate after:** use six-character, globally unique, type-marked strings `s<split><world_hex3><target_hex1>` and `b<split><world_hex3><target_hex1>`, where split is `t`, `d`, or `c`. Full world IDs, seeds, split labels, bijections, templates, record IDs, and all metadata remain unchanged.
- **Scientific invariants preserved:** 320 disjoint worlds; 10,240 records; eight sources, eight bridges, and eight labels per world; all bridge and terminal facts; exact 1,024-token length; 48/640 bridge gaps; matched equal-token-length swaps; sealed CONFIRMATION; zero cross-split entity and exact-token overlap.
- **Approval basis:** entity surface forms change every prompt, so Section 5.3 reopened Block B. The researcher explicitly approved A4 after reviewing the exact candidate encoding and full-tokenizer audit; no model output or CONFIRMATION representation existed before approval.
- **Candidate validation:** all 320 worlds materialize with the exact tokenizer; the largest mandatory prefix is 161 tokens, the earliest far-slot start is 370, leaving a 209-token margin. The complete dataset preflight passes with 2,560 records per condition and zero cross-split entity overlap; all 37 tests pass.
- **Execution consequence:** exactly one Reserved job using the A3 image and fresh `_r2` root is eligible. No other data, model, analysis, or decision field changes.
- `2026-08-11T13:03:17Z`: the sole eligible post-A4 run was submitted as ACP job `pt-yiwkpm3p`, display name `a15-08-01-layerwise-new-info-20260811T130300Z`. The ACP record independently verifies the requested A3 image, `quota_type=RESERVED`, one `n6ls.iu.i40.4` worker, four H100 devices, and the fresh `_r2` root.
- `2026-08-11T13:08:22Z`: job `pt-yiwkpm3p` stopped as designed at the four-rank identity gate. Replay maximum absolute logit error and $Z-X-U$ reconstruction error were both zero on every rank, while the registered $o_{proj}$ check reported 0.00290--0.00334 and failed its unchanged $10^{-5}$ threshold. No full extraction, TRAIN fit, CONFIRMATION access, or scientific result occurred; the job and `_r2` root are ineligible.

### Execution Repair A5: Shape-preserving output-projection identity check

- **Recorded at:** `2026-08-11T13:11:03Z` as an in-scope measurement-conformance repair after a pre-analysis guard stop.
- **Before:** the original attention projection executed one BF16 linear operation on `[batch, sequence, hidden]`, while the checker sliced the decision token first and recomputed a different `[batch, hidden]` GEMM shape. The checker therefore mixed mathematical identity with shape-dependent low-precision kernel differences.
- **After:** only during the identity smoke, recompute the projection on the original complete input tensor shape and then compare the decision-token output. Full extraction still stores the actual model output and does not perform this duplicate projection.
- **Unchanged contract:** the measured module, decision token, relative-error formula, and $10^{-5}$ threshold are unchanged. Replay and $Z-X-U$ guards are unchanged.
- **Validation:** a BF16 full-shape regression test passes and all 38 model-independent tests pass.
- **Evidence eligibility:** `_r2` remains permanently ineligible. One Reserved replacement on the A3 image may use the fresh `_r3` root; it must pass all four unchanged identity thresholds before full extraction.
- `2026-08-11T13:11:35Z`: the A5 replacement was submitted as Reserved ACP job `pt-b45k84qc`, display name `a15-08-01-layerwise-new-info-20260811T131100Z`, using the A3 image and fresh `_r3` root.
- `2026-08-11T13:19:33Z`: job `pt-b45k84qc` passed data, all four identity receipts, full 36-layer extraction, bridge-swap extraction, and shard merge, then stopped before ridge selection because block-1 TRAIN $X_1$ had exactly zero pooled scale. TRAIN has 4,096 exactly identical $X_1$ rows and DEVELOPMENT has 2,048 exactly identical rows. No standardizer, ridge penalty, common basis, or CONFIRMATION label was opened. Under the frozen Section 8.1 rule, the registered E01 outcome is **H2_LC_INSUFFICIENT_STANDARDIZATION**; `_r3` cannot yield the registered H2 verdict.

### Candidate Successor A6: Structural constant-null handling for block-1 old state

**State: AWAITING_HUMAN_CONFIRMATION; E01 CLOSED AT INSUFFICIENT.**

- **Observed structural fact:** at the shared answer-preceding token, $X_1=N_1(h_1)$ is exactly identical across every TRAIN and DEVELOPMENT prompt. This is expected because the block-1 pre-attention residual sees the same terminal token before any context-dependent attention write. Layers 2--36 have positive TRAIN scales.
- **Why E01 cannot be silently repaired:** Section 8.1 says a scale at most $10^{-12}$ makes the layer invalid. Replacing that rule after extraction would change a frozen computational definition, so it requires a successor experiment rather than another E01 retry.
- **Candidate E02 rule:** only when finite TRAIN $X_1$ is exactly constant and DEVELOPMENT matches its frozen mean, encode standardized $X_1$ as the zero matrix, freeze `scale=0` and `constant_null=true`, and require any later evaluation state to match the same mean exactly. This yields an intercept-free null old-state readout while preserving the actual $U_1$ correction. $U$, $H$, and every nonconstant layer keep strict Section 8.1 standardization.
- **Estimand preserved:** block 1 remains in the registered early median; no layer is dropped, no epsilon scale is introduced, and the five H2-LC clauses and thresholds remain unchanged.
- **Freshness requirement:** if approved, create `A15_08_01_E02` before execution and use a new sealed CONFIRMATION seed. E01 extraction may serve only as provenance and failure evidence, not as E02 confirmation evidence.
- **Candidate implementation validation:** explicit opt-in, exact-match, and evaluation-drift tests pass; all 39 model-independent tests pass. No E02 remote job is authorized until human confirmation.
