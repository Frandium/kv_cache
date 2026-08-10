---
anchor_id: 15_07_layerwise_conditional_fine_discriminant_directions
status: draft_human_review_required
canonical_language: en
chinese_companion: 15_07_layerwise_conditional_fine_discriminant_directions_anchor_cn.md
thinking_card: 15_07_layerwise_conditional_fine_discriminant_directions_thinking_card_cn.md
parent_line: 15_spectral_representation_and_functional_routing
created: 2026-08-10
updated: 2026-08-10
---

# A15_07 Layerwise Conditional-Fine Discriminant Directions

Researcher judgment: [Chinese Thinking Card](15_07_layerwise_conditional_fine_discriminant_directions_thinking_card_cn.md). This Anchor is a candidate design awaiting human review. No Protocol exists, and no implementation or execution is authorized.

## 1. Problem Definition

A15_02 found layer- and local-parameter-rank-dependent semantic variance, but no stable fine-specific relocation law. A15_04 then showed that retained local rank relocation did not enter the registered cross-layer shared F9--F16 broad tail. A15_05 further showed that fixed covariance bands can change token-to-expert dispatch without improving the registered functional targets. Band position is therefore not an admissible rule for selecting a semantic coordinate.

This Anchor parks the A15_03 middle-band audit and fixes **conditional-fine** as its only semantic object: the task is to distinguish children after the parent is given. It neither compares coarse against fine nor assumes that the signal belongs to head, middle, or tail.

The `AI_PROPOSAL` representation is the actual attention-induced MLP-input increment at ten frozen Qwen3-8B layers:

$$
x_\ell=\Delta n_\ell
=RMSNorm_\ell(h_\ell+a_\ell)-RMSNorm_\ell(h_\ell).
$$

Its complete 4,096-dimensional coordinates in the frozen orthogonal parameter basis may be used because a complete orthogonal change of coordinates does not change the generalized discrimination problem. The sampled layers are 1/5/9/13/17/21/25/29/33/36. Layer 36 is a displayed terminal boundary and is excluded from the primary decision.

**Exactly one decision question:**

> On the frozen conditional-fine data and actual $\Delta n_\ell$, does a low-rank layer-local subspace learned from one set of expressions continue to distinguish within-parent children on the other expressions, while beating equal-rank Haar and within-parent label-permutation controls, thereby earning a cross-expression layerwise linear semantic-coordinate certificate?

The decision only admits or rejects candidate directions for a later functional-admission question. It does not test native MLP use, expert utility, or Router-training benefit.

## 2. Physical Priors

1. **Discriminative signal must be separated from expression noise.** A useful conditional-fine direction should separate child centers while remaining stable when the same child is rephrased or instantiated with another fact bundle.
2. **High-dimensional small-sample estimation requires shrinkage.** The representation has 4,096 dimensions and few expressions per child. An unregularized within-child covariance is singular, and its largest generalized eigenvalue can be an accidental near-zero-noise direction.
3. **Directions require expression-held-out validation.** Construction eigenvalues, training accuracy, and visual separation are not evidence of reproduction.
4. **Each layer may own a different subspace.** $S_\ell$ and $S_m$ need not share vector identity. Equal rank or equal band name is not cross-layer direction identity.
5. **Semantic separation is not functional use.** A stable child-discriminating direction need not predict which expert is useful for a token.

## 3. Falsifiable Hypotheses

**H1 -- stable low-rank conditional-fine directions.** The top $r$ generalized discriminant directions constructed independently from TRAIN or DEVELOPMENT expressions improve within-parent child balanced accuracy on CONFIRMATION beyond equal-rank Haar and within-parent sample-to-child assignment permutation. The two construction splits also produce subspaces whose overlap exceeds the equal-rank random baseline.

**R1 -- expression shortcut or small-sample overfit.** Construction eigenvalues or accuracy are high, but held-out advantage does not beat random or reverses when the expression halves are exchanged. The directions are not a stable fine-semantic certificate.

**R2 -- no low-rank linear concentration certificate.** A regularized full-space linear readout separates children, but the registered low-rank candidate does not beat random. The signal may be distributed over a larger linear space. If full-space linear capability also fails, the outcome is capability-insufficient. Neither case establishes that fine information is absent or that a nonlinear Router is required.

**Pass:** at least one preregistered layer has a grouped-bootstrap 95% simultaneous lower bound of $G_\ell$ above zero; the TRAIN-built and DEVELOPMENT-built results are both positive at that layer; cross-expression subspace overlap beats equal-rank Haar q95; and all full-space capability and data-reliability guards pass. Pass admits only the specific layers satisfying every condition.

**Fail:** full-space linear capability passes, but low-rank held-out advantage is precisely nonpositive or the construction/confirmation directions stably reverse. Fail rejects only the registered $B/W$ construction and rank range.

**Insufficient:** full-space capability is absent, expression splits are not independent, labels or cached representations fail validation, precision crosses zero, or $r/\alpha$ selection leaks confirmation data.

## 4. Mathematical Model

### 4.1 Concrete objects in $B_\ell$ and $W_\ell$

Let $x_{pce}^{(\ell)}\in\mathbb R^{4096}$ be the actual $\Delta n_\ell$ for layer $\ell$, parent $p$, child $c$, and expression $e$. Each parent has $C_p=8$ children. Using only the current construction half, define

$$
\mu_{pc}^{(\ell)}=\frac{1}{E_{pc}}\sum_e x_{pce}^{(\ell)},
\qquad
\mu_p^{(\ell)}=\frac{1}{C_p}\sum_c\mu_{pc}^{(\ell)}.
$$

The within-parent between-child covariance is

$$
B_\ell=
\frac{1}{P}\sum_{p=1}^{P}\frac{1}{C_p}
\sum_{c=1}^{C_p}
(\mu_{pc}^{(\ell)}-\mu_p^{(\ell)})
(\mu_{pc}^{(\ell)}-\mu_p^{(\ell)})^\top.
$$

The within-child expression covariance is

$$
W_\ell=
\frac{1}{P}\sum_{p=1}^{P}\frac{1}{C_p}
\sum_{c=1}^{C_p}\frac{1}{E_{pc}}
\sum_e
(x_{pce}^{(\ell)}-\mu_{pc}^{(\ell)})
(x_{pce}^{(\ell)}-\mu_{pc}^{(\ell)})^\top.
$$

$B_\ell$ measures child-identity differences, whereas $W_\ell$ measures expression changes within one child. Centering each child against its parent removes the parent-common component; these are not mixed coarse/fine quantities.

### 4.2 Origin and role of the generalized eigenproblem

For a direction $v$, define the discriminative Rayleigh quotient

$$
J_\ell(v)=
\frac{v^\top B_\ell v}
{v^\top(W_\ell+\rho I)v}.
$$

The numerator is between-child center variance along $v$. The denominator is within-child expression variance plus shrinkage. Maximizing the numerator subject to a unit denominator gives

$$
\max_v v^\top B_\ell v
\quad\text{s.t.}\quad
v^\top(W_\ell+\rho I)v=1.
$$

The Lagrangian is

$$
\mathcal L(v,\lambda)=
v^\top B_\ell v-
\lambda\left[v^\top(W_\ell+\rho I)v-1\right].
$$

Setting $\nabla_v\mathcal L=0$ yields

$$
B_\ell v=\lambda(W_\ell+\rho I)v.
$$

The equation is therefore the first-order optimality condition for maximizing child separation per unit expression noise, not an empirical identity. At a solution, $\lambda=J_\ell(v)$. A large $\lambda$ describes only the construction data; it is not held-out accuracy, information quantity, or Router utility.

Equivalently, define

$$
C_\ell=(W_\ell+\rho I)^{-1/2}
B_\ell(W_\ell+\rho I)^{-1/2}.
$$

Solve $C_\ell u=\lambda u$ and map back with $v=(W_\ell+\rho I)^{-1/2}u$. The method first whitens within-child expression variability and then finds the directions of largest child-center variation.

$\rho>0$ makes $W_\ell+\rho I$ invertible and suppresses accidental near-zero-variance directions. It may not be tuned on confirmation expressions. With eight parents and eight children per parent, $\operatorname{rank}(B_\ell)\le8(8-1)=56$; more than 56 nonzero discriminant directions have no support from the registered class structure.

### 4.3 One primary metric

Let $S_\ell^{T}(r,\alpha)$ be constructed only from TRAIN, with $r$ and $\alpha$ selected by a preregistered 5/5 internal TRAIN expression cross-validation. Freeze those hyperparameters and construct $S_\ell^{D}$ independently from DEVELOPMENT. Each construction split uses its own $\rho=\alpha\operatorname{tr}(W_\ell)/4096$. Normalize the columns of $V_\ell$ so that $V_\ell^\top(W_\ell+\rho I)V_\ell=I$ and use $z=V_\ell^\top x$ as the classification coordinate. Each classifier uses only its own construction-split within-parent child centroids and predicts each CONFIRMATION item by its nearest projected child center within the known parent.

Balanced accuracy averages correctness equally over children and parents. Let $BA_\ell^{disc}$ be the mean CONFIRMATION balanced accuracy of the TRAIN-built and DEVELOPMENT-built subspaces. For the selected rank at each layer, compute 512 equal-rank Haar subspaces and 512 within-parent sample-to-child assignment permutations with the identical split and classifier. Define

$$
G_\ell=BA_\ell^{disc}
-\max\left\{q_{0.95}(BA_\ell^{Haar}),
q_{0.95}(BA_\ell^{perm})\right\},
$$

Let $LCB_{0.95}^{sim}(G_\ell)$ be the grouped-bootstrap lower bound simultaneously corrected over the nine non-terminal sampled layers, and define the admissible layer set

$$
\mathcal L^*=\left\{\ell\in\mathcal L_{dec}:\,
LCB_{0.95}^{sim}(G_\ell)>0\right\}.
$$

The layerwise $G_\ell$ curve, measured in absolute balanced-accuracy points, is this Anchor's **one primary metric**; $\mathcal L^*$ is only its decision set. This avoids hiding a layer-local result in a cross-layer average. The `AI_PROPOSAL` Pass rule requires $\mathcal L^*\ne\varnothing$ and admits only layers in that set. Cross-layer medians and late-minus-early change are explanatory and cannot replace $G_\ell$.

Euclidean-QR-orthonormalize the generalized eigenvectors constructed from TRAIN and DEVELOPMENT and let $P_\ell^T,P_\ell^D$ be the resulting equal-rank orthogonal projectors. Then

$$
O_\ell=\frac{\operatorname{tr}(P_\ell^TP_\ell^D)}{r}
$$

is a direction-stability guard. It must exceed equal-rank independent-Haar q95 but is not a second primary metric. Generalized eigenvalues, construction accuracy, band energy, and cross-layer overlap cannot replace $G_\ell$.

## 5. Computational Realization

1. Freeze `/data/share/Qwen3-8B`, the A15_02_07 ten layers, tokenization, final `Classification:` readout, and complete 4,096-dimensional actual-$\Delta n_\ell$ coordinates.
2. Freeze A15_02_07 TAX data with SHA-256 `ce91dbbd3c5071e17beeccf0d86a280dc8a3e48e0fdbf2178868da45eea18af4`. The eight parents are mathematics, physics, chemistry, biology, computer science, economics, medicine, and linguistics. Each parent has eight children, and each child has 10 TRAIN + 10 DEVELOPMENT + 10 CONFIRMATION expressions, for 1,920 texts. Use only the eight complex/conditional-fine tasks; exclude the simple condition. Weight parents and children equally. Chance balanced accuracy is $1/8=12.5\%$.
3. `AI_PROPOSAL`: divide TRAIN by frozen expression ID into two 5-expression halves and use bidirectional internal cross-validation to select one shared $r,\alpha$. Then construct $S_\ell^T$ from all TRAIN and $S_\ell^D$ from all DEVELOPMENT; adjudicate both only on CONFIRMATION.
4. `AI_PROPOSAL`: $r\in\{1,2,4,8,16,32,56\}$ and $\rho=\alpha\operatorname{tr}(W_\ell)/4096$ for $\alpha\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$. Break ties toward smaller $r$ and larger $\alpha$.
5. All centers, $B_\ell$, $W_\ell$, $r$, $\alpha/\rho$, and directions come only from TRAIN/DEVELOPMENT. CONFIRMATION cannot select sign, rank, regularization, layer, sample, or plot range. These CONFIRMATION expressions were inspected under earlier metrics, so they are analysis-held-out for the newly frozen method, not a new population replication.
6. Freeze seeds for 512 Haar and 512 assignment-permutation controls. Within each parent, reassign individual TRAIN/DEVELOPMENT expressions to child bins while preserving every child count; a simple label renaming would leave $B_\ell$ unchanged and is not a valid control. The permutation pipeline uses the true candidate's frozen rank and reruns $B/W$ construction and CONFIRMATION evaluation.
7. The full-space capability guard uses a 4,096-dimensional regularized multiclass linear readout under the same TRAIN-only hyperparameter discipline and is adjudicated on CONFIRMATION. A layer passes only if the simultaneous lower bound of $BA_\ell^{full}$ minus its within-parent assignment-permutation q95 is above zero. This establishes full-space linear task capability and cannot replace low-rank $G_\ell$.
8. Do not form a dense 4,096-by-4,096 inverse merely for convenience. The Protocol must specify a stable ridge-whitened solve in the sample span and verify residuals, orthogonality, and agreement with a direct implementation on a small check.
9. Do not adjudicate head, middle, or tail. After the primary verdict is frozen, the parameter-rank energy of $S_\ell$ may be described, but it cannot change the result or select a band post hoc.
10. The grouped bootstrap treats the eight parents as outer uncertainty units and resamples expressions within each parent. Every replicate must rebuild directions and classification centroids. One max-statistic simultaneously corrects the nine decision layers.

## 6. Minimal Falsification Tests

| Typed outcome | Inspectable rule | Knowledge update |
| --- | --- | --- |
| `STABLE_LOW_RANK_LINEAR_DIRECTIONS` | full-space capability passes; $\mathcal L^*$ is nonempty; TRAIN-built and DEVELOPMENT-built results are both positive and subspace overlap beats Haar q95 for every admitted layer; parent/expression-source groups do not reverse | only layers in $\mathcal L^*$ receive a cross-expression low-rank conditional-fine candidate coordinate; they may enter separate functional admission but not Router training |
| `EXPRESSION_OR_SMALL_SAMPLE_OVERFIT` | construction eigenvalue or accuracy is high, but all $G_\ell$ values fail to beat random, TRAIN-built and DEVELOPMENT-built results reverse, or subspace overlap does not beat Haar | the candidate is dominated by expression shortcuts, sampling noise, or unstable estimation; do not use it for Router design |
| `DISTRIBUTED_LINEAR_SIGNAL` | a regularized full-space linear readout reliably beats chance/Haar at named layers, but those layers have no low-rank $G_\ell$ advantage | fine linear information exists without registered low-rank concentration; next ask about higher-rank, sparse, or local discriminative structure, not band location |
| `INSUFFICIENT_TASK_CAPABILITY` | full-space capability does not reliably beat chance, or model/data/cache guards fail | the experiment cannot adjudicate low-rank directions; repair capability before invoking nonlinear routing |

**Minimal counterexample:** add a construction-only template offset that covaries with child labels and reverses independently in confirmation expressions. It creates a large construction $\lambda$ and training accuracy while producing $G_\ell\le0$ and opposite outer-fold results. This falsifies any inference from generalized eigenvalue alone to stable semantics.

## 7. Current Evidence

1. [A15_02_05](../evidence/a15_02_05/summary.md) established reproducible conditional-fine residuals but no stable extra local-rank relocation beyond coarse at actual $\Delta n$; named-case directions were heterogeneous. It did not solve the $B/W$ discriminant problem.
2. [A15_02_07 TAX](../evidence/a15_02_07/summary.md) found no stable fine-specific rank relocation over the same ten layers. Existing linear readability is not a low-rank cross-expression direction certificate.
3. [A15_04](../evidence/a15_04/summary.md) failed its global shared F9--F16 claim: local relocation did not enter a reusable broad tail. This directly motivates replacing band position with direction identity.
4. [A15_05_04](../evidence/a15_05_04/summary.md) and [A15_05_05](../evidence/a15_05_05/summary.md) did not admit fixed M/T/N bands. A changed high-dimensional neighborhood or dispatch is not functional evidence.
5. No A15_07 $B_\ell/W_\ell$ direction, cross-expression $G_\ell$, or subspace-overlap result exists. Every A15_07 result is `NOT RUN`.

## 8. Claim Boundary And Next Decision

This Anchor can establish only whether conditional-fine differences concentrate into small, expression-reproducible, layer-local linear subspaces for one frozen model, one previously inspected 1,920-text English TAX bank, one representation site, and one registered regularized method.

Even a Pass cannot establish native functional use, attention-specific new information, cross-model sharing, fixed-band identity, expert prediction, reduced expert conflict, or Router NLL gain. To advance a layerwise Router, a later independent functional admission must freeze candidate directions in the **actual Router input of the same MoE** and test whether they predict expert-specific utility or validated same-expert compatibility beyond native-score, equal-rank Haar, and wrong-layer controls.

**Exactly one next decision:** the researcher confirms or corrects six design choices: actual $\Delta n$ as the representation; conditional-fine as the only semantic object; the TRAIN/DEVELOPMENT/CONFIRMATION reuse boundary; $G_\ell$ as the layerwise primary metric; the $r/\alpha$ search space and $\rho$ scaling; and a Pass opening functional admission rather than Router training.

**Completion criterion:** all six choices are recorded explicitly, with A15_03 remaining parked, no band adjudication in A15_07, and no promotion from semantic separation to function.

**Resume action:** after confirmation, write one `DRAFT_NOT_EXECUTABLE` Protocol from this Anchor. Implementation or smoke requires a separately approved execution scope after Protocol review.
