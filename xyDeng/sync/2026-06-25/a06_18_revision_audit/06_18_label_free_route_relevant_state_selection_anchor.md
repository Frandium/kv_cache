# A06_18: Label-Free Route-Relevant State Selection

Document status: result-updated anchor after the A06_18 revision audit.

## 0. Thinking Card

**Phenomenon:** A06_17 shows a sharp mismatch: route-position hidden states
recover the feature partition perfectly, but all-position clustering often
merges whole features.

**Mechanism guess:** The failed object is the clustering population. All-position
k-means mixes route-relevant states with non-route, neutral, or role-specific
states, so centers are spent on the wrong geometry.

**Key variables:** hidden-state pool, label-free route-state selector,
representation map, selected pool purity, cluster centers, held-out
route-position assignment.

**Causal relation:** If the selector keeps route-relevant states and rejects
non-route states, clustering the selected pool should recover the held-out
route-position feature partition better than all-position clustering.

**Observable metric:** held-out route-position `feature_NMI`. Load, active
experts, nuisance NMI, and a gated real-DCLM compatibility read are guards, not
the main claim.

**Alternative explanation:** Route feature geometry may be absent, or the
selector may simply encode slot position, nuisance, or a handcrafted route mask.

**Decision:** Keep this as A06_18, not A06_19. Split-stability, PCA,
bottleneck AE, and SAE-code clustering did not produce a selector that
approaches route-only. The next selector must add an explicit route-local or
route-readout constraint.

## 1. Problem Definition

**Parent problem:** A06 asks whether hidden-state geometry can support
label-free proxy-level gate initialization.

**Sharper subproblem:** Before real-DCLM proxy extraction, decide whether we can
identify which controlled hidden states should be clustered for routing.

**Decision question:** Can a label-free selector choose a hidden-state pool whose
cluster centers recover the held-out route-position feature partition better
than all-position clustering?

**Not in scope:** real-DCLM semantics, SAE as a method claim, training
preservation, expert utility, or scaling. The real-DCLM touchpoint is a
compatibility guard, not a primary method claim.

## 2. Physical Prior

**P1: Route-relevant geometry is population-dependent.**  
The route position can contain clean feature geometry even when all hidden
states do not share one clusterable feature geometry. This prior is wrong if
all candidate pools fail in the same way despite a strong route-only control.

**P2: A selector must improve the routed readout, not just cluster appearance.**  
A good selector is defined by held-out route-position `feature_NMI`, not by
silhouette, sparse-looking features, or balanced load.

**P3: Representation transforms are only useful as route-state selectors.**  
SAE sparsity, bottleneck compression, or reconstruction quality are not
evidence by themselves. They matter only if they produce hidden-space centers
that recover held-out route-position features.

## 3. Falsifiable Hypothesis

**H1: Sample-pool mismatch is the active bottleneck.**  
Supported if a label-free selected pool gives higher held-out route-position
`feature_NMI` than all-position clustering and is close to the route-only
positive control. Weakened if the selected pool matches all-position failure or
only improves nuisance/load metrics.

**H2: A representation map can improve the selected clustering population.**  
Supported if SAE-code clustering or low-dimensional bottleneck clustering
beats all-position and the failed split-stability selector. Weakened if the
representation reconstructs well but route-position `feature_NMI` does not
improve.

## 4. Mathematical Model

**Objects:** $A$ is the all-position hidden-state pool, $R$ is the unknown
route-relevant pool, $N$ is non-route states, $g_\phi(h_i)$ is a label-free
representation map, $s(h_i)$ is the selector, and $\hat R$ is the selected
pool.

**Core model:** Hidden states are split into route-relevant and non-route
populations:

$$
A = R \cup N,\qquad h_i \in A.
$$

The selector proposes:

$$
\hat R=\{h_i:s(h_i)=1\}.
$$

For representation clustering, compute

$$
u_i=g_\phi(h_i),
$$

cluster in $u$-space, then convert each cluster back to hidden-space centers
by averaging the original hidden states:

$$
c_k^h=\frac{1}{|C_k|}\sum_{i\in C_k}h_i.
$$

**Mechanism relation:** Clustering succeeds only if $\hat R$ preserves the
feature residual used at the route position:

$$
\operatorname{Cluster}(\hat R) \rightarrow
\text{high held-out route-position feature_NMI}.
$$

**Observable metric:** held-out route-position `feature_NMI`, with max load,
active experts, and nuisance NMI as guards.

**Falsifier:** no label-free selector improves held-out route-position
`feature_NMI` over all-position clustering while the route-only positive
control remains high.

## 5. Computational Realization

**Input objects:** A06_17/A06_18 controlled bridge hidden states and
route-position held-out labels; optional A06_10/A06_11-style real-DCLM hidden
states only after a controlled pass.

**Computed variables:** representation codes, selected pool $\hat R$,
hidden-space cluster centers, held-out route-position assignments,
`feature_NMI`, load, nuisance NMI, and selected-role composition.

**Algorithm stages:** fit representation maps without feature labels; cluster
candidate pools or representation codes; map clusters back to hidden-space
centers by original-state averaging; evaluate every center set on held-out
route-position states; use labels only for evaluation and failure
interpretation.

**Stage-local evidence:** selected-pool composition, route-feature heatmap,
load/nuisance guard table, and optional real-DCLM stability/readout table.

**Expected artifacts:** `protocol.md`, `summary.md`, `detailed.md`,
`tables/selector_comparison.csv`, `tables/representation_comparison.csv`, and
one route-feature heatmap. Real-DCLM artifacts are gated until controlled
success.

## 6. Minimal Falsification Test

| Test | Comparison | Primary metric | Pass / fail / insufficient | Failure means |
|---|---|---|---|---|
| First controlled selector audit | all-position baseline, route-only positive control, split-stability selector | held-out route-position `feature_NMI` | Failed: split-stability does not beat all-position | split stability is not route relevance |
| A06_18 revision audit | all-position baseline, route-only positive control, slot-offset positive control, failed split-stability selector, overcomplete SAE-code clustering, low-dimensional bottleneck/PCA clustering | held-out route-position `feature_NMI` | Pass if a revised selector beats all-position and approaches route-only; fail if it matches all-position or improves only load/reconstruction; insufficient if held-out route states are missing | representation transform did not solve sample-pool mismatch |
| Gated real-DCLM touchpoint | selected real-DCLM pool/readout vs raw all-position or prior proxy-center baseline | split stability, nuisance guards, optional step-0 `proxy_route_NMI` | Compatible if the selected readout is stable above nuisance controls and does not collapse at step 0; incompatible if it is unstable or nuisance-driven; insufficient if metadata/proxy labels are missing | controlled selector may not transfer, or the real-DCLM state object is still misdefined |

Guards: active experts, max load, nuisance/slot-position NMI, selected-role
composition, reconstruction error for autoencoder variants, and the real-DCLM
touchpoint only after the controlled selector passes.

## 7. A06_18 Revision Audit

**Goal:** improve A06_18 without changing the decision question.

**Do not open A06_19:** this is still the same selector operationalization
problem. The revision tested whether representation maps can fix the failed
all-position clustering population.

**Selector families to test:**

| Family | Representation | Cluster space | Hidden-space center |
|---|---|---|---|
| Baseline | raw hidden state | $h$ | k-means center in $h$ |
| Failed control | split-stability positions | $h$ | original A06_18 center |
| Overcomplete SAE | sparse code, e.g. `4*d_model` or `8*d_model` with top-k / L1 sparsity | SAE code $z$ | mean original $h_i$ per code cluster |
| Low-dimensional PCA | $q=4,8,16,32$ | PCA latent $u$ | mean original $h_i$ per latent cluster |
| Bottleneck AE | $q=4,8,16,32$ | bottleneck latent $u$ | mean original $h_i$ per latent cluster |

**Primary comparison:** revised selector `feature_NMI` vs all-position and
route-only on held-out route-position states.

**Result:** no representation selector passed. PCA q=4 weakly improved mean
`feature_NMI` over raw all-position (`0.871` vs `0.831`) but remained unstable
and reached only 2/8 perfect seeds. SAE-code clustering reconstructed all
states well but did not recover route-position features; SAE L1 8x reached
reconstruction MSE `0.0034` but only `feature_NMI=0.729`.

**Interpretation:** representation-only selection is insufficient. The next
selector needs an explicit route-readout constraint rather than another generic
representation transform.

## 8. Current Evidence

**Observation:** A06_17 found route-only and slot offset 3 clustering reach
`feature_NMI=1.0` across 8/8 seeds, while all-position has mean 0.797 and only
1/8 perfect seeds.

**Observation:** A06_10/A06_11 show real-DCLM proxy clusters can be read out
and converted to step-0 router rows, while A06_12 shows ordinary early training
does not preserve that partition.

**Observation:** A06_18 full local two-GPU run found route-only and slot offset
3 remain perfect (`feature_NMI=1.0` in 8/8 seeds), while split-stability
selectors do not beat all-position: top-1 mean 0.745, top-3 mean 0.778,
threshold mean 0.674, versus all-position mean 0.797.

**Observation:** A06_18 revision audit found raw all-position mean
`feature_NMI=0.831`, PCA q=4 mean `0.871`, PCA q=16 mean `0.851`, bottleneck
AE q=32 mean `0.814`, SAE L1 8x mean `0.729`, and SAE top-k variants below
`0.65`. Route-only and slot offset 3 remained `1.0` in 8/8 seeds.

**Interpretation:** feature geometry exists at the route position. The tested
split-stability selector operationalization is not a route-relevance criterion,
and generic representation learning does not solve the selection problem.
The real-DCLM touchpoint should not be run from these selectors.

**Boundary:** this evidence is controlled-bridge evidence. It does not prove
real-DCLM semantic proxy extraction or preservation.

**Evidence links:**

- `Projects/from-attention-to-search/main/experiments/A06/A06_17_all_position_route_relevant_feature_discovery/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_17_all_position_route_relevant_feature_discovery/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_10_real_dclm_proxy_feature_operationalization/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_11_real_dclm_proxy_center_router_initialization/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_12_real_dclm_proxy_init_training_preservation/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_18_label_free_route_relevant_state_selector/summary.md`
- `Projects/from-attention-to-search/main/experiments/A06/A06_18_label_free_route_relevant_state_selector/detailed.md`

## 9. Claim Boundary And Next Decision

**Can claim now:** route-position feature geometry remains reachable, but the
tested split-stability and representation-only selector families do not
identify a reliable better clustering population than all-position hidden
states.

**Cannot claim:** no label-free selector can work, real-DCLM semantic proxies,
training preservation, utility, or that SAE is needed.

**Next decision:** Move to an explicit route-local or route-readout-constrained
selector. Do not claim SAE failure in general, do not run the real-DCLM
touchpoint from these selectors, and do not open A06_19 until a controlled
selector approaches route-only and is ready for slot early-training tests.
