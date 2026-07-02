# Protocol: A06_24_synthetic_common_rare_transformer_moe

## 0. Approval Snapshot

Approval status: user requested anchor, protocol, experiment execution, result report, and 4-GPU full run in the current turn.

Purpose: Test whether common subtraction separates rare features or only repairs concentration in a no-position one-layer Transformer plus one-layer MoE synthetic common/rare task.

Primary anchor: `../../../problem_anchors/06_geometry_proxy_preservation/06_24_synthetic_common_rare_transformer_moe_anchor.md`

Anchor decision question: In a no-position one-layer Transformer plus one-layer MoE synthetic common/rare task, does common subtraction create rare-feature expert separation, or is it only a load/concentration control while rare separation requires route-relevant centers and preservation controls?

Anchor physical prior tested: global common subtraction can improve concentration without supplying route relevance or rare-rare separation.

Anchor core model term tested: the split $h_i=c+r_i$ and whether routing on $r_i$ improves rare-feature NMI beyond load repair.

Anchor falsifier: common-subtracted all-position routing reaches high rare-feature NMI and positive rare margins across slot lengths while slot-start NMI stays low and task loss is not worse.

Experiment role: root-cause audit plus method-readiness gate.

Primary metric from anchor: rare-feature NMI on a balanced held-out route-position evaluation set.

Claim boundary from anchor: the run can support or weaken the common-subtraction boundary in synthetic Transformer-MoE training, but cannot claim real-DCLM transfer or semantic experts.

Minimal setup: one-layer causal Transformer, one weighted top-1 MoE layer, no position embeddings, one high-frequency common feature, three low-frequency rare features, repeated feature slots, neutral background, and feature-specific targets.

Basic configuration: seeds 0--7, slot lengths 1/2/4/8, four features, four experts, balanced held-out evaluation, imbalanced calibration/training, checkpoints 0/10/40/80/160 for full.

Conditions to run: random raw, random common-subtracted, all-position k-means raw, all-position common-subtracted k-means, route-position k-means raw, route-position residual-input k-means, oracle raw, oracle residual input, and oracle row projection.

Pass: common-subtracted all-position variants fail to produce rare-feature separation while route-position/oracle centers do, and preservation controls improve final rare margin or sign flips without task-loss regression.

Fail: common-subtracted all-position routing matches route-position/oracle rare-feature NMI and margins across slot lengths without position leakage.

Insufficient: target accuracy fails, slot-start NMI is high, route-position positive controls fail, or the run cannot complete enough seeds/slot lengths.

Cannot claim: real-DCLM behavior, natural-language semantics, deployable routing, or optimizer optimality.

Approval decision: execute full synthetic run.

## 1. Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit or formula | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Common feature | High-frequency synthetic feature | Feature id 0 sampled with probability 0.70 during calibration/training | Probability mass | Tests frequency-dominated routing | Real common semantics |
| Rare feature | Low-frequency synthetic feature | Feature ids 1--3 sampled with total probability 0.30 | Feature id | Tests rare-rare separation | Real rare semantics |
| Route position | Task-relevant audited state | Last feature-slot token, whose logits predict the target token | Sequence position | Keeps routing metric tied to the task | A real route selector |
| No-position model | Model without explicit position embeddings | Token embedding plus causal attention only | Architecture | Guards against position shortcut | All context-length effects |
| Rare-feature NMI | Rare feature/expert assignment agreement | NMI(feature id, route) restricted to rare eval samples | 0--1 | Primary decision metric | Expert utility |
| Rare margin | Matched rare score advantage | $z_{i,m(f_i)}-\max_{e\ne m(f_i)}z_{i,e}$ | Logit difference | Tests basin thickness | Utility |
| Joint feature score | Combined common/rare and rare-feature separation | `rare_feature_NMI * common_rare_NMI` | Dimensionless score | Prevents overclaim when only one separation axis passes | Expert utility |
| Sign flip | Preserved matched margin crosses zero | Step-0 positive rare margin becomes non-positive later | Fraction | Tests preservation | Cause of failure alone |
| Slot-start NMI | Position nuisance agreement | NMI(slot_start, route) | 0--1 | Position leakage guard | Absence of every position effect |

## 2. Anchor Alignment

Decision question: Does common subtraction create rare-feature separation, or only reduce concentration?

Physical prior tested: common subtraction can remove a shared bias but does not identify route-relevant feature centers.

Core model term tested: $w_e^\top c$ versus $w_e^\top r_i$ in router scores.

Falsifier: common-subtracted all-position centers match route-position/oracle centers on rare-feature NMI and rare margins.

Claim boundary: synthetic no-position Transformer-MoE only.

## 3. Tested Hypothesis

H1 is the primary hypothesis: common subtraction is a load/concentration repair, not a rare-feature separator. H2 and H3 are guards: route-position centers should be stronger than all-position centers, and residual input or row projection should be evaluated only after valid initialization.

## 4. Rival Explanations

- Load-only improvement: max load improves but rare-feature NMI does not.
- Position leakage: routed experts track slot start rather than feature id.
- Binary common/rare separation: common and rare separate, but rare features merge.
- Oracle leakage: only label-based centroids pass.
- Training failure: routing method looks bad because the model does not learn the synthetic target.

## 5. Data / Model / Algorithm / Objective

Data: sequences contain neutral background tokens, a repeated feature slot, and a target token after the slot. Feature 0 is common; features 1--3 are rare. Slot starts are balanced across valid starts, and no explicit position embeddings are used.

Model: one-layer causal self-attention followed by one weighted top-1 MoE layer and a language-model head. The selected expert output is weighted by the selected softmax gate probability so router rows receive gradient.

Objective: cross-entropy at the route position to predict the feature-specific target token.

Discovery: extract step-0 hidden states, fit centers on all positions or route positions, optionally subtract common means, and initialize router rows from equal-norm centers.

Training: train each condition under the imbalanced feature distribution and evaluate on a balanced held-out route-position set.

## 6. Conditions, Seeds, And Checkpoints

| Item | Anchor clause tested | Model term / rival explanation | Why needed | Primary or secondary evidence | Pass | Fail | Insufficient | Figure/table |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `random_raw` | baseline concentration | random top-1 routing | Establish baseline | rare NMI, load | baseline low | n/a | target fails | `training_trajectory.csv` |
| `random_common_subtract` | H1 | common subtraction only | Tests load-only repair | rare NMI vs load | load improves without rare NMI | rare NMI improves | high slot NMI | `step0_discovery.csv` |
| `allpos_kmeans_raw` | H2 | wrong hidden-state pool | Tests all-position centers | rare NMI | below route centers | matches route centers | unstable | `step0_discovery.csv` |
| `allpos_kmeans_common_subtract` | H1/H2 | common-subtracted wrong pool | Tests whether common subtraction rescues pool mismatch | rare NMI, rare margin | still below route centers | matches route centers | high slot NMI | `step0_discovery.csv` |
| `route_kmeans_raw` | H2 | route-relevant pool | Positive label-free route-pool control | rare NMI | beats all-position | fails like all-position | hidden geometry absent | `training_trajectory.csv` |
| `route_kmeans_residual_input` | H3 | residual router input | Preservation control | final rare margin, sign flip | improves preservation | no improvement | loss regression | `training_trajectory.csv` |
| `oracle_raw` | positive control | label-known feature centers | Tests reachability | rare NMI, margin | passes | fails | data/model invalid | `training_trajectory.csv` |
| `oracle_residual_input` | H3 | residual router input after valid init | Tests preservation | final rare margin | improves or matches without regression | worsens | loss regression | `training_trajectory.csv` |
| `oracle_row_projected` | H3 | router-row projection | Tests row common control | final rare margin | improves or matches without regression | worsens | loss regression | `training_trajectory.csv` |

Seeds: 0--7 for full, one or two seeds for smoke. Slot lengths: 1, 2, 4, 8. Checkpoints: 0, 10, 40, 80, 160 in full.

## 7. Primary Metric

Rare-feature NMI is the primary metric because the main question is whether rare features are separated from one another. Overall load or overall feature NMI can hide a rare-feature merge when the common feature is frequent. Joint feature score is a required guard because rare features can separate while common still shares an expert with rare features.

False positive cost: promoting common subtraction as a feature-specialization method when it only balances load. False negative cost: parking a useful common-control method too early; route-position and oracle controls protect against this by showing whether rare geometry is available.

## 8. Secondary Metrics

Overall feature NMI, common/rare binary NMI, joint feature score, rare margin mean and p05, sign-flip rate, max load, effective experts, target accuracy, route-position loss, and slot-start NMI.

## 9. Known Good / Known Bad / Known Confusing Cases

Known good: oracle feature centers should produce high rare-feature NMI at step 0.

Known bad: random raw routing should not reliably separate rare features.

Known confusing: common subtraction may reduce max load or binary common/rare concentration while rare features still merge.

## 10. Stage-Level Profiling Plan

| Stage | Local question | Input evidence | Pass / fail / unclear rule | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| Data audit | Is frequency skew present and slot starts balanced? | generated counts | pass if feature and slot distributions match config | `data_audit.csv` | detailed |
| Step-0 discovery | Can centers separate rare features before training? | `step0_discovery.csv` | route/oracle controls should pass | heatmaps | summary/detailed |
| Training | Does target objective learn? | loss and accuracy | pass if target accuracy high | trajectory tables | detailed |
| Preservation | Do rare margins survive? | margin and sign flip | pass if controls improve over raw | trajectory figures | summary/detailed |
| Position guard | Are routes position-driven? | slot-start NMI | pass if low | nuisance table | summary/detailed |

## 11. Algorithm Specification, If Nontrivial

input: seed, slot length, common probability, number of features, number of experts, train steps, and condition list.

parameters: full run uses eight seeds, four slot lengths, hidden size 64, four attention heads, four experts, and 160 training steps.

intermediate variables: route hidden states, all-position hidden states, common means, k-means centers, router mappings, margins, route assignments.

steps:

1. Generate calibration and balanced evaluation batches.
2. Initialize the model without position embeddings.
3. Extract step-0 hidden states.
4. Fit all-position and route-position k-means centers.
5. Build router initializations.
6. Evaluate step-0 discovery conditions.
7. Train each selected condition on imbalanced batches.
8. Evaluate checkpoints on balanced route-position data.
9. Aggregate tables and generate figures.

outputs: CSV tables, PNG figures, JSON run summary, and logs.

debug artifacts: per-seed/slot trajectory rows, data audit, heatmap rows, run config, and submission record.

pass conditions: route/oracle controls work; common-subtracted all-position does not match them; preservation controls do not regress loss.

fail conditions: common-subtracted all-position matches route/oracle controls or positive controls fail.

failure reasons: load-only effect, pool mismatch, position leakage, insufficient hidden geometry, or training failure.

## 12. Success / Failure / Insufficient Evidence

Success for the current boundary: common subtraction improves concentration but not rare separation; route-position or oracle centers separate rare features; residual controls improve or preserve final rare margins.

Failure: common-subtracted all-position centers produce rare-feature separation comparable to route-position/oracle centers across slot lengths.

Insufficient: positive controls fail, target accuracy fails, slot-start NMI is high, or too few full cells complete.

## 13. What This Cannot Claim

This cannot claim real DCLM transfer, semantic expert specialization, expert utility in natural language, a final deployable router method, or optimizer optimality.

## 14. Review Notes And Protocol Changes

The old vector-only A06_26 result was renumbered to A06_24_toy and treated as mechanism background. This protocol is the main A06_24_synthetic check because it uses a trained one-layer Transformer plus one-layer MoE and explicitly removes position embeddings.
