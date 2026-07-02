# Summary: A06_24_synthetic_common_rare_transformer_moe

Primary anchor:
`../../../problem_anchors/06_geometry_proxy_preservation/06_24_synthetic_common_rare_transformer_moe_anchor.md`

Protocol:
`protocol.md`

## Result Snapshot

Verdict: Supported, with a refined preservation boundary.

What we established: In a no-position one-layer Transformer plus one-layer MoE synthetic common/rare task, common subtraction is not a reliable feature separator. It does not match route-position or oracle centers on the joint common/rare plus rare-feature separation metric, and all-position common-subtracted routing keeps weak rare margins.

What the experiment shows: Route-position centers and oracle centers produce clean step-0 separation. During training, every condition learns the target, so target accuracy is not a specialization metric. Router-row projection preserves the clean common/rare partition better than residual router input; residual input preserves rare-rare separation but can lose common-vs-rare separation.

What we do next: Treat simple global common subtraction as a parked method. The next method anchor should test either a task-aware route-relevant state selector or a row-projection / margin-preserving router update, not load repair alone.

Execution: Full 4-GPU ACP run completed successfully.

- job id: `pt-hb9swzcm`
- run name: `a06_24_synthetic_full_4gpu_20260702_172558`
- completed cells: `32` seed/slot cells
- training rows: `1280`

## Purpose

This experiment replaces the old vector-only A06_24_toy mechanism audit with a trained synthetic surface: one causal Transformer layer, one weighted top-1 MoE layer, no explicit position embeddings, imbalanced common/rare features, and multiple feature slot lengths.

## Terminology / Definitions

| Term | Plain meaning | Concrete object / computation | Unit or formula | Why it matters | Cannot prove |
| --- | --- | --- | --- | --- | --- |
| Common feature | High-frequency synthetic feature | Feature id 0, sampled at about 70% in calibration/training | Probability | Tests frequency-dominated routing | Natural-language common semantics |
| Rare features | Low-frequency synthetic features | Feature ids 1--3, total about 30% in calibration/training | Feature ids | Tests rare-rare separation | Real rare semantics |
| Route position | Task-relevant audited token | Last token of the repeated feature slot | Sequence index | Ties routing to target prediction | A real route selector |
| Rare-feature NMI | Rare feature / expert agreement | NMI(feature id, route), restricted to rare eval examples | 0--1 | Primary rare separation metric | Common-vs-rare separation |
| Common/rare NMI | Binary common-vs-rare agreement | NMI(feature is rare, route) | 0--1 | Tests whether common separates from rare | Rare-rare separation |
| Joint feature score | Product of rare separation and common/rare separation | `rare_feature_NMI * common_rare_NMI` | 0--about 0.637 here | Prevents overclaim when only one part passes | Expert utility |
| Rare margin p05 | Lower-tail rare routing margin | 5th percentile of matched rare score gap | Logit difference | Tests basin thickness | Functional value |
| Slot-start NMI | Position nuisance agreement | NMI(slot_start, route) | 0--1 | Position leakage guard | No possible context-length effect |

## Exact Setup

Research question: Does common subtraction create common/rare and rare-rare expert separation, or does it mainly repair concentration?

Data / model / objective: Synthetic sequences contain neutral background tokens, one repeated feature slot, and a feature-specific target token after the slot. The model has one causal attention layer, one weighted top-1 MoE layer, and no explicit position embeddings. The loss is cross-entropy at the route position predicting the target token.

Conditions: random raw, random common-subtracted, all-position k-means raw, all-position common-subtracted k-means, route-position k-means raw, route-position residual-input k-means, oracle raw, oracle residual input, and oracle row projection.

Changed variables: router initialization pool, common subtraction at router input, and row projection.

Held fixed: data generator, model size, no-position architecture, feature distribution, slot-start balancing, seeds, train steps, and evaluation set design.

Seeds / checkpoints: seeds `0--7`; slot lengths `1,2,4,8`; checkpoints `0,10,40,80,160`.

Paths:

- runner: `scripts/run_a06_24_synthetic_common_rare_transformer_moe.py`
- tables: `tables/`
- figures: `figures/`
- run summary: `summary.json`
- ACP log: excluded from this curated `xyDeng` sync package; job id and completeness checks are preserved in `detailed.md`.

Known limitation: The model is synthetic and no-position, and the target objective is much cleaner than language modeling. This makes it a mechanism gate, not a real-DCLM method validation.

## Primary Metric

Definition: rare-feature NMI is the normalized mutual information between rare feature id and routed expert on balanced held-out route-position examples.

Unit: 0--1.

Why it decides: The old A06 evidence already showed load can improve without feature specialization. The current question asks whether rare features separate, so rare-feature NMI is the first metric. The joint feature score is a required guard because rare features can separate while common still shares an expert with rare features.

False-positive cost: promoting common subtraction as a specialization method when it only changes load or binary concentration.

## Result

### Step-0 Discovery

Common-subtracted all-position centers do not match route-position or oracle centers. At step 0:

| Condition | Rare-feature NMI | Common/rare NMI | Joint feature score | Max load | Rare margin p05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-position common-subtracted | 0.690 | 0.639 | 0.405 | 0.516 | -2.759 |
| route-position raw | 0.948 | 0.648 | 0.612 | 0.305 | 6.802 |
| route-position residual input | 1.000 | 0.637 | 0.637 | 0.250 | 11.657 |
| oracle raw | 1.000 | 0.637 | 0.637 | 0.250 | 11.561 |

The result supports the sample-pool boundary: the route-relevant population matters. Common subtraction helps some aspects of all-position clustering, but it leaves weaker joint separation and negative rare-margin lower tail.

### Training Preservation

All conditions learn the synthetic target by step 160, so target accuracy cannot decide specialization. Final results:

| Condition | Rare-feature NMI | Common/rare NMI | Joint feature score | Max load | Rare margin p05 | Target acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random raw | 0.903 | 0.526 | 0.472 | 0.438 | -0.955 | 1.000 |
| random common-subtracted | 0.881 | 0.415 | 0.371 | 0.516 | -1.033 | 1.000 |
| all-position common-subtracted | 0.733 | 0.615 | 0.432 | 0.508 | -5.427 | 1.000 |
| route-position raw | 0.948 | 0.658 | 0.620 | 0.305 | 5.227 | 1.000 |
| route-position residual input | 1.000 | 0.355 | 0.355 | 0.444 | 9.144 | 1.000 |
| oracle raw | 1.000 | 0.637 | 0.637 | 0.250 | 8.807 | 1.000 |
| oracle residual input | 1.000 | 0.376 | 0.376 | 0.427 | 8.924 | 1.000 |
| oracle row-projected | 1.000 | 0.636 | 0.636 | 0.250 | 8.646 | 1.000 |

The refined preservation result is important: residual input protects rare-rare separation and rare margins, but it can let common-vs-rare separation degrade. Row projection preserves the full common/rare plus rare-feature partition better in this setup.

### Position Guard

Step-0 slot-start NMI is low in all conditions, with maximum mean `0.024`. This supports the no-position guard: the main result is not explained by explicit slot-start routing.

## Key Figures

### Figure: Step-0 Joint Feature Score

![Step-0 joint feature score](figures/step0_joint_feature_score_by_condition.png)

What this tests: whether a condition separates rare features and also separates common from rare before training.

Anchor question: Does common subtraction create feature-level separation?

Protocol question: Do all-position common-subtracted centers match route-position or oracle centers?

Metric shown: joint feature score.

Metric definition: `rare_feature_NMI * common_rare_NMI`.

Metric unit: dimensionless score.

Data source: `tables/condition_aggregate_step0.csv`.

Aggregation level: mean over 8 seeds and 4 slot lengths.

How to read: higher means stronger combined common/rare and rare-rare separation.

Expected if supported: route-position/oracle conditions exceed all-position common-subtracted conditions.

Expected if weakened or incomplete: all-position common-subtracted matches route-position/oracle.

What this figure decides: common subtraction does not close the gap to route-relevant centers at step 0.

Observed result: all-position common-subtracted is `0.405`; route-position residual and oracle are `0.637`.

Allowed claim: route relevance remains necessary in this synthetic surface.

What this does not prove: real-DCLM transfer or semantic feature discovery.

Anchor update implication: keep common subtraction parked as a standalone separator.

### Figure: Final Joint Feature Score

![Final joint feature score](figures/final_joint_feature_score_by_condition.png)

What this tests: whether the partition survives training.

Observed result: oracle raw and oracle row-projected remain near `0.637`; route-position raw remains high at `0.620`; all-position common-subtracted remains lower at `0.432`.

Take-home: target learning does not imply clean common/rare feature routing.

### Figure: Final Rare Margin Lower Tail

![Final rare margin p05](figures/final_rare_margin_p05_by_condition.png)

What this tests: whether rare examples remain inside a positive routing basin.

Observed result: all-position common-subtracted has strongly negative rare margin p05 (`-5.427`), random variants are also negative, while route-position raw (`5.227`) and oracle/row-projected conditions stay positive.

Take-home: common subtraction can leave rare-feature routing fragile even when target accuracy is perfect.

### Figure: Slot-Start NMI Guard

![Step-0 slot-start NMI guard](figures/step0_slot_start_nmi_guard.png)

What this tests: whether route assignments mainly track slot starts.

Observed result: maximum mean step-0 slot-start NMI is `0.024`.

Take-home: the main result is not a visible slot-start shortcut.

## Claim Boundary

Can claim:

- In this synthetic no-position Transformer-MoE surface, simple common subtraction is not a reliable feature separator.
- Route-position centers and oracle centers produce much stronger common/rare plus rare-feature separation than all-position common-subtracted centers.
- Target accuracy alone is not evidence of specialization.
- Row projection is a better preservation candidate than residual input when the desired claim includes common-vs-rare separation, not only rare-rare separation.

Cannot claim:

- real-DCLM transfer;
- semantic expert formation;
- utility of the specialized experts;
- that every task-aware common operator fails;
- that residual input is useless, because it still preserves rare-rare separation and rare margins.

## Next Decision

Use A06_24_synthetic as the boundary-setting experiment: park simple common subtraction as the next main method. The next anchor should test a task-aware route-relevant state selector or a row-projected preservation update on a harder bridge, with joint feature score and rare margin as primary guards.
