# Summary: A06_08 Label-Free Feature Discovery Initialization

Primary anchor:
`../../problem_anchors/06_geometry_proxy_preservation/06_08_label_free_feature_discovery_initialization_anchor.md`

Protocol:
`protocol.md`

## Purpose

A06_08 tests whether label-free feature discovery can recover pseudo-feature centers from initialized hidden states when simple global controls have already failed. This is a step-0 routing replay, not a training experiment.

## Exact Setup

- synthetic A06 replay surface with four strictly uniform `(slot,target)` pair features;
- `slot_token_len=4`, repeated `SLOT_s` span, route position at the last slot token;
- four experts, no training;
- seeds `20260521..20260528`;
- depths `num_layers in {1,2,4}`;
- readouts `post_ln` and `pre_ln_replay`;
- 4096 samples per pair, split 2048 / 2048 calibration/evaluation per pair.

Validity audit:
all route-position rows are valid, all split rows are balanced, and no label-free discovery condition uses `pair_id` for construction.

## Primary Metric

Primary metric:
held-out `feature_NMI = NMI(pair_id, routed_expert)`.

Why it decides:
A06_07 already showed that load can improve without feature specialization. Therefore A06_08 passes only if a label-free method improves feature_NMI and closes at least half of the oracle gap.

Pass gate:
`NMI_delta_vs_baseline >= 0.25`, `oracle_gap_fraction >= 0.50`, no leakage, and no many-to-one feature merge.

## Result

A06_08 supports label-free feature discovery in this synthetic initialized replay setting.

K-means and spherical k-means on common-centered route-position residuals recover the oracle feature partition perfectly across the full grid: 48/48 cells pass for each method, with mean `feature_NMI=1.0` and load $L=0.0`.

Spectral clustering and dictionary learning are weaker: they sometimes improve NMI, but they are not stable enough to be the main method family.

Primary readout (`num_layers=1`, `post_ln`, mean over 8 seeds):

| Condition | Feature NMI | Oracle gap | Load $L$ | Pass cells |
| --- | ---: | ---: | ---: | ---: |
| baseline raw | 0.1894 | 0.0000 | 0.5767 | 0/8 |
| calibration mean | 0.2072 | 0.0199 | 0.2691 | 0/8 |
| k-means residual K4 | 1.0000 | 1.0000 | 0.0000 | 8/8 |
| spherical k-means residual K4 | 1.0000 | 1.0000 | 0.0000 | 8/8 |
| spectral residual K4 | 0.5482 | 0.4280 | 0.4666 | 1/8 |
| dictionary residual K4 | 0.4723 | 0.3343 | 0.4944 | 0/8 |
| oracle centroid | 1.0000 | 1.0000 | 0.0000 | upper bound |

Across all depths/readouts, k-means and spherical k-means pass 48/48 cells each. Spectral passes 13/48 and dictionary K4 passes 8/48.

## Key Figures

### Figure: Feature NMI By Method

![Feature NMI by method](figures/nmi_by_method.png)

What this tests:
whether label-free discovery methods move routing from random/global-control NMI toward the oracle feature partition.

Observed result:
k-means and spherical k-means reach the oracle level. Spectral and dictionary are partial and unstable.

Allowed claim:
in this synthetic initialized hidden-state surface, route-position residual clusters are label-free recoverable.

What this figure does not prove:
training stability, real-text transfer, expert utility, or that every label-free discovery family works.

### Figure: Load Versus Oracle Gap

![Load versus oracle gap](figures/load_oracle_gap_by_method.png)

What this tests:
whether a method is only balancing load or actually closing the feature-NMI gap.

Observed result:
global calibration mean improves load but closes only about 2% of the oracle gap. K-means methods close the full gap and also balance load.

Allowed claim:
the successful clustering result is not merely a load-balancing artifact.

### Figure: Route Heatmap By Method

![Route heatmap by method](figures/route_heatmap_by_method.png)

What this tests:
whether features are assigned one-to-one to experts rather than merged.

Observed result:
k-means and spherical k-means show clean one-to-one routing; spectral and dictionary are mixed.

Allowed claim:
the passing clustering methods avoid the many-to-one merge failure in the tested grid.

## Claim Boundary

Can claim:
label-free route-position clustering can recover pseudo-feature centers in the synthetic initialized A06 hidden-state setting with `slot_token_len=4`.

Cannot claim:
ordinary training preserves this partition, real DCLM behavior, semantic feature discovery, expert utility, or that spectral/dictionary methods are reliable here.

## Next Decision

Proceed to A06_09. Use `route_kmeans_residual_K4` as the best A06_08 pseudo-center initialization for the conditional pseudo-init training run.
