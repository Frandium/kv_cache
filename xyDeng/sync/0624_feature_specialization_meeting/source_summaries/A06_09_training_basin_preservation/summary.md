# Summary: A06_09 Training Basin Preservation

Primary anchor:
`../../problem_anchors/06_geometry_proxy_preservation/06_09_training_basin_preservation_anchor.md`

Protocol:
`protocol.md`

## Purpose

A06_09 tests whether a feature-aligned router initialization is preserved by ordinary top-1 MoE training. This is the training-dynamics follow-up to A06_08: A06_08 asks whether label-free discovery can find pseudo-feature centers; A06_09 asks whether those centers remain useful after training begins.

## Exact Setup

- synthetic four-feature task with strictly uniform pair features;
- `slot_token_len=4`, repeated `SLOT_s` span, route position at the last slot token;
- four experts and four feature pairs;
- route-position target prediction objective: logits at the last slot token predict the following `TARGET_s`;
- seeds `20260521..20260528`;
- depths `num_layers in {1,2,4}`;
- checkpoints `0, 100, 400, 800, 1600`;
- full run on four ACP GPUs, job id `pt-epdfv6pd`;
- A06_08 dependency selected `route_kmeans_residual_K4`.

The protocol uses route-position target prediction because random background tokens are not learnable. Full-sequence next-token loss is therefore not the primary evidence for this anchor.

## Primary Metric

Primary metric:
held-out `feature_NMI(t) = NMI(pair_id, routed_expert)` through training.

Why it decides:
the question is not whether the model can learn the synthetic target. The question is whether a feature-level routing partition survives training. Accuracy is only a constraint.

Support gate:
final `feature_NMI >= 0.80`, NMI drop no more than `0.15`, and target accuracy at least `0.95`.

## Result

A06_09 supports the training-basin hypothesis in this synthetic setting.

Oracle feature-centroid initialization and A06_08 pseudo-center initialization both preserve perfect feature routing across the full grid: `24/24` supported cells for each condition, final `feature_NMI=1.0`, final target accuracy `1.0`, and final load $L=0.0$.

Random, equal-norm random, and label-shuffled guards also learn the target, but they do not provide clean supported specialization. Their final NMI can improve during training, but the final route partition is less reliable and has nonzero load imbalance.

Final aggregate by condition across all depths and seeds:

| Condition | Step-0 NMI | Final NMI | Target acc. | Load $L$ | Supported cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| A06_08 pseudo center | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 24/24 |
| oracle centroid | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 24/24 |
| random Gaussian | 0.2399 | 0.8100 | 1.0000 | 1.1277 | 0/24 |
| equal-norm random | 0.2394 | 0.8027 | 1.0000 | 1.1667 | 0/24 |
| label-shuffled centroid guard | 0.4969 | 0.8409 | 1.0000 | 1.0000 | 0/24 |

## Key Figures

### Figure: NMI Trajectory

![Feature NMI trajectory](figures/nmi_trajectory.png)

What this tests:
whether initial feature routing is preserved through training.

Observed result:
oracle and A06_08 pseudo init remain at `feature_NMI=1.0` from step 0 to step 1600. Random conditions improve but do not become clean supported feature partitions.

Allowed claim:
in this synthetic route-position training setting, feature-aligned initialization defines a stable training basin.

What this figure does not prove:
real-text transfer, full next-token training stability, or semantic expert specialization.

### Figure: Accuracy Trajectory

![Target accuracy trajectory](figures/accuracy_trajectory.png)

What this tests:
whether high NMI is achieved while the model still learns the task.

Observed result:
all conditions reach target accuracy `1.0`. Therefore accuracy alone cannot distinguish clean specialization from weaker or imbalanced routing.

Allowed claim:
the oracle and pseudo initializations preserve routing without sacrificing the synthetic target objective.

### Figure: Route Heatmaps

![Route heatmaps](figures/route_heatmaps_step0_final.png)

What this tests:
whether feature groups remain one-to-one with experts rather than merging.

Observed result:
oracle and pseudo init keep clean one-to-one routing and load $L=0.0$. Random and guard conditions show less reliable partitions.

Allowed claim:
the supported conditions preserve a clean feature-to-expert partition in the tested grid.

## Claim Boundary

Can claim:
feature-aligned initialization alone preserves specialization through route-position target training in the synthetic uniform `slot_token_len=4` setting. The A06_08 k-means residual pseudo-centers are good enough to enter the same basin as oracle centers here.

Cannot claim:
real DCLM behavior, full next-token training stability, semantic feature discovery, expert utility, deployable router design, or that anti-lockin is solved in harder settings.

## Next Decision

Continue the initialization route. The next natural test is to weaken the feature proxy and training assumptions: less controlled feature discovery, more realistic objectives, or real-DCLM proxy metrics. Anti-lockin should remain parked until preservation fails in a less controlled setting.
