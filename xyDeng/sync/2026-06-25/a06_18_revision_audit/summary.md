# Summary: A06_18 Revision Audit

Primary anchor:
`../../../problem_anchors/06_geometry_proxy_preservation/06_18_label_free_route_relevant_state_selection_anchor.md`

Protocol: `protocol.md`

## Purpose

Test whether label-free representation clustering can fix the A06 all-position
sample-pool failure.

## One-Sentence Result

No revised selector passed: PCA q=4 weakly improves mean `feature_NMI`, but no
representation variant approaches the route-only / slot-offset-3 controls.

## Setup

- Data: A06_17 `C4_all_position_scope` controlled no-position bridge.
- Model: one-layer Transformer plus top-1 MoE from A06_16.
- Route position: slot offset 3.
- Features / experts: 4 / 4.
- Seeds: `20260623` to `20260630`.
- Evaluation: held-out route-position states.
- Representation rule: fit without feature labels, cluster in latent/code
  space, average original hidden states per latent cluster into gating centers.
- Local execution: two H100 GPUs, two shards.

## Primary Metric

Held-out route-position `feature_NMI`.

## Main Result

Decision: weakened / not passed.

| Pool | Mean `feature_NMI` | Min | Max | Perfect seeds | Mean max load |
|---|---:|---:|---:|---:|---:|
| Route-only | 1.000 | 1.000 | 1.000 | 8/8 | 0.250 |
| Slot offset 3 | 1.000 | 1.000 | 1.000 | 8/8 | 0.250 |
| PCA q=4 | 0.871 | 0.637 | 1.000 | 2/8 | 0.469 |
| PCA q=16 | 0.851 | 0.637 | 1.000 | 2/8 | 0.469 |
| Raw all-position | 0.831 | 0.637 | 1.000 | 2/8 | 0.469 |
| Bottleneck AE q=32 | 0.814 | 0.637 | 1.000 | 2/8 | 0.531 |
| Split-stability top-3 | 0.778 | 0.587 | 1.000 | 1/8 | 0.452 |
| SAE L1 4x | 0.749 | 0.637 | 0.866 | 0/8 | 0.562 |
| SAE L1 8x | 0.729 | 0.000 | 0.866 | 0/8 | 0.594 |
| SAE top-k 8x | 0.641 | 0.492 | 0.811 | 0/8 | 0.695 |
| SAE top-k 4x | 0.620 | 0.400 | 0.820 | 0/8 | 0.695 |

Full table: `tables/pool_comparison_aggregate.csv`.

## Interpretation

Route geometry is still present: route-only and slot-offset-3 controls remain
perfect.

PCA gives a small, unstable gain over raw all-position: PCA q=4 improves mean
`feature_NMI` by only `+0.040`, improves 4/8 seeds, and worsens 3/8 seeds.
This is not close to the positive controls.

SAE reconstruction is not the right objective. SAE L1 8x reaches very low
reconstruction MSE (`0.0034`) but has mean `feature_NMI=0.729`; top-k SAE is
sparser but more load-collapsed and worse.

## What Updated

Supported:

- A06_17's sample-pool mismatch diagnosis remains right.
- Generic representation learning does not by itself define route relevance.
- Reconstruction quality is not a sufficient proxy for route-readout quality.

Not supported:

- SAE-code clustering as the next main selector.
- PCA/AE latent clustering as a sufficient route-relevant pool selector.

## Next Decision

Do not create A06_19 from this result alone. The next selector needs an
explicit route-readout or route-local constraint. Candidate direction:
score candidate centers by held-out route-position readout on an unlabeled
split, then test the winning initialization in slot early training.
