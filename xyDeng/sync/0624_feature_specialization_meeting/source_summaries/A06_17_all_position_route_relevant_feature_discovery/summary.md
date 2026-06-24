---
experiment_id: A06_17_all_position_route_relevant_feature_discovery
anchor_id: A06_17_all_position_route_relevant_feature_discovery
status: result_updated
run_name: a06_17_route_relevant_pool_audit_full_20260623_1
job_id: pt-9mt77o49
canonical_language: en
---

# A06_17 Summary: Route-Relevant States Are Required For Feature Discovery

## Conclusion

A06_17 supports the sample-set mismatch explanation. The feature geometry is clean at the route position, but all-position clustering is not a reliable proxy for route-relevant feature initialization.

Primary metric: held-out route-position `feature_NMI`.

Main result:

| Fit pool | Mean route `feature_NMI` | Min | Perfect seeds | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Oracle route upper bound | 1.000 | 1.000 | 8/8 | route feature geometry exists |
| Route-only pseudo | 1.000 | 1.000 | 8/8 | label-free discovery works when the right state pool is known |
| Slot offset 3, the route position | 1.000 | 1.000 | 8/8 | the last slot token is the stable route-relevant state |
| All positions | 0.797 | 0.637 | 1/8 | all-position clustering often merges complete features |
| Slot mixed | 0.900 | 0.866 | 2/8 | mixing all slot roles is still not enough |
| Role-balanced all-position | 0.854 | 0.637 | 1/8 | fixing role counts alone does not solve role-geometry mixing |
| Non-slot no-target | 0.664 | 0.346 | 0/8 | feature information leaks outside the slot, but not as a clean route partition |

Decision: do not use all hidden states as the feature discovery object. The next anchor should build a label-free route-relevant state selector, then run clustering only on selected route-like states.

## Key Evidence

![A06_17 pool feature NMI](figures/pool_feature_nmi.png)

What this figure tests: whether different hidden-state pools can produce centers that route held-out slot-end states by feature.

How to read it: higher `feature_NMI` means the fitted centers recover the feature partition at the route position. Orange dots are seeds.

Observed result: oracle, route-only, and slot offset 3 are always perfect. All-position, mixed slot, role-balanced all-position, and non-slot pools are seed-sensitive and often collapse multiple features into one expert.

Take-home: the route-position feature signal exists, but the center-fitting population must be route-relevant.

![A06_17 feature NMI by seed and pool](figures/pool_feature_nmi_heatmap.png)

What this figure tests: whether the result is stable across seeds.

Observed result: route-only and slot offset 3 are stable across all seeds. All-position fails in 7/8 seeds by the strict perfect-partition criterion, with mean `feature_NMI=0.797`.

What it does not prove: it does not identify a real-language state selector; it only shows which controlled hidden-state population is valid in this bridge.

## Failure Mode

All-position failure is not nuisance alignment or slot-start alignment. In all-position rows, `nuisance_NMI=0` and `slot_start_NMI=0`.

The failure is whole-feature merge. All-position keeps `feature_purity=1.0`, but uses too few experts on route evaluation: mean active experts is 2.75, with three seeds using only two experts.

The merged features are not consistently the closest route-feature centroids. The all-position merged pair distance ranks range from 1 to 6. This weakens the explanation that route feature geometry itself is insufficient.

All-position k-means is dominated by non-route states: 81.2% of its fit pool is neutral states, while route states are only 3.1%. Role-balanced sampling improves the route-state fraction to 25%, but still fails in 7/8 seeds. Therefore the issue is not only count imbalance; mixing different position roles in one k-means objective is itself harmful.

## Claim Boundary

Can claim: in the corrected no-pos A06_16 bridge, route-position residual k-means discovers feature centers perfectly, while all-position clustering is not a reliable route-initialization proxy.

Cannot claim: real-DCLM semantic feature discovery, RoPE behavior, whole-slot compositional semantics, or a final deployable router initializer.

## Next Decision

Next anchor: build and test a label-free route-relevant state selector. Plainly: first identify which hidden states are likely to be used for routing, then cluster those states; do not cluster every hidden state together.
