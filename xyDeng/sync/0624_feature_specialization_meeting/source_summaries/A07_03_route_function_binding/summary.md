# Summary: A07_03 Route-Function Binding

## Purpose

Test whether the A07_02 rare-loss gain is backed by conflict-conditioned expert utility.

## Conclusion

Supported for controlled synthetic D07, with one important reading rule: raw expert IDs permute across seeds, so use modal-aligned route and ablation metrics for the main judgment.

Primary metric: route-conditioned utility advantage is about `0.950` for each conflict group.

## Exact Setup

Run: `a07_common_rare_conflict_full_20260623_1`

Seeds: `20260623` to `20260630`.

Inputs: A07_02 common-control route assignments, expert IDs, conflict groups, forced-routing utility matrix, fixed-route ablation.

## Key Evidence

| Check | Result | Judgment |
| --- | ---: | --- |
| selected-best rate | `1.0000` | pass |
| chance baseline | `0.2500` | pass |
| common-control modal route probability | `1.0000` per group | pass |
| route-conditioned utility advantage | `0.9498-0.9502` | pass |
| modal ablation delta | `0.6800` | pass |
| nonmodal ablation delta | `0.0000` | pass |
| early utility advantage | `0.7408-0.7411` | pass |

## Central Figure

![A07_03 route-function heatmap](figures/route_function_binding_heatmap.png)

This heatmap shows forced-routing loss by conflict group and raw expert ID. Because expert IDs permute across seeds, the figure is auxiliary. The decisive evidence is the modal-aligned selected-best, route-conditioned utility, and modal ablation tables.

## Claim Boundary

This supports controlled D07 route-function binding. It does not claim stable semantic expert IDs across seeds, real-language experts, or one-feature-one-expert structure.

## Next Decision

Decide whether to port this diagnostic to a real trained checkpoint or first run a neural D07 training version.
