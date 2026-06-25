# Report Card: A06_18 Revision Audit

## One-Line Claim

Representation-only selector revision did not solve A06_18: route geometry is
present, but PCA/AE/SAE clustering does not recover route-position features
reliably from all-position states.

## Decision

**Status:** weakened / not passed.

Do not create A06_19 from this result alone. Do not promote SAE as the main
selector. Do not run the real-DCLM touchpoint from these selectors.

## Primary Metric

Held-out route-position `feature_NMI`.

## Key Numbers

| Pool | Mean `feature_NMI` | Perfect seeds |
|---|---:|---:|
| Route-only | 1.000 | 8/8 |
| Slot offset 3 | 1.000 | 8/8 |
| Raw all-position | 0.831 | 2/8 |
| PCA q=4 | 0.871 | 2/8 |
| Bottleneck AE q=32 | 0.814 | 2/8 |
| SAE L1 8x | 0.729 | 0/8 |
| SAE top-k 4x | 0.620 | 0/8 |

## Interpretation

Route-position geometry exists because route-only and slot-offset-3 controls
are perfect.

PCA q=4 gives only a weak and unstable mean gain over all-position. It does
not approach the positive controls.

SAE reconstruction is not route relevance: SAE L1 8x has reconstruction MSE
`0.0034` but only `feature_NMI=0.729`.

## Updated Belief

Generic representation learning cannot replace a route-relevant selector.
The next selector must include an explicit route-local or route-readout
constraint.

## Next Step

Build a route-readout-constrained selector. Only after it approaches
route-only in controlled A06 should it be tested in slot early training and
promoted into a short A06_19.
