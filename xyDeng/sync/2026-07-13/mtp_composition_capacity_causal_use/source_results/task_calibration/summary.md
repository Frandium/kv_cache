# Summary: A11_23 Composition Calibration

Primary anchor: `../../../problem_anchors/11_long_horizon_mtp_objective/11_23_composition_calibration_anchor.md`
Protocol: `protocol.md`

## Result Snapshot

**Verdict:** weakened. Depth, training duration, and width do not repair the original serialization.
**What we established:** `2L-64d-600`, `2L-64d-3000`, `3L-64d-3000`, and `3L-128d-3000` all remain near chance.
**What the experiment shows:** the strongest condition has train accuracy `0.1249` and validation accuracy `0.1211`, so the blocker precedes generalization.
**What we do next:** isolate within-row/key-value binding rather than add more scale.

## Terminology / Definitions

Calibration means learning eligibility, not MTP benefit. Chance is `0.125`; pass threshold is the operational value `0.70`.

## Exact Setup

Globally disjoint target permutations; train 8192, validation 1024, test 2048; exact 155-token serialization; AdamW `3e-4`; seed-971 bounded ladder.

## Primary Metric And Result

Maximum validation answer accuracy across the ladder is below `0.130`. For `3L-128d-3000`, train answer accuracy is `0.1249`, validation `0.1211`, while format accuracy is `1.0`.

## Claim Boundary

The result rejects the bounded depth/time/width repair. It does not show that all Transformers cannot compose relations.

## Next Decision

Test an explicit-binding serialization under A11_23b.
