# Summary: A11_23b Explicit-Binding Calibration

## Result Snapshot

**Verdict:** weakened. Atomic role-specific pair tokens still do not produce learned composition.
**What we established:** test answer accuracy is `0.1274--0.1353` across five seeds after 3000 steps; format accuracy is `1.0`.
**What the experiment shows:** removing within-row token binding is insufficient; the from-scratch Transformer still does not acquire algorithmic composition.
**What we do next:** stop tuning the sequence learner and isolate rank competition on oracle-provided factors.

## Exact Setup

Two-layer width-64 Transformer; 16 role-specific mapping tokens plus query/format/decision; globally disjoint train/validation/test permutations; five seeds; 3000 steps.

## Key Figure

![Explicit-binding calibration curves](figures/curves.png)

The curves remain around chance rather than approaching the `0.70` threshold.

## Claim Boundary

This weakens pair-token serialization as a calibration repair. It does not falsify relation composition in pretrained or explicitly structured models.
