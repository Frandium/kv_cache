# Summary: A11_24 Task-Rank Competition

## Result Snapshot

**Verdict:** supported in the stated linear oracle-factor model.
**What we established:** rank 1 creates a reproducible MTP standard-loss penalty; rank 2 removes it.
**What the experiment shows:** the penalty depends on an explicit representation bottleneck rather than parameter count alone.
**What we do next:** test whether curriculum can Pareto-improve the rank-1 tradeoff.

## Definitions

The standard objective predicts independent factors $u$ and $z$ with weights 2 and 1. MTP adds a second weight-2 prediction of $z$. The rank intervention changes only the shared encoder rank.

## Primary Result

At rank 1, MTP-minus-NTP standard-loss gaps are `0.9084--1.0218` in five seeds, and MTP improves future-factor MSE in `5/5`. At rank 2, the gap is approximately zero (`|gap| <= 1.23e-13`).

## Key Figure

![Rank-conditioned MTP interference](figures/rank_phase.png)

The MTP penalty exists only when one shared direction must serve two independent factors.

## Claim Boundary

This is a controlled linear rank-causality result, not evidence that Transformer size or parameter count explains MTP scaling.
