# Detailed: A14_E07 Reachable-Space Accounting

Primary package:
[A14 probabilistic sparse-activation tree handoff](../../README.md)

Protocol: [protocol.md](protocol.md)
Summary: [summary.md](summary.md)

## 0. Quick Recap

Purpose: audit the exact implementation of the refined reachable-space
dimension recurrence.

Conclusion: pass. The maximum registered prediction-versus-measurement
mismatch was 0 dimensions over 950 records.

Boundary: synthetic mathematical operationalization only.

## 1. Protocol Compliance Audit

| Check | Result |
| --- | --- |
| Approved conditions match actual conditions | pass: 190 conditions across five rotations |
| Primary metric exists | pass: maximum absolute dimension mismatch |
| Central figures exist | pass: equality plot and mechanism-growth plot |
| Seeds recorded | pass: rotations 0--4 |
| Known good/bad/confusing cases reviewed | pass |
| Decision rule applied | pass: required and observed mismatch both 0 |

## 2. Setup

- Ambient space: $\mathbb R^{512}$.
- Exact coordinate subspaces were constructed and then globally rotated.
- Branching factors: 2, 3, and 4.
- Leaf dimensions: 4 or 8 in growth regimes.
- Arithmetic and rank measurement: float64 SVD with the registered adaptive
  tolerance and spectral-gap guard.
- Growth regimes: worst expansion, compression only, within-layer reuse only,
  within- plus cross-layer reuse, and bounded new directions.
- Activation regimes: full, deficient, and shape-limited.
- Runtime: 13.57 seconds.

No text, neural model, optimization, noise, or pretrained checkpoint was used.

## 3. Main Results

### Decision evidence

- Registered records: 950.
- Maximum absolute dimension mismatch: 0.
- Construction failures: 0.
- Coordinate-oracle/SVD mismatches: 0.
- Rotation-invariance failures: 0.
- Spectral-gap failures: 0.

### Negative controls

Seven checks were registered against deliberately wrong accounting rules. All
seven rejected the wrong rule; failures: 0.

### Activation evidence

- Full activation equality: 20/20 records.
- Deficient activation strict gap: 20/20 records.
- Shape-limited activation strict gap rate: 1.0.

This separates mechanism reachability from the rank observed in a finite
sample matrix.

### Representative growth endpoints

For a binary tree with leaf dimension 4 and depth 5:

| Regime | Final layer dimension | Cumulative global dimension |
| --- | ---: | ---: |
| Worst expansion | 128 | 252 |
| Compression only | 4 | 24 |
| Within-layer reuse only | 4 | 24 |
| Within- and cross-layer reuse | 4 | 4 |
| Bounded new directions | 9 | 9 |

The compression-only and within-layer-only cases have fixed layer dimension
but growing global dimension. Cross-layer reuse is the additional mechanism
that keeps the global dimension fixed.

## 4. Failure Decomposition

Falsified implementation: none.
Falsified mathematical model: none within the registered exact cases.
Falsified physical prior: none; no language prior was tested.
Remaining empirical uncertainty: whether any real language representation has
small role-image dimensions and substantial within- or cross-layer reuse.

## 5. Visualization Results

### Predicted versus measured dimensions

![Predicted versus measured dimensions](figures/predicted_vs_measured_dimension.png)

Purpose: verify exact implementation agreement. Every point must lie on the
equality line; every registered point did. This cannot prove the theorem or a
language property.

### Mechanism growth curves

![Mechanism growth curves](figures/mechanism_growth_curves.png)

Purpose: distinguish mechanisms that can have the same layer rank but different
global rank. The observed trajectories match the coordinate oracle. This does
not identify which trajectory describes a Transformer.

## 6. Claim Boundary And Next Decision

The run establishes only that the synthetic implementation realizes the exact
accounting and its activation boundary. The model-general real-language
shared-linear propagation claim remains closed.

Exactly one next decision: approve or reject a researcher judgment record for constrained
conditional local low-rank composition. No protocol is yet authorized.

## 7. Artifact Map

- Runner:
  `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/tree_low_rank/scripts/run_reachable_space_accounting.py`
- Config:
  `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/tree_low_rank/configs/exp7_reachable_space_accounting.yaml`
- Result directory:
  `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/tree_low_rank/outputs/A14_E07_full/`
- Machine summary:
  `Projects/from-attention-to-search/XingyuD/Attention_Search_Experiments/active/tree_low_rank/outputs/A14_E07_full/summary.json`
- Primary table: `recurrence_audit.csv`.
- Activation table: `activation.csv`.
- Growth table: `growth_curves.csv`.
- Manifest: `run_manifest.json`.
