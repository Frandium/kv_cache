# Detailed: A11_23 Composition Calibration

## 0. Quick Recap

**Purpose:** calibrate a learned composition checkpoint.
**Hypothesis:** depth-first bounded repair reaches `0.70`.
**Conclusion:** weakened; all conditions remain at chance, including train fit.
**Evidence:** `tables/screening_curves.csv`, `tables/train_fit_diagnostic.json`.

## 1. Protocol Compliance

All four ladder stages ran in order; global target-permutation overlap is zero; no configuration passed screening, so five-seed confirmation was correctly not opened. The post-run train-fit diagnostic is explanatory and does not change the verdict.

## 2. Results

| Condition | Final validation answer | Maximum validation answer |
|---|---:|---:|
| 2L-64d-600 | 0.1250 | 0.1299 |
| 2L-64d-3000 | 0.1250 | 0.1299 |
| 3L-64d-3000 | 0.1250 | 0.1279 |
| 3L-128d-3000 | 0.1211 | 0.1289 |

The final train-fit diagnostic for `3L-128d-3000` is answer accuracy `0.1249`, format accuracy `1.0`, and answer loss `2.1173`. Failure is not merely held-out generalization.

## 3. Failure Decomposition

Depth-only, longer-training, and width repairs are weakened. Remaining causes are weak key-value/role binding in serialization or lack of an appropriate algorithmic inductive bias.

## 4. Boundary And Artifacts

Runner: `active/synthetic_data_understanding/scripts/run_a11_23_composition_calibration.py`. Raw result: `active/synthetic_data_understanding/results/a11_23_calibration_full/`. Local GPU run; no job id.
