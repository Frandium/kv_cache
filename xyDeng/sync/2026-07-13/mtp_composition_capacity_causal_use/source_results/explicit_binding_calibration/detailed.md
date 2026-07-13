# Detailed: A11_23b Explicit-Binding Calibration

## 0. Quick Recap

Purpose: remove the identified within-row binding blocker. Hypothesis: explicit pair tokens enable two-layer composition. Conclusion: weakened in 5/5 seeds.

## 1. Results

Test answer accuracy by seed: `0.1289, 0.1294, 0.1353, 0.1274, 0.1279`. Both one-table controls remain near chance, but the full task also remains near chance. Format accuracy is `1.0` throughout.

## 2. Interpretation

The failure is not explained by cross-split reuse or three-token row binding. The remaining sequence-level problem is algorithmic composition/inductive bias. More ad hoc sequence tuning is parked.

## 3. Artifact Map

Runner: `active/synthetic_data_understanding/scripts/run_a11_23b_explicit_binding_calibration.py`; raw result: `active/synthetic_data_understanding/results/a11_23b_full/`; curated table: `tables/confirmation.csv`; figure: `figures/curves.png`; local GPU run.
