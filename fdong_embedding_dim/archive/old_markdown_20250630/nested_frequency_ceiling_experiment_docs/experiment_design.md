# Experiment Design

## Factorial conditions

Primary grid:

| Objective | Moon sharing |
|---|---|
| `zipf_raw` | shared / split |
| `uniform_raw` | shared / split |
| `zipf_reweight` | shared / split |

Run matched seeds and initialization. The main dimension is 8. Dimension 16 checks whether conclusions survive additional unused capacity. Dimension 3 is optional and is not allowed to support a claim unless its baseline reaches the Bayes convergence thresholds.

## Optimization contract

- exact population loss every step;
- Adam optimizer;
- no weight decay in the first run;
- identical matrix initialization across all conditions for a seed;
- `moon_H` and `moon_T` receive identical initial vectors in split mode;
- evaluate every 10 steps;
- run 500 steps in the first sweep; the smoke run converged before step 100, and both first-stable and fixed-final checkpoints are saved;
- do not early-stop individual conditions in the primary matched-step comparison.

## Convergence

A family is stable at the first evaluation where its metric remains below threshold for five consecutive evaluations:

- `the_bayes_gap <= 0.03`;
- `a_bayes_gap <= 0.03`;
- `cake_loss <= 0.03`.

Overall convergence requires all three simultaneously. Also report cumulative pattern exposure at the convergence step.

The thresholds are operational tolerances, not theoretical constants. Curves and final values are retained so conclusions do not depend only on threshold crossings.

## Required comparisons

### Frequency test

Compare `zipf_raw` against `uniform_raw` with moon sharing fixed.

Support for frequency-driven inefficiency requires:

- slower cake/tail convergence in `zipf_raw`;
- smaller frequency-weighted tail gradient norm in `zipf_raw`;
- comparable raw per-pattern gradient norms after controlling for current loss;
- `zipf_reweight` tracking `uniform_raw` under matched initialization.

If reweight does not recover uniform learning, simple gradient-share insufficiency is incomplete.

### Nested-sharing test

Compare shared versus split moon with objective fixed.

Support for “nested changes geometry but not speed” requires:

- shared moon produces greater spectral concentration or common-direction projection;
- shared/split convergence steps remain similar relative to the much larger frequency effect;
- any speed difference is small and inconsistent across objectives/seeds.

If splitting moon materially accelerates the tail under fixed weights, nested parameter sharing contributes to inefficiency.

### Direction-usage test

At convergence, compare common-only and residual-only cake losses.

- common reuse: common-only retains low cake loss or causes less damage than residual-only;
- new-direction usage: residual-only retains low cake loss or carries most tail contrast energy;
- mixed usage: neither ablation is sufficient.

This result describes functional usage, not whether the representation spectrum looks visually flat.

## Stage-level evidence

Save:

- `history.csv`: family losses, Bayes gaps, accuracies, spectra, common projection, causal ablations;
- `gradient_history.csv`: raw and weighted gradient norm/common/residual components by pattern;
- `summary.csv`: first stable steps, final spectra, final causal metrics;
- `aggregate_summary.csv`: seed aggregates;
- `learning_curves.png`;
- `spectral_curves.png`;
- `gradient_contributions.png`;
- `representation_geometry.png`.

## Failure and insufficient-evidence conditions

The run is insufficient if:

- dimension/capacity prevents uniform conditions from reaching the Bayes thresholds;
- `zipf_reweight` and `uniform_raw` do not have numerically matching effective objectives at initialization;
- split moon aliases are not initialized identically;
- a claimed common direction has negligible high-pattern energy;
- conclusions change sign across most seeds.

Potential falsifications:

- nested sharing does not measurably sharpen any parameter or representation spectrum;
- split moon is consistently much faster at matched frequency;
- weighted tail gradient is not smaller in Zipf;
- uniform/reweight does not improve tail convergence;
- tail prediction is primarily residual-only even in the fastest, flattest condition.
