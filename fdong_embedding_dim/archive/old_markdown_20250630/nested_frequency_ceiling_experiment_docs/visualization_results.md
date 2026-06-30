# Visualization Results

Expected artifacts:

- `results/history.csv`;
- `results/gradient_history.csv`;
- `results/summary.csv`;
- `results/aggregate_summary.csv`;
- `results/learning_curves.png`;
- `results/spectral_curves.png`;
- `results/gradient_contributions.png`;
- `results/representation_geometry.png`.

## Reading contract

`learning_curves.png` compares Bayes gaps and deterministic cake loss. It must not present top-1 accuracy on ambiguous prefixes as if 100% were attainable.

`spectral_curves.png` separates parameter spectra from macro contextual-representation spectra. A training-weighted spectrum is shown separately because repeated Zipf rows mechanically concentrate covariance.

`gradient_contributions.png` separates raw per-pattern gradient norm from the frequency-weighted contribution used by the optimizer.

`representation_geometry.png` reports contextual similarities, common projection mass, and common-only/residual-only cake loss. It is the causal usage test.

## Current run

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fdong_embedding_dim/nested_frequency_ceiling_experiment/run_experiment.py
```

Configuration:

- hidden dimensions 8 and 16;
- seeds 0 through 4;
- 500 exact-population Adam steps;
- evaluation every 10 steps;
- first-stable window: five evaluations;
- Bayes-gap thresholds 0.03;
- cake-loss threshold 0.03.

All `uniform_raw` and `zipf_reweight` trajectories were numerically identical for every seed, sharing mode, dimension, recorded loss, and recorded spectral metric. The maximum recorded trajectory difference was exactly 0. This validates the frequency-reweight control.

## Result 1: frequency weighting is a major cause of slow tail learning

Mean first-stable-all step:

| Dimension | Sharing | Zipf raw | Uniform / Zipf reweight |
|---:|---|---:|---:|
| 8 | shared | 98 | 40 |
| 8 | split | 54 | 32 |
| 16 | shared | 110 | 46 |
| 16 | split | 40 | 30 |

The paired `zipf_raw - uniform_raw` delay in the shared condition was positive for all ten dimension/seed pairs:

- dimension 8: `10, 40, 90, 50, 100` steps;
- dimension 16: `90, 100, 80, 10, 40` steps.

At initialization in the shared condition:

| Dimension | Objective | Raw tail/high gradient norm | Weighted tail/high | Weighted common component tail/high | Weighted residual component tail/high |
|---:|---|---:|---:|---:|---:|
| 8 | Zipf | 0.584 | 0.097 | 0.010 | 0.137 |
| 8 | Uniform | 0.584 | 0.584 | 0.059 | 0.825 |
| 16 | Zipf | 0.596 | 0.099 | 0.013 | 0.130 |
| 16 | Uniform | 0.596 | 0.596 | 0.077 | 0.782 |

The raw per-pattern tail gradient was not zero, but Zipf weighting reduced the optimizer-visible tail contribution by approximately six times. This suppression occurred both along the high-pattern gradient direction and in its orthogonal residual.

This supports the frequency-gradient-share mechanism directly.

## Result 2: fast uniform learning still uses the common direction

At the first stable checkpoint in the shared uniform condition:

| Dimension | Tail hidden energy in high top direction | Common-only cake loss | Residual-only cake loss |
|---:|---:|---:|---:|
| 8 | 0.931 | 0.0005 | 5.661 |
| 16 | 0.971 | 0.0000 | 6.104 |

Thus the fastest condition did not need to move cake prediction into a new residual direction. The high-pattern top direction alone retained almost all cake-prediction function, while the residual-only representation failed.

This supports the weaker ceiling conclusion:

> Reusing a common direction is not intrinsically inefficient. With sufficient gradient share, the model learns the tail quickly while functionally using that common direction.

It does not yet prove that common reuse is globally optimal, because no forced-residual counterfactual was included in this run.

## Result 3: nested sharing is not zero-cost; it interacts with frequency

The paired `shared - split` convergence delay was:

- under Zipf:
  - dimension 8: mean 44 steps, positive in all five seeds;
  - dimension 16: mean 70 steps, positive in all five seeds;
- under uniform/reweight:
  - dimension 8: mean 8 steps;
  - dimension 16: mean 16 steps, driven partly by one 60-step seed difference.

Therefore the strong statement “nested sharing does not affect efficiency” is falsified by this operationalization. A better statement is:

> Nested token sharing creates an optimization conflict, but its cost is much larger when the tail gradient is frequency-suppressed. Uniform weighting mostly neutralizes that cost.

The shared Zipf causal decomposition reinforces this interaction. At convergence:

- dimension 8: common-only cake loss 1.335, residual-only 0.897;
- dimension 16: common-only cake loss 6.017, residual-only 0.209.

Unlike uniform training, the underweighted shared-tail task could not reliably install cake prediction into the high-pattern common direction and instead required mixed or residual computation.

## Result 4: “nested sharing creates the large singular direction” was not supported

At first stable convergence, split-token models were generally **more** spectrally concentrated than shared-token models. For example in dimension 8:

| Condition | Bqk top-1 energy | Bvo top-1 energy | Macro representation top-1 energy |
|---|---:|---:|---:|
| shared Zipf | 0.855 | 0.826 | 0.975 |
| shared uniform | 0.848 | 0.737 | 0.948 |
| split Zipf | 0.985 | 0.945 | 0.997 |
| split uniform | 0.959 | 0.916 | 0.992 |

Zipf often sharpened `Bvo`, centered embedding, and representation spectra relative to uniform, but the direction was not universal for every matrix/dimension. More importantly, removing moon parameter sharing did not flatten the model.

This falsifies the current split-alias operationalization of:

> nested sharing itself necessarily causes the dominant singular direction.

One plausible interpretation is that independent moon aliases let the model align both routes with one simple dominant cake map, whereas a shared moon must serve incompatible high- and tail-frequency roles and therefore uses more directions. This interpretation requires a new control; it is not established by this run.

## Result 5: static cosine is secondary and not robust enough for the main claim

In shared Zipf at convergence, cake was more similar to moon than to banana/fruit, especially in dimension 16. Under uniform weighting, cake became approximately symmetric across its three noun parents and was not consistently closest to moon. The exact cosine ordering varied with dimension.

The causal common-only/residual-only result is therefore more reliable than raw token cosine for deciding which subspace carries cake prediction.

## Updated conjecture

The run supports:

1. frequency weighting strongly suppresses tail gradients in both common and residual components;
2. inverse-frequency reweight exactly recovers the uniform trajectory in this global-batch toy;
3. fast uniform learning can functionally reuse the high-pattern common direction;
4. nested sharing and frequency interact: sharing is mildly costly under uniform weighting and strongly costly under Zipf weighting.

The run does not support:

1. nested sharing alone is the cause of spectral concentration;
2. nested sharing has no learning-efficiency cost;
3. the uniform condition is a truly flat-spectrum ceiling—the representation top-1 energy remains high;
4. forcing new directions is harmful, because a matched forced-residual counterfactual has not yet been run.

## Next decisive experiment

Use the same five-pattern task and compare, under uniform/shared training:

1. Adam natural representation;
2. Muon update flattening;
3. a forced common-to-residual branch;
4. an unconstrained matched-capacity branch.

If all fast conditions still use common-only prediction and the forced-residual branch is slower, that would complete the intended ceiling argument. If Muon or forced residual shifts causal prediction into residual directions without slowing learning, then new-direction usage remains viable.
