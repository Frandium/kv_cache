# Experiment Design

Main comparison axes:

- optimizer: `adam` vs `muon`
- data: `withK_uniform` vs `withK_zipf`
- batch: `population` vs `64` vs `16`
- loss:
  - `raw`
  - `sqrt_reweight = 1 / sqrt(f_target)` with expectation renormalized to 1

Learning-rate selection:

- select one LR per optimizer
- selection benchmark: `withK_uniform + population + raw`
- score: mean `first_stable_all_groups_full_accuracy_step`, tie-broken by final population raw loss

Evaluation:

- `first_stable_all_groups_full_accuracy_step`
- `first_stable_all_examples_full_accuracy_step`
- `first_stable_internal_accuracy_step`
- final `common_accuracy`, `tail_accuracy`, `internal_accuracy`
- final `common_loss`, `tail_loss`
- `Bqk = Wq^T Wk` spectrum:
  - `top1_energy`
  - `effective_rank`
- step-1 hidden gradient/update spectra

Additional falsification check:

At initialization, compare:

- exact population hidden gradient
- minibatch hidden gradient estimator
- Muon-transformed hidden update estimator

This checks whether minibatch noise is merely high-variance or whether Muon's nonlinear transform also changes the mean hidden update.
