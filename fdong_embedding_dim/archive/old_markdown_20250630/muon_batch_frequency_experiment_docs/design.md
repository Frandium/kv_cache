# Muon, frequency imbalance, and batch coverage

## Objective

Test the existence claim:

> A matrix optimizer that flattens the spectrum of each update can remove the
> common-versus-tail learning-speed gap when every feature is represented in
> every update, while stochastic batches weaken this benefit because rare
> features are often absent and the nonlinear Muon transform is applied to an
> incomplete update matrix.

This is a controlled optimizer-mechanism test. It is not yet a claim about a
full Transformer or real language data.

## Physical priors

1. A population with probabilities `p_i` gives common features more gradient
   mass than tail features under the ordinary expected cross-entropy loss.
2. Inverse-frequency loss weighting changes the population objective to equal
   feature mass. With known global probabilities, its mini-batch gradient is
   unbiased, but its variance grows when tail features are rarely sampled.
3. Muon applies a nonlinear matrix function to the momentum update. Therefore
   even an unbiased batch gradient does not imply an unbiased Muon update:

   `E[Muon(G_batch)] != Muon(E[G_batch])` in general.
4. An exactly absent orthogonal feature direction contributes no gradient in
   that direction. Matrix orthogonalization cannot recover target information
   that is not present in the batch.

## Mathematical model

There are `N=16` feature types. The input matrix `X` is a fixed random
orthogonal matrix, so feature `i` is the dense row `X[i, :]`; the correct class
is `i`. The only trainable parameter is
`W in R^(N x N)`:

`logits_i = X[i, :] W`.

Orthogonal feature directions make the task exactly separable while the dense
coordinates prevent Adam's elementwise normalization from reducing the task to
sixteen independent parameter rows.

Four common features receive 90 percent of population mass. Twelve tail
features share the remaining 10 percent. The uniform control assigns `1/N` to
every feature.

The population loss is:

`L_pop(W) = sum_i p_i CE(X[i, :] W, i)`.

The inverse-frequency objective uses `a_i = 1 / (N p_i)`:

`L_bal(W) = sum_i p_i a_i CE(W[i, :], i)
          = (1/N) sum_i CE(W[i, :], i)`.

For a sampled mini-batch `B`, the implementation uses the mean
`mean_{i in B} a_i CE_i` without renormalizing weights inside the batch. This
is required for an unbiased estimator of `L_bal`.

Muon maintains a momentum buffer `M` and computes:

`M_t = beta M_(t-1) + (1-beta) G_t`

`U_t = beta M_t + (1-beta) G_t` for Nesterov momentum, written in the same
`lerp` form as the reference implementation.

Five quintic Newton-Schulz steps approximate the zeroth matrix power of `U_t`.
The parameter update is `W <- W - lr * NS5(U_t)`.

## Implementation contract

Input:

- distribution: uniform or 90/10 common-tail;
- batch regime: exact population gradient, batch 64, or batch 16;
- optimizer: AdamW or Muon;
- loss: ordinary or known-global inverse-frequency weighting;
- seed and learning rate.

Procedure:

1. Initialize the same `16 x 16` matrix for paired runs.
2. Compute either the exact population loss or a categorical mini-batch loss.
3. Form AdamW or canonical momentum-plus-NS5 Muon update.
4. Evaluate all 16 features after every update.
5. Save per-step common accuracy, tail accuracy, macro loss, population loss,
   parameter spectrum, and applied-update spectrum.
6. Require ten consecutive fully correct evaluations for a stable convergence
   step.
7. At initialization, repeatedly sample batches and compare the mean raw and
   Muon-transformed batch updates with their exact-population counterparts.

Outputs:

- `results/history.csv`: per-step trajectories;
- `results/summary.csv` and `summary.json`: convergence summaries;
- `results/estimator_diagnostics.csv`: batch estimator bias and variance;
- `results/learning_curves.png`: common/tail learning curves;
- `results/batch_gap.png`: stable convergence versus batch regime.

## Claim boundary

Passing this experiment shows existence in a direct matrix-classification toy.
It does not show that canonical hybrid Muon solves the same problem in a
Transformer. Canonical Muon normally applies to hidden 2D weights while input
embeddings, output heads, gains, and biases use AdamW. A follow-up attention
experiment is required before making an architecture-level claim.
