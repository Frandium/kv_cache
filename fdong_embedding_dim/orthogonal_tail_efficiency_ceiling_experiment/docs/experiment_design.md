# Experiment design

## Data

The vocabulary is

`<pad>, the, a, sun, moon, banana, fruit, cake`.

Stage 1 trains only two equally weighted patterns:

- `the sun <pad>`
- `the moon <pad>`

The pad target is masked, so the model learns the Bayes-optimal ambiguous prediction `the -> {sun, moon}`.

Stage 2 trains only three equally weighted patterns:

- `a moon cake`
- `a banana cake`
- `a fruit cake`

Its first prediction has Bayes loss \(\log 3\); its second prediction is deterministically `cake`. The best possible mean per-sequence loss is therefore \(\log(3)/2\approx0.5493\).

## Model

The model is a one-layer, one-head, causal attention-only Transformer:

\[
Q=XW_q^\top,\quad K=XW_k^\top,\quad V=XW_v^\top,
\]

\[
H=X+\operatorname{softmax}(QK^\top/\sqrt d)VW_o^\top,
\]

\[
\text{logits}=HE^\top.
\]

There is no MLP or layer normalization. Input and output embeddings are tied. Hidden dimensions are 8 and 16.

## Training protocol

Stage 1 trains the entire base model for 400 full-batch Adam steps at learning rate 0.03. Stage 2 freezes all base parameters.

Each stage-2 branch contains:

- \(A\in\mathbb{R}^{2\times2}\): four parameters;
- five rank-2 contextual tied-embedding coefficient rows for `a`, `moon`, `banana`, `fruit`, and `cake`: ten parameters.

Each branch therefore has exactly 14 trainable scalars. All are zero initialized, so common and spectral-tail variants begin with exactly identical logits.

The LR sweep is `0.003, 0.01, 0.03, 0.1, 0.3, 1.0`. Every run uses 1,000 full-batch stage-2 updates. Evaluation occurs every 5 updates.

## Convergence metric

A checkpoint passes when both conditions hold:

\[
L(a\rightarrow\{moon,banana,fruit\})-\log 3\leq0.03,
\]

\[
L(\{moon,banana,fruit\}\rightarrow cake)\leq0.03.
\]

`first_stable_tail_step` is the first checkpoint for which both inequalities remain true for five consecutive evaluations. This avoids counting a transient threshold crossing.

## Hyperparameter selection and held-out evaluation

Seeds 0–4 select one LR independently for each dimension and branch. Those LRs are then frozen.

- dimension 8: common 0.3, spectral tail 0.3;
- dimension 16: common 0.3, spectral tail 1.0.

Held-out evaluation uses seeds 5–19 for dimension 8 and seeds 5–49 for dimension 16. Dimension 16 was expanded after an initial trend was observed, so its p-values are strong exploratory evidence rather than a preregistered confirmatory test.

## Contract checks

The saved smoke contract verifies:

- maximum initial-logit difference: exactly 0;
- trainable parameters: 14 versus 14;
- top/bottom output-basis overlap: approximately \(10^{-7}\);
- frozen base maximum drift: exactly 0 in every run;
- learned map and embedding deltas remain numerically confined to their assigned subspace.

For the representative dimension-8 seed-0 checkpoint, \(B_{vo}\)'s largest and smallest singular values are 19.53 and 0.0162, so the treatment genuinely uses a direction from the weak spectral tail.

## Reproduction

Main LR sweep:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 \
  fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment/run_experiment.py \
  --output-dir fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment/results
```

Held-out runs and analysis:

```bash
bash fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment/run_heldout.sh
```

The analysis consumes the main sweep's seeds 0–4 only for LR selection and disjoint seed runs for reporting.
