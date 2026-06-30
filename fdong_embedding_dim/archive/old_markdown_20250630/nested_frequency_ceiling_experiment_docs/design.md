# Nested Sharing vs Frequency Ceiling Experiment

## Objective

Separate two possible causes of slow long-tail learning:

1. **Nested sharing:** a token such as `moon` appears in both `the moon` and `a moon cake`, causing common spectral directions and forcing the tail pattern to reuse them.
2. **Frequency weighting:** long-tail patterns contribute too little weighted gradient, in the common direction and in the full parameter space.

The expected story is treated as a falsifiable conjecture:

> Nested sharing may sharpen parameter and representation spectra and may cause
> tail predictions to reuse common directions, but it does not by itself slow
> learning. Slow learning is caused by the small frequency-weighted gradient of
> tail patterns.

## Data

Five unique sequences are used:

```text
the sun <pad>
the moon <pad>
a moon cake
a banana cake
a fruit cake
```

`<pad>` is excluded from attention and loss. Each sequence first averages its own valid next-token losses, so the three-token tail sequences do not receive twice the per-example weight of the two-token high-frequency sequences.

The two distributions are:

- `zipf_raw`: counts `(6, 6, 1, 1, 1)`;
- `uniform_raw`: counts `(3, 3, 3, 3, 3)`.

The third objective is:

- `zipf_reweight`: the data counts remain `(6, 6, 1, 1, 1)`, but inverse pattern-frequency coefficients make the effective objective identical to uniform weighting.

Because every optimization step uses the exact five-pattern population objective, this experiment tests deterministic loss weighting. It does not test minibatch omission noise.

## Nested-sharing intervention

Two tokenizations are compared:

- `shared`: the same `moon` embedding is used in `the moon` and `a moon cake`;
- `split`: `the moon_H` and `a moon_T cake` use independent embedding rows initialized to the same vector.

The split condition preserves the surface computation while removing direct parameter sharing through the moon embedding. This is an operational intervention on nested lexical sharing, not a claim that real tokenizers literally contain `moon_H` and `moon_T`.

## Model

The model is a one-layer, one-head, attention-only causal Transformer:

- tied input/output embedding;
- `Wq`, `Wk`, `Wv`, and `Wo`;
- causal attention;
- residual connection;
- no MLP and no LayerNorm;
- hidden dimensions 8 and 16 for the main tests; dimension 3 is an optional capacity-stress control.

For token representations `X`:

\[
Q=XW_Q^\top,\quad K=XW_K^\top,\quad V=XW_V^\top,
\]

\[
H=\operatorname{softmax}(QK^\top/\sqrt d+M_{causal})VW_O^\top+X,
\]

\[
\operatorname{logits}=HE^\top.
\]

## Why pattern accuracy is not the primary convergence metric

The prefixes are intentionally ambiguous:

\[
p(\text{sun}\mid\text{the})=p(\text{moon}\mid\text{the})=1/2,
\]

\[
p(\text{moon}\mid\text{a})=p(\text{banana}\mid\text{a})=p(\text{fruit}\mid\text{a})=1/3.
\]

Therefore a deterministic top-1 classifier cannot make every sequence correct. Convergence is measured by:

\[
G_{the}=\operatorname{CE}(q_{the},p_\theta)-\log2,
\]

\[
G_a=\operatorname{CE}(q_a,p_\theta)-\log3,
\]

and the mean deterministic cake loss:

\[
L_{cake}=\frac13\sum_{n\in\{moon,banana,fruit\}}-\log p_\theta(\text{cake}\mid a,n).
\]

These metrics use the full-vocabulary probabilities, so probability leakage to unrelated tokens remains penalized.

## Gradient decomposition

At diagnostic checkpoints, compute one raw gradient vector per unique pattern:

\[
g_i=\nabla_\theta L_i.
\]

Define the macro high-pattern direction:

\[
g_H=\frac{g_{the\ sun}+g_{the\ moon}}{2},\qquad c_g=g_H/\|g_H\|.
\]

For each pattern, report:

\[
\|g_i\|,
\]

\[
|g_i^\top c_g|,
\]

\[
\|g_i-(g_i^\top c_g)c_g\|.
\]

Report both raw per-pattern values and frequency-weighted contributions. The frequency hypothesis predicts that raw tail gradients need not be small, while their weighted contributions are small under `zipf_raw` and restored by `uniform_raw` or `zipf_reweight`.

## Spectral and causal representation tests

Record spectra of:

- tied embedding `E` using a canonical semantic row set;
- `Bqk = Wq.T @ Wk`;
- `Bvo = Wo @ Wv`;
- contextual hidden states.

Representation spectra are computed twice:

1. macro: each unique valid prediction context has equal weight;
2. training-weighted: contexts follow the current pattern objective weights.

The common representation direction is the top uncentered singular direction of hidden states from the two high patterns. For tail noun states, report projection mass onto that direction and causal cake loss under:

- common-only hidden states;
- residual-only hidden states.

Natural common reuse is supported only when common-only representations retain substantial cake-prediction ability. A cosine or singular value alone is not causal evidence.

## Claim boundary

This toy can show whether frequency weighting, lexical parameter sharing, and common-direction usage separate under a controlled objective. It cannot establish that all natural-language nested structure behaves the same way, and uniform data is not assumed to create a flat spectrum unless the measured spectra actually show it.

