# Design

## Objective and falsifiable conjecture

For sequences `P_i B_j_0 B_j_1 S_i`, test whether MTP=3 makes the early state
`h(P_i)` more useful for predicting `S_i` than MTP=1 (NTP).

The falsifiable conjecture is: after both backbones are frozen and their
original heads are discarded, an identically trained fresh probe predicts
`S_i` from the MTP=3 prefix state more accurately, or with fewer examples,
than from the NTP prefix state.

## Data prior and mathematical model

`S_i` is a deterministic function of `P_i`. The bone is independent of that
pair. The dataset is the Cartesian product of all prefix-suffix pairs and all
bones:

```text
D = {(P_i, B_j_0, B_j_1, S_i) : i in [N_prefix], j in [N_bone]}.
```

MTP=1 supervises only the next token at each state. MTP=3 attaches losses for
offsets 1, 2, and 3 to every state for which that target exists. Therefore only
MTP=3 directly attaches the suffix error to `h(P_i)`.

## Model contract

All models have a learned token embedding, one causal backbone, and independent
linear vocabulary heads. The backbone variants are:

1. Linear: causal cumulative mean followed by a linear map.
2. MLP: causal cumulative mean followed by a two-layer GELU MLP.
3. Attention: one causal self-attention block with a feed-forward sublayer.

The cumulative mean is required in the first two variants so that the state at
`B_j_1` can read `P_i` and solve the ordinary next-token suffix task.

## Claim boundary

This experiment measures early-state future-token decodability, not natural
language semantics. Linear-probe failure means failure under a linear readout;
it does not prove information-theoretic absence. Spectrum flatness is a
correlate and is not by itself evidence that the extra directions encode the
suffix.

## Compositional extension

The second-stage dataset is `X_a Y_b B_j_0 B_j_1 S_f(a,b)`, with
`f(a,b)=(a+b) mod M`. The probe position is after `Y_b`; the suffix remains at
offset 3. Entire `(a,b)` combinations are held out, so success requires
composition rather than memorizing one label for each observed prefix state.
