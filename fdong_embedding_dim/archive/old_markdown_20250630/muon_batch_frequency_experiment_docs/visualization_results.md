# Visualization and results

## Outcome

The controlled result supports the narrow version of the hypothesis.

1. With an exact population update, raw-loss Muon removed the 90/10
   common-versus-tail delay in this fixed-feature matrix task.
2. Exact inverse-frequency reweighting did the same for AdamW.
3. Both methods degraded as batch coverage decreased, while common features
   remained immediately learnable.
4. The degradation has two measured components: high stochastic variance from
   rare-feature absence, and an additional bias after the nonlinear Muon
   transform.

This is an existence and mechanism result for a `16 x 16` matrix classifier. It
is not yet evidence that canonical hybrid Muon completely solves the problem in
a Transformer.

## Setup

- Sixteen fixed dense orthogonal feature directions.
- Four common features carry 90 percent of probability mass.
- Twelve tail features share 10 percent.
- Labels are feature identities, so a perfect full-rank solution exists.
- AdamW and canonical momentum-plus-NS5 Muon are tuned on uniform exact-
  population training, then held fixed across all other conditions.
- Selected learning rates: AdamW `0.1`, Muon `0.1`.
- Three paired seeds; 400 steps; stable accuracy requires ten consecutive fully
  correct evaluations.
- Batch regimes: exact population expectation, 64 categorical samples, and 16
  categorical samples.

## Algorithm checks

For input singular values `[10, 1, 0.1, 0]`, NS5 produced approximately
`[0.709, 0.702, 0.697, 0]`.

Interpretation:

- existing nonzero singular directions were flattened;
- the exactly absent direction stayed exactly zero;
- Muon amplified weak directions but did not invent missing information.

The exact 90/10 inverse-frequency population gradient matched the uniform
feature objective with relative error `0.0`.

## Learning result

Median stable all-feature step across three seeds:

| Distribution | Optimizer and loss | Population | Batch 64 | Batch 16 |
|---|---|---:|---:|---:|
| Uniform | AdamW, raw | 1 | 1 | 7 |
| Uniform | Muon, raw | 1 | 2 | 6 |
| 90/10 | AdamW, raw | 14 | 21 | 36 |
| 90/10 | AdamW, balanced | 1 | 5 | 19 |
| 90/10 | Muon, raw | 1 | 4 | 15 |
| 90/10 | Muon, balanced | 1 | 4 | 15 |

The seed ranges at batch 16 were:

- AdamW raw: `30..44`;
- AdamW balanced: `14..41`;
- Muon raw: `10..39`;
- Muon balanced: `10..39`.

In this orthogonal-feature task, Muon raw and Muon balanced were identical. The
loss weights change singular magnitudes along already orthogonal feature
directions, and the zeroth-power update removes those magnitudes. This equality
is special to the controlled construction and should not be assumed for a
Transformer.

## Update-spectrum evidence

At step 1 on 90/10 raw-loss data, averaged across seeds:

| Optimizer | Batch | Update top-1 energy | Update effective rank | Tail/common functional-update norm |
|---|---:|---:|---:|---:|
| AdamW | Population | 0.687 | 3.12 | 0.282 |
| AdamW | 64 | 0.626 | 3.91 | 0.334 |
| AdamW | 16 | 0.640 | 3.64 | 0.377 |
| Muon | Population | 0.155 | 14.32 | 0.804 |
| Muon | 64 | 0.121 | 9.69 | 0.571 |
| Muon | 16 | 0.260 | 5.62 | 0.128 |

The exact-population Muon update was high-rank and gave tail directions almost
the same functional update scale as common directions. With batch 16, its
effective rank collapsed and tail functional update fell to 12.8 percent of the
common update. This is direct evidence for the feature-coverage mechanism.

After 400 exact-population raw-loss steps:

| Optimizer | Parameter top-1 energy | Parameter effective rank | Macro loss |
|---|---:|---:|---:|
| AdamW | 0.237 | 11.43 | 0.00282 |
| Muon | 0.108 | 14.81 | approximately 0 |

Muon therefore produced both flatter applied updates and a flatter learned
parameter matrix in this toy.

## Estimator evidence

For 90/10 data with known-global inverse-frequency weighting:

| Batch | Mean tail-feature coverage | Raw relative bias | Raw RMS error | Muon relative bias | Muon RMS error |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.407 | 0.064 | 1.180 | 0.390 | 0.748 |
| 16 | 0.126 | 0.103 | 2.383 | 0.738 | 0.852 |

The raw reweighted estimator remains theoretically unbiased; the nonzero Monte
Carlo bias is finite-sample error and is much smaller than its RMS variation.
The important failure is variance: at batch 16, only 12.6 percent of tail
feature types appear on average and raw RMS error is 2.38 times the exact
gradient norm.

Muon adds a second issue. Because the matrix zeroth-power map is nonlinear, the
mean transformed batch update differs strongly from the transformed exact
population update. At batch 16 the measured relative bias was `0.738`.

## How to read the plots

`learning_curves.png` shows mean common and tail accuracy over the first 100
steps. The population Muon panel shows simultaneous one-step learning. The
batch-16 panels show common accuracy saturating immediately while tail accuracy
rises only as rare feature directions appear.

`batch_gap.png` plots median stable convergence step against batch regime. The
raw AdamW curve retains the frequency delay even for the population update.
Balanced AdamW and Muon remove it only at exact population coverage; their cost
increases as coverage decreases.

## Conjecture update

Supported in this controlled setting:

> Muon can act as an implicit frequency equalizer when the complete set of
> orthogonal feature directions is present in the update matrix. Small batches
> weaken this effect because the update is low-rank in the missing feature
> directions, and the nonlinear Muon transform of a stochastic gradient is not
> the exact population Muon update.

Not established:

- that mini-batching is the only reason Muon is incomplete on real data;
- that semantic features are orthogonal or identifiable within one matrix;
- that Muon can recover a feature absent from the batch through momentum from
  earlier batches;
- that canonical Transformer Muon, which excludes embedding and output layers,
  eliminates long-tail learning delay;
- that global batch is computationally or statistically optimal for language
  modeling.

## Next experiment

Use the existing K-token single-head attention task with:

1. AdamW on all parameters;
2. canonical hybrid Muon on separate `Wq`, `Wk`, and `Wv`, AdamW on embedding
   and output parameters;
3. raw and inverse-square-root target-frequency losses;
4. exact population, large mini-batch, and small mini-batch regimes;
5. common/tail stable step, per-layer update rank, tail/common update projection,
   and `Bqk` top-channel ablation.

That follow-up tests whether the controlled mechanism survives shared learned
representations and attention routing.
