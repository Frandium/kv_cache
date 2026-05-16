# ABC Motif Transformer Spectral Study

## 1. Problem Formulation

This experiment studies whether a very small causal Transformer can learn a simple high-probability local statistical pattern in a token stream, and whether the learned pattern is visible in the spectrum and singular directions of the model's feed-forward network.

The synthetic data uses only uppercase English characters:

```text
A B C ... Z
```

The primary dataset contains exactly `1,000,000` tokens. It is generated from two sources:

1. Inserted `ABC` motifs.
2. Uniform random background characters.

The inserted `ABC` motif accounts for approximately `20%` of all emitted tokens. The remaining approximately `80%` are independent uniform random characters from the 26-character alphabet.

The model is trained with next-token prediction. The most important behavioral question is whether the model learns:

```text
inserted A -> predict B
inserted B -> predict C
```

The most important spectral question is whether the FFN matrices develop anisotropic singular spectra, and whether high singular directions align with the internal representation of the learned `ABC` pattern.

---

## 2. Hypotheses

### 2.1 Learning Hypothesis

A one-layer causal Transformer with latent dimension `100` should easily learn the local `ABC` motif through next-token prediction.

Expected behavior:

- Validation loss should decrease.
- Motif-specific loss should decrease more sharply than background loss.
- Prediction accuracy at inserted `A -> B` and inserted `B -> C` positions should approach `100%`.

### 2.2 Spectral Hypothesis

The FFN matrices should become more anisotropic when trained on data with a strong statistical motif than when trained on fully random data.

Expected behavior:

- The `ABC` model should have slightly more concentrated FFN singular spectra than the random-only control.
- The random-only control should have a flatter spectrum because the stream contains no predictive structure beyond uniform marginal token probabilities.

### 2.3 Alignment Hypothesis

If the FFN spectrum contains task-relevant anisotropy, then high singular directions should align with representations of the `ABC` motif.

Expected behavior:

- The internal representation used to predict `B` after inserted `A` should align with one or more high singular directions.
- The internal representation used to predict `C` after inserted `B` should align with one or more high singular directions.
- The top singular direction may or may not correspond directly to the `ABC` motif; it could instead represent a global/background activation direction.

---

## 3. Experiment Setup

### 3.1 Dataset A: ABC Motif Stream

Dataset composition:

| Quantity | Value |
|---|---:|
| Total tokens | `1,000,000` |
| Inserted motif | `ABC` |
| Inserted motif tokens | `199,998` |
| Background random tokens | `800,002` |
| Motif token fraction | `0.199998` |

Overall character frequencies are intentionally not uniform because `A`, `B`, and `C` are boosted by motif insertion:

| Character Group | Probability |
|---|---:|
| `A` | `0.097251` |
| `B` | `0.097328` |
| `C` | `0.097593` |
| Other characters | approximately `0.03045` to `0.03100` |

However, the background-only component is uniform:

| Background-Only Frequency | Value |
|---|---:|
| Expected uniform probability | `0.038462` |
| Observed min probability | `0.038062` |
| Observed max probability | `0.038752` |

### 3.2 Dataset B: Random-Only Control

The control dataset also contains exactly `1,000,000` tokens, but every token is sampled independently and uniformly from the 26-character alphabet.

Observed random-only frequency range:

| Quantity | Value |
|---|---:|
| Expected uniform probability | `0.038462` |
| Observed min probability | `0.038079` |
| Observed max probability | `0.038808` |

This confirms that the random-only generator is uniform up to finite-sample noise.

### 3.3 Model

Both datasets use the same model architecture:

| Component | Value |
|---|---:|
| Model type | one-layer causal Transformer |
| Latent dimension / `d_model` | `100` |
| Attention heads | `4` |
| Context length | `64` |
| FFN hidden dimension | `400` |
| Vocabulary size | `26` |
| Training objective | next-token prediction |
| Training steps | `500` |

The FFN contains two learned matrices:

```text
fc1: 400 x 100
fc2: 100 x 400
```

The spectral analysis computes singular values for both matrices.

---

## 4. Training Results

### 4.1 ABC Motif Training Loss

![ABC training loss](figures/abc_training_loss.svg)

Final ABC run metrics:

| Metric | Value |
|---|---:|
| Overall validation loss | `2.8460` |
| Perplexity | `17.2187` |
| ABC motif loss | `0.2000` |
| Background loss | `3.3375` |
| `A -> B` loss | `0.3654` |
| `B -> C` loss | `0.0346` |
| `A -> B` accuracy | `1.0000` |
| `B -> C` accuracy | `1.0000` |
| Combined ABC accuracy | `1.0000` |

The model learns the inserted motif quickly. The motif-specific loss drops much more sharply than the background loss, which is expected because the background is mostly random.

### 4.2 ABC Prediction Accuracy

![ABC prediction accuracy](figures/abc_prediction_accuracy.svg)

The model reaches `100%` accuracy on both motif prediction positions:

```text
inserted A -> B
inserted B -> C
```

Average predicted probabilities after training:

| Conditional Prediction | Average Probability |
|---|---:|
| `P(B | inserted A)` | `0.695` |
| `P(C | inserted B)` | `0.967` |

The `B -> C` prediction is more confident than the `A -> B` prediction. This is reasonable because an observed `B` is more diagnostic of being inside an inserted `ABC` motif than an observed `A`, since `A` also appears in random background positions.

### 4.3 Random-Only Control Training

Final random-only metrics:

| Metric | Value |
|---|---:|
| Validation loss | `3.2595` |
| Perplexity | `26.0371` |
| Background loss | `3.2595` |

For a uniform 26-character stream, the irreducible cross entropy is:

$$
\log(26) \approx 3.258
$$

The random-only model converges almost exactly to this value, indicating that it correctly learns that there is no useful predictive structure.

---

## 5. FFN Spectral Analysis

### 5.1 ABC Versus Random-Only SVD Comparison

![FFN SVD comparison](figures/ffn_svd_comparison.svg)

Summary:

| Condition | Matrix | Full Rank | Top-1 Energy | Top-10 Energy | Stable Rank |
|---|---|---:|---:|---:|---:|
| ABC motif | `fc1` `400 x 100` | `100` | `5.22%` | `24.07%` | `19.16` |
| Random only | `fc1` `400 x 100` | `100` | `4.97%` | `22.55%` | `20.14` |
| ABC motif | `fc2` `100 x 400` | `100` | `8.71%` | `33.15%` | `11.48` |
| Random only | `fc2` `100 x 400` | `100` | `8.11%` | `28.80%` | `12.32` |

The ABC model is directionally more anisotropic than the random-only model:

- Higher top-10 energy in both FFN matrices.
- Lower stable rank in both FFN matrices.

However, the effect is modest. The FFN matrices do not collapse to rank one. Both remain full-rank.

### 5.2 Interpretation

The raw FFN spectrum alone is not sufficient to identify the `ABC` mechanism.

The model can store and use motif information across several components:

- token embeddings
- positional embeddings
- attention matrices
- FFN matrices
- unembedding matrix
- residual stream geometry

Therefore, the absence of a rank-one FFN does not mean the model failed to learn the motif. Behaviorally, the model clearly learned it.

---

## 6. Singular Vector Alignment Analysis

### 6.1 Method

The alignment analysis asks whether learned singular directions correspond to the internal representation of the `ABC` motif.

The main `ABC` representation is defined as the mean FFN-input activation at motif-prediction positions:

```text
after inserted A: hidden state used to predict B
after inserted B: hidden state used to predict C
ABC predictive mean: average of the above two representations
```

For `fc1`, the relevant 100-dimensional singular directions are the right singular vectors:

```text
fc1 right singular vectors: FFN input directions
```

For `fc2`, the relevant 100-dimensional singular directions are the left singular vectors:

```text
fc2 left singular vectors: FFN output directions
```

Cosine similarity is computed between the normalized representation vector and each singular direction. Absolute cosine is used in the plot because singular-vector sign is arbitrary.

### 6.2 Alignment Results

![ABC SVD alignment](figures/abc_svd_alignment.svg)

The `ABC` representation aligns strongly with high singular directions of `fc1`.

| Representation | Strongest Aligned `fc1` Singular Vector | Singular Value | Absolute Cosine |
|---|---:|---:|---:|
| after inserted `A` | `#2` | `2.366` | `0.693` |
| after inserted `B` | `#3` | `1.898` | `0.567` |
| ABC predictive mean | `#3` | `1.898` | `0.601` |
| token embedding `A` | `#2` | `2.366` | `0.712` |
| token embedding `B` | `#2` | `2.366` | `0.467` |
| token embedding `ABC` mean | `#3` | `1.898` | `0.482` |

The alignment with `fc2` left singular vectors is weaker and less concentrated in the highest singular directions.

For the ABC predictive mean:

| Matrix Direction | Strongest Singular Vector | Singular Value | Absolute Cosine |
|---|---:|---:|---:|
| `fc2` left singular vectors | `#39` | `0.662` | `0.243` |

This suggests the motif is more visible in `fc1` input geometry than in `fc2` output geometry.

---

## 7. What Does The First Singular Vector Represent?

The largest singular vector of `fc1` is not the main `ABC` motif direction.

For the ABC-trained model:

| Quantity | Cosine With `fc1` Right Singular Vector #1 |
|---|---:|
| Mean token embedding | `-0.401` |
| ABC embedding mean | `0.127` |
| Non-ABC embedding mean | `-0.476` |

For the random-only model:

| Quantity | Cosine With `fc1` Right Singular Vector #1 |
|---|---:|
| Mean token embedding | `-0.702` |
| ABC embedding mean | `-0.356` |
| Non-ABC embedding mean | `-0.650` |

This suggests that singular vector #1 mostly represents a broad global/background activation direction, not the `ABC` motif itself.

The more specific motif directions are:

```text
fc1 singular vector #2: strongly aligned with inserted A / token A
fc1 singular vector #3: strongly aligned with inserted B and the ABC predictive mean
```

So the current interpretation is:

| Direction | Tentative Meaning |
|---|---|
| `fc1` singular vector #1 | broad/global/background direction |
| `fc1` singular vector #2 | `A` / start-of-motif-related direction |
| `fc1` singular vector #3 | `B` / ABC predictive direction |
| lower directions | mixed or weaker token/statistical features |

---

## 8. Conclusions

### 8.1 Behavioral Conclusion

The one-layer causal Transformer with latent dimension `100` learns the inserted `ABC` motif very quickly.

The behavioral result is unambiguous:

```text
A -> B accuracy: 100%
B -> C accuracy: 100%
```

### 8.2 Spectral Conclusion

The ABC-trained FFN spectrum is more anisotropic than the random-only control, but only modestly.

The FFN does not become rank one. Therefore, raw singular values alone are too crude to identify the learned motif.

### 8.3 Alignment Conclusion

The singular-vector alignment analysis is more informative than spectrum-only analysis.

The `ABC` representation aligns strongly with high singular directions of `fc1`, especially singular vectors `#2` and `#3`.

This supports the idea that learned statistical structure can appear as anisotropic geometry in FFN input directions, even when the full FFN matrix remains high-rank.

### 8.4 Top Singular Vector Conclusion

The largest singular vector appears to be a global/background direction rather than the motif direction.

This is important because it warns against assuming:

```text
largest singular value = most semantically meaningful task feature
```

In this experiment, the task-relevant motif directions are high-ranking but not necessarily first.

---

## 9. Pending Issues And Limitations

Several questions cannot be answered from the current experiment alone.

### 9.1 Mechanistic Localization

The current analysis does not prove where the model stores the `ABC` rule.

Possible locations include:

- attention heads
- token embeddings
- FFN input geometry
- FFN output geometry
- unembedding directions
- interactions among all of the above

The FFN alignment results show correlation with motif representations, but not causal necessity.

### 9.2 Causal Role Of Singular Directions

We have not yet ablated singular directions.

A stronger test would remove or project out individual singular directions and measure whether `ABC` prediction accuracy drops.

For example:

```text
remove fc1 singular vector #2 -> does A -> B fail?
remove fc1 singular vector #3 -> does B -> C fail?
remove fc1 singular vector #1 -> does general/background performance change?
```

This would distinguish correlation from causal importance.

### 9.3 Effect Of Marginal Token Frequency

In the ABC dataset, `A`, `B`, and `C` are more frequent than other characters. This is intended, but it creates two overlapping signals:

1. `ABC` sequential structure.
2. Higher marginal frequency of `A`, `B`, and `C`.

A cleaner future control should preserve unigram frequencies while destroying sequential order. For example:

```text
shuffle the ABC dataset tokens globally
```

This would keep `A/B/C` frequent but remove the `A -> B -> C` motif.

### 9.4 Multiple Seeds

The current study uses one seed for each condition.

A more reliable conclusion requires multiple random seeds and confidence intervals for:

- final loss
- singular value concentration
- stable rank
- alignment cosine
- rank position of motif-aligned directions

### 9.5 Architecture Dependence

The result may depend on this exact small architecture:

- one Transformer layer
- `d_model = 100`
- FFN dimension `400`
- context length `64`
- 4 attention heads

Different model widths, FFN ratios, initialization scales, or optimizers may change the spectrum.

### 9.6 Representation Definition

The phrase "ABC representation" is not unique.

This report used FFN-input activation means at motif-prediction positions. Other definitions may reveal different structure:

- token embedding mean of `A`, `B`, `C`
- residual stream before attention
- residual stream after attention
- FFN hidden activations after `fc1`
- logits or unembedding-space directions
- contrastive directions such as inserted `A` minus background `A`

The contrastive version may be especially useful because it separates motif-specific `A` from generic `A`.

### 9.7 From Local Motifs To Hierarchical Structure

This experiment only tests a local three-token motif.

It does not yet test hierarchical semantic memory, graph structure, long-context retrieval, or multi-resolution abstraction. It is best viewed as a pipeline sanity check:

```text
Can we generate controlled data, train a tiny model, observe learning, and inspect spectral geometry?
```

The answer is yes.

---

## 10. Reproducibility Artifacts

Experiment scripts:

- `../../experiments/abc_motif/run_abc_motif_experiment.py`
- `../../experiments/abc_motif/analyze_abc_svd_alignment.py`

Packaged data summaries:

- `data/abc_summary.json`
- `data/random_summary.json`
- `data/comparison_summary.json`
- `data/abc_svd_alignment_summary.json`

Packaged figures:

- `figures/abc_training_loss.svg`
- `figures/abc_prediction_accuracy.svg`
- `figures/ffn_svd_comparison.svg`
- `figures/abc_svd_alignment.svg`

Primary generated experiment directories:

- `../../experiments/abc_motif/out`
- `../../experiments/abc_motif/random_out`
- `../../experiments/abc_motif/comparison`

