# A15_01_09_E01 Detailed Evidence Ledger

## 1. Question And Typed Verdict

**Question.** In frozen Qwen3-8B, is the high-gain MLP input subspace uniquely more shared across layers, and does fine-relative-to-coarse semantic variance move to later local parameter ranks with depth in isolated attention writes, actual MLP-input increments, and nonlinear MLP responses?

**Typed verdict:** `location_without_commonality`; nonlinear effective-use clause Fail.

| Clause | Verdict | Direct reading |
| --- | --- | --- |
| Parameter head-specific commonality | Fail | Head exceeds random and middle, but tail exceeds head |
| Overall semantic right shift | Descriptive Pass | Both coarse and fine move substantially right with depth |
| Raw fine-specific right shift | Fail | Fine does not move farther right than coarse |
| RMSNorm-folded gain-weighted write | Pass | Fine-relative shift is positive after coordinate matching and gain |
| Gain-weighted actual MLP-input increment | Primary Pass | Positive, bootstrap-stable, split/parent/width robust |
| Nonlinear MLP band response | Fail | Positive point estimate; 95% interval crosses zero |

## 2. Architecture And Physical Objects

For decoder block $\ell$:

```text
historical residual x_l
-> attention heads -> concatenated pre-o_proj state
-> o_proj -> isolated write a_l
-> residual sum x_l + a_l
-> post-attention RMSNorm -> actual MLP input n_l
-> MLP -> residual-stream update
```

The isolated write is newly produced by the current attention branch, but it depends on $x_\ell$ and is not information-theoretically pure new information. The attention-induced MLP-input increment is

$$
\Delta n_\ell=RMSNorm_\ell(x_\ell+a_\ell)-RMSNorm_\ell(x_\ell).
$$

The nonlinear response intervention is

$$
\Delta m_{\ell,k}=MLP_\ell(n_{old,\ell}+P_{\ell,k}\Delta n_\ell)-MLP_\ell(n_{old,\ell}),
$$

for broad H/M/T parameter bands. Median H/M/T non-additivity is 0.170 relative Frobenius norm, so these are band interventions rather than an additive decomposition.

## 3. Parameter Spectrum

The joint input-side MLP operator is

$$
K_\ell=W_{gate,\ell}^{\top}W_{gate,\ell}+W_{up,\ell}^{\top}W_{up,\ell}
=V_\ell\Gamma_\ell V_\ell^{\top}.
$$

`down_proj` is excluded because it maps MLP hidden units back to the residual stream. The RMSNorm-folded control is $K_\ell^{eff}=D_\ell K_\ell D_\ell$. Each spectrum has 16 equal-rank 256-dimensional bands: F1=head, F2--F8=middle, F9--F16=tail.

Cross-layer commonality compares projectors, not individual eigenvectors:

$$
O_{\ell m k}=tr(P_{\ell,k}P_{m,k})/r.
$$

For two independent Haar-random rank-$r$ subspaces in 4096 dimensions, the expectation is $r/4096$.

## 4. Data And Fair Coarse/Fine Estimator

| Item | Frozen value |
| --- | --- |
| Model | `/data/share/Qwen3-8B` |
| Shape | 36 blocks; hidden 4096; 32 attention heads; MLP width 12,288 |
| Semantic cube | 8 parents × 8 children × 4 templates × 2 fact bundles = 512 |
| Parents | mathematics, physics, chemistry, biology, computer science, economics, medicine, linguistics |
| Readout | final shared `Classification:` colon, padded absolute position 57 |
| Natural length | 41--58 tokens; median 49 |
| Frozen dataset identifier | `cb440b98d81bac3f9813344f85e6efdbd994b7b988d8009ba64e207e64a11859` |
| Cached write run | `a15-01-05-e04-post-o-proj-20260806T172600Z` |

Example parent/children: `mathematics -> algebra / analysis / combinatorics / geometry / number theory / probability / statistics / topology`. One algebra record is:

> Topic description: This topic studies symbolic expressions and equations with unknown quantities; it also uses abstract operations satisfying closure and inverse properties; a central concern is structure-preserving maps between formal systems. Identify the broad academic field and the specific subfield. Classification:

The correct labels are absent from the description. Every parent-child cell has four design and four confirmation expressions.

Coarse covariance uses parent means; fine covariance uses child means centered within each parent. Both are population-weighted over the identical balanced cube. One within-child expression covariance is used only as a reliability quantity. The location metric normalizes each role's between-class variance over the full spectrum, so no unequal coarse/fine denominator remains.

## 5. Metrics And Their Meanings

Raw band variance is

$$E^{raw}_{\ell g k}=tr(P_{\ell,k}B_{\ell,g})/256.$$

It measures geometric occupancy in the parameter basis. Gain-weighted variance is

$$E^{gain}_{\ell g k}=tr(V_{\ell,k}\Gamma_{\ell,k}V_{\ell,k}^{\top}B_{\ell,g})/256.$$

It measures the corresponding linear preactivation-energy contribution. Neither by itself is a nonlinear MLP-output effect.

Within role, $p_{\ell g k}=E_{\ell g k}/\sum_jE_{\ell g j}$. The local-rank centroid is

$$C_{\ell,g}=\sum_kp_{\ell g k}(k-0.5)/16.$$

The registered fine-specific statistic removes the layer-common shift:

$$
T=median_{25:35}(C_{fine}-C_{coarse})-median_{1:12}(C_{fine}-C_{coarse}).
$$

$T>0$ means fine moves farther right than coarse. It does not mean a vector traveled across layers, because every layer has its own basis.

## 6. Execution And Hard Guards

All jobs used one `n12lp.nn.i10a.8` worker in `computing-cluster-5090-01g`, 8×5090, spot quota, normal priority, image `ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04`.

| Role | Job | Run | State |
| --- | --- | --- | --- |
| Smoke | `om-ltgmx70x` | `a15-01-09-e01-smoke-20260807T060900Z` | SUCCEEDED, 0 retries |
| Parameter P | `om-zn6g16x8` | `a15-01-09-e01-p-20260807T061000Z` | SUCCEEDED, 0 retries |
| Semantic S | `om-fnauw0g7` | `a15-01-09-e01-s-20260807T061000Z` | SUCCEEDED, 0 retries |
| Response R | `om-dph5vhje` | `a15-01-09-e01-r-20260807T061000Z` | SUCCEEDED, 0 retries |

| Guard | Maximum/result |
| --- | ---: |
| Native/folded eigensolver probe residual | $7.60\times10^{-7}$ / $1.95\times10^{-6}$ |
| Native/folded orthogonality | $2.68\times10^{-6}$ / $2.72\times10^{-6}$ |
| Semantic projection-energy relative error | $2.39\times10^{-7}$ |
| Rerun write vs frozen E04 cache | exact, max absolute error 0 |
| Full-basis $\Delta n$ reconstruction | $2.73\times10^{-6}$ |
| Coverage | 36/36 layers; 512/512 records |

All seven rendered figures were opened at original resolution after consolidation. Layer numbering, masked heatmap diagonals, log-scale overlap colorbars, early/late windows, block-36 boundary, fine/coarse sign convention, and the pooled-q98 clipping labels are present and readable. Clipping affects color saturation only; all decisions use un-clipped numeric tables.

## 7. Parameter Commonality Result

Equal-rank all-layer-pair overlap divided by Haar expectation is:

| Operator | head-256 | middle-256 | tail-256 | Registered reading |
| --- | ---: | ---: | ---: | --- |
| Native $K$ | 2.199 | 1.041 | 3.263 | head > middle, tail > head |
| RMSNorm-folded $K^{eff}$ | 2.047 | 1.127 | 7.036 | stronger tail commonality |

The ordering is unchanged for rank 128 and 512. At rank 256, native head-minus-tail overlap is -0.0665 with bootstrap interval entirely below zero; folded head-minus-tail is -0.3118 with interval entirely below zero. The 16-band profile is U-shaped: native F1/F16 are 2.20/3.26 times Haar, while central bands are about 1.04; folded F1/F16 are 2.05/7.04.

Adjacent layers are more aligned, but the tail conclusion is not only adjacency: for layer gaps at least 18, native head/middle/tail are 1.22/1.02/2.32 times Haar, and folded values are 1.23/1.05/4.83.

![Layer-pair overlap](figures/figure1_parameter_commonality_heatmaps.png)

![Bandwise overlap and wrong-band controls](figures/figure1b_parameter_bandwise_cross_overlap.png)

This refutes “common equals parameter head.” A shared tail may be a common near-null or low-gain geometry rather than shared processed information; the present audit does not identify which.

## 8. Overall Layerwise Right Shift

Both semantic roles move substantially toward later local ranks. These are descriptive early/late median centroids:

| Site/weighting | Coarse early→late | Fine early→late | Interpretation |
| --- | ---: | ---: | --- |
| write, native raw | 0.342→0.479 | 0.353→0.470 | shared raw redistribution |
| write, native gain | 0.134→0.303 | 0.125→0.302 | shared gain-weighted redistribution |
| actual $\Delta n$, raw | 0.370→0.480 | 0.370→0.471 | shared input-increment redistribution |
| actual $\Delta n$, gain | 0.115→0.314 | 0.120→0.311 | shared gain-weighted input redistribution |
| nonlinear H/M/T response | 0.159→0.309 | 0.172→0.317 | shared broad-band response redistribution |

For raw writes, coarse head/middle/tail shares change from about 0.266/0.425/0.301 to 0.119/0.403/0.475; fine changes from 0.268/0.427/0.313 to 0.116/0.419/0.463. Thus “later layers shift right” is a clear local-rank observation, but it does not distinguish semantic granularity.

## 9. Fine-Specific Location And Effective-Use Chain

| Quantity | $T$ | 95% interval | Robustness | Verdict |
| --- | ---: | ---: | --- | --- |
| raw write, native basis | -0.01071 | [-0.01784, 0.00447] | design/confirmation negative; 0/8 LOO positive | Fail |
| gain write, native basis | +0.01371 | [-0.00072, 0.01950] | 8/8 LOO positive; CI crosses 0 | Fail |
| gain write, folded basis | +0.01306 | [0.00165, 0.02301] | both splits, 8/8 LOO, all widths positive | Pass |
| actual $\Delta n$, raw | +0.00880 | [-0.00267, 0.01684] | splits/LOO/width positive; CI crosses 0 | Fail |
| actual $\Delta n$, gain | +0.01630 | [0.00599, 0.02833] | both splits, 8/8 LOO, all widths positive | Primary Pass |
| nonlinear MLP H/M/T | +0.00514 | [-0.00298, 0.01620] | interval crosses 0 | Fail |

The primary effect is 0.01630 of the local spectrum, or 66.8 of 4096 ranks. It is a relative allocation result, not an absolute claim that middle/tail has more variance than head at every layer.

![Layerwise centroid trajectories](figures/figure3_layerwise_centroid_curves.png)

![MLP response and non-additivity](figures/figure4_mlp_response_and_nonadditivity.png)

## 10. Relation To The Senior Experiment

The senior report used Qwen3-0.6B, an MLP-input SAE, layers 3/14/26, and constituent-exclusive versus composite-exclusive feature populations. It separated raw feature projection variance from singular-value-weighted transmitted variance. This experiment retains that raw/gain distinction but changes the semantic object and adds missing guards:

- balanced coarse versus conditional-fine class covariance instead of unequal SAE role populations;
- Qwen3-8B and all 36 layers;
- post-`o_proj` write plus exact RMSNorm-induced MLP-input increment;
- cross-layer projector overlap rather than assuming equal rank means equal direction;
- nonlinear H/M/T MLP response.

Therefore the two results are complementary, not direct replications. The senior result motivates parameter coordinates; it cannot establish the current coarse/fine or cross-layer claims.

## 11. Claim Ledger

| Claim | Verdict | Boundary |
| --- | --- | --- |
| Later layers show an overall right shift in their own parameter-rank coordinates | Supported descriptively | Both coarse and fine; one model/taxonomy |
| Fine raw write moves farther right than coarse | Not supported | Registered relative statistic fails |
| Gain-weighted actual MLP input has a fine-relative right shift | Supported | Static input-energy diagnostic |
| MLP parameter commonality is concentrated only in head | Refuted | Tail-256 exceeds head-256 |
| Nonlinear MLP effectively uses a fine-specific later-rank coordinate | Not established | Response CI crosses zero |
| Frequency or rare/high-level knowledge causes the pattern | Not tested | No frequency-matched intervention |
| The coordinate should guide a Router | Not tested | No utility experiment admitted |

## 12. Artifact Map

- [Protocol](protocol.md)
- [Summary](summary.md)
- [Typed verdict](provenance/verdict.json)
- [Layerwise centroid table](tables/layerwise_centroids.csv)
- [Parameter commonality table](tables/parameter_commonality.csv)
- [Figures](figures/)
- Worker surface: `XingyuD/MoE_Routing_Experiments/active/a15_01_09_e01_qwen3_mlp_parameter_commonality_attention_write_semantic_location/`
- Consolidated run: `a15-01-09-e01-consolidated-20260807T061000Z`

## 13. One Next Decision

Decide whether to run one independent balanced-taxonomy replication of the gain-weighted $\Delta n$ effect, with nonlinear response as the admission gate. Completion requires both $T_{\Delta n,gain}$ and $T_{MLP}$ to have positive 95% lower bounds on the new taxonomy. If either fails, close the fine-specific effective-location mechanism and retain only the broad shared layerwise redistribution plus U-shaped parameter-commonality observations.
