# A15_01_09_E01 Summary: MLP Parameter Commonality And Semantic Location

## Verdict

The typed verdict is **`location_without_commonality`**, with the nonlinear effective-use clause failing.

- A large descriptive right shift exists with model depth, but it is shared by coarse and fine semantics. In raw post-`o_proj` writes, the coarse/fine local-rank centroids move from 0.342/0.353 in blocks 1--12 to 0.479/0.470 in blocks 25--35.
- The proposed head-specific common space is false. Native head-256 overlap is 2.20 times the equal-rank Haar expectation and exceeds middle-256 at 1.04 times, but tail-256 is still larger at 3.26 times; after folding RMSNorm scale the values are 2.05, 1.13, and 7.04.
- After removing the common layer effect, raw fine semantics do not move farther right than coarse semantics: $T_{write,raw}=-0.0107$ with 95% interval $[-0.0178,0.00447]$.
- The registered primary statistic does pass on the actual MLP-input increment after parameter-gain weighting: $T_{\Delta n,gain}=+0.01630$, equivalent to 66.8 local ranks, with interval $[0.00599,0.02833]$. Design/confirmation templates, 8/8 leave-one-parent contrasts, and 128/256/512 rank widths agree.
- The nonlinear MLP H/M/T intervention has a positive point estimate but is not confirmed: $T_{MLP}=+0.00514$ with interval $[-0.00298,0.01620]$. Therefore the experiment does not establish effective nonlinear use of a later-rank fine-semantic coordinate.

The narrow knowledge update is:

> Later layers allocate both coarse and fine attention-write variance to later ranks of their own MLP parameter spectra. That is a broad layer effect, not raw evidence that fine semantics move farther right. Parameter gain reveals a stable fine-relative shift at the actual MLP input, but the proposed head-only common-space mechanism and nonlinear-use claim do not survive.

![Decisive evidence](figures/figure0_decisive_composite.png)

## What “Right Shift” Means

Every layer diagonalizes its own MLP input operator

$$
K_\ell=W_{gate,\ell}^{\top}W_{gate,\ell}+W_{up,\ell}^{\top}W_{up,\ell}.
$$

The horizontal coordinate is that layer's locally sorted parameter-rank percentile. A centroid increase means that a larger share of semantic between-class variance lies in lower-gain ranks of the same layer. It never means that layer $\ell$'s direction 500 is the same vector as layer $m$'s direction 500. Cross-layer sharing is measured separately by projector overlap.

## Main Evidence

| Clause | Direct statistic | Result |
| --- | --- | --- |
| Head-specific parameter commonality | native H/M/T overlap divided by Haar: 2.20/1.04/3.26 | Fail: tail exceeds head |
| Overall raw-write shift | coarse +0.137; fine +0.117 centroid | Strong descriptive shared shift |
| Raw fine-specific shift | $T=-0.0107$, CI crosses zero | Fail |
| RMSNorm-folded gain-weighted write | $T=+0.0131$, CI $[0.00165,0.0230]$ | Pass |
| Gain-weighted actual input increment | $T=+0.0163$, CI $[0.00599,0.02833]$ | Primary Pass |
| Nonlinear MLP response | $T=+0.00514$, CI crosses zero | Fail |

The complete layer×layer and bandwise parameter heatmaps show a U-shaped commonality pattern rather than head-only alignment.

![Layer-by-band parameter commonality](figures/figure1c_layer_band_parameter_commonality.png)

The semantic heatmaps use within-role normalized variance shares, so coarse and fine have the same denominator meaning. Red means fine allocates more of its own semantic variance than coarse to that local band; it does not mean fine has more absolute variance.

![Semantic-location heatmaps](figures/figure2_semantic_location_heatmaps.png)

## Validity And Boundary

All 36 layers and all 512 frozen semantic records are covered. Parameter eigensolver residual is at most $1.95\times10^{-6}$, orthogonality error at most $2.73\times10^{-6}$, semantic projection-energy error $2.39\times10^{-7}$, cached post-`o_proj` replay error exactly zero, and $\Delta n$ reconstruction error at most $2.73\times10^{-6}$. Smoke and all three full jobs succeeded with zero retries.

This result does not identify word frequency, rare knowledge, knowledge-learning sites, individual cross-layer singular directions, causal necessity, SAE feature identity, or Router utility. The one next decision is whether an independent balanced taxonomy should attempt to reproduce the gain-weighted $\Delta n$ shift and convert the currently non-significant nonlinear MLP response into a valid functional admission signal. No Router experiment is admitted by this result.

