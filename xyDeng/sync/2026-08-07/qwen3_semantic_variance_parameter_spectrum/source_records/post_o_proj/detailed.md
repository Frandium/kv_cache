# A15_01_05_E04 Detailed Evidence Ledger

## 1. Question And Verdict

**Question.** After Qwen3-8B mixes all attention heads through `o_proj` but before residual addition, does the full attention branch retain the late-layer increase in fine-relative-to-coarse semantic discriminability, and where does that signal lie in its natural-text covariance spectrum?

Typed verdict: **`depth_pass_spectral_fail`**.

| Clause | Verdict | Direct reading |
| --- | --- | --- |
| Post-`o_proj` depth effect | Pass | Late fine-relative discriminability is larger than early discriminability |
| Actual-variance head dominance | Pass | F1 has maximum $b$ for both roles in every block 1--35 |
| Stable middle relative enrichment | Pass, local clause | +0.063 exceeds the registered random q95 and all local stability guards |
| Stable tail relative enrichment | Fail | +0.159 fails the independent-half per-late-layer direction guard |
| Joint spectral clause | Fail | The registered joint rule requires stable middle and tail evidence |

## 2. Exact Architecture Boundary

For block $\ell$:

$$
a_\ell=\operatorname{Concat}(head_1,\ldots,head_{32}),
\qquad
g_\ell=W_{O,\ell}a_\ell.
$$

E04 captures $g_\ell$, the output of `self_attn.o_proj`, before residual addition. The compared boundaries are:

```text
pre-o_proj a -> post-o_proj g -> residual addition -> RMSNorm -> MLP input
```

This prevents the pre-`o_proj` concatenation, post-`o_proj` branch, block residual, and MLP input from being treated as the same representation.

## 3. Frozen Model And Data

| Item | Frozen value |
| --- | --- |
| Model | `/data/share/Qwen3-8B` |
| Architecture | 36 decoder blocks, hidden size 4096, 32 query heads, head dimension 128 |
| Forward | frozen bfloat16, no fine-tuning |
| Semantic data | 8 parents × 8 children × 4 templates × 2 fact bundles = 512 records |
| Readout | shared final `Classification:` colon, absolute position 57 |
| Semantic dataset SHA-256 | `cb440b98d81bac3f9813344f85e6efdbd994b7b988d8009ba64e207e64a11859` |
| Natural calibration | 128 DCLM documents × 512 tokens = 65,536 tokens |
| Extracted calibration tensor SHA-256 | `d88620267dbdc9c2b87ec24c2f51d9e6fb317cb8c051ec96503414ea9bff60b9` |

All actual semantic strings, token ids, hierarchy labels, templates, fact bundles, and readout spans are in [actual_semantic_text_sequences.json](provenance/actual_semantic_text_sequences.json). The independent natural-text source and order are in [calibration_manifest.json](provenance/calibration_manifest.json). Model identity is in [model_manifest.json](provenance/model_manifest.json).

## 4. Metrics

For semantic role $s$:

$$
D_s=\frac{\operatorname{tr}(B_s)}{\operatorname{tr}(W_s)+\epsilon},
\qquad
R=\log\frac{D_{fine}}{D_{coarse}}.
$$

$D$ measures separation of semantic class centers relative to within-class template/fact variation. $R>0$ means fine classes are relatively more separable than coarse classes at that block. It is not prediction accuracy, expert utility, or training benefit.

The primary depth statistic is

$$
T_{post}
=\operatorname{median}_{25:35}R_\ell
-\operatorname{median}_{1:12}R_\ell.
$$

Natural-background covariance $\Sigma=U\Lambda U^\top$ defines 16 equal-rank 256-dimensional bands: F1=head, F2--F8=middle, F9--F16=tail. Per band:

- $b$: actual between-class variance per direction;
- $q$: $b$ normalized by natural-background variance;
- $j$: band-local between/within discriminability;
- $e$: band $b$ divided by the role's full-spectrum per-direction mean, used to compare relative variance allocation.

## 5. Implementation And Execution

Worker surface:

`XingyuD/MoE_Routing_Experiments/active/a15_01_05_e04_qwen3_post_o_proj_full_attention_semantic_variance_atlas/`

The code reuses E03 semantic construction, calibration ordering, moment accumulation, bootstrap, null controls, and plotting. The only scientific representation change is the hook from the direct input of `o_proj` to its output.

Formal ACP job:

| Field | Value |
| --- | --- |
| Job | `om-5y1d8uf1` |
| Platform | SCO ACP |
| Resource | one node, 8×5090 SPOT, `n12lp.nn.i10a.8` |
| Run | `a15-01-05-e04-post-o-proj-20260806T172600Z` |
| Terminal state | `SUCCEEDED`, zero retries |

The earlier job `om-1hc00w00` stopped during smoke because its diagnostic changed the bfloat16 matrix-multiplication shape. It never entered valid scientific analysis. The repaired guard recomputes the full native tensor shape; no data, model, metric, split, seed, threshold, or treatment changed.

## 6. Hard Guards

| Guard | Result |
| --- | ---: |
| Native post-`o_proj` output vs direct full-shape linear replay | absolute 0; relative 0 |
| FP32 Gram vs direct FP64 maximum relative error | $7.22\times10^{-8}$ |
| Layer coverage | 36/36 |
| Semantic records | 512/512 |
| Calibration tokens | 65,536/65,536 |
| Covariance reconstruction and projection energy | Pass |
| H/M/T calibration-half basis stability | Pass |
| Template, parent leave-one-out, nuisance and floor guards | Pass |

The rendered versions of all four central figures were opened and checked. Axes, legends, colorbars, blocks 35/36, and sign conventions are readable.

## 7. Depth Evidence

| Quantity | Value |
| --- | ---: |
| Early blocks 1--12 median $R$ | 0.131; fine/coarse ratio 1.14 |
| Late blocks 25--35 median $R$ | 0.762; fine/coarse ratio 2.14 |
| $T_{post}$ | 0.631 |
| Hierarchical-bootstrap 95% interval | [0.553, 1.148] |
| Block 35 | $R=0.757$; ratio 2.13 |
| Block 36 | $R=0.857$; ratio 2.36 |

All template contrasts are positive: 0.541, 0.328, 0.389, 0.669. All leave-one-parent contrasts are positive: 0.553, 0.733, 0.614, 0.648, 0.797, 0.691, 0.540, 0.654.

![Cross-site depth comparison](figures/figure1_cross_site_depth_comparison.png)

The pre- and post-`o_proj` curves are close and both retain the late increase. The MLP-input curve has a different early baseline and rises later. These are descriptive architecture-boundary observations, not module effects.

## 8. Spectral Evidence

Actual between-class variance $b$ is maximal in F1 for coarse and fine semantics in every block 1--35.

| Group | Late fine/coarse log enrichment | 95% interval | Multiplicative reading | Registered result |
| --- | ---: | ---: | ---: | --- |
| Head | -0.106 | [-0.165, -0.064] | 0.900×; 10.0% less relative share | descriptive depletion |
| Middle | +0.063 | [0.053, 0.108] | 1.065×; 6.5% more relative share | local Pass |
| Tail | +0.159 | [0.135, 0.249] | 1.172×; 17.2% more relative share | Fail: half-basis instability |

The equal-dimensional random-direction late q95 is 0.059. Middle is narrowly above that threshold and passes design/confirmation, both calibration-half bases, nuisance residualization, and eigenvalue-floor checks. Tail is larger but is not positive in every late layer under both independent-half bases. The joint spectral clause therefore fails.

![Post-o_proj layer-band atlas](figures/figure2_post_o_proj_layer_band_heatmaps.png)

![Post-o_proj band-local discriminability](figures/figure3_decisive_post_o_proj_band_discriminability.png)

The middle result concerns relative actual variance allocation $e$. It does not say that middle has the largest absolute semantic variance, the best $B/W$ discriminability, or any expert-routing value.

## 9. Descriptive Architecture-Boundary Comparison

| Representation | Early $R$ | Late $R$ | $T$ | Block 35 $R$ | Block 36 $R$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-`o_proj` attention branch | 0.131 | 0.762 | 0.631 | 0.757 | 0.857 |
| Pre-`o_proj` concatenated heads | 0.153 | 0.764 | 0.611 | 0.686 | 0.792 |
| Raw block residual | 0.001 | 0.452 | 0.451 | 0.601 | 0.609 |
| MLP input | -0.260 | 0.819 | 1.079 | 1.133 | 1.075 |

The shared conclusion across boundaries is late fine-relative enhancement without final-block reversal. Numerical differences cannot be attributed causally to `o_proj`, residual addition, RMSNorm, or MLP preparation because the representations, covariance bases, scales, and nonlinear $B/W$ ratios differ.

## 10. Claim Ledger

| Claim | Verdict | Boundary |
| --- | --- | --- |
| Late fine-relative discriminability exists after `o_proj` and before residual addition | Pass | One frozen Qwen3-8B and one taxonomy |
| Actual coarse/fine semantic variance is F1-dominant | Pass | Per-direction $b$, blocks 1--35 |
| A modest late middle relative-share enrichment exists | Local Pass | $e$ only; one taxonomy; not expert utility |
| A stable tail coordinate exists | Fail | Independent-half per-late-layer direction guard fails |
| `o_proj` causes the middle enrichment | Not tested | No causal intervention |
| Middle should guide a Router | Not tested | No compatibility or matched training |

## 11. Artifact Map

- Protocol: [protocol.md](protocol.md), [protocol_cn.md](protocol_cn.md)
- Summary: [summary.md](summary.md), [summary_cn.md](summary_cn.md)
- Typed verdict: [verdict.json](provenance/verdict.json)
- Complete tables: [tables/](tables/)
- Figures: [figures/](figures/)
- Actual semantic sequences: [actual_semantic_text_sequences.json](provenance/actual_semantic_text_sequences.json)
- Calibration manifest: [calibration_manifest.json](provenance/calibration_manifest.json)
- Model manifest: [model_manifest.json](provenance/model_manifest.json)

## 12. Next Decision

Use one independent, balanced, label-leakage-free taxonomy to test whether both positive post-`o_proj` depth contrast and late middle relative enrichment reproduce. Only a replicated middle signal should enter a later expert-utility gate.
