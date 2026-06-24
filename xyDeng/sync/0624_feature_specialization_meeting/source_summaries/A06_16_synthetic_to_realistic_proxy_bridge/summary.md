# Summary: A06_16 Synthetic-To-Realistic Proxy Bridge

Primary anchor:
`../../problem_anchors/06_geometry_proxy_preservation/06_16_synthetic_to_realistic_proxy_bridge_anchor.md`

Protocol:
`protocol.md`

## Purpose

A06_16 asks whether the bridge from clean synthetic feature routing to more realistic input mixtures fails first at label-free discovery or at same-model training preservation.

The corrected experimental object removes learned absolute positional embeddings. The model still has causal attention, but `pos_emb` is zeroed and frozen. RoPE is outside this anchor.

## Exact Setup

Canonical full run:

- ACP job `pt-vcl6tl57`;
- run name `a06_16_proxy_bridge_no_pos_same_model_audit_full_20260623_1`;
- one-layer Transformer + top-1 MoE;
- `d_model=128`, `ffn_dim=256`, 4 experts, 4 heads;
- sequence length 32, movable feature slot length 4;
- route position is the last slot token, `slot_start + 3`;
- feature id, nuisance id, and slot start are strictly balanced;
- C0-C4, 8 seeds;
- C0-C3 train for 1600 steps after pseudo discovery;
- C4 tests all-position discovery only;
- preservation reuses the same discovery model state: `reuse_discovery_model_state=true`.

Old raw A06_16 result directories were deleted before this rerun because the previous preservation stage used cross-model center transfer. That old trajectory is not used for the current conclusion.

## Primary Metric

Primary metric: `first_failed_stage`.

The decisive submetrics are step-0 pseudo `feature_NMI`, final same-model training `feature_NMI`, final target accuracy, load imbalance, and feature-to-expert mapping changes.

## Result

Same-model A06_16 passes C0-C3 end to end.

| Cell | Step-0 pseudo NMI | Final feature NMI | Final target acc | Final load imbalance | Stage |
| --- | ---: | ---: | ---: | ---: | --- |
| C0 clean | 1.000 | 1.000 | 1.000 | 0.000 | pass |
| C1 mild nuisance | 1.000 | 1.000 | 1.000 | 0.000 | pass |
| C2 moderate nuisance | 1.000 | 1.000 | 1.000 | 0.000 | pass |
| C3 weak feature + nuisance | 1.000 | 1.000 | 1.000 | 0.000 | pass |
| C4 all-position | 0.871 | n/a | n/a | n/a | 7/8 pass |

Training heatmap audit confirms there is no hidden feature merge in C0-C3 after training: final dominant-feature fraction is `1.0` for every feature in every seed, and final dominant expert changes are `0/128`.

C4 is the only remaining within-anchor boundary. Its all-position pseudo NMI averages `0.871`, with 7/8 seeds passing and one seed at `0.637`. The C4 failure mode is whole-feature merge at route evaluation, not nuisance or slot-start alignment: pseudo nuisance NMI and slot-start NMI are both `0.0`.

## Key Figures

### Figure: Effective Geometry By Cell

![Effective geometry by cell](figures/effective_geometry_by_cell.png)

What this tests: whether the input nuisance knobs change measured hidden-state geometry.

Metric shown: mean $\rho_{\mathrm{eff}}=\beta_{\mathrm{eff}}/(\lambda_{\mathrm{eff}}+\delta)$.

Observed result: C0/C1 have high $\rho_{\mathrm{eff}}$ around 586/562; C2/C3 drop to about 26/25.

Allowed claim: nuisance knobs still create harder hidden geometry without learned absolute positional embeddings.

What this does not prove: RoPE behavior, real-language semantics, or whole-slot composition.

### Figure: Step-0 Discovery

![Bridge first failed stage](figures/bridge_first_failed_stage.png)

What this tests: whether label-free residual k-means finds feature centers before training.

Observed result: C0-C3 pseudo and oracle discovery are both `1.0`; C4 all-position pseudo NMI averages `0.871`.

Allowed claim: route-position discovery is not the bottleneck in the corrected no-pos bridge.

### Figure: Same-Model Preservation

![NMI trajectory by bridge cell](figures/nmi_trajectory_by_bridge_cell.png)

What this tests: whether ordinary target training preserves routing after valid pseudo initialization in the same model.

Observed result: C0-C3 remain at `feature_NMI=1.0` through 1600 steps, with target accuracy reaching `1.0`.

Allowed claim: in this controlled no-pos slot-end bridge, same-model pseudo initialization is preserved through training.

## Claim Boundary

Can claim:

- learned absolute positional embeddings caused the earlier false C0 discovery failure;
- without learned absolute positional embeddings, route-position residual k-means discovers feature centers in C0-C3;
- same-model training preserves the discovered C0-C3 partition for 8/8 seeds;
- C4 all-position discovery mostly works but is not fully stable.

Cannot claim:

- RoPE behavior;
- robust all-position discovery under every position encoding;
- whole-slot compositional semantics, because slot-end tokens remain feature-specific;
- real-DCLM preservation;
- expert utility or deployable gating.

## Next Decision

Do not treat preservation as the next bottleneck for this controlled no-pos bridge. The next clean uncertainty is C4-style all-position robustness or a separate boundary branch: RoPE, whole-slot composition, or real-DCLM preservation.
