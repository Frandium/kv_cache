# Detailed: A06_18 Revision Audit

Primary anchor:
`../../../problem_anchors/06_geometry_proxy_preservation/06_18_label_free_route_relevant_state_selection_anchor.md`

Protocol: `protocol.md`

Summary: `summary.md`

## 0. Quick Recap

Purpose: test whether representation clustering can replace the failed
A06_18 split-stability selector.

Decision question: can a label-free representation of all-position hidden
states produce hidden-space gating centers that recover held-out route-position
features better than raw all-position k-means?

Conclusion: no variant passes. PCA q=4 is weakly informative but not strong
enough. SAE and bottleneck AE do not solve the route-relevant selection problem.

## 1. Protocol Compliance

approved conditions match actual conditions: yes.

primary metric exists: yes, held-out route-position `feature_NMI`.

guards exist: yes, max load, active experts, nuisance NMI, slot-start NMI,
cluster role composition, cluster position composition, and reconstruction MSE.

seeds recorded: yes, `20260623` to `20260630`.

no labels used for fitting: yes for all non-control variants. Feature labels
are used only for final evaluation and interpretation.

no A06_19 created: yes.

## 2. Setup

Data:

- A06_17 `C4_all_position_scope`.
- Sequence length: 32.
- Feature slot length: 4.
- Slot starts: 2, 10, 18.
- Route position: slot offset 3.
- Calibration samples per `(feature, nuisance, slot_start)`: 256.
- Evaluation samples per `(feature, nuisance, slot_start)`: 256.

Model:

- One-layer Transformer plus top-1 MoE from A06_16.
- Positional embedding disabled.
- `d_model=128`, `n_heads=4`, `ffn_dim=256`.
- Features / experts: 4 / 4.

Compared pools:

- raw all-position k-means;
- route-only positive control;
- slot-offset-3 positive control;
- A06_18 split-stability top-1/top-3/threshold selectors;
- PCA latent clustering `q=4,8,16,32`;
- bottleneck AE latent clustering `q=4,8,16,32`;
- SAE-code clustering with `d_sae=4*d_model, 8*d_model`, L1/top-k variants.

Representation rule:

```text
fit representation without feature labels
-> cluster all-position states in representation space
-> average original hidden states per latent cluster
-> evaluate those hidden-space centers on held-out route-position states
```

## 3. Main Table

| Pool | Mean `feature_NMI` | Min | Max | Perfect seeds | Mean max load | Mean active experts |
|---|---:|---:|---:|---:|---:|---:|
| Route-only | 1.000 | 1.000 | 1.000 | 8/8 | 0.250 | 4.0 |
| Slot offset 3 | 1.000 | 1.000 | 1.000 | 8/8 | 0.250 | 4.0 |
| PCA q=4 | 0.871 | 0.637 | 1.000 | 2/8 | 0.469 | 3.1 |
| PCA q=16 | 0.851 | 0.637 | 1.000 | 2/8 | 0.469 | 3.0 |
| Raw all-position | 0.831 | 0.637 | 1.000 | 2/8 | 0.469 | 2.9 |
| PCA q=32 | 0.818 | 0.637 | 0.866 | 0/8 | 0.531 | 2.8 |
| Bottleneck AE q=32 | 0.814 | 0.637 | 1.000 | 2/8 | 0.531 | 2.9 |
| Bottleneck AE q=4 | 0.809 | 0.637 | 0.866 | 0/8 | 0.562 | 2.8 |
| Bottleneck AE q=8 | 0.806 | 0.637 | 1.000 | 1/8 | 0.531 | 2.8 |
| PCA q=8 | 0.806 | 0.637 | 1.000 | 1/8 | 0.531 | 2.8 |
| Bottleneck AE q=16 | 0.789 | 0.637 | 0.866 | 0/8 | 0.562 | 2.6 |
| Split-stability top-3 | 0.778 | 0.587 | 1.000 | 1/8 | 0.452 | 3.4 |
| SAE L1 4x | 0.749 | 0.637 | 0.866 | 0/8 | 0.562 | 2.4 |
| Split-stability top-1 | 0.745 | 0.627 | 1.000 | 1/8 | 0.498 | 3.0 |
| SAE L1 8x | 0.729 | 0.000 | 0.866 | 0/8 | 0.594 | 2.6 |
| Split-stability threshold | 0.674 | 0.533 | 0.866 | 0/8 | 0.534 | 3.0 |
| SAE top-k 8x | 0.641 | 0.492 | 0.811 | 0/8 | 0.695 | 2.5 |
| SAE top-k 4x | 0.620 | 0.400 | 0.820 | 0/8 | 0.695 | 2.6 |

Full aggregate: `tables/pool_comparison_aggregate.csv`.

## 4. Guard Read

Nuisance NMI and slot-start NMI are near zero for representation variants, so
the failure is not a simple nuisance/slot-start shortcut.

Load remains a problem for weak variants. SAE top-k has max load around `0.695`
and never reaches a perfect seed.

Reconstruction is decoupled from route readout:

| Pool | Mean `feature_NMI` | Reconstruction MSE | Mean active code |
|---|---:|---:|---:|
| PCA q=4 | 0.871 | 0.9145 | 4.00 |
| PCA q=16 | 0.851 | 0.7218 | 16.00 |
| Bottleneck AE q=32 | 0.814 | 0.5962 | 15.98 |
| SAE L1 4x | 0.749 | 0.0126 | 258.79 |
| SAE L1 8x | 0.729 | 0.0034 | 517.53 |
| SAE top-k 4x | 0.620 | 0.1262 | 8.00 |
| SAE top-k 8x | 0.641 | 0.0314 | 8.00 |

Interpretation: SAE can reconstruct the all-position hidden states very well,
but its code clusters still do not recover the route-position feature
partition.

## 5. Failure Decomposition

| Stage | Evidence | Status | Interpretation |
|---|---|---|---|
| Route geometry guard | route-only and slot-offset-3 are `1.0` in 8/8 seeds | passed | feature geometry exists at the route position |
| Raw all-position baseline | mean `0.831`, 2/8 perfect seeds | reproduced | sample-pool mismatch remains |
| Split-stability selector | means `0.674` to `0.778` | failed | stability does not imply route relevance |
| PCA latent clustering | best mean `0.871`, only 2/8 perfect | weak / not passed | small denoising gain, not reliable route selection |
| Bottleneck AE clustering | best mean `0.814` | failed | compressed reconstruction objective does not target route readout |
| SAE code clustering | best mean `0.749`; top-k worse | failed | sparse reconstruction is not route relevance |

The key failure is not absence of route-position geometry. It is that generic
unsupervised representations optimize population reconstruction or variance,
while the desired object is a small route-readout partition.

## 6. What This Updates

Falsified operationalization:

- split-stability selector;
- PCA/AE/SAE latent clustering as a sufficient selector;
- reconstruction MSE as a proxy for route-relevant partition quality.

Still alive:

- route-relevant geometry is population-dependent;
- a selector may need route-local, pre-target, or route-readout constraints;
- a passing controlled selector should later be tested in slot early training.

## 7. Next Step

Do not promote SAE as the main method. The next A06 revision should make the
selector explicitly route-readout aware while staying label-free. Minimal
candidates:

- route-local selector: only pre-target / route-neighborhood states are
  eligible;
- self-supervised route-readout selector: score candidate center sets by
  agreement/stability on held-out route-position states without feature labels;
- causal route sensitivity selector: choose states whose perturbation changes
  next-token route-position readout more than neutral positions.

Only after a controlled selector approaches route-only should we write a short
A06_19 and test the same initialization in slot early training.

## 8. Full Run Record

Run name:

```text
a06_18_revision_full_local_2gpu_20260625
```

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 python active/synthetic_data_understanding/scripts/run_a06_18_label_free_route_relevant_state_selector.py --config active/synthetic_data_understanding/configs/a06_18_label_free_route_relevant_state_selector.json --run-name a06_18_revision_full_local_2gpu_20260625 --run-stage full --num-shards 2 --shard-index 0 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python active/synthetic_data_understanding/scripts/run_a06_18_label_free_route_relevant_state_selector.py --config active/synthetic_data_understanding/configs/a06_18_label_free_route_relevant_state_selector.json --run-name a06_18_revision_full_local_2gpu_20260625 --run-stage full --num-shards 2 --shard-index 1 --device cuda:0
python active/synthetic_data_understanding/scripts/run_a06_18_label_free_route_relevant_state_selector.py --config active/synthetic_data_understanding/configs/a06_18_label_free_route_relevant_state_selector.json --run-name a06_18_revision_full_local_2gpu_20260625 --run-stage full --aggregate-only
```

Artifacts:

- `tables/pool_comparison.csv`
- `tables/pool_comparison_aggregate.csv`
- `tables/representation_audit.csv`
- `tables/cluster_role_composition.csv`
- `tables/cluster_position_composition.csv`
- `figures/pool_feature_nmi.png`
- `figures/pool_feature_nmi_heatmap.png`
