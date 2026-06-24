# Summary: A06_10 Real-DCLM Proxy Feature Operationalization

## Purpose

This experiment asks whether real DCLM router-input hidden states contain a stable label-free proxy feature that can replace synthetic `pair_id` for the next gating-initialization experiments.

## Setup

- Model: random-initialized Qwen-style top-1 MoE, 6 layers, hidden size 512, 8 experts.
- Data: packed DCLM bin stream, sequence length 256, same loader family as A06_02.
- Seeds: 0, 1, 2.
- Checkpoints: step 0 and step 10.
- Layers: 0, 3, 4, 5.
- Splits: two independent calibration splits and one held-out evaluation split.
- Core method: fit cluster centers on calibration A and B, match centers, then compare their assignments on the same held-out evaluation tokens.

Run:

```text
job_id: pt-796e62re
run_name: a06_10_real_dclm_proxy_4gpu_20260622_full01
run_dir: runs/real_dclm_proxy_feature_operationalization/a06_10_real_dclm_proxy_4gpu_20260622_full01
```

## Primary Metric

Primary metric: held-out proxy assignment stability, measured as `eval_assignment_nmi` after center matching.

This metric decides the anchor because a real-data proxy is useful only if two independent calibrations assign the same held-out tokens similarly.

## Result

Decision: supported, with an important boundary.

Real DCLM hidden geometry contains stable label-free proxy clusters above random. The strongest signal appears after 10 training steps in late layers. However, common-centering is not what creates the clusters: raw and residual k-means give effectively the same proxy assignments because k-means is translation-invariant. Therefore A06_10 supports a real proxy-feature audit, but weakens the claim that simple common-centering is the key discovery operation.

Key numbers:

| Checkpoint | Condition | Mean stability NMI | Random NMI | Frequency NMI | Position NMI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | residual | 0.2565 | 0.0002 | 0.0215 | 0.0134 |
| 10 | residual | 0.5211 | 0.0002 | 0.0401 | 0.0736 |
| 10 | spherical residual | 0.4854 | 0.0002 | 0.0435 | 0.0569 |
| 10 | whitened residual | 0.4339 | 0.0002 | 0.0394 | 0.0735 |

Layer-specific residual stability:

| Checkpoint | Layer 0 | Layer 3 | Layer 4 | Layer 5 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.4489 | 0.2005 | 0.1959 | 0.1806 |
| 10 | 0.4488 | 0.4681 | 0.5172 | 0.6501 |

The top-frequency removal guard did not remove the signal. For step-10 layer-5 residual clusters, the best rows remain near 0.68--0.79 NMI after removing top 1% or top 5% frequent tokens.

## Central Figures

![Proxy stability by condition](figures/full01_proxy_stability_by_condition.png)

This figure tests whether independent proxy-center estimates produce stable held-out assignments. Step-10 residual/raw/spherical conditions are well above random. It does not prove semantic meaning or expert utility.

![Frequency nuisance NMI](figures/full01_nuisance_frequency_nmi_by_condition.png)

This figure tests whether proxy labels are mostly frequency buckets. Frequency NMI is much lower than proxy stability NMI, so frequency alone does not explain the proxy. It does not rule out finer token-identity effects.

![Position nuisance NMI](figures/full01_nuisance_position_nmi_by_condition.png)

This figure tests whether proxy labels are mostly sequence-position buckets. Position NMI is low compared with stability NMI, though it increases at step 10 and remains a guard for later anchors.

## Claim Boundary

Can claim:

- a reproducible real-DCLM proxy partition exists in hidden-state geometry;
- the proxy is not explained mainly by the measured frequency or position nuisances;
- step-10 late-layer proxy structure is much stronger than step-0 late-layer structure.

Cannot claim:

- semantic feature discovery;
- expert specialization;
- training preservation;
- that common-centering improves k-means proxy discovery;
- that the proxy is already linearly routable by a router.

## Next Decision

Proceed to A06_11, but carry the boundary forward:

1. Use A06_10 proxy labels as evaluation targets.
2. Test whether equal-norm proxy centers can be converted into linear router rows.
3. Separate actual raw-input routing from centered-input diagnostic routing, because A06_10 did not prove common-centering is enough for a real router.
