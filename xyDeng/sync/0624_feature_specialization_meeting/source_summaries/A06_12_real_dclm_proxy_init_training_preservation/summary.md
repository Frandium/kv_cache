# Summary: A06_12 Real-DCLM Proxy-Init Training Preservation

## Purpose

This experiment tests whether the A06_11 raw-center proxy router initialization survives early real-DCLM next-token training.

## Setup

- Seeds: 0, 1, 2.
- Layers: 3, 4, 5.
- Variants: `random`, `equal_norm_random`, `raw_center`, `residual_center`.
- Training: standard next-token loss, checkpoints 0, 1, 2, 3, 5, 10, 20, 50.
- Fixed target: step-0 proxy labels on the held-out eval tokens.
- Job: `pt-8577iw38`.
- Run: `runs/real_dclm_proxy_init_training_preservation/a06_12_proxy_init_training_4gpu_20260622_full01`.

## Result

Decision: not supported.

Raw-center initialization creates strong proxy routing at step 0, but ordinary DCLM training rapidly overwrites it. The collapse is already near-complete by step 5 and random-like by step 10. LM loss is comparable to random, so this is not an optimization crash; it is training feedback overriding the proxy partition.

| Variant | Step 0 NMI | Step 5 NMI | Step 10 NMI | Step 50 NMI | Step 10 max load | Step 10 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.0379 | 0.0147 | 0.0047 | 0.0339 | 0.9830 | 10.7190 |
| equal-norm random | 0.0351 | 0.0166 | 0.0037 | 0.0297 | 0.9939 | 10.7171 |
| raw-center | 0.7549 | 0.0410 | 0.0131 | 0.0337 | 0.9645 | 10.7222 |
| residual-center | 0.2712 | 0.0282 | 0.0038 | 0.0399 | 0.9953 | 10.7246 |

## Central Figures

![Proxy route NMI trajectory](figures/full01_trajectory_proxy_route_nmi.png)

This figure tests preservation. Raw-center starts high and collapses to random-like values by step 10.

![Route max load trajectory](figures/full01_trajectory_route_max_cluster_load.png)

This figure tests whether collapse coincides with route concentration. All variants show strong early concentration around step 5--10.

![LM loss trajectory](figures/full01_trajectory_lm_loss.png)

This figure tests whether proxy-init failure is simply worse training. Loss is comparable across variants.

## Claim Boundary

Can claim: ordinary DCLM next-token training does not preserve the proxy-center initialization in this setup.

Cannot claim: proxy features are useless, expert utility is absent, or no preservation objective can work.

## Next Decision

Run A06_13 failure decomposition. The leading diagnosis is training feedback override, not proxy discovery failure or step-0 linear bridge failure.
