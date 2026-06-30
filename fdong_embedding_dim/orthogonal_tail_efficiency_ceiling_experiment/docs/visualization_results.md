# Results and visualization guide

## Primary held-out result

| hidden dim | top-2 common | bottom-2 spectral tail | unrestricted full output |
|---:|---:|---:|---:|
| 8 | 70 | 65 | **50** |
| 16 | 80 | **65** | **65** |

All entries are median Stage-2 stable tail steps over 45 held-out seeds. At dimension 8 unrestricted output beats top-2 by 21.8 mean steps and bottom-2 by 13.4; both paired bootstrap intervals exclude zero. At dimension 16 unrestricted output beats top-2 by 16 mean steps, but differs from bottom-2 by only 4 mean steps with a bootstrap interval `[-13.0, 4.11]` and one-sided Wilcoxon `p=0.454`.

Thus unrestricted output is the clear speed ceiling at dimension 8 and statistically tied with spectral tail at dimension 16.

## How to read the paired plot

`three_way_heldout_paired.png` plots one seed per point. The x-axis is a constrained branch's stable step and the y-axis is the unrestricted branch's stable step.

- points below the diagonal favor unrestricted output;
- dimension 8 strongly favors unrestricted output over both restricted branches;
- dimension 16 favors unrestricted output over top-2 but not over bottom-2.

![Three-way held-out paired comparison](/Users/bytedance/kv_cache/fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment/results/three_way_heldout_paired.png)

## LR sweep

`lr_sweep.png` is diagnostic rather than the primary test. It shows that comparing both branches at an arbitrary shared learning rate can reverse or hide the conclusion. The branches therefore receive independent LR tuning on seeds 0–4 before held-out evaluation.

`best_lr_comparison.png` summarizes the best LR on all ten sweep seeds. It is useful for debugging, but it is not the inferential result because those seeds participate in LR selection.

## Geometry checks

The final map and tied-embedding deltas have essentially unit energy in their assigned basis:

- common branch: common energy fraction approximately 1;
- spectral-tail branch: bottom-r residual energy fraction approximately 1;
- cross-subspace leakage is at floating-point noise scale;
- high-pattern retention is identical because its oracle gate uses the frozen stage-1 embedding and base model.

Thus the measured difference is not caused by unequal parameter counts, different initial predictions, base-model drift, or accidental leakage back into the common output basis.

## What the result does and does not establish

It establishes that perfect routing and an isolated unrestricted branch explain the fastest branch learning. It does not support the claim that a spectral-tail restriction is necessary for speed. At dimension 16, bottom-2 remains interesting as a parameter-efficient solution: it matches unrestricted median speed with 14 rather than 112 parameters.

None of the oracle branches beats the earlier full-model uniform/reweight median of roughly 40 steps. The practical question is therefore still whether learned routing or online subspace allocation can provide isolation without oracle information and without losing the efficiency of full-model frequency balancing.
