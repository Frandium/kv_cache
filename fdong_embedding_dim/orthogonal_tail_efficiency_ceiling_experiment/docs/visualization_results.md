# Results and visualization guide

## Primary held-out result

| hidden dim | held-out seeds | common LR | spectral-tail LR | common median | tail median | mean tail-common | tail wins / ties / losses | one-sided Wilcoxon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 15 | 0.3 | 0.3 | 65 | 65 | +2.67 | 7 / 0 / 8 | 0.533 |
| 16 | 45 | 0.3 | 1.0 | 80 | 65 | -12.0 | 28 / 2 / 15 | 0.008 |

Negative `tail-common` means the spectral-tail branch converged sooner. At dimension 16 the paired bootstrap 95% interval for the mean difference is `[-22.56, -1.78]` steps; the one-sided sign-test p-value is 0.033. At dimension 8 the interval crosses zero widely and the directions split evenly.

The dimension-16 median reduction is 15 steps, or 18.75% relative to the common branch. This is the constructive existence result.

## How to read the paired plot

`heldout_paired_comparison.png` plots one seed per point. The x-axis is the common branch's stable step and the y-axis is the spectral-tail branch's stable step.

- points below the diagonal favor the spectral tail;
- points above the diagonal favor common reuse;
- dimension 8 is balanced around the diagonal;
- dimension 16 has a visible majority below it, but retains real seed-to-seed failures and outliers.

![Held-out paired comparison](/Users/bytedance/kv_cache/fdong_embedding_dim/orthogonal_tail_efficiency_ceiling_experiment/results/heldout_paired_comparison.png)

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

It establishes that a more efficient spectral-tail setting can exist. This closes the specific logical gap left by the frequency-reweighting experiment: common-space reuse is not the only potentially efficient solution.

It does not establish that the tail basis is universally superior. Dimension 8 is a direct counterexample. It also does not identify why dimension 16 helps. The likely candidates are cleaner parameter isolation and better conditioning at a larger learning rate, but the current tied embedding means the chosen basis also changes how tail-context embedding deltas interact with frozen Q/K/V maps. A follow-up untied-readout or fixed-input ablation is needed to isolate that mechanism.

