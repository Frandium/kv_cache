# Summary: A15_01_01_E01 Controlled Four-Layer Shallow-Head Pilot

Primary anchor: [A15_01_01 controlled four-layer shallow-head pilot](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_01_shallow_head_guided_deep_routing/subanchors/15_01_01_controlled_four_layer_shallow_head_pilot_anchor.md)  
Protocol: [protocol.md](protocol.md)  
Detailed record: [detailed.md](detailed.md)

## Result Snapshot

**Registered terminal verdict:** `insufficient_stage_a_capability`.

**Direct result:** the full job completed correctly, but the Stage-A
specificity check was saturated. In all 10 task-seed states, the layer-2 head
decoded the coarse variable with accuracy 1.0, while the 95th percentile over
256 same-dimensional random-subspace probes was also 1.0. Therefore the
registered strict difference, head accuracy minus random q95, was 0 in 10/10
states and passed in 0/10 states.

**Knowledge update:** this setup shows that the coarse variable is readable
from the head, but not that it is concentrated specifically in the covariance
head. A generic 64-dimensional view already retains enough information for
perfect linear decoding. The current Stage-A operationalization therefore has
no specificity for the proposed shallow-head mechanism.

**Consequence:** the fail-closed implementation correctly stopped. B0, Stage 0
compatibility, and B1 four-arm training each produced zero records. Thus H2 was
never compared with N4, R2, or SH2.

**What this does not mean:** it is not evidence that shallow-head information
is absent, that H2 fails compatibility, that random routing is equally useful,
or that shallow-head guidance cannot improve training.

## Purpose And Decision Question

The full Protocol asks whether layer-2 head coefficients first provide
compatibility information beyond native and matched controls, and only then
whether they improve deeper-layer held-out loss per matched training FLOP.
Stage A is a validity gate: before using those coefficients as a treatment, it
must show that the intended coarse signal is specifically more accessible in
the head than in a generic subspace of the same dimension.

This run reached only that gate. Its decision question is therefore narrower:
**does the registered Stage-A probe distinguish head-specific capture from
generic 64-dimensional access?** The answer is no.

## Terms And Metric

- **Head probe:** a linear decoder of the coarse variable using ranks 1--64 of
  the covariance basis of the actual layer-2 Gate input.
- **Random q95:** the 95th percentile of held-out decoder accuracy across 256
  frozen Haar-random 64-dimensional subspaces. It controls for dimension and
  generic linear accessibility.
- **Specificity gap:** for task-seed state $s$,

$$
D_s=\operatorname{Acc}_{\mathrm{head},s}
-Q_{0.95}\!\left(\operatorname{Acc}_{\mathrm{random64},s}\right).
$$

Accuracy is a fraction of correctly decoded held-out examples; $D_s$ is in
accuracy points. The registered strict guard requires $D_s>0$ in every state.
It decides whether this Stage-A proxy distinguishes the head from random views.
It cannot decide compatibility or training benefit.

## Exact Setup

- **Tasks:** informative and nuisance synthetic controls.
- **Seeds:** 3101--3105, giving 10 task-seed states.
- **Model:** four residual top-1 MoE layers; width 256; eight experts per layer;
  expert width 512; side dimension 64.
- **Stage A:** 500 AdamW steps, batch 512, constant learning rate $3\times10^{-4}$,
  betas (0.9, 0.95), weight decay 0.01.
- **Evaluation:** 4,096 validation examples; two independent 2,048-example
  covariance-calibration halves; 2,048 probe-fit and 4,096 probe-test examples;
  256 random probes.
- **Other guards:** coarse accuracy at least 0.90, content explained variance
  at least 0.80, head accuracy at least 0.85, and split-half projector overlap
  at least 0.80.
- **Routing balance:** no load-balance auxiliary loss; the later registered
  arms share a non-gradient expert-score bias, but those arms were not reached.
- **Execution:** one idle ACP SPOT node with eight 5090 GPUs.

The frozen contract and full runner hashes were respectively
`84478ead8bfffd6b3b25710ad25ea3145cb8d1b00aeb54822fad840b98ae2a4a`
and
`218b9538e58eeba247333a3e6c153f6067c0eb3efd9349b7c09fb7bf68862e8f`.

## Key Evidence

| Registered check | Direct result | Pass count | Decision role |
| --- | ---: | ---: | --- |
| Head probe accuracy | 1.000 in every state | 10/10 | Confirms coarse readability from head |
| Random-subspace q95 | 1.000 in every state | not a pass guard alone | Reveals generic 64D saturation |
| Strict specificity, head $>$ random q95 | gap 0.000 in every state | **0/10** | Blocks the head-specific treatment claim |
| Coarse accuracy $\ge 0.90$ | 1.000 in every state | 10/10 | Confirms proxy task learned |
| Content explained variance $\ge 0.80$ | 0.994957--0.996855 | 10/10 | Guards against discarding content |
| Split-half overlap $\ge 0.80$ | 0.845114--0.899591 | 10/10 | Confirms projector stability |
| B0 / Stage 0 / B1 records | 0 / 0 / 0 | correctly blocked | Prevents unsupported downstream claims |

The random-probe minimum was also 1.0 in nine states and 0.999756 in the
remaining state. This is not a noisy near-threshold miss.

## Central Figure

![Stage-A specificity gate](figures/stage_a_specificity_gate.png)

**Figure contract.** The figure asks whether the covariance head gives more
held-out coarse-decoding access than a random 64-dimensional subspace. The
left panel reports both accuracies for every task-seed state without averaging;
the right reports $D_s$ in accuracy points and the number of downstream
records. Source: the formal full-run `stage_a_gate.json`. The figure supports
only the conclusion that this Stage-A specificity test is saturated and cannot
admit the head treatment. It does not compare compatibility or training
trajectories, because none were run.

## Three-Run Evidence Chain

| Run | Role | Outcome | Scientific use |
| --- | --- | --- | --- |
| `om-7c8jvl98` | Initial smoke | Invalid: downstream loss had no Gate-gradient path | Retained engineering failure only |
| `om-9xae345b` | Repaired smoke | 11/11 engineering guards passed | Authorized interpretation of the full runner |
| `om-demeqowk` | Formal full | `insufficient_stage_a_capability`; downstream blocked | Current terminal evidence |

The first smoke was not overwritten. The repair multiplied the selected expert
output by its selected softmax probability and added a mandatory Router
learning-path gradient guard. The full run then enforced the registered
Stage-A gate exactly.

## Claim Boundary

**Established:** the formal execution was valid; the coarse target and content
proxy were learned; the layer-2 projector was stable; both head and generic 64D
subspaces decoded the coarse target perfectly; the registered specificity
guard failed; and downstream stages were correctly not launched.

**Not established:** absence of shallow-head information, failure of H2,
compatibility equivalence with controls, matched-FLOP training benefit or harm,
Router learning dynamics under H2, from-scratch benefit, or natural-language
transfer. The anchor is updated only to record that the current Stage-A
specificity test is non-discriminating; the physical prior and H2 remain open.

## Next Decision

Exactly one decision remains: **whether to approve a new Protocol with a
non-saturated Stage-A specificity criterion before any B1 training is
authorized.** Completion requires a criterion that can distinguish
head-concentrated access from generic same-dimensional access on held-out data;
the present failed gate must not be bypassed or resumed.
