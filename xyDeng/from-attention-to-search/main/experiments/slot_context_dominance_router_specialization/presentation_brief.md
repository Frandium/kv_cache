# Presentation Brief: Slot-Context Dominance

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

## One Sentence

```text
slot 变长增强了 B hidden state 中的 slot feature，所以 router 更容易按 slot 分；但 multi-B 里 B identity 仍然干扰，所以只得到部分 specialization。
```

## What To Claim

| Result | Interpretation | Claim boundary |
|---|---|---|
| fixed-B long slot reaches clean routing | slot context can control B-position routing when B identity is removed | this is a positive control, not the main multi-B claim |
| multi-B long slot improves but stays imperfect | slot feature is visible and useful, but identity variation and top-1 dynamics still interfere | visibility + slot init is not sufficient for robust functional specialization |
| bridge r-B/AB/CB/DB supports long context | old semantic-prior decay and current fixed-B positive result are compatible | semantic init alone is not a functional-specialization proof |

## Three Figures

1. Main short-vs-long / init-vs-final routing comparison:

![random vs slot-centroid init, init vs final](figures/discussion_random_vs_slot_init_init_final_route_heatmaps.png)

Use this to say: slot-centroid init plus long context gives the cleanest routing prior; random init does not reliably discover it.

2. Bridge final NMI by seed:

![bridge final NMI by seed](../slot_context_bridge_abcd_context_length/figures/bridge_abcd_final_nmi_by_seed_heatmap.png)

Use this to say: fixed-B long semantic is stable across seeds; multi-B improves but remains seed-dependent.

3. Bridge short-vs-long route heatmaps:

![bridge short vs long init-final route heatmaps](../slot_context_bridge_abcd_context_length/figures/bridge_abcd_short_long_init_final_route_heatmaps.png)

Use this to say: long context makes the semantic route prior more stable.

## Three Questions To Prepare

| Question | Short answer |
|---|---|
| Is this normal pretraining NTP? | Yes. Training uses full-sequence causal NTP. B-position metrics are evaluation, not supervised routing loss. |
| Does NMI prove specialization? | No. NMI proves route-slot alignment only. Functional specialization requires route alignment plus expert utility alignment. |
| Did this prove the original hypothesis? | It supports the weak version: stronger context helps routing. It weakens the strong version: slot visibility plus centroid init is not enough for robust multi-B specialization. |

## Next Step

```text
Do not extend context length again.
Run multi-B with an explicit route-function binding signal.
Judge by route NMI, route heatmap, forced utility heatmap, and Assign-Utility together.
```
