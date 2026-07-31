# A15 Experiment Index

Status: Q1 E01/E02 completed on 2026-07-30. E03-S passed its controlled
covariance-causality test; E03-R ended `insufficient_load_guard` because real
DCLM routing collapsed before a valid formation comparison. The four-layer
shallow-head pilot ended `insufficient_stage_a_capability` because every tested
64-dimensional view decoded the coarse target perfectly, so downstream
training did not unlock. A15_02_01_E01 separately completed with an operational
Pass and scientific Fail, leaving A15_02_E02 blocked.

| Experiment | Role | Status | Reading path |
| --- | --- | --- | --- |
| A15_00_E01 actual Router-input band response | Frozen coarse/fine access, native use, and 30k/40k/80k allocation audit | completed; endpoint pass, persistent strengthening fail | [summary_cn.md](15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md) |
| A15_00_E02 early head-alignment onset | LB/batch-gradient 10k/20k/30k onset and fixed-basis trajectory audit | completed; 10k onset pass, 10k--30k progressive strengthening fail | [summary_cn.md](15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md) |
| A15_00_E03_S controlled spectral dynamics | Flat/anisotropic/whitened fixed-advantage causal test, with conditional trainable experts | completed; controlled S0/S1 scientific Pass; S2 not launched | [summary.md](15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md) |
| A15_00_E03_R real early spectral dynamics | From-initialization DCLM trajectory through at most 2B tokens; no LB loss | completed fail-closed; `insufficient_load_guard` at step 100, valid heavy evidence through step 120 | [summary.md](15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md) |
| A15_01_01_E01 four-layer shallow-head pilot | Native four-layer control plus compatibility-gated layer-2 head guidance for layer-3/4 Gates | completed fail-closed; `insufficient_stage_a_capability`; downstream B1 not launched | [summary.md](15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md) |
| A15_02_01_E01 middle/long-tail compatibility gate | Static residual novelty plus bidirectional one-step same-expert compatibility | completed; operational pass, functional admission fail | [summary_cn.md](15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md) |
| A15_02_E02 matched spectral dispatch training | Conditional native vs selected-band vs equal-dimensional random 8x5090 training | not run; blocked by E01 fail | [protocol_cn.md](15_02_middle_tail_functional_resolution/A15_02_E02_matched_spectral_dispatch_training/protocol_cn.md) |

E03-S/R remain Q1 dynamics experiments. E03-S establishes a controlled finite-
time learning-speed effect; E03-R does not yet establish that effect in a real
MoE because its no-load-loss condition violated the registered stability
guard. A15_01_01 is a separate shallow-to-deep functional pilot, and its
saturated admission task cannot adjudicate utility. The 2026-07-30 conditional
approval for A15_02_E02 did not activate because A15_02_01_E01 selected no
admissible band; resuming E02 under the current fixed M/T/N definition would
require a new research decision.
