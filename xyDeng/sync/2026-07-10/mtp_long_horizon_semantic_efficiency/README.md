# MTP Long-Horizon Semantic Efficiency Handoff

Date: 2026-07-10

## Scope

This package updates the July 8 representation-level MTP story with the
current progression:

low-loss representation guarantee -> first-order hidden semantic velocity ->
controlled hidden-state finite-step theorem -> Transformer
parameter-transmission gap.

Read meeting_brief_cn.md first, then story_cn.md for the full synthesis.

## Source Files

- Meeting brief: meeting_brief_cn.md
- Integrated source brief: source_brief_hidden_to_parameter_cn.md
- Story: story_cn.md
- Current anchor: anchor/11_10_finite_step_semantic_efficiency_anchor.md
- Finite-step experiment: experiments/A11_10_finite_step/
- Direct-versus-indirect experiment: experiments/A11_10_indirect_transfer/
- General-K summaries: experiments/A11_08_general_k/summary.md and
  experiments/A11_09_next_k/summary.md
- Central figure: figures/mtp_a11_10_direct_native_margin_split.png

## Current Claim

In the controlled model, covering the first future target that carries the
early variable gives the current hidden state a direct semantic constraint and
a positive first-order semantic velocity. Under fixed readout directions and
explicit hidden-state assumptions, this velocity yields a finite-step recovery
bound. Controlled Transformer experiments support persistent positive early
velocity and stronger native future prediction under direct supervision.

## Claim Boundary

This package does not establish a uniform Transformer parameter-space
hitting-time theorem, natural-language sample-efficiency superiority, general
MTP-over-NTP superiority, or monotonic benefit from increasing K.

## Next Action

Audit the K=2 semantic tangent-kernel transmission decomposition: direct
semantic transmission, background interference, and nonlinear remainder.

## Excluded

Raw logs, checkpoints, datasets, full result directories, and experiment code
are intentionally excluded.
