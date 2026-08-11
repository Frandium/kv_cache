# Design

## Claim

At matched training steps, the proposed router reduces tail-expert cache misses
without collapsing expert use, remains predictable from earlier
representations, and follows the same compute/performance trend as baseline.

## Operational definitions

- Healthy routing: every layer/expert receives tokens and normalized load
  entropy remains high; continuity alone is not evidence of health.
- Continuity: LRU cache loads and evictions under per-layer capacities 1, 2, 4,
  and all 8 experts.
- Same-token predictability: representation at layer i predicts the selected
  expert at layer j for i < j.
- Next-token predictability: current-token representation at layer i predicts
  the next token's selected expert at layer j for i >= j.
- Recall@k: because routing is Top-1, one test item is correct when its selected
  expert appears among the predictor's top k classes.
- Exact decode swapping latency: synchronous pinned-host-to-device expert copies into
  fixed GPU slots during token-by-token KV-cached decoding.
- TTFT: wall time from beginning the prompt H2D transfer through complete
  batch-size-one causal prefill and production of the first output logits.
- Scaling compute: active forward FLOPs from checkpoint architecture multiplied
  by tokens trained and a documented forward/backward factor of three.

## Claim boundary

Latency is exact for this Python/PyTorch swapping runtime on the tested PPU. TTFT
uses expert-grouped full-sequence prefill, not token-by-token prompt decoding. It
does not claim to be the optimal latency of a fused production runtime. FLOPs
are an architectural estimate and exclude small elementwise/norm operations.
