# Experiment design

## Inputs

Latest diagnostics use baseline/proposed checkpoints at the largest common step
for L and M. Scaling uses the closest common checkpoint to each fixed global
FLOPs target that has been reached, plus the latest common checkpoint. This
gives multiple M and L points in one compute coordinate system without changing
old target points when newer checkpoints appear.

## Stages and pass conditions

1. Preflight: eight CUDA/PPU devices, data, tokenizer, Transformers, and lm-eval
   are available before any expensive work starts.
2. Manifest: four checkpoints load and filename step equals payload step.
3. Routing: expert shares sum to one per layer; cache loads are non-increasing
   as capacity grows; no model silently omits a layer.
4. Predictability: train/test sequences are disjoint; all valid upper/lower
   matrix cells have nonzero test counts; recall@1 <= recall@2 <= recall@4.
5. Decode swapping: real copies report nonzero bytes for finite-cache misses; the
   8-slot run has zero decode-time misses after prompt warmup; timings synchronize
   the device before and after decode.
6. TTFT: full causal prefill produces first-token logits at prompt lengths
   32/128/512/1024/2048; cold and warm cache rows exist for every budget.
7. Scaling: every selected M/L checkpoint has positive FLOPs, train loss, test
   loss, and downstream metrics; baseline/proposed use the same step per size.
8. Packaging: every required CSV is nonempty and the archive is below 5 MB.

Failure writes `FAILED`, the active stage, and the exit code to `STATUS.tsv`.
Relaunching skips completed checkpoint-level jobs with matching protocol and
configuration keys.
