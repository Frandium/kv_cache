# Real-data common/long-tail MoE

This directory contains the 4-layer, 16K-vocabulary real-data comparison:

| variant | common `d_ff` | tail experts | tail `d_ff` | router input |
|---|---:|---:|---:|---|
| baseline | 768 | 4 | 768 | full post-attention residual |
| proposed | 384 | 4 | 864 | causal local mean of attention output |

Both variants use hidden size 768, 12 attention heads, tied input/output
embeddings, one always-active common expert, and one top-1 routed tail expert.
Their total expert capacity is equal because `768 + 4*768 = 384 + 4*864`.

The `dense` variant uses one `d_ff=1248` SwiGLU FFN per layer and no router.
Its active FFN width matches the proposed model because `384 + 864 = 1248`.

Run both experiments sequentially:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/run_baseline_then_proposed.sh
```

Checkpoints include model, optimizer, exact DCLM stream position, buffered
tokens, random-number states, model config, and train arguments. The launcher
uses `--resume auto`; rerunning it resumes each unfinished experiment from its
`latest.pt`. A checkpoint is written every 250 optimizer steps, while only the
latest three numbered checkpoints plus `latest.pt` are retained to control disk
usage.

Each run also appends the complete console output to `train.log` and writes
plot-ready `step`, `loss`, `perplexity`, `learning_rate`, and `route_shares`
records to `metrics.jsonl` every logging interval.

The common/tail output-space constraint defaults to off. Enable it for a
separate proposed-model run with `--orthogonalize-tail`; its rank defaults to
16 and its detached common basis is refreshed every 50 optimizer steps.

For a fresh v2 comparison that records training metrics every 10 steps, retains
all 20 numbered checkpoints per model, uses identical DCLM batch order, plots
the loss curve, and runs final route-continuity analysis:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/run_baseline_then_proposed_v2.sh
```

Run the active-width-matched dense control on the same seeded data order:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/run_dense_active_matched.sh
```
