# Qwen3 K-cache And Attention Energy Analysis

This project runs Qwen3-0.6B on a short prefix from a DCLM text shard and writes:

- per-token next-token loss and PPL
- per-layer/head attention energy captured by the top 30% attention entries
- per-token top-k counts needed to reach 50%, 75%, 90%, 95%, 98%, and 100% attention energy
- model loss/PPL after pruning attention positions to those energy thresholds
- the original K-cache norm summaries

Default inputs:

```text
model: /mnt/workspace/Qwen3-0.6B
text:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
tokens: 3000
```

Loss/PPL rows are aligned to the prediction made at a query token. For example,
`query_token_index=10` predicts `target_token_index=11`; the loss/PPL values are
for token 11.

Attention energy is computed from attention probabilities. For each query token,
layer, and head, the script sorts attention scores descending. Energy is:

```text
sum(selected attention scores) / sum(all attention scores)
```

Because causal attention has a different number of valid keys at each position,
100% energy uses the full valid causal context length for that query token.

The pruned loss/PPL experiment first runs the full model to get the original
attention distribution. For each threshold, it then re-runs the chunk while
masking out attention positions below that layer/head/query token's top-k set.
The threshold is applied to all layers and all attention heads at the same time.

## Run

```bash
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

Useful overrides:

```bash
MAX_TOKENS=3000 \
CHUNK_SIZE=128 \
MODEL_PATH=/mnt/workspace/Qwen3-0.6B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

For a quick smoke test:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

The script prints progress for every forward chunk and attention-processing step.
`ATTN_IMPLEMENTATION=eager` is the default because Hugging Face attention outputs
are required for the energy experiments.

## Outputs

By default, files are written to:

```text
ymluo/projects/qwen3_kcache_norm_analysis/outputs/kcache_norms/
```

New attention/loss files:

- `token_loss_ppl.csv`: one row per predicted token with token id, token text,
  loss, and PPL.
- `attention_top_fraction_energy_by_head.csv`: one row per `(layer, head)`
  summarizing how much attention energy is captured by the top 30% valid keys.
- `attention_energy_thresholds_by_head.csv`: one row per `(layer, head,
  threshold)` summarizing how many tokens are needed to reach each energy
  threshold.
- `attention_token_topk.csv`: one row per `(layer, head, query token)` with the
  exact top-k count and fraction for each energy threshold plus the aligned
  loss/PPL.
- `attention_pruned_loss_ppl_by_threshold.csv`: one row per energy threshold
  with the model's loss/PPL after attention pruning. Threshold `1.0` is the
  unpruned baseline.
- `attention_pruned_token_loss_ppl.csv`: one row per `(energy threshold, target
  token)` with the pruned model's token-level loss/PPL.
- `summary.json`: metadata, global loss/PPL summary, output paths, and the
  original K-cache norm summary.

Original K-cache norm files are still written:

- `summary_by_head.csv`
- `summary_by_layer.csv`
- `histogram_by_head.csv`
- `histogram_by_layer.csv`
- `histogram_global.csv`

## Options

```bash
TOP_FRACTION=0.30 \
ENERGY_THRESHOLDS=50,75,90,95,98,100 \
SAVE_ATTENTION_TOKEN_ROWS=true \
COMPUTE_PRUNED_LOSS_PPL=true \
SAVE_PRUNED_TOKEN_ROWS=true \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

Set `SAVE_ATTENTION_TOKEN_ROWS=false` if you only want summary CSVs and want to
avoid writing the larger per-token/per-head file.

Set `COMPUTE_PRUNED_LOSS_PPL=false` if you only want the attention energy/top-k
statistics. Keeping it enabled is slower because each non-100% threshold requires
an additional forward pass for every chunk.

Default K-cache norm percentiles are:

```text
1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99
```

Override them with:

```bash
PERCENTILES=1,5,10,20,30,50 \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

## Notes

- Attention heads are query attention heads. The K-cache norm files still report
  KV heads, which can be fewer than query heads for GQA models.
- `MAX_TOKENS=3000` truncates after tokenization. The first token has no
  next-token loss, so loss/PPL files contain `MAX_TOKENS - 1` token rows.
- `CHUNK_SIZE` controls how many new tokens are processed per forward call.
  Smaller chunks reduce temporary attention memory.
- Set `SAVE_NORM_TENSORS=true` if you also want a PyTorch file containing the
  `[kv_heads, tokens]` norm matrix for each layer.
