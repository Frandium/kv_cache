# Qwen3 K-cache Norm Analysis

This project builds a long Qwen3-0.6B KV cache from a DCLM text shard and
summarizes the L2 norm distribution of every K vector for every layer and KV
head.

Default inputs:

```text
model: /mnt/workspace/Qwen3-0.6B
text:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
```

The script runs the base Qwen3 model with `use_cache=True` in chunks, extracts
the final K cache, computes vector norms over the head dimension, and writes
per-head, per-layer, and global statistics.

## Run

```bash
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

Useful overrides:

```bash
MAX_TOKENS=65536 \
CHUNK_SIZE=512 \
MODEL_PATH=/mnt/workspace/Qwen3-0.6B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

For a quick smoke test:

```bash
MAX_TOKENS=2048 CHUNK_SIZE=256 \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

## Outputs

By default, files are written to:

```text
ymluo/projects/qwen3_kcache_norm_analysis/outputs/kcache_norms/
```

Main files:

- `summary_by_head.csv`: one row per `(layer, KV head)` with mean, std,
  variance, min/max, RMS, MAD, CV, skewness, excess kurtosis, and percentiles.
- `summary_by_layer.csv`: the same statistics aggregated across all KV heads in
  each layer.
- `summary.json`: metadata plus global, layer-level, and head-level summaries.
- `histogram_by_head.csv`: fixed-bin norm distribution per `(layer, KV head)`.
- `histogram_by_layer.csv`: fixed-bin norm distribution per layer.
- `histogram_global.csv`: fixed-bin norm distribution across the full K cache.

Default percentiles are:

```text
1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99
```

Override them with:

```bash
PERCENTILES=1,5,10,20,30,50 \
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

## Notes

- The script reports KV heads, not necessarily query heads. For GQA models,
  `num_key_value_heads` can be smaller than `num_attention_heads`.
- `MAX_TOKENS` controls cache length. Larger values produce longer K-cache
  matrices but need more GPU memory and time.
- `CHUNK_SIZE` controls how many new tokens are processed per forward call.
  Smaller chunks reduce temporary attention memory when the cache is long.
- Set `SAVE_NORM_TENSORS=true` if you also want a PyTorch file containing the
  `[kv_heads, tokens]` norm matrix for each layer.
