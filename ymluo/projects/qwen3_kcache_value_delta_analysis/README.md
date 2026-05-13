# Qwen3 K-cache Value And Delta Analysis

This project profiles one Qwen3 forward pass, extracts the final K cache, and
analyzes both:

- `k_i`: K vectors for each layer and KV head.
- `delta(k_i) = k_i - k_{i-1}`: adjacent-token K-vector changes.

Default inputs:

```text
model: /mnt/workspace/Qwen3-8B
text:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
tokens: 5000
```

The forward pass is chunked so the cache is built incrementally with
`past_key_values`, but the resulting K cache is the same object of interest:
one K matrix per `(layer, KV head)`, shaped approximately
`[tokens, head_dim]`.

## Run

```bash
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

Useful overrides:

```bash
MAX_TOKENS=5000 \
CHUNK_SIZE=512 \
MODEL_PATH=/mnt/workspace/Qwen3-8B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

For a quick smoke test:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 SAVE_HEAD_HISTOGRAMS=false \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

## Outputs

By default, files are written to:

```text
ymluo/projects/qwen3_kcache_value_delta_analysis/outputs/kcache_value_delta/
```

Main files:

- `summary_by_head.csv`: exact per `(metric, layer, head)` statistics.
- `summary_by_layer.csv`: exact per `(metric, layer)` statistics.
- `summary_global.csv`: exact global mean/std/min/max plus sampled global
  percentiles.
- `histogram_global.csv`: exact global histogram counts for every metric.
- `histogram_by_layer.csv`: exact per-layer histogram counts.
- `histogram_by_head.csv`: exact per-head histogram counts, enabled by default.
- `profile_timings.csv`: elapsed seconds for each forward chunk.
- `summary.json`: run metadata, cache shapes, histogram ranges, and plot paths.
- `plots/*.png`: global distribution histograms and layer/head heatmaps.

Optional tensor dump:

```bash
SAVE_K_TENSORS=true SAVE_DELTA_TENSORS=true \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

This writes `kcache_tensors.pt`, which can be large.

## Metrics

Scalar component metrics:

- `k_value`: signed K-vector components.
- `abs_k_value`: absolute K-vector components.
- `delta_value`: signed components of `k_i - k_{i-1}`.
- `abs_delta_value`: absolute components of `k_i - k_{i-1}`.

Vector metrics:

- `k_l2_norm`, `k_l1_norm`, `k_linf_norm`, `k_mean_abs`.
- `delta_l2_norm`, `delta_l1_norm`, `delta_linf_norm`, `delta_mean_abs`.
- `relative_delta_l2 = ||k_i - k_{i-1}||_2 / max(||k_{i-1}||_2, eps)`.
- `cosine_prev = cosine(k_i, k_{i-1})`.

The default percentile columns are:

```text
0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9
```

## Plot Notes

The global histogram PNGs use exact histogram counts. Histogram ranges are
chosen from a global sample at `HISTOGRAM_CLIP_PERCENTILE=99.9`, so the plots are
not dominated by a few outliers. Tail mass is shown as underflow/overflow text
inside each plot.

The heatmaps summarize head-level means across layer and KV head for:

- mean absolute K value
- mean absolute delta value
- mean K L2 norm
- mean delta L2 norm
- mean relative delta L2
- mean adjacent-token cosine similarity
