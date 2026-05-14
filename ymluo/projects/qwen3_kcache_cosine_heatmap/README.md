# Qwen3 K-cache Cosine Heatmap

This project profiles Qwen3-0.6B on a DCLM text prefix, extracts the final K
cache, and computes token-token cosine similarity for each selected
`(layer, KV head)` K matrix.

Default inputs:

```text
model: /mnt/workspace/Qwen3-0.6B
text:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
tokens: 5000
```

For a selected layer/head, the K cache is reshaped to:

```text
[tokens, head_dim]
```

The script L2-normalizes the token vectors and computes:

```text
cosine_matrix = normalized_k @ normalized_k.T
```

so each heatmap is a `tokens x tokens` pairwise cosine matrix. The diagonal is
self-similarity and should be near `1.0`.

## Run

```bash
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

Useful overrides:

```bash
MAX_TOKENS=5000 \
CHUNK_SIZE=512 \
MODEL_PATH=/mnt/workspace/Qwen3-0.6B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

For a quick smoke test on one layer/head:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

To generate one full 5k-token heatmap for a single layer/head:

```bash
LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

To generate every layer/head heatmap, leave `LAYERS=all HEADS=all` as the
default. This can produce many large PNGs.

## Outputs

By default, files are written to:

```text
ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/
```

Main files:

- `plots/layer_XX_head_YY_cosine.png`: token-token cosine heatmap for one
  `(layer, KV head)`.
- `plots/layer_head_offdiag_mean_heatmap.png`: layer/head overview of mean
  off-diagonal cosine.
- `plots/layer_head_offdiag_std_heatmap.png`: layer/head overview of
  off-diagonal cosine standard deviation.
- `summary_by_head.csv`: per `(layer, head)` summary statistics for the full
  cosine matrix and the off-diagonal entries.
- `profile_timings.csv`: elapsed time for each forward chunk used to build the
  K cache.
- `tokens.csv`: token index, token id, tokenizer piece, and decoded text.
- `summary.json`: run metadata and output paths.

Optional tensor dump:

```bash
SAVE_SIMILARITY_TENSORS=true LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

This writes `similarity_tensors/layer_XX_head_YY_cosine.pt`. A single 5000 x
5000 float16 matrix is about 50 MB, so avoid enabling this for all heads unless
you have enough disk space.

## Options

Layer/head selection:

```bash
LAYERS=0,7,15 HEADS=0-3 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

Plot downsampling:

```bash
PLOT_MAX_TOKENS=1500 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

The cosine matrix is still computed on all tokens; only the rendered heatmap is
strided if the token count exceeds `PLOT_MAX_TOKENS`. Set
`PLOT_MAX_TOKENS=0` to force plotting every token.

## Notes

- Heads are KV heads, not query attention heads. Qwen-style GQA models can have
  fewer KV heads than query heads.
- The default summary percentile columns sample up to `SUMMARY_SAMPLE_SIZE`
  values per matrix to avoid spending most of the run sorting 25M entries per
  head. Mean, std, min, max, and RMS are computed on the full matrix.
- `SIMILARITY_DEVICE=auto` uses CUDA when available; otherwise it falls back to
  CPU.
