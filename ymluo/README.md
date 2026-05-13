# Qwen3 KV Cache Research Workspace

This workspace collects Qwen3 KV-cache experiments around one research theme:
treating long-context KV memory as an indexed retrieval system instead of a
flat token sequence that every decode step scans densely.

The main design directions are:

- block- or chunk-level candidate recall before exact attention;
- learned compression of older KV memory while preserving recent and anchor
  tokens;
- attention-energy analysis to estimate how much context can be dropped with
  limited loss impact;
- K-cache value and adjacent-token delta profiling to understand what the cache
  actually stores across layers and heads.

For the broader motivation and search-system analogy, start with
`KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md`.

Commands below assume they are launched from the parent repository root
(`kv_cache`). If you run them after `cd ymluo`, remove the leading `ymluo/`
path component.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md` | Research notes mapping search, vector indexing, hierarchy, and knowledge-graph ideas onto KV-cache lookup. |
| `projects/qwen3_chunk_routing` | Training harness for Qwen3-0.6B chunk attention with `baseline`, `oracle`, and learned `router` modes. |
| `projects/pyramid_kv_compression` | Continued-pretraining experiment that replaces older middle-layer KV blocks with learned summaries. |
| `projects/qwen3_kcache_avg_topk` | Inference-time sparse decode experiment using averaged K-cache block summaries for top-k block selection. |
| `projects/qwen3_kcache_norm_analysis` | Qwen3-0.6B K-cache norm, attention-energy, and pruning loss/PPL analysis on DCLM text. |
| `projects/qwen3_kcache_value_delta_analysis` | Qwen3-8B K-cache component, norm, and adjacent-token delta distribution analysis. |
| `logs/` | Historical logs or pushed workspace snapshots. Not required for normal experiment entry points. |
| `utils/` | Reserved for shared utilities. |

Each project has its own README and scripts; this file is the high-level index.

## Projects

### `qwen3_chunk_routing`

This project compares three Qwen3-0.6B attention modes:

- `baseline`: original full attention.
- `oracle`: full scores are computed, then valid past tokens are split into 20
  chunks; chunk 1, the recent chunk, and the top 3 middle chunks by attention
  mass are kept.
- `router`: a lightweight learned router predicts the top 3 middle chunks from
  chunk summaries before exact attention.

Run examples:

```bash
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh baseline
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh oracle
bash ymluo/projects/qwen3_chunk_routing/scripts/run_8gpu.sh router
```

The script uses `torchrun --nproc_per_node=8` by default. It reads tokenizer and
config from `MODEL_PATH`, but initializes model weights from scratch unless the
project code is changed.

### `pyramid_kv_compression`

This project patches Qwen3 attention for continued pretraining with a
pyramid-shaped KV memory:

```text
early layers:   full KV
middle layers:  compressed older KV
final layers:   full KV
```

The hidden-state sequence length stays unchanged. Only selected layers shorten
the attention memory by replacing older middle blocks with learned weighted
K/V summaries. Anchor tokens and recent tokens remain raw.

Recommended stage order:

```bash
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh sanity
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
bash ymluo/projects/pyramid_kv_compression/scripts/run_8gpu.sh full
```

`METHOD_NOTES.md` records the current caution: the aggressive default schedule
has produced high losses, so the next serious runs should start with weak center
layer compression, larger anchor/recent windows, and the `attention` stage
before considering full-model training.

### `qwen3_kcache_avg_topk`

This is an inference-only sparse decode experiment. Layers 0-2 use the original
attention path. Layers 3-27 split the current K cache into blocks, average keys
inside each block, score blocks with the current query, keep the top block
fraction, and then run exact attention over the original K/V tokens inside the
selected blocks.

Generate text:

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_generate.sh
```

Evaluate baseline vs sparse decode:

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B \
DATA_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10 \
bash ymluo/projects/qwen3_kcache_avg_topk/scripts/run_eval.sh
```

Default sparse settings are `BLOCK_SIZE=10`, `TOPK_RATIO=0.30`,
`FIRST_SPARSE_LAYER=3`, and `LAST_SPARSE_LAYER=27`.

### `qwen3_kcache_norm_analysis`

This project runs Qwen3-0.6B on a DCLM text prefix and writes:

- token-level next-token loss and PPL;
- original K-cache norm summaries by layer and KV head;
- attention-energy summaries by layer and query head;
- top-k counts needed to reach several attention-energy thresholds;
- loss/PPL after pruning attention positions below those thresholds.

Run:

```bash
bash ymluo/projects/qwen3_kcache_norm_analysis/scripts/run_analysis.sh
```

The current summary experiment on 3000 tokens shows:

- 90% attention energy is close to full attention: loss increases by `0.014206`,
  about `1.0143x` PPL.
- 95% attention energy is nearly lossless in this sample: loss increases by
  `0.002030`, about `1.0020x` PPL.
- 50% and 75% energy pruning are too aggressive for quality.
- the number of tokens needed to reach the same energy threshold varies strongly
  by layer and head, so a fixed token top-k is less appropriate than adaptive
  energy-based selection.

See `projects/qwen3_kcache_norm_analysis/attention_energy_loss_summary.md` for
the detailed result table and interpretation.

### `qwen3_kcache_value_delta_analysis`

This project profiles a Qwen3-8B forward pass, builds the final K cache with
`past_key_values`, and analyzes both K vectors and adjacent-token changes:

```text
k_i
delta(k_i) = k_i - k_{i-1}
```

It writes per-head, per-layer, and global statistics, exact histograms, timing
rows, and optional plots. The default run uses 5000 DCLM tokens.

Run:

```bash
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

Useful smoke test:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 SAVE_HEAD_HISTOGRAMS=false \
bash ymluo/projects/qwen3_kcache_value_delta_analysis/scripts/run_analysis.sh
```

## Common Defaults

Most scripts can be configured with environment variables. Common ones are:

```bash
MODEL_PATH=/path/to/Qwen3
DATA_PATH=/path/to/dclm
TEXT_PATH=/path/to/part-00000.txt
OUTPUT_DIR=/path/to/output
MAX_TOKENS=3000
CHUNK_SIZE=256
DEVICE=cuda
DTYPE=bfloat16
```

The training scripts default to Hugging Face streaming for large DCLM-style
directories. Keep `STREAMING=true` unless you intentionally want to build an
Arrow cache and have enough disk space.

## Suggested Reading Order

1. `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md` for the retrieval-system
   framing.
2. `projects/qwen3_kcache_norm_analysis/attention_energy_loss_summary.md` for
   the current attention-pruning evidence.
3. `projects/qwen3_kcache_avg_topk/README.md` for the deployable block-selection
   inference baseline.
4. `projects/qwen3_chunk_routing/README.md` for oracle/router sparse chunk
   training.
5. `projects/pyramid_kv_compression/METHOD_NOTES.md` before running continued
   pretraining with compressed KV.
6. `projects/qwen3_kcache_value_delta_analysis/README.md` for K-cache value and
   delta distribution profiling.

## Practical Notes

- Use project-local README files for exact command options and output paths.
- Analysis scripts write outputs under each project's `outputs/` directory by
  default.
- Long-context analysis can be memory-heavy; reduce `MAX_TOKENS` and
  `CHUNK_SIZE` for smoke tests.
- The sparse decode and routing experiments still need kernel-aware
  implementation work before their theoretical KV-read reduction becomes a
  real serving speedup.
