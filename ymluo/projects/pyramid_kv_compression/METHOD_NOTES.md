# Pyramid KV Compression Method Notes

This document summarizes the current Qwen3-0.6B pyramid KV compression idea,
implementation, and recommended experiment order.

## Goal

Modify Qwen3-0.6B so that training already follows a pyramid-shaped KV memory:

```text
early layers:   full KV
middle layers:  shorter/compressed KV
final layers:   full KV
```

The hidden sequence length is not shortened. Every layer still receives and
outputs:

```text
hidden_states: [batch, seq_len, hidden_size]
```

Only the attention KV memory length changes in selected layers:

```text
Q: [batch, q_heads, seq_len, head_dim]
K: [batch, kv_heads, memory_len, head_dim]
V: [batch, kv_heads, memory_len, head_dim]
```

For compressed middle layers:

```text
memory_len < seq_len
```

## Compression Structure

Each compressed layer splits KV memory into three parts:

```text
anchor_raw + compressed_middle + recent_raw
```

- `anchor_raw`: first tokens, kept uncompressed.
- `compressed_middle`: old middle tokens, block-compressed.
- `recent_raw`: latest tokens, kept uncompressed.

Example:

```text
seq_len = 16
anchor_tokens = 2
recent_tokens = 4
block_size = 4

original positions:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

compressed memory:
0 1 C5 C9 10 11 12 13 14 15
```

Where:

```text
C5 summarizes positions 2,3,4,5
C9 summarizes positions 6,7,8,9
```

Incomplete middle blocks are kept raw, not compressed.

## Learned Compressor

The compressor is trainable. For each block:

```text
K_block: [batch, kv_heads, block_size, head_dim]
V_block: [batch, kv_heads, block_size, head_dim]
```

It computes token scores:

```text
score_i = MLP([K_i, V_i])
alpha_i = softmax(score_i)
```

Then produces one summary KV:

```text
K_summary = sum_i alpha_i * K_i
V_summary = sum_i alpha_i * V_i
```

$$

$$





This is learned weighted pooling, not simple averaging.

## RoPE Handling

The implementation compresses raw K before RoPE:

```text
raw K + V -> learned compressor -> compressed raw K + compressed V
compressed raw K -> apply RoPE
```

Each compressed summary uses the last position in its block as its representative
position:

```text
[2,3,4,5] -> summary position 5
```

This helps preserve causality and avoids averaging already-rotated K vectors.

## Current Default Layer Shape

The current default block-size schedule for a 28-layer model is:

```text
1,1,1,1,2,2,2,3,3,3,3,3,4,4,4,4,3,3,3,3,3,2,2,2,1,1,1,1
```

Meaning:

```text
block_size = 1: no compression
block_size = 2: 2 middle tokens -> 1 summary
block_size = 3: 3 middle tokens -> 1 summary
block_size = 4: 4 middle tokens -> 1 summary
```

This default is aggressive for adapting from an already trained Qwen3 checkpoint.

## Training Stages

### 1. sanity

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh sanity
```

Purpose:

```text
random token data
short run
checks model loading, patching, forward, backward, DDP, and logging
```

The loss from this stage is not meaningful.

### 2. compressor

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
```

Trainable:

```text
pyramid_kv_compressor only
```

Frozen:

```text
all original Qwen3 parameters
```

This stage is safest, but may be too weak if the compression is aggressive.

### 3. attention

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
```

Trainable:

```text
pyramid_kv_compressor
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
self_attn.q_norm
self_attn.k_norm
```

Frozen:

```text
MLP
embedding
LM head
most non-attention parameters
```

This is the recommended main adaptation stage after sanity checks.

### 4. full

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh full
```

Trainable:

```text
all model parameters
```

This is full-parameter continued pretraining/fine-tuning. It has the highest
capacity but also the highest risk. Do not use it before confirming that the
compression structure works in weaker settings.

## Current Observation

With the aggressive default schedule, observed losses were high:

```text
compressor mode: loss around 30
attention mode:  loss around 26-27, slowly decreasing
```

This means:

```text
training has gradients and can move
but the current compression is too disruptive or needs more debugging
```

Do not jump directly to `full` under this condition. Full training may adapt the
whole model to a broken or overly aggressive structure and damage the pretrained
weights.

## Recommended Debug Order

### Step 1: run a baseline

First confirm that the same data/tokenization path gives reasonable loss without
KV compression.

If baseline loss is also very high, the issue is likely data loading, text
column selection, tokenization, or labels.

### Step 2: use very weak compression

Only compress the center four layers with `2 -> 1`, while preserving a larger
anchor and recent window:

```bash
LAYER_BLOCK_SIZES=1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1 \
ANCHOR_TOKENS=512 \
RECENT_TOKENS=2048 \
MAX_STEPS=500 \
LOGGING_STEPS=10 \
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
```

If this still gives loss above 20, suspect an implementation or data issue.

### Step 3: gradually increase compression

Recommended progression:

```text
A. center 4 layers, 2 -> 1
B. center 8 layers, 2 -> 1
C. center 8 layers, 3 -> 1
D. wider schedule with 3 -> 1
E. only then try 4 -> 1
```

## Recommended Current Command

For the next serious experiment, use:

```bash
LAYER_BLOCK_SIZES=1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1 \
ANCHOR_TOKENS=512 \
RECENT_TOKENS=2048 \
MAX_STEPS=500 \
LOGGING_STEPS=10 \
DATA_FILES_GLOB="**/*.parquet" \
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
```

Use `DATA_FILES_GLOB="**/*.txt"` instead if the dataset is text files.

## Data Loading Note

Large DCLM directories should use streaming:

```bash
STREAMING=true
```

Non-streaming `load_dataset(...)` may try to generate a huge Arrow cache and can
fill the disk:

```text
OSError: [Errno 28] No space left on device
```

If this happens, inspect and clean the Hugging Face dataset cache:

```bash
du -sh ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/datasets
```

Only remove the cache after confirming it is safe for the current machine.
