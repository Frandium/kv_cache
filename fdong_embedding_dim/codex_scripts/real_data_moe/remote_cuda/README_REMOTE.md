# CUDA 8-GPU Common/Tail MoE Experiments

## Fixed configuration

- 8 CUDA GPUs, 80 GB each, launched with NCCL DDP.
- Qwen3-style backbone: 28 layers, hidden size 1024, 16 query heads,
  8 KV heads, head dimension 128, and vocabulary size 151936.
- Qwen3 tokenizer files are bundled; no Hugging Face download is required.
- Sequence length 1024, per-GPU micro batch 4, gradient accumulation 8.
- Global batch: `4 * 8 * 8 = 256` sequences.
- Tokens per optimizer step: `256 * 1024 = 262,144`.
- BF16 autocast. Activation checkpointing is disabled because the 1.653B
  model fits comfortably on each 80GB GPU and recomputation is unnecessary.
- Peak learning rate `1e-4`; 200-step warmup; cosine decay to `5e-6` at
  step 50000; constant `5e-6` afterwards.
- Training is unbounded by default (`MAX_STEPS=0`).
- A checkpoint is saved every 4000 steps, approximately every 1.049B tokens.
  Every checkpoint is retained.

All runs use seed 42 by default. Each DDP rank receives a deterministic,
disjoint subset of input files. Checkpoints retain model, optimizer, every
rank's data offset, token buffer, and RNG state.

## Extract and configure

```bash
tar -xzf real_data_moe_cuda_2b.tar.gz
cd real_data_moe_cuda_2b
```

Defaults:

```bash
DATA_DIR=/mnt/workspace/dclm/global-shard_01_of_10
OUTPUT_ROOT=/mnt/workspace/fmoe_cuda_2b_outputs
NPROC_PER_NODE=8
MAX_STEPS=0
```

Override them before launching when needed.

## Launch one experiment per instance

Create the shared output root once:

```bash
mkdir -p /mnt/workspace/fmoe_cuda_2b_outputs
```

Baseline, common/tail `3072/3072` (about 1.653B total parameters):

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_baseline.sh \
  > /mnt/workspace/fmoe_cuda_2b_outputs/baseline_launcher.log 2>&1 &
```

Total-parameter-matched proposed model, common/tail `1536/3456`:

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_proposed_total_matched.sh \
  > /mnt/workspace/fmoe_cuda_2b_outputs/proposed_launcher.log 2>&1 &
```

Routing-only model, common/tail `3072/3072`:

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_routing_only.sh \
  > /mnt/workspace/fmoe_cuda_2b_outputs/routing_launcher.log 2>&1 &
```

The launch scripts use `--resume auto`. Restarting the same command resumes
from that experiment's `latest.pt`.

## Monitor

```bash
tail -f /mnt/workspace/fmoe_cuda_2b_outputs/baseline/train.log
tail -f /mnt/workspace/fmoe_cuda_2b_outputs/proposed_total_matched/train.log
tail -f /mnt/workspace/fmoe_cuda_2b_outputs/routing_only/train.log
```

Each log reports loss, accumulated token count, LR, seconds per step, time to
the next checkpoint, and time until the LR reaches its floor. Since training is
unbounded, there is no final completion ETA.

Stop cleanly with `kill <torchrun-pid>` after a checkpoint. To impose a finite
limit, launch with an environment override, for example:

```bash
MAX_STEPS=100000 bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_baseline.sh
```

## Analyze at any time

Plot all currently available training curves:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/plot_loss.sh
```

Evaluate each latest checkpoint on fixed DCLM samples:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/test_loss.sh
```

Analyze expert continuity:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/continuity.sh
```

Override any checkpoint without editing scripts:

```bash
BASELINE_CKPT=/path/to/checkpoint.pt \
PROPOSED_CKPT=/path/to/checkpoint.pt \
ROUTING_CKPT=/path/to/checkpoint.pt \
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/test_loss.sh
```

The same override works for `continuity.sh`. Analysis outputs are written to
`$OUTPUT_ROOT/analysis`.
