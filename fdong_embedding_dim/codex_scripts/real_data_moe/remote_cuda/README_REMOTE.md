# CUDA 8-GPU Common/Tail MoE Experiments

## Fixed configuration

- 8 CUDA GPUs, 80 GB each, launched with NCCL DDP.
- Qwen3-style backbone: 28 layers, hidden size 1024, 16 query heads,
  8 KV heads, head dimension 128, and vocabulary size 151936.
- Every layer has one always-on common expert and eight routed tail experts.
  Every expert uses SwiGLU with intermediate size 2048; each token selects
  exactly one of the eight tail experts.
- Qwen3 tokenizer files are bundled; no Hugging Face download is required.
- Sequence length 1024, per-GPU micro batch 8, gradient accumulation 4.
- Global batch: `8 * 8 * 4 = 256` sequences.
- Tokens per optimizer step: `256 * 1024 = 262,144`.
- BF16 autocast. Activation checkpointing is disabled because the 1.917B
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
tar -xzf real_data_moe_cuda_2b_8e.tar.gz
cd real_data_moe_cuda_2b_8e
```

Defaults:

```bash
DATA_DIR=/mnt/workspace/dclm/global-shard_01_of_10
OUTPUT_ROOT=/mnt/workspace/fmoe_cuda_2b_8e_outputs
NPROC_PER_NODE=8
MAX_STEPS=0
RESUME=auto
MICRO_BATCH_SIZE=8
GRADIENT_ACCUMULATION=4
```

Override them before launching when needed.

The baseline and load-balanced proposed run use Switch-style auxiliary-loss
weight `0.01`. Override these independently with `BASELINE_LB_WEIGHT` and
`PROPOSED_LB_WEIGHT`. The no-load-balance run always uses weight zero.

Micro-batch 8 is the primary setting. If an instance runs out of memory, use
`MICRO_BATCH_SIZE=4 GRADIENT_ACCUMULATION=8`; this preserves the same global
batch and token count per optimizer step.

## Launch one experiment per instance

Create the shared output root once:

```bash
mkdir -p /mnt/workspace/fmoe_cuda_2b_8e_outputs
```

Baseline routing with load balancing:

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_baseline_lb.sh \
  > /mnt/workspace/fmoe_cuda_2b_8e_outputs/baseline_lb_launcher.log 2>&1 < /dev/null &
```

Attention-mean routing without load balancing:

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_proposed_no_lb.sh \
  > /mnt/workspace/fmoe_cuda_2b_8e_outputs/proposed_no_lb_launcher.log 2>&1 < /dev/null &
```

Attention-mean routing with load balancing:

```bash
nohup bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_proposed_lb.sh \
  > /mnt/workspace/fmoe_cuda_2b_8e_outputs/proposed_lb_launcher.log 2>&1 < /dev/null &
```

The launch scripts use `--resume auto`. Restarting the same command resumes
from that experiment's `latest.pt`.

## Monitor

```bash
tail -f /mnt/workspace/fmoe_cuda_2b_8e_outputs/baseline_lb/train.log
tail -f /mnt/workspace/fmoe_cuda_2b_8e_outputs/proposed_no_lb/train.log
tail -f /mnt/workspace/fmoe_cuda_2b_8e_outputs/proposed_lb/train.log
```

Each log reports loss, accumulated token count, LR, seconds per step, time to
the next checkpoint, and time until the LR reaches its floor. Since training is
unbounded, there is no final completion ETA.

Stop cleanly with `kill <torchrun-pid>` after a checkpoint. To impose a finite
limit, launch with an environment override, for example:

```bash
MAX_STEPS=100000 bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_baseline_lb.sh
```

Example with a different load-balance coefficient:

```bash
PROPOSED_LB_WEIGHT=0.02 bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/run_proposed_lb.sh
```

To write a new run directory while initializing from an existing checkpoint,
override both `OUTPUT_ROOT` and `RESUME`.

## Analyze at any time

Plot all training curves through their shared latest step:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/plot_loss.sh
```

Evaluate the latest checkpoint step shared by all three runs on fixed DCLM samples:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/test_loss.sh
```

Analyze expert continuity at the same shared checkpoint. The default uses 16
sequences of 1024 consecutive tokens and reports both switch counts and rates:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/continuity.sh
```

Run lm-evaluation-harness downstream tasks for one checkpoint:

```bash
RUN_NAME=baseline_lb bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/lm_eval.sh
RUN_NAME=proposed_no_lb bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/lm_eval.sh
RUN_NAME=proposed_lb bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/lm_eval.sh
```

The default task set is
`arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,siqa,race,winogrande`.
Override `TASKS`, `CHECKPOINT`, `EVAL_BATCH_SIZE`, `EVAL_DEVICE`, or
`EVAL_OUTPUT_DIR` when needed.

Launch all three downstream evaluations at their latest common checkpoint on
CUDA devices 7, 6, and 5, then summarize the resulting task table:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/lm_eval_common_ckpt_3gpu.sh
python3 fdong_embedding_dim/codex_scripts/real_data_moe/remote_cuda/summarize_lm_eval_results.py
```

Set `CHECKPOINT_STEP` to evaluate a specific common checkpoint with
`test_loss.sh` or `continuity.sh`. Analysis filenames include that checkpoint
step and are written under `$OUTPUT_ROOT/analysis`.

Load-balance loss is used only while optimizing the router. It adds no model
parameters and is not evaluated during inference, test loss, continuity, or
lm-evaluation-harness runs. Existing checkpoint loading and evaluation APIs
therefore remain unchanged.
