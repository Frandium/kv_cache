---
name: kv-cache-remote-collaboration
description: Use this skill when collaborating on the kv_cache project under restricted cloud-server access, where Codex cannot SSH directly and must use GitHub plus user-run remote commands to ship code, run experiments, collect logs, and iterate.
---

# KV Cache Remote Collaboration

## Context

The kv_cache project is developed locally and executed on a cloud GPU service.

The cloud service has an access restriction:

- The user can log in through RAM / browser IDE, such as web VSCode.
- Direct SSH access is only allowed from public IPs on a whitelist.
- Codex usually cannot SSH or rsync directly to the server.

Therefore, do not assume direct remote shell access. Use GitHub as the code and result exchange channel.

## Collaboration Principle

Codex writes and reviews code locally.

The user runs commands on the remote server.

GitHub is the synchronization layer between local development and remote execution.

The loop is:

```text
Codex edits locally
-> Codex commits/pushes to GitHub
-> user pulls on remote server
-> user runs debug/training
-> user collects logs/results
-> user commits/pushes results or pastes logs
-> Codex pulls/analyzes/fixes
-> repeat
```

## Standard Workflow

### 1. Local Code Changes

When asked to modify code:

1. Edit files locally in the current repository.
2. Run lightweight local checks when possible, such as syntax checks.
3. Commit and push changes when the user asks.

Do not include large training artifacts in commits.

Prefer committing:

- source code
- shell scripts
- config snippets
- small logs
- metrics summaries
- experiment notes

Do not commit:

- checkpoints
- large model weights
- large raw logs
- caches
- `__pycache__`
- `.DS_Store`

### 2. Remote Pull

Ask the user to run this in the browser IDE terminal:

```bash
cd /path/to/kv_cache
git pull origin main
```

If the project is under a subdirectory, use the actual path shown by the user.

### 3. Remote Single-Thread Debug First

Before distributed training, ask the user to run single-thread debug.

Example:

```bash
cd /path/to/kv_cache/fdong/scripts
python single_thread_debug_qwen.py \
  --dataset_type synthetic_indexed \
  --local_batch_size 1 \
  --global_batch_size 1 \
  --seq_len 1024 \
  --attention_stride_pattern 1,1,1,4,4,4,1,1 \
  --residual_source_pattern -1,-1,-1,-1,-1,-1,-1,-1
```

The exact command may change with the current experiment. Keep the command in the repo when possible, such as in a shell script or experiment note.

### 4. Remote Training

After debug succeeds, ask the user to run the training entrypoint.

Example:

```bash
cd /path/to/kv_cache/fdong/scripts
bash pretrain_baseline.sh
```

or:

```bash
torchrun \
  --nproc_per_node=8 \
  --master_addr=localhost \
  --master_port=12345 \
  pretrain_qwen.py \
  --dataset_type synthetic_indexed \
  --seq_len 1024 \
  --local_batch_size 16 \
  --global_batch_size 512
```

### 5. Collect Results

Ask the user to collect:

- git commit hash
- exact command
- last 100-300 lines of logs
- error stack trace if any
- loss values or metrics
- relevant config values
- GPU / memory error messages if any

Useful remote commands:

```bash
git rev-parse HEAD
```

```bash
tail -n 200 path/to/log_file.log
```

```bash
nvidia-smi
```

```bash
ps aux | grep pretrain_qwen
```

For structured experiment records, prefer a small directory like:

```text
experiments/YYYY-MM-DD-short-name/
  command.sh
  config.json
  metrics.csv
  train_tail.log
  notes.md
```

Keep checkpoints outside git or ignored by `.gitignore`.

### 6. Bring Results Back

There are two acceptable paths.

Path A: User pastes logs directly into the chat.

Use this for quick errors and short debug traces.

Path B: User commits and pushes result summaries.

Ask the user to run:

```bash
git add experiments/YYYY-MM-DD-short-name
git commit -m "Add experiment results for <short-name>"
git push origin main
```

Then Codex can pull locally and analyze the files.

## Debugging Rules

- If remote execution fails, first ask for the exact command and full stack trace.
- If a change affects both debug and distributed training, put the shared logic in common files, not only in one entrypoint.
- For this project, prefer shared training utilities over duplicating logic between `single_thread_debug_qwen.py` and `pretrain_qwen.py`.
- When changing model architecture, run synthetic data first before expensive real-data training.
- When changing data generation, print or inspect one sample before launching long training.

## Git Safety

- Do not force push unless the user explicitly asks.
- Pull/rebase before pushing if remote has new commits.
- Do not commit large artifacts.
- Do not delete remote checkpoints or logs unless explicitly requested.
- If collaborators are active, keep commits focused and use clear messages.

## Expected Handoff Message

When Codex has pushed code and needs the user to run remote commands, provide:

```text
I pushed commit <hash>.
On the remote server, please run:

cd <remote_repo>/fdong/scripts
git pull origin main
<debug_or_train_command>

Then send back:
1. git rev-parse HEAD
2. exact command
3. last 200 lines of log
4. any error trace
```

