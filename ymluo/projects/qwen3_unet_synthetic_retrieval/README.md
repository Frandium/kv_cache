# Qwen3 U-Net Synthetic Retrieval

This project evaluates the `fdong` mask-based U-Net Transformer checkpoints on
the controlled synthetic retrieval tasks described in `fdong/unet_transformer.md`
section 7.

The evaluator generates token-id sequences directly. This avoids tokenizer
segmentation effects and keeps the task aligned with the paper note:

- Variant A: fixed 4-token patterns.
- Variant B: random 3-token content blocks followed by a shared anchor marker.

For each checkpoint and variant, the script reports answer-only loss and
accuracy for:

- full-sequence forward;
- teacher-forced decode with full KV cache;
- teacher-forced decode with anchor-only KV cache.

It also compares full-KV and anchor-only answer logits and reports average cache
token counts.

## Default Checkpoints

The script defaults to the checkpoint paths supplied for this experiment:

```text
/mnt/workspace/df-unet-transformer/fdong/checkpoints/baseline/145000.pth
/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4/103000.pth
/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4-8-4/70000.pth
/mnt/workspace/df-unet-transformer/fdong/checkpoints/unet-4-8-16-8-4/24000.pth
```

When a checkpoint directory has `runtime_config.json`, the evaluator uses it.
Otherwise it falls back to the known stride schedules for these run names.

## Run

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

Quick smoke test:

```bash
NUM_SAMPLES=8 BATCH_SIZE=1 \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

Useful overrides:

```bash
CONFIG_DIR=/mnt/workspace/Qwen3-0.6B \
NUM_SAMPLES=256 \
BATCH_SIZE=4 \
VARIANTS=A,B \
DEVICE=cuda \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_synthetic_eval.sh
```

Outputs:

```text
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.json
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/synthetic_eval/metrics.csv
```

## Train With Answer-Only Loss

Train a model on generated synthetic retrieval data:

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

Run the same training in the background with `nohup`:

```bash
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/nohup_train_synthetic.sh
```

Watch the log:

```bash
tail -f ymluo/projects/qwen3_unet_synthetic_retrieval/logs/train_*.log
```

By default this trains an `unet-4` schedule on Variant B. The loss is:

```text
prefill = source[:, :-1]
query = source[:, -1:]
cross_entropy(model(query, past_key_values=prefill_kv).logits[:, -1, :], answer)
```

No loss is applied to the first 1023 next-token positions, so backpropagation is
driven only by the final answer prediction. The default `TRAIN_MODE` is
`anchor_kv_decode`, so the training path uses the same anchor-only KV pruning
path that the evaluator tests. Set `TRAIN_MODE=full_sequence` if you want the
cheaper single-forward version.

Useful overrides:

```bash
MODEL_NAME=unet-4-8-4 \
VARIANT=A \
RUN_NAME=unet-4-8-4-variant-a-answer-only \
TOTAL_STEPS=20000 \
BATCH_SIZE=8 \
TRAIN_MODE=anchor_kv_decode \
bash ymluo/projects/qwen3_unet_synthetic_retrieval/scripts/run_train_synthetic.sh
```

Use `VARIANT=mix` to randomly alternate Variant A and Variant B by batch.

Training outputs:

```text
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_unet_synthetic_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
```

## Notes

The synthetic trainer initializes from the Qwen config unless `--init_checkpoint`
is provided. It saves plain `state_dict` checkpoints compatible with the fdong
evaluation code.
