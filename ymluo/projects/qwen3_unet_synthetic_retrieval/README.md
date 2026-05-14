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

## Notes

These checkpoints were trained on normal text data. A low zero-shot score on
Variant A/B is therefore not by itself evidence against the synthetic hypothesis.
The main use of this project is to make the controlled evaluation path concrete;
meaningful positive results require models trained or fine-tuned on the same
synthetic distribution.
