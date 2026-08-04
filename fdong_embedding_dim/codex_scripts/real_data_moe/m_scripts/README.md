# M-scale experiment scripts

This directory contains all launch and evaluation entry points for the M-scale
experiment. It depends only on the sibling `../moe/` Python package.

Default model configuration:

- hidden size 768, 24 layers
- 12 attention heads, 6 KV heads, head dimension 128
- 8 tail experts, Top-1 routing
- common and tail intermediate size 1536
- 152,000 training steps (approximately 40B tokens)
- output root `/mnt/workspace/fmoe_cuda_m_8e_outputs`

Launch the two runs on separate 8-GPU workers:

```bash
bash m_scripts/launch_baseline.sh
bash m_scripts/launch_proposed.sh
```

Evaluate their latest common checkpoint:

```bash
bash m_scripts/plot_loss.sh
bash m_scripts/test_loss.sh
bash m_scripts/continuity.sh
RUN_NAME=baseline bash m_scripts/lm_eval.sh
RUN_NAME=proposed bash m_scripts/lm_eval.sh
```

To select a checkpoint explicitly, set `CHECKPOINT_STEP`, for example:

```bash
CHECKPOINT_STEP=0080000 bash m_scripts/test_loss.sh
```

The launch scripts reject a second launch while their recorded process is still
alive. Override paths or training length with `DATA_DIR`, `TOKENIZER_DIR`,
`M_OUTPUT_ROOT`, or `M_MAX_STEPS` when needed.
