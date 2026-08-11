# L/M MoE remote evaluation suite

This is one incremental suite for both the latest-checkpoint diagnostics and
multi-checkpoint scaling curves. It never compares mismatched baseline/proposed
steps within a model size.

Run from the remote bundle root:

```bash
bash eval_scripts/launch_all.sh
```

Monitor:

```bash
tail -f eval_outputs/logs/master.log
cat eval_outputs/STATUS.tsv
```

Success is indicated by a `DONE` status and this file:

```text
eval_outputs/final_results.tar.gz
```

The final archive contains only compact CSV files and is checked against a
5,000,000-byte limit. Large predictor weights and intermediate results stay in
`eval_outputs/raw/` on the remote server.

## Default experiment sizes

- Routing load/continuity: 64 sequences × 2048 tokens, distributed over 8 GPUs.
- Predictability: 512 train and 64 disjoint test sequences × 256 tokens, four
  epochs, distributed over 8 GPUs.
- Real swapping latency: batch-size-one cached autoregressive decode, 32-token
  prompt plus 2048 generated tokens, three repeats, budgets 1/2/4/8.
- TTFT: batch-size-one full causal prefill through the first output logits,
  prompt lengths 32/128/512/1024/2048, cold and warm expert caches, five repeats,
  and budgets 1/2/4/8. Tokenization and checkpoint construction are excluded;
  prompt H2D transfer is included.
- Held-out loss: 32 sequences × 1024 tokens.
- Downstream tasks: ARC Challenge/Easy, HellaSwag, LAMBADA OpenAI, PIQA, SIQA,
  RACE, and Winogrande.

Environment variables with the corresponding uppercase names can reduce the
workload for a smoke run, for example:

```bash
ROUTING_SEQUENCES=8 ROUTING_LENGTH=128 \
PREDICT_TRAIN_SEQUENCES=64 PREDICT_TEST_SEQUENCES=16 PREDICT_EPOCHS=1 \
SWAP_DECODE_TOKENS=64 SWAP_REPEATS=1 TEST_SEQUENCES=2 \
bash eval_scripts/launch_all.sh
```

Scaling checkpoints are selected against a fixed global FLOPs grid
`1.25e19,2.5e19,5e19,1e20,2e20,4e20`; the latest common checkpoint is always
included. Override this with `SCALING_FLOPS_TARGETS`, but keep the same value
across reruns if old points must remain exactly comparable.

Completed jobs are content-keyed under `eval_outputs/jobs/` by checkpoint,
protocol, and evaluation configuration. Adding checkpoints and relaunching
therefore runs only new work. Existing results from the first evaluation bundle
are migrated and reused automatically.

To update an existing remote installation, extract the update archive over the
bundle root. Do not delete `eval_outputs/`.
