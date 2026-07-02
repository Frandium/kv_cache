# Real-data common/long-tail MoE

This directory contains the 4-layer, 16K-vocabulary real-data comparison:

| variant | common `d_ff` | tail experts | tail `d_ff` | router input |
|---|---:|---:|---:|---|
| baseline | 768 | 4 | 768 | full post-attention residual |
| proposed | 384 | 4 | 864 | causal local mean of attention output |

Both variants use hidden size 768, 12 attention heads, tied input/output
embeddings, one always-active common expert, and one top-1 routed tail expert.
Their total expert capacity is equal because `768 + 4*768 = 384 + 4*864`.

Run both experiments sequentially:

```bash
bash fdong_embedding_dim/codex_scripts/real_data_moe/run_baseline_then_proposed.sh
```

Checkpoints include model, optimizer, exact DCLM stream position, buffered
tokens, random-number states, model config, and train arguments. The launcher
uses `--resume auto`; rerunning it resumes each unfinished experiment from its
`latest.pt`. A checkpoint is written every 250 optimizer steps, while only the
latest three numbered checkpoints plus `latest.pt` are retained to control disk
usage.

The common/tail output-space constraint defaults to off. Enable it for a
separate proposed-model run with `--orthogonalize-tail`; its rank defaults to
16 and its detached common basis is refreshed every 50 optimizer steps.
