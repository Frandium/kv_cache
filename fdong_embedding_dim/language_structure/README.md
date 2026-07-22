# Language structure pilot

第一轮建议运行 10,000 篇文档，每篇最多 1,024 个 Qwen3 tokens：

```bash
python3 fdong_embedding_dim/language_structure/mine_hierarchy.py \
  --data-dir /Users/bytedance/Desktop/dclm \
  --tokenizer-dir fdong/Qwen3-0.6B \
  --output-dir fdong_embedding_dim/language_structure/results/pilot_10k \
  --max-documents 10000
```

100 篇文档速度测试需要降低 support 阈值：

```bash
python3 fdong_embedding_dim/language_structure/mine_hierarchy.py \
  --data-dir /Users/bytedance/Desktop/dclm \
  --tokenizer-dir fdong/Qwen3-0.6B \
  --output-dir fdong_embedding_dim/language_structure/results/smoke_100 \
  --max-documents 100 \
  --max-documents-per-file 10 \
  --level1-min-train-support 3 \
  --level1-min-valid-support 1 \
  --level2-min-train-support 2 \
  --level2-min-valid-support 1 \
  --progress-every 10
```

程序持续打印 documents、tokens、docs/s、tokens/s、事件数和 ETA。最终结果见 `summary.json`、`c4_patterns.csv`、`c8_real_patterns.csv`、`c8_null_patterns.csv` 和 `examples.jsonl`。

生成 real/null 分位数、复用度和内容类型汇总：

```bash
python3 fdong_embedding_dim/language_structure/analyze_results.py \
  fdong_embedding_dim/language_structure/results/pilot_10k
```

结果写入 `analysis_summary.json`。
