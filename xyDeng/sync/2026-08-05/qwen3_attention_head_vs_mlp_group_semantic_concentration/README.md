# 0805 Qwen3 粗细语义集中性更新

本目录是一份可独立阅读的策展同步包，回答：Qwen3-8B 的粗粒度领域与领域内细粒度概念区分，是否集中在少数 `o_proj` 前真实注意力头，还是集中在进入 MLP 的固定连续通道组。

## 阅读顺序

1. [一页认识更新](focus.md)
2. [自包含实验报告](docs/qwen3_attention_head_vs_mlp_group_semantic_concentration/report.md)
3. [全部实际语义文本](docs/qwen3_attention_head_vs_mlp_group_semantic_concentration/data/actual_semantic_text_sequences.json)
4. [自然语料 calibration 清单](docs/qwen3_attention_head_vs_mlp_group_semantic_concentration/data/calibration_manifest.json)

报告旁边的 `figures/`、`tables/` 与 `data/` 是其全部本地依赖。同步包不包含模型权重、原始激活、集群日志、checkpoint 或 worker 代码。

## 阶段性结论

- 少数真实注意力头的粗、细和细相对粗区分度超过随机同形状切分。
- 典型注意力头没有普遍优势，MLP 输入固定连续通道组也没有超过随机切分。
- 候选头内固定八频带没有超过随机 16 维方向，因此当前支持稀疏头身份，不支持固定 covariance rank 身份。

这些是冻结表示的统计定位结论，不证明注意力头因果计算语义，也不证明该坐标能改善 Router 或训练效率。
