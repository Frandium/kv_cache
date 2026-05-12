# Attention Energy Pruning Loss/PPL 实验总结

## 实验设置

本实验使用 `part-00000.txt` 截取后的 3000 个 token。先用完整 attention 跑一遍模型，得到每层、每个 head、每个 query token 的 attention 分布；然后按 attention 分数从高到低选取 token，使累计 attention energy 分别达到 50%、75%、76%、78%、80%、82%、84%、86%、88%、90%、95%、98%、100%，再重新 forward 计算模型 loss。

这里的 energy 定义为：

```text
被保留 attention 分数之和 / 完整 attention 分数之和
```

`energy_threshold=1.0` 是完整 attention baseline。

## 主要结果

| energy_threshold | loss_mean | 相对完整 attention 的 loss 变化 | PPL 倍率 |
|---:|---:|---:|---:|
| 0.50 | 4.581851 | +0.600312 | 1.8227x |
| 0.75 | 4.135016 | +0.153477 | 1.1659x |
| 0.76 | 4.123737 | +0.142198 | 1.1528x |
| 0.78 | 4.105289 | +0.123750 | 1.1317x |
| 0.80 | 4.084739 | +0.103200 | 1.1087x |
| 0.82 | 4.064307 | +0.082768 | 1.0863x |
| 0.84 | 4.040527 | +0.058988 | 1.0608x |
| 0.86 | 4.019267 | +0.037728 | 1.0384x |
| 0.88 | 4.008092 | +0.026553 | 1.0269x |
| 0.90 | 3.995745 | +0.014206 | 1.0143x |
| 0.95 | 3.983569 | +0.002030 | 1.0020x |
| 0.98 | 3.981308 | -0.000231 | 0.9998x |
| 1.00 | 3.981539 | +0.000000 | 1.0000x |

PPL 倍率按 `exp(loss_mean - baseline_loss)` 计算，用来表示相对完整 attention 的困惑度变化。

## 结论

1. attention 的有效信息高度集中，保留到 90% energy 已经接近完整 attention。

   完整 attention 的 loss 是 3.981539；90% energy pruning 后 loss 是 3.995745，只增加 0.014206，对应 PPL 约增加 1.43%。这说明低能量的 attention tail 从整体 next-token loss 看贡献很小。

2. 95% energy 基本没有可观测损失。

   95% energy 的 loss 只比 baseline 高 0.002030，PPL 只增加约 0.20%。如果目标是在尽量不影响模型质量的前提下减少 attention token，95% 是很保守的阈值。

3. 98% energy 和完整 attention 几乎完全一致。

   98% 的 loss 比 100% baseline 还低 0.000231。这个差异极小，更合理的解释是有限样本和数值误差，不应解读为 pruning 一定提升模型质量。它说明 98% 和完整 attention 在这个实验上没有实质差别。

4. 低于 80% energy 后质量退化明显。

   75% energy 的 loss 增加 0.153477，PPL 约变为 baseline 的 1.1659 倍；50% energy 的 loss 增加 0.600312，PPL 约变为 baseline 的 1.8227 倍。说明只保留一半 attention energy 会丢掉大量对预测有用的信息。

5. 比较合理的折中点在 90% 到 95% 之间。

   从 50% 提高到 90%，loss 持续明显下降；但从 90% 提高到 100%，loss 只再下降 0.014206。也就是说，90% 之后边际收益快速变小。如果后续目标是加速或减少 KV 读取，90% 可以作为激进方案，95% 可以作为稳妥方案。

## 图中 Head 9, Threshold 0.9 的含义

这张图展示的是在 `head=9`、`energy_threshold=0.9` 时，每一层平均需要保留多少个 token 才能覆盖 90% attention energy。

图中可以看到，不同层之间需要的 top-k token 数差异很大：平均值约为 48.77，最大值出现在 layer 12，约为 325；最小值出现在 layer 11，约为 1。layer 1、layer 6、layer 12、layer 15 到 17、layer 25、layer 27 需要的 token 数相对更多，而很多层只需要很少 token 就能覆盖 90% energy。

这个结果说明 attention 稀疏程度不是所有层都一样，也不是所有 head 都一样。因此，用固定 top-k token 数做剪枝可能不理想：同一个 k 对某些层过大，对另一些层又过小。按 energy threshold 自适应选择 top-k 更合理，因为它允许每个 layer/head/query token 使用不同的保留 token 数。

## 如何从实验结果推出这些结论

判断方法是把每个阈值的 `loss_mean` 和完整 attention 的 `loss_mean=3.981539` 做差：

```text
delta_loss = loss_mean(threshold) - loss_mean(1.0)
```

如果 `delta_loss` 很小，说明剪掉低 attention energy 的 token 后，模型预测几乎不变；如果 `delta_loss` 很大，说明这些被剪掉的 token 对模型输出仍然重要。

从表格看，90% energy 的 `delta_loss=0.014206`，95% energy 的 `delta_loss=0.002030`，都很小；而 75% energy 的 `delta_loss=0.153477`，50% energy 的 `delta_loss=0.600312`，明显变大。因此可以判断：attention energy 的前 90% 到 95% 覆盖了绝大多数对 loss 有贡献的信息，而 50% 到 75% 的剪枝过于激进。

同时，Head 9 的图说明达到同一个 90% energy 所需 token 数在不同层差异很大。因此结论不是“固定保留某个 token 数就够了”，而是“按 attention energy 自适应决定每层每头每个 token 的 top-k，更符合实际 attention 分布”。

## 注意事项

- 这个结论基于当前 3000-token 文本片段，后续最好在更多文本 shard 上重复验证。
- 当前 pruning 使用完整 attention 先验来决定 top-k，属于分析实验；如果要做真正推理加速，还需要设计在线估计或可部署的选择策略。
- 98% loss 略低于 100% 的差异太小，不建议解读为 pruning 提升模型，只能认为二者等价。
