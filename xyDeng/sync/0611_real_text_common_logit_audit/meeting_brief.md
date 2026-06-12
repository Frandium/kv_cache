# Meeting Brief: Real-Text Common-Logit Audit

## **Question**

真实 DCLM 文本中，随机初始化的线性稀疏混合专家路由器是否已经存在专家选择偏置？这个偏置主要来自隐藏状态的共同分量、数据中的高频tokens，还是训练早期反馈放大？

这里的共同分量指所有被审计tokens的路由输入均值 $c$。负载集中指少数专家接收过多token，但其他专家不一定完全死亡。硬坍塌指绝大多数tokens进入极少数专家，其余专家几乎不被使用。

## **Conclusion**

本轮实验检验的是：在真实 DCLM 文本中，随机初始化的线性路由器是否已经因为隐藏状态中的共同分量而产生专家负载集中。

1. **真实文本数据下，支持router随机初始化下已经存在 token 分发集中的结论，不支持之前合成数据实验上“第 0 步共同分量已经主导路由”的结论**  
   1. 但是随机初始化下路由是不均匀的：`top-1` 下最大专家负载为 `0.2781`，明显高于 8 个专家均匀分发时的 `0.125`；前 2 个专家累计接收 `0.4467`，前 4 个专家累计接收 `0.7150`。
   2. 第 0 步主要专家分数gap来自 token 自身差异，也就是 $h_{\text{residual}} = h - c$，而不是共同分量 $c$。

2. **common分量仍然是可测量的负载偏置源。**  
虽然共同分量不是第 0 步最大的分数来源，但减去共同分量后，token 偏向少数专家的现象明显缓解。这说明common分量会把路由分布推向某些专家，只是它在第 0 步还没有主导路由。

1. **这个初始化偏置不是高频 token 单独造成的，并且会被早期训练过程快速放大。**  
数据上可以分成三层看。
   1. `top-k` 读取显示单专家硬选择最容易暴露初始化偏置：最大专家选择占比（所有tokens中，tokens选择最频繁的专家，吸收了多少tokens）从 `top-1=0.2781` 降到 `top-2=0.2303`，再降到 `top-4=0.1794`；
   2. 高频 token 不是主因：top200 高频 token 已经占全部 token 的 `52.72%`，但去掉 top200 后，`top-1` 最大专家选择占比仍为 `0.2739`，和全量 `0.2781` 几乎相同。
   3. 训练早期会显著放大这个偏置：第 0 步最大专家负载为 `0.2781`，第 10 步上升到 `0.8507`；有效专家数从约 `6.79` 降到约 `1.96`，但专家激活比例仍为 `1.0000`。问题不是专家全部死亡，而是主要 token 被少数专家吸收。

**更新：**  
在合成数据上的 claim“随机初始化几何直接导致硬坍塌”应该修正为：

1. 真实文本中随机初始化路由器已经存在负载偏置；
2. 这个偏置不是硬坍塌，也不是高频 token 单独造成。
3. 第 0 步共同分量不是主要分数间隔来源，但它是可测量的负载偏置源；随后早期训练过程会把该偏置快速放大，最终形成严重专家负载集中。

放大的具体来源仍需要在下一轮拆分路由器更新、表征空间的变化和专家输出反馈。

## **Key Evidence**

**Evidence 1: 第 0 步共同分量没有主导分数间隔。**

主判据是共同分量占优比：

```text
dominance_ratio = common_margin / residual_margin
```

结果：

```text
common_margin = 0.1237
residual_margin = 0.2364
dominance_ratio = 0.5251
dominance_ratio > 1 cases = 1 / 18 layer-seed cases
```

解释：真实文本第 0 步的单token路由决策主要不是由共同分量决定；原先“共同分量在初始化时已经主导路由”的强版本需要削弱。

**Evidence 2: 共同分量仍然造成负载偏置。**

```text
raw_max_load = 0.2781
centered_max_load = 0.1561
delta_max_load = 0.1220
```

解释：原始路由直接使用 $h_i$；去共同分量路由使用 $h_i-c$。去掉共同分量后最大专家负载下降约 12.2 个百分点，说明共同分量虽然不是最大分数间隔来源，但会把更多token推向少数专家。

**Evidence 3: 单专家硬选择最容易暴露初始化偏置。**

这里的 `top-k` 是读取同一个初始化路由器分数最高的 k 个专家，不是重新训练 `top_k=2/4` 模型。均匀情况下，每个专家选择占比应接近 `1/8 = 0.125`。

| k | 最大专家平均选择占比 | 前 2 专家累计占比 | 前 4 专家累计占比 | 有效专家数 | 负载不均匀系数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2781 | 0.4467 | 0.7150 | 6.7855 | 0.3027 |
| 2 | 0.2303 | 0.3967 | 0.6719 | 7.2267 | 0.2385 |
| 4 | 0.1794 | 0.3334 | 0.6095 | 7.6663 | 0.1510 |

解释：初始化时已经存在专家选择倾向，`top-1` 最明显，`top-2` 减弱，`top-4` 更接近均匀。这支持“单专家硬选择会暴露并强化早期偏置”，但不能证明 `top-k` 训练已经解决问题。

**Evidence 4: 高频token不是初始化集中的主因。**

```text
top10 token ids fraction = 0.1958
top50 token ids fraction = 0.3748
top200 token ids fraction = 0.5272
```

| token组 | token fraction | top-1 max selection share |
|---|---:|---:|
| all tokens | 1.0000 | 0.2781 |
| not top10 | 0.8042 | 0.2749 |
| not top50 | 0.6252 | 0.2751 |
| not top200 | 0.4728 | 0.2739 |

解释：如果专家集中主要由高频token造成，去掉 top200 高频token后最大专家占比应明显下降；实际从 `0.2781` 只降到 `0.2739`。高频token会轻微加重集中，但不是主因。需要区分数据中的高频token和隐藏状态中的共同分量 $c$。

**Evidence 5: 训练早期会快速放大偏置。**

实际专家数为 8 时：

```text
step10 dominance_ratio = 21.7874
step10 raw_max_load = 0.8507
step10 effective_experts ~= 1.96
step10 raw_active_experts_ratio = 1.0000
```

解释：第 10 步 8 个专家都至少被激活，但最大专家吸收 85.07% token，真正承担主要负载的专家约为 2 个。因此这不是硬坍塌，而是强负载集中。

## **Detailed Setup**

**Data:**

```text
data_path = /data/share/109_cache_dir/hf_data/dclm_bin
sample_span = 257 tokens
input_tokens = positions 0..255
target_tokens = positions 1..256
padding = none
audit_sequences = 8192
train_sequences = 32768
```

**Model:**

```text
model = random-initialized Qwen-style causal LM
pretrained = false
num_layers = 6
hidden_size = 512
attention_heads = 8
kv_heads = 4
expert_hidden_dim = 2048
vocab_size = 151936
initializer_range = 0.02
```

**Router and MoE:**

```text
moe_type = standard sparse MoE
router_type = linear
router_input = exact hidden state entering the gate
top_k = 1
shared_expert = false
load_balance_loss = 0.0
norm_topk_prob = false
oracle_gating = false
multihead_routing = false
```

`norm_topk_prob=false` 表示被选中专家输出仍乘以概率函数给出的专家概率。因此任务损失会通过概率函数分母更新未被选中的路由向量，但只有被选中的专家前馈网络参与该token的前向计算。

**Run:**

```text
job_id = pt-mrx0wq1v
run_name = real_text_common_logit_audit_v5_4gpu_20260611_r3
status = succeeded
model_seeds = 0, 1, 2
checkpoints = 0, 1, 10, 50, 100, 300
phase1_rows = 18
phase2_rows = 20736
phase3_rows = 108
phase4_rows = 432
```

## **Current Interpretation**

当前最稳妥的解释是：

```text
真实文本隐藏状态存在整体几何偏置。
随机线性路由器会把该偏置转成专家选择倾向。
top-1 硬选择让这种倾向更明显。
训练早期进一步放大该偏置，形成严重负载集中。
```

这不是在说真实文本第 0 步已经共同分量主导，也不是在说高频token解释了全部现象，更不是在说所有专家已经死亡。

## **Boundary**

当前结论覆盖：

- DCLM 打包文本。
- 随机初始化的小型 Qwen 风格因果语言模型。
- 线性稀疏单专家路由器。
- 关闭共享专家。
- 关闭负载均衡损失。
- 训练前 300 步。
- 路由负载、共同分量分数间隔、残差分数间隔、专家使用比例、有效专家数。

当前结论不覆盖：

- 预训练大模型的混合专家行为。
- 长训练后的最终专家功能分工。
- 专家是否学到可解释、可复用的功能。
- `top-k` 训练能否解决问题。
- 任何可部署的去共同分量路由方法。
- 真实大模型已经发生同样问题的结论。

## **Next Step**

下一步应定位第 10 步共同分量偏置被放大的来源。保持同一数据、同一模型规模、同一路由审计方法，做三个最小因果拆分：

1. 冻结路由器权重，判断没有路由器更新时第 10 步集中是否仍出现。
2. 冻结隐藏状态生成层，判断主干隐藏状态变化是否是主因。
3. 冻结专家输出反馈或切断专家输出反馈，判断早期被选中的专家是否继续强化已有路由。

判断标准：

```text
如果某个冻结条件显著压低 step10 dominance_ratio 和 step10 raw_max_load，
则该部分是早期放大的主要来源。
```
