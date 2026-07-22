# `feature_context_frozen`：基于冻结 SAE feature partition 的 LoRA-MoE

## 1. 方法概览

`feature_context_frozen` 的目标不是替换 Qwen3-0.6B 原有的 MLP，而是在第 14 层原始 MLP 旁边增加一条可训练的低秩 MoE 残差支路：

```text
layer-14 hidden state x
        |
        +----------------------> frozen original MLP --------------------+
        |                                                               |
        +--> frozen SAE --> frozen feature partition --> Top-2 experts  |
                                   |                         |           |
                                   +--> expert-owned x_e ----+           |
                                                             v           |
                                               trainable LoRA-MoE delta  |
                                                             |           |
                                                             +-----------+
                                                                         v
                                                   output = MLP(x) + delta
```

整个 Qwen 模型、原始第 14 层 MLP、SAE 和 feature-to-expert assignment 都被冻结。训练期间只更新新加入的 LoRA expert 参数。

与 standard MoE 的根本区别是：

- `standard` 直接从 token hidden state 学习一个线性 router，并把同一个完整 hidden state 交给被选中的 LoRA experts；
- `feature_context_frozen` 使用 SAE 把 hidden state 分解成稀疏 feature，由预先优化并冻结的 feature partition 决定路由；每个 expert 除了看到完整 hidden state，还会看到只由归属于自己的 SAE features 重构出的子空间输入。

因此，`feature_context_frozen` 同时改变了两件事：

1. **如何选择 expert**：从自由学习的 token router 改为由可解释 SAE features 聚合得到的固定语义路由；
2. **expert 如何处理 token**：从只处理完整上下文，改为同时处理完整上下文和 expert 专属 feature 子空间。

---

## 2. 相比 standard MoE 的好处

### 2.1 在参数量匹配下取得了更好的验证集语言模型性能

当前 medium 实验中，两种方法的可训练参数量相同：

| 方法 | Expert 结构 | Router | 可训练参数量 | Mean NLL | Perplexity |
|---|---|---|---:|---:|---:|
| `standard` | 8 个 rank-64 普通 LoRA experts | 可训练线性 router | 1,056,768 | 3.4694767 | 32.11993 |
| `feature_context_frozen` | 8 个 rank-43 双输入 LoRA experts | 冻结 SAE partition | 1,056,768 | **3.4653483** | **31.98760** |

相对 `standard`：

- Mean NLL 降低约 **0.0041284**；
- Perplexity 降低约 **0.13233**，约为 **0.41%**；
- 改善不是通过增加参数量获得的。

这一结果说明：在当前设置和单次实验中，固定的 SAE 语义路由加上 feature-aware expert processing，比自由学习的标准 token router 更有效地利用了相同规模的新增参数。它仍需要多随机种子或置信区间实验来确认统计稳定性。

### 2.2 参数量和主要 adapter 计算量与 standard 基本对齐

Qwen3-0.6B 的 hidden size 为 1024，expert 数为 8。

Standard MoE 每个 expert 有两张 LoRA 矩阵：

```text
A_e: 1024 -> 64
B_e: 64 -> 1024
```

再加上线性 router `1024 -> 8`，参数量为：

```text
8 * (1024 * 64 + 64 * 1024) + 1024 * 8
= 1,056,768
```

`feature_context_frozen` 每个 expert 有两张独立的 down projection 和一张共享的 up projection：

```text
A_context,e: 1024 -> 43
A_feature,e: 1024 -> 43
B_e:         43 -> 1024
```

partition 和 SAE 都不训练，因此参数量为：

```text
8 * (1024 * 43 + 1024 * 43 + 43 * 1024)
= 1,056,768
```

对每个激活 expert，standard 的低秩宽度工作量近似为 `2 * 64 = 128`，新方法为 `3 * 43 = 129`。两者也近似匹配。因此当前性能差异不容易被“更多参数”或“明显更多 adapter 计算”解释。

### 2.3 路由不会在语言模型训练中漂移

`feature_context_frozen` 将 partition logits 注册为 frozen buffer，而不是 `nn.Parameter`：

```text
P(e | f) = softmax(L_f,e / temperature)
```

训练前后的 feature assignment 完全相同，assignment drift 为零。这样带来几个好处：

- expert 的 feature 归属在训练全过程中保持稳定；
- 训练后仍可直接使用 partition 中的 feature 语义解释 expert；
- 不会出现语言模型 loss 将原本的 feature partition 逐渐改写成普通 token router 的情况；
- 可以把性能变化更清楚地归因于固定 partition 与 expert processing，而不是在线 feature-logit 学习。

当前实验里，可学习 assignment 的 `feature_context` 与冻结版本的 NLL 只相差约 `0.000025`，几乎没有区别。这说明该实验中的在线 assignment 学习没有提供可见收益，冻结 partition 是更简单且更容易解释的选择。

### 2.4 显式向 expert 提供其负责的 feature 内容确实有用

`feature_context_no_feature` 是严格匹配的对照：

- 使用相同冻结 partition；
- 使用相同的 feature-based Top-2 路由；
- 使用相同 rank-43 双输入 expert；
- 参数量完全相同；
- 唯一变化是将每个 expert 的 feature reconstruction 输入替换为全零。

结果为：

| 方法 | Mean NLL | Perplexity |
|---|---:|---:|
| `feature_context_frozen` | **3.4653483** | **31.98760** |
| `feature_context_no_feature` | 3.4731682 | 32.23872 |

去掉 feature 内容后：

- Mean NLL 恶化约 **0.0078200**；
- Perplexity 恶化约 **0.25112**，约为 **0.79%**；
- `no_feature` 甚至比 standard 的 NLL 差约 **0.0036916**。

由于两者的路由、参数量和 expert 选择完全一致，这个消融支持一个重要结论：**收益不只是来自固定 feature router；expert 实际接收到与其 feature 归属对应的 SAE 子空间内容也很重要。**

feature reconstruction 的实际输入并非可以忽略的小数值。评估中：

```text
expert_feature_input_rms     = 0.2771
expert_context_input_rms     = 1.0286
feature/context RMS ratio    = 0.2698
```

也就是说，feature 分支输入的 RMS 约为完整上下文的 27%，足以为 expert 提供明显的条件信号。

### 2.5 把“完整上下文”和“feature 专属信息”解耦处理

只输入 expert-owned SAE reconstruction 的 `feature_reconstruction_frozen`，NLL 为 3.4690554，只比 standard 改善约 0.0004213；而加入完整上下文的 `feature_context_frozen` 改善明显更大。

这表明只给 expert 一个稀疏重构子空间会丢失必要信息。新方法保留两条互补路径：

- `A_context,e(x)` 保留 token 的完整上下文、SAE reconstruction residual、非 SAE 信息和跨 feature 交互；
- `A_feature,e(x_e)` 强调 expert 被分配负责的 feature 子空间；
- 两者在低秩空间相加后再由 `B_e` 联合映射回 hidden space。

因此，这一结构不是让 SAE reconstruction 取代原始表征，而是把它作为结构化的 expert 条件输入。

### 2.6 专家分工具有预先定义的语义锚点

Standard router 学到的是 `W_router x`，expert 编号本身没有固定含义，而且不同训练或随机种子之间可能发生置换、负载集中或语义漂移。

`feature_context_frozen` 中 expert 的含义来自同一个 feature partition：expert `e` 负责 assignment probability `P(e|f)` 较高的一组 SAE features。由此可以：

- 统计每个 expert 负责哪些 feature；
- 使用已有的 feature 语义与触发样例解释 expert；
- 分析某类 feature 被送到哪个 expert，以及该 expert 学到了怎样的残差变换；
- 在持续学习场景中，把稳定 partition 作为跨阶段不变的功能接口。

这是一种结构和分析上的优势；当前语言模型实验尚未直接证明它一定提升持续学习或减少遗忘。

### 2.7 对 crowding 结果应作准确解读

新方法的验证结果支持“更好的 feature-aware processing”，但不能简单概括为“所有定义下的 crowding 都低于 standard”。尤其在做了负载校正后，`feature_context_frozen` 的 crowding 并没有稳定优于 standard。

因此，目前最稳健的结论是：

> 在相同参数量和近似计算量下，固定 feature partition 提供稳定的语义路由；同时向 expert 输入完整上下文及其专属 SAE 子空间，能够改善语言模型性能。当前改善不能只归因于更低 crowding，feature 分发之后如何被 expert 处理是关键因素。

还不能仅凭这些结果声称：

- 新方法在所有 crowding 指标上都优于 standard；
- partition 中每个 feature 的具体归属一定是最优的；
- 性能改善已经在多随机种子下显著；
- 该方法已经证明能改善持续学习。

---

## 3. `feature_context_frozen` 的完整工作流程

## 3.1 阶段 A：训练并冻结 SAE

首先在 Qwen3-0.6B 第 14 层 MLP 的输入 hidden state 上训练 Top-K SAE。对于 hidden state `x`，SAE 执行：

```text
x_norm = (x - activation_mean) / activation_scale
pre    = W_enc x_norm + b_enc
z      = TopK(ReLU(pre))
```

本实验 SAE 有 16,384 个 feature，每个 token 保留 32 个 active features。SAE 训练完成后，其 encoder、decoder、bias、activation mean 和 activation scale 全部冻结。

SAE 在后续实验中扮演两个角色：

1. 将 token hidden state 转成稀疏、带强度的 feature activations；
2. 将属于某个 expert 的 feature 子集解码回 hidden-state 子空间。

## 3.2 阶段 B：离线构建 feature crowding/coactivation graph

`build_feature_partition.py` 从 SAE checkpoint、验证/校准 feature statistics 和一批 calibration tokens 构建 feature graph。

Activation-aware partition 使用 feature activation energy 作为 importance，并基于 decoder 几何关系以及 token 上的共激活关系构造稀疏边。对 feature pair `(i,j)`，graph 可记录：

- decoder direction 的几何相似性；
- firing 或 activation-energy importance；
- coactivation overlap；
- token separability，即二者是否能够通过 token 路由被分开；
- mutual coactivation，用于判断是否应当作为 must-link bundle。

直觉是：

- 容易被不同 token 分开的、存在 crowding 风险的 feature pair，尽量分给不同 experts；
- 经常必须在同一个 token 上共同出现的 feature，不强行拆开，而是允许或鼓励组成可路由的 bundle；
- 同时保持 feature-weighted 和 token-level expert load 平衡；
- 保持每个 token 的 Top-2 experts 覆盖足够多的 active feature mass。

## 3.3 阶段 C：优化并保存 feature partition

为每个 feature `f` 学习一组 assignment logits：

```text
L_f in R^(num_experts)
P(e | f) = softmax(L_f,e / T)
```

partition 优化器组合以下目标：

- graph crowding loss；
- empirical token crowding loss；
- feature importance balance；
- token routing load balance；
- must-link violation；
- assignment entropy；
- Top-K fragmentation/coverage。

温度在优化过程中逐步退火，使最终 assignment 接近离散 feature ownership。完成后保存：

```text
feature-partition.pt
├── assignment_logits
├── graph
├── summary
└── config
```

`feature_context_frozen` 使用 activation-aware partition，即 experiment runner 中的 `feature-activation-partition/feature-partition.pt`。

## 3.4 阶段 D：在 Qwen 第 14 层安装并行模块

训练脚本载入 Qwen、SAE 和 partition，然后用 `ParallelLoRAMoE` 替换原来的 `layer[14].mlp`。原 MLP 被保存在 wrapper 内部并冻结。

对每个输入 `x`，主路径仍然计算：

```text
y_base = original_mlp(x)
```

新增模块只学习残差 `delta(x)`，最终输出为：

```text
y = y_base + delta(x)
```

所有 LoRA `B_e` 矩阵采用零初始化，因此训练开始时 `delta(x)=0`，模型初始行为与原始 Qwen 完全一致，避免新模块在第一步就破坏原模型输出。

## 3.5 阶段 E：在线提取 token 的稀疏 SAE features

对第 14 层每个 token 的 MLP 输入 `x_t`，冻结 SAE 产生：

```text
{(f_t1, z_t1), ..., (f_tK, z_tK)}, K = 32
```

其中 `f_tk` 是 feature index，`z_tk >= 0` 是 activation value。该过程在 `torch.no_grad()` 下执行，不为 SAE 构建梯度。

## 3.6 阶段 F：从 active feature mass 计算 Top-2 token 路由

先对每个 token 的 active feature activation 做归一化。默认 `activation_power=1`：

```text
w_tk = z_tk / sum_j z_tj
```

再用冻结 partition 将 feature mass 聚合到 experts：

```text
m_t,e = sum_k w_tk * P(e | f_tk)
```

选取 mass 最大的两个 experts：

```text
S_t = Top2_e(m_t,e)
g_t,e = m_t,e / sum_(j in S_t) m_t,j,  e in S_t
```

这里需要区分两类权重：

- `w_tk` 只用于路由时衡量 active feature 的相对 mass；
- 后续 feature reconstruction 仍使用原始 activation `z_tk`，不会因路由归一化而丢失 feature 强度。

## 3.7 阶段 G：为每个被选 expert 重构其专属 SAE 子空间

对 token `t` 和被选 expert `e`，只解码该 expert 对 active features 的 ownership：

```text
x_t,e = activation_scale * sum_k z_tk * P(e | f_tk) * D_f_tk
```

其中 `D_f` 是 SAE decoder 中 feature `f` 的方向。

这里有两个重要实现细节：

1. 使用原始 `z_tk`，保留 feature activation 的绝对强度；
2. 只乘回 `activation_scale`，不加入全局 `activation_mean`。

不加入 mean 的原因是：mean 是所有 token 和 experts 共享的上下文，不属于任何一个 feature。如果给每个 expert 都添加 mean，会把同一份公共信号重复伪装成 expert-owned content。完整上下文路径已经直接接收原始 `x_t`。

## 3.8 阶段 H：expert 同时处理完整上下文和 feature 子空间

每个 expert 使用 `ContextFeatureLoRAExpert`：

```text
h_t,e = A_context,e(dropout(x_t))
        + A_feature,e(dropout(x_t,e))

delta_t,e = (alpha / rank) * B_e(h_t,e)
```

当前参数为：

```text
rank = 43
alpha = 43
alpha / rank = 1
dropout = 训练配置中的 lora_dropout
```

两条输入先通过不同的 down projection，再在 rank-43 latent space 中相加，最后共享同一个 up projection。它允许 expert 学习：

- 如何利用完整 token 上下文；
- 如何识别和变换自己负责的 SAE feature 内容；
- 如何在低秩空间中组合两类信息。

## 3.9 阶段 I：聚合 Top-2 expert residual

每个 token 只执行选中的两个 experts。最终 LoRA 残差为：

```text
delta_t = sum_(e in S_t) g_t,e * delta_t,e
```

再与冻结 MLP 输出相加：

```text
y_t = original_mlp(x_t) + delta_t
```

所以 token 仍满足 Top-2 expert 计算限制，但 expert 的选择由 active SAE feature mass 决定。

## 3.10 阶段 J：仅用 LM loss 训练 LoRA experts

训练前，整个 Qwen 模型先执行 `requires_grad_(False)`。安装 wrapper 后，只有以下参数可训练：

```text
experts[e].context_a.weight
experts[e].feature_a.weight
experts[e].lora_b.weight
```

以下部分全部冻结：

- Qwen embedding、attention、所有原始 MLP 和 LM head；
- 第 14 层原始 MLP；
- SAE encoder/decoder 和 normalization statistics；
- feature partition logits；
- graph 和统计量。

由于 partition 被冻结，load balance、fragmentation、graph crowding、must-link 等辅助量只用于 telemetry，不具有可训练梯度，也不加入实际优化目标。实际更新只来自 causal language modeling loss：

```text
loss_train = loss_LM
```

这与 standard 不同：standard 的线性 router 是可训练参数，并可接收 load-balance loss 和 router z-loss。

## 3.11 阶段 K：验证与诊断

评估脚本重新安装相同结构并载入 adapter checkpoint，在独立 validation files 上统计：

- mean NLL 和 perplexity；
- 不同 feature-frequency bucket 的 conditioned NLL；
- expert selection load、importance 和 load CV；
- weighted、routeable、load-corrected crowding；
- feature-expert mutual information 和 route entropy；
- Top-K feature mass coverage；
- must-link violation；
- feature assignment drift；
- expert feature-input RMS、context-input RMS 及二者比例。

对 `feature_context_frozen`，assignment drift 应严格为零；对 `feature_context_no_feature`，`expert_feature_input_rms` 应严格为零。这两项也是检查消融实现是否正确的重要 sanity checks。

---

## 4. 与 standard MoE 的逐项对照

| 维度 | `standard` | `feature_context_frozen` |
|---|---|---|
| 基础模型 | 冻结 Qwen3-0.6B | 冻结 Qwen3-0.6B |
| 插入位置 | layer-14 MLP 并行残差 | layer-14 MLP 并行残差 |
| Expert 数 | 8 | 8 |
| Token 激活 experts | Top-2 | Top-2 |
| 路由输入 | 完整 hidden state `x` | Top-K SAE feature activations |
| 路由函数 | 可训练 `softmax(W_router x)` | 冻结 `sum_f normalized(z_f) P(e|f)` |
| Assignment 是否在线更新 | 是 | 否 |
| Expert 输入 | 完整 `x` | 完整 `x` + expert-owned `x_e` |
| Expert 结构 | `B_e A_e x` | `B_e(A_context,e x + A_feature,e x_e)` |
| Expert rank | 64 | 43 |
| 可训练参数 | 1,056,768 | 1,056,768 |
| 训练目标 | LM loss + router auxiliary losses | LM loss only |
| Expert 语义锚点 | 无预定义含义 | 冻结 SAE feature group |
| Assignment drift | 可能发生 | 严格为 0 |

---

## 5. 具体代码文件与职责

### `qwen_moe/adapters.py`

核心在线模型实现：

- `LoRAExpert`：standard 使用的普通两矩阵 LoRA expert；
- `ContextFeatureLoRAExpert`：完整上下文与 feature reconstruction 双输入 expert；
- `ParallelLoRAMoE.__init__`：创建 experts、standard router 或冻结 feature logits；
- `ParallelLoRAMoE._encode_features`：用冻结 SAE 提取 Top-K features；
- `ParallelLoRAMoE._standard_gate`：standard 的线性 Top-K router；
- `ParallelLoRAMoE._feature_gate`：按 active feature mass 聚合冻结 assignment 并执行 Top-2；
- `ParallelLoRAMoE._expert_feature_reconstruction`：解码 expert-owned SAE 子空间；
- `ParallelLoRAMoE.forward`：执行原始 MLP、路由、expert processing、门权重聚合及残差相加；
- `install_parallel_lora_moe`：将 wrapper 安装到指定 Transformer layer 的 `.mlp`。

### `qwen_moe/routing.py`

Feature partition 和路由数学实现：

- `CrowdingGraph`：保存 feature graph 及 frequency、energy、routeable、must-link 等权重；
- `build_sparse_crowding_graph`：根据 decoder geometry 和 feature importance 构造稀疏 crowding graph；
- `add_coactivation_structure`：从 calibration tokens 加入 coactivation、separability 和 must-link 结构；
- `token_expert_mass`：计算在线路由的 `m_t,e`；
- `FeaturePartitioner`：维护 feature-to-expert logits；
- `FeaturePartitioner.losses`：计算离线 partition 的 crowding、balance、fragmentation 和 must-link 等损失。

### `build_feature_partition.py`

离线 partition 入口：

- 加载 SAE 和 feature statistics；
- `collect_feature_samples` 从 DCLM calibration 文本提取 token features；
- 构建 activation-aware crowding/coactivation graph；
- 优化 `FeaturePartitioner`；
- 输出 `feature-partition.pt`、`crowding-graph.pt`、`partition-summary.json` 和训练 metrics。

### `train_qwen_lora_moe.py`

单个方法的训练入口：

- 加载 Qwen、SAE 和 partition；
- 冻结整个基础模型；
- 调用 `install_parallel_lora_moe` 在第 14 层安装模块；
- 校验没有 wrapper 之外的参数被意外解冻；
- 对 frozen partition 跳过无梯度 auxiliary loss，只用 LM loss 更新 experts；
- 保存 `adapter-weights.pt`、`config.json`、`metrics.jsonl` 和训练文件列表。

### `run_lora_moe_experiments.py`

完整实验编排入口：

- 将外部实验名 `feature_context_frozen` 映射到内部模型方法 `feature_context`；
- 自动添加 `--freeze-feature-assignment`；
- 为 feature-processing 方法选择 activation-aware partition；
- 默认把 `feature_context` rank 设为约 `2/3 * standard rank`，实现参数量匹配；
- 支持 standard、balanced standard、feature variants 和消融方法从头训练；
- 支持不同方法分配到不同 GPU，并把训练输出保存到各方法的 `train.log`。

### `evaluate_qwen_lora_moe.py`

统一验证入口：

- 根据 checkpoint config 重建相同 wrapper；
- 加载训练后的 adapter 参数；
- 在公共验证数据和公共 reference graph 上计算 NLL、perplexity、路由负载、crowding、frequency-conditioned NLL；
- 计算 load-corrected crowding 和 feature assignment drift；
- 统计 feature/context 输入 RMS，用于验证 feature processing 和 no-feature 消融。

### `summarize_lora_moe_experiments.py`

汇总所有方法的 evaluation JSON，生成统一的 `comparison.json`，包括：

- 各方法绝对指标；
- `feature_context_frozen - standard` 等成对差值；
- frozen、unfrozen、no-feature 等消融对比。

### `qwen_moe/checkpoints.py`

负责加载 SAE checkpoint、feature partition、adapter checkpoint 和数据文件列表，保证训练与评估使用一致的配置格式。

### `tests/test_qwen_moe.py`

覆盖核心结构和消融行为，包括 feature-context expert、冻结 assignment、feature reconstruction、no-feature 零输入以及参数保存/加载等回归测试。

### `README_QWEN_LORA_MOE.md`

项目级实验说明，记录所有方法、partition 目标、数据切分、启动参数以及评估指标定义。

---

## 6. 当前实验能够支持的核心结论

当前结果最直接支持以下结论：

1. 在相同可训练参数量和近似 adapter 计算量下，`feature_context_frozen` 的验证 NLL 和 perplexity 优于 standard MoE；
2. 冻结 assignment 与可学习 assignment 的结果几乎相同，因此当前收益不依赖在线 feature assignment drift；
3. 相同路由下将 expert-owned feature reconstruction 置零会显著恶化性能，说明收益不只是 feature routing，feature 内容的后续处理确实重要；
4. 只处理 SAE 子空间的收益很小，而完整上下文与 feature 子空间联合输入效果更好；
5. 当前结果尚不能证明收益来自更低的 load-corrected crowding，也尚未证明具体 feature ownership 优于任意非零 SAE reconstruction。后者需要“所有 experts 接收相同 full SAE reconstruction”或 shuffled ownership 等进一步匹配消融。

