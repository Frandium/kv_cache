# Inverse KV: Feature-Based Memory Layout

## 1. 核心问题

之前的 `main_seq_compress` 分支从 sequence 维度出发：如果语言数据具有层次化、组合式结构，那么模型不一定需要在所有层保留完整 token-level memory。通过 stride / anchor memory，可以沿时间轴压缩 KV cache。

`main_inverse_kv` 分支关注另一条可能的组织轴：feature 维度。

如果语言 token 背后存在可复用的 feature，例如实体、主题、句法角色、代码变量、函数职责、事实类型、推理状态等，那么模型内部 memory 也许不应该只按 sequence order 平铺存储，而应该按 feature organization 建立索引。

核心问题不是先去观察现有 MoE 模型是否碰巧已经学到了这种结构，而是：

> 如果数据本身由 hierarchical, compositional features 生成，我们能否设计一种 MoE / KV 结构，使模型主动学习到 feature-based routing，并让同 feature token 形成可用于 KV retrieval 的 memory bucket？

## 2. 语言数据假设

我们仍然沿用同一个底层假设：

> 人类语言数据是 hierarchical, compositional 的 sequential data。

这里的重点不是 token 的表面顺序，而是 token 背后的 feature 结构：

- 低层 token 组合成短语、实体、事件、变量、函数、约束；
- 高层 feature 在长文本中反复出现，并通过指代、引用、因果、调用关系连接；
- 不同位置的 token 可能距离很远，但共享同一个 semantic / functional feature；
- 推理时真正需要访问的历史信息，往往不是“最近 token”，而是“和当前 query 共享相关 feature 的 token”。

因此，KV memory 的自然组织方式可能不是：

```text
token_1, token_2, ..., token_N
```

而更像：

```text
feature_bucket_1 -> tokens carrying feature 1
feature_bucket_2 -> tokens carrying feature 2
...
feature_bucket_M -> tokens carrying feature M
```

## 3. MoE Gate 假设

在 MoE 模型中，gate/router 接收 token hidden state，并把 token 分发给 expert：

$$
g(h_i) \rightarrow e_i
$$

通常我们把这个过程理解成 compute routing：不同 token 被送到不同 FFN expert，以提升模型容量和计算效率。

但这里我们关心的不是已有 MoE gate 是否自然产生了这个现象，而是把 MoE gate 作为一种可设计的结构假设：

> MoE gate 不应只被设计成 compute router，也可以被设计成 feature router：它按照 token hidden state 中的 latent feature 对数据进行分组。

如果模型在合成的层次化数据上学会了这种 routing，那么 expert assignment 可以被看成一种 learned feature index：

$$
h_i \mapsto \text{feature bucket}
$$

这进一步导出一个可验证推论：

> 被分到同一个 expert 的 token，应该在 attention 计算中表现出更高相关性。

形式化地说，对 token $i,j$，如果：

$$
e_i = e_j
$$

那么我们期望：

$$
q_i^\top k_j
$$

或 attention mass 相比不同 expert 的 token pair 更高。

## 4. Inverse KV 的直觉

标准 attention 可以看成：

```text
current query q -> scan all historical K/V -> weighted read V
```

也就是每个 query 都在平铺 KV cache 上做一次全量检索。

`inverse_kv` 想反过来问：

> 如果模型被设计成必须识别当前 token 属于哪些 feature，那么能否用 gate / expert assignment 作为 KV cache 的索引入口？

也就是说，KV cache 不再只是按时间顺序存储：

```text
K/V[time]
```

而是额外组织成：

```text
K/V[expert][time]
```

推理时，query 可以先通过 gate 得到相关 expert，再优先检索同 expert 或相关 expert 的 historical K/V。

如果成立，这会得到一种 feature-indexed KV memory：

```text
query hidden state
-> gate predicts relevant feature buckets
-> retrieve K/V from selected buckets
-> exact attention inside selected candidates
```

这里 KV cache 节省不是最初目的，而是 byproduct：模型不再需要对所有历史 token 做 flat scan，而是先通过 feature index 缩小候选集合。

## 5. 与 Seq-Compress 分支的关系

两个分支共享同一个底层语言假设，但压缩轴不同。

| 分支 | 组织轴 | 核心结构 | 主要问题 |
|---|---|---|---|
| `main_seq_compress` | sequence / time hierarchy | stride, anchor, U-shaped mask | token-level memory 是否能沿时间轴层次压缩 |
| `main_inverse_kv` | feature hierarchy | MoE gate, expert bucket, feature index | token memory 是否能按 feature 组织和检索 |

因此，`inverse_kv` 不是 `seq_compress` 的替代，而是另一种 memory layout 假设：

- `seq_compress`：远处 token 可以被压入时间轴上的 anchor memory；
- `inverse_kv`：相关 token 可以被组织进 feature bucket，并通过 feature routing 被检索。

## 6. 主线验证问题

### 6.1 合成数据是否具有明确 feature structure

第一步应当构造可控数据，而不是直接分析现有模型。

合成数据需要显式包含：

- latent feature id；
- feature popularity 分布；
- feature composition 规则；
- token sequence 生成规则；
- query / target 依赖某个 feature bucket，而不是依赖最近 token；
- 可评估 ground truth：每个 token 属于哪些 feature，答案应该从哪个 feature bucket 中检索。

这样才能判断模型是否真的学到了我们希望的 feature-based memory，而不是在自然语言噪声中做事后解释。

### 6.2 MoE gate 是否学会 feature routing

在可控数据上，expert assignment 应该和真实 latent feature 对齐。

需要测：

- expert assignment 与 ground-truth feature id 的 mutual information；
- 每个 feature 是否稳定路由到少量 experts；
- 每个 expert 是否主要承载少量 coherent features；
- feature popularity 是否反映到 expert load 或 expert specialization；
- top-k gate 是否能覆盖 token 的 compositional features。

如果 gate 不能恢复 synthetic feature id，那么这个方向的核心假设就没有被验证。

### 6.3 Same-feature / same-expert token 是否更容易互相 attend

这是连接 MoE routing 和 KV retrieval 的关键问题。

对每层、每个 attention head，统计：

- same-expert token pair 的平均 QK logit；
- different-expert token pair 的平均 QK logit；
- same-ground-truth-feature token pair 的平均 QK logit；
- different-ground-truth-feature token pair 的平均 QK logit；
- query token 的 attention mass 有多少落在 same-expert historical tokens 上；
- top-attended historical tokens 中有多少与 query 属于同 expert；
- expert assignment 与 attention target 的 mutual information / enrichment。

必须做 permutation baseline：

- 随机打乱 expert assignment；
- 保持 expert 使用频率不变；
- 比较真实 expert assignment 是否显著优于随机 assignment。

### 6.4 Gate 是否能作为 KV retrieval router

即使 same-expert attention 更高，也还要验证 gate 是否足够用于检索。

需要测：

- 只保留 same-expert K/V 时，attention mass retained 多少；
- 只保留 top-k related experts 时，attention mass retained 多少；
- exact attention output 与 full attention output 的误差；
- next-token loss / retrieval accuracy 是否接近 full KV；
- 不同层、不同 head 是否需要不同 expert routing。

## 7. 最小实验设计

第一阶段应当训练一个小模型，而不是分析现有 MoE 模型。

### 7.1 Synthetic data

输入：

- vocabulary 中包含 content tokens、feature tokens、query tokens、answer tokens；
- 每条序列由多个 latent features 生成；
- 每个 feature 生成若干 token 或 block；
- query 指定某个 feature，target 要求模型从该 feature bucket 中找回对应信息；
- feature popularity 可以服从 Zipf / long-tail 分布；
- feature 可以组合，例如 `(topic, entity, relation)` 或 `(module, variable, attribute)`。

目标是让任务必须依赖 feature-level grouping，而不能只靠局部 n-gram 或最近窗口解决。

### 7.2 Model

模型应当包含：

- causal Transformer backbone；
- MoE FFN 或显式 feature router；
- 可记录 gate logits / selected experts；
- 可记录 attention Q/K 或 attention weights；
- 可选：限制 attention 只能访问 selected expert buckets，用于测试 gate-as-KV-router。

### 7.3 Metrics

输出统计：

1. task loss / retrieval accuracy；
2. expert assignment 与 ground-truth feature 的 mutual information；
3. same-feature vs different-feature hidden similarity；
4. same-expert vs different-expert QK logit；
5. attention mass retained by same feature / same expert；
6. 只访问 same expert 或 top-k experts 时的 accuracy；
7. 与 shuffled expert assignment 的对比。

如果结果显示：

```text
expert assignment predicts ground-truth feature
and
same-expert attention mass >> shuffled-expert attention mass
and
expert-restricted retrieval preserves accuracy
```

则说明这种 MoE / KV 结构确实能学习 feature-indexed memory。

如果结果不明显，则说明当前数据生成规则或模型结构没有迫使模型学习 feature-indexed memory，需要调整 synthetic task 或 router 约束。

## 8. 当前风险

- 如果只使用普通 MoE FFN，gate 的训练目标仍然不是 attention retrieval，而是 FFN expert selection。
- load balancing loss 会人为拉平 expert 使用频率，可能掩盖 feature popularity。
- 一个 expert 可能包含多个 unrelated features。
- 一个 feature 也可能被拆散到多个 experts。
- attention 是 layer/head-specific 的，不能假设某层 gate 能预测所有 attention heads。
- causal LM 中 query token 的 expert assignment 和历史 token expert assignment 未必在同一层语义空间中对齐。
- synthetic task 如果太简单，模型可能用 shortcut 解决，而不学习 feature routing。
- synthetic task 如果太人工，可能证明不了自然语言中同样存在该结构。

## 9. 当前定位

`inverse_kv` 的目标不是简单复用现有 MoE 来省 KV cache，而是验证一个更基础的问题：

> 对于具有 hierarchical, compositional feature structure 的 sequential data，模型是否可以通过 MoE / routing 结构学习 feature-indexed memory，并把这种 memory 用作 KV retrieval 的组织方式？

如果成立，MoE 不只是 conditional computation，也可能是 feature-indexed memory 的自然入口。

现有 MoE 模型的离线统计可以作为后续诊断，但不应当是当前主线。当前主线应是：先构造可控数据，再设计能捕捉该结构的小模型，验证 feature routing 和 KV retrieval 是否同时成立。
