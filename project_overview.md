# KV Cache Compression: 逻辑闭环

## 背景

现代大模型推理的核心瓶颈之一是 KV cache。它的规模近似为：

$$
O(L \cdot N \cdot d)
$$

其中 $L$ 是层数，$N$ 是上下文长度，$d$ 是 hidden dimension。长上下文场景下，主要压力来自 sequence length $N$。

因此，如果要显著降低 KV cache，最自然的方向是压缩 KV cache 的 sequence 维度。

我们的直觉是：人处理长文本时，并不是平铺地记住所有 token，而是层次化地理解、组织和检索信息。因此，模型也不一定需要在所有层都保存完整 token-level KV。

## 第 1 轮：KV sequence 维度是否可压缩

- **假设**：KV cache 的 sequence 维度可以被压缩。中间层不必缓存所有 token 的 K/V，只需要缓存一部分更高信息密度的 anchor positions。

- **验证状态**：初步真实语料实验支持该假设。在 `unet-4` 设置下，中间层只长期保留约 1/4 的 anchor KV，但 training loss 仍然非常接近 dense baseline。

- **当前结论**：至少在当前实验范围内，KV sequence 维度存在可压缩性。

## 第 2 轮：第一轮结论是否稳健

- **小假设 1：压缩比例可能不够激进。**  
  `unet-4` 可能太容易了；如果压缩到 `unet-8` 或更高比例，模型能力可能明显下降。  
  **验证状态**：正在等待更激进 stride 结构的训练 loss 和能力结果。

- **小假设 2：训练形态成立，不一定代表推理形态成立。**  
  训练时使用 attention mask，并不自动等价于推理时真的可以删除非 anchor KV。  
  **验证状态**：当前 anchor-only KV decode 实验初步支持推理形态下的压缩。在 `unet-4` checkpoint 上，只保留 stride 决定的 anchor KV 后，teacher-forced decode loss 几乎不变，同时实际 KV cache token 数减少约 24%。

- **小假设 3：当前数据和任务可能太简单。**  
  普通网页/新闻语料和常规下游任务未必强依赖长程精确信息，因此 loss 接近 baseline 不一定说明长程信息压缩成功。  
  **验证状态**：普通下游任务已有初步结果，但仍需要更难、更长程、更可控的任务，例如 long-context retrieval、exact copy、synthetic key-value retrieval 和 code reference tracking。

- **当前结论**：`unet-4` 已经在训练 loss 和推理 cache 形态上给出正向证据；下一步要回答的是，这种可压缩性在更激进压缩比例和更难长程任务上能撑到哪里。

详细实验记录见 [`fdong/unet_transformer.md`](fdong/unet_transformer.md)。

