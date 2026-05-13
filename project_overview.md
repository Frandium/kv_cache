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

- **假设**：KV cache 的 sequence 维度可以被压缩。

- **验证状态**：支持。

- **当前结论**：KV sequence 维度存在可压缩性。

## 第 2 轮：第一轮实验是否依赖特定 setting

- **假设 1：压缩比例不够激进。**  
  **结论**：更激进压缩仍然可行，45%的压缩仅带来1%的能力代价。

- **假设 2：训练形态成立，不一定代表推理形态成立。**  
  **结论**：推理形态依旧成立，预期的 kv 节省可以实现。

- **假设 3：当前数据和任务可能太简单。**  
  **结论**：未完全排除。常识问答任务能力可以保持，但长程信息能力还需要进一步验证。

- **当前结论**：强化了“KV sequence 维度可压缩”这一核心判断，并显示出清晰的压缩-能力 trade-off。

详细实验记录见 [`fdong/unet_transformer.md`](fdong/unet_transformer.md)。
