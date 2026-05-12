# KV Cache Compression Research TODO

## 1. 过去工作的调研

我们的第一性理解是：模型当年就不应该在所有层、所有位置都产生完整 token-level KV cache。人类阅读长文本时不会平铺地记住每个 token，而是会建立目录、索引、层次化摘要和可检索的记忆结构。因此，KV cache 瓶颈不只是系统实现问题，也可能暴露了 Transformer 算法结构本身的问题。

调研过去工作时，不应该只问“有没有人做过类似 mask”，而应该问：

- 他们当初的动机是什么；
- 他们如何理解长上下文或 KV cache 问题；
- 他们的方法是在训练时改变模型，还是在推理时压缩/管理 cache；
- 他们优化的是 attention compute、显存占用、KV cache 容量、还是 decode latency；
- 他们的效果如何，限制在哪里；
- 他们和我们“训练模型学会可丢弃 KV memory layout”的思路有什么关系。

### 1.1 Sparse attention：把 full attention matrix 稀疏化

代表工作：Sparse Transformer、Longformer、BigBird、Reformer、Routing Transformer、MInference。

这条线最早的核心动机不是 KV cache，而是 full self-attention 的 $O(n^2)$ time/memory 太贵，导致长序列训练、prefill 或长文档建模不可扩展。它们理解的问题是：dense attention matrix 中并不是所有 pairwise interaction 都必须显式计算，只要 sparse pattern 仍然允许信息跨位置传播，模型就可以处理更长序列。

- **Sparse Transformer**：提出 fixed / strided sparse attention，把每个位置能看的 key 数量降到约 $O(\sqrt n)$，从而把 attention 复杂度从 $O(n^2)$ 降到 $O(n\sqrt n)$。这是训练时算法，训练和推理 forward 都使用 sparse attention pattern。它和我们的关系最直接：fixed / strided attention pattern 这件事本身已经存在；但它主要优化 attention compute，不是从 decode-time KV cache discard 出发。参考：[OpenAI blog](https://openai.com/index/sparse-transformer/)、[paper](https://huggingface.co/papers/1904.10509)。

- **Longformer**：用 local sliding window attention 加 task-motivated global attention，将长文档 self-attention 做到线性复杂度。它的动机是让 Transformer 能处理几千 token 的长文档，并在 QA、summarization 等任务上有效。它是训练/finetune 时的架构替换，不是 KV cache eviction 方法。参考：[paper](https://huggingface.co/papers/2004.05150)。

- **BigBird**：用 local window + random attention + global tokens，把 sparse attention 做到线性复杂度，并证明在一定条件下保留 full attention 的表达能力。它的问题理解是：如果 sparse graph 具有合适连通性和少量 global tokens，就可以近似 full attention，同时处理更长序列。它是训练时 sparse attention 架构，不是专门为 decode KV cache 设计。参考：[Google Research](https://research.google/pubs/big-bird-transformers-for-longer-sequences/)、[paper](https://huggingface.co/papers/2007.14062)。

- **Reformer / Routing Transformer**：进一步把 sparse pattern 从固定结构推进到 hashing 或 content-based routing。它们关注的是如何用近似最近邻、聚类或路由减少 attention 计算。它们仍然主要面向长序列 attention compute 和训练/forward memory，而不是显式训练一个可删除 KV 的 cache topology。参考：[Reformer](https://huggingface.co/papers/2001.04451)、[Routing Transformer](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00353/97776/Efficient-Content-Based-Sparse-Attention-with)。

- **MInference**：面向 million-token prompt prefill，利用长上下文 LLM attention 的动态稀疏性和若干稳定空间模式，通过在线近似 sparse indices 和定制 kernel 加速 prefill。它是 training-free 推理优化，主要解决长 prompt prefill latency，也提到 KV cache 存储/传输成本，但核心不是让模型训练时学习可丢弃 KV。参考：[Microsoft Research project](https://www.microsoft.com/en-us/research/project/minference-million-tokens-prompt-inference-for-long-context-llms/)。

这条线给我们的启发是：固定 stride/fixed sparse pattern 不是新东西。我们的区别不能落在“提出了 strided sparse attention”，而应落在“这个 mask 被设计成一种未来访问不变量，使非 anchor KV 在 decode 时可安全删除”。

### 1.2 KV cache eviction / compression：在已有模型上选择保留哪些 KV

代表工作：Scissorhands、H2O、StreamingLLM、SnapKV。

这条线最接近现代 LLM serving 场景。它们的核心动机是：decode 阶段 KV cache 随上下文长度和 batch size 线性增长，显存占用和 memory bandwidth 成为瓶颈。它们通常不改变训练算法，而是在推理时判断哪些 token 的 KV 应该保留。

- **Scissorhands**：提出 persistence of importance hypothesis：过去重要的 token 往往未来仍然重要。因此，在固定 cache budget 下优先保留 pivotal tokens。它是 test-time KV cache compression，不需要 finetuning；目标是降低推理显存，提升 batch/throughput。参考：[paper](https://huggingface.co/papers/2305.17118)。

- **H2O**：提出 heavy hitter tokens，认为少量 token 对 attention 贡献巨大，删除它们会严重伤害质量。因此 H2O 动态保留 recent tokens 和 heavy hitters，把 KV eviction 表述为动态子模优化问题。它是推理时 cache eviction policy，不改模型训练。参考：[paper](https://huggingface.co/papers/2306.14048)。

- **StreamingLLM**：发现只保留 recent window 会失败；保留初始 attention sink tokens 加 recent window 可以稳定 streaming generation。它的问题理解是 LLM 中存在并不一定语义重要、但被 attention 强烈依赖的 sink token。它是 training-free 推理框架，也提出预训练时加入 dedicated sink token 可进一步改善。参考：[paper](https://huggingface.co/papers/2309.17453)。

- **SnapKV**：认为模型在 generation 前已经通过 prompt attention 暴露了未来会关注哪些位置，于是用 prompt 末尾 observation window 的 attention pattern 选择重要 KV，并按 head 聚类压缩。它是 fine-tuning-free 推理压缩方法，目标是长输入下的 memory 和 decoding speed。参考：[paper](https://huggingface.co/papers/2404.14469)。

这条线和我们的共同点是都直接面对 KV cache 瓶颈。关键区别是：它们大多是 post-hoc / training-free 的 token selection，默认模型已经产生完整 KV，再选择保留哪些；我们的想法是训练时就规定未来只能访问 anchor KV，让模型从一开始适配这种可丢弃 memory topology。

### 1.3 Layer-wise / pyramidal KV compression：不同层需要不同 cache budget

代表工作：PyramidKV。

PyramidKV 和我们的直觉最接近。它观察到 LLM 长上下文处理中存在 Pyramidal Information Funneling：低层 attention 更分散，高层逐渐聚焦到关键 token / attention sink。因此，不同层不应该使用相同 KV budget：低层保留更多，高层可以保留更少。

- **PyramidKV**：动态调整不同层的 KV cache size，把更多 cache 分配给低层、更少分配给高层。它是推理时 KV cache compression 方法，不重新训练模型；实验显示在 LongBench 上保留很少比例的 KV 也能接近 full KV。参考：[Microsoft Research](https://www.microsoft.com/en-us/research/publication/pyramidkv-dynamic-kv-cache-compression-based-on-pyramidal-information-funneling/)、[paper](https://huggingface.co/papers/2406.02069)。

这条线说明“层次化/金字塔式 KV cache”不是我们独有的直觉。它和我们最大的区别仍然是训练时还是推理时：PyramidKV 在已有模型中观察信息 funneling 并做动态 cache allocation；我们试图通过训练 mask 主动塑造这种 funneling，让模型学会把信息压入未来会保留的 anchor KV。

### 1.4 Long-context serving systems：默认 full KV 存在，优化如何管理它

代表工作：PagedAttention / vLLM、vAttention、RingAttention、blockwise attention 系统。

这条线的问题理解更系统化：KV cache 很大、动态增长、容易造成显存碎片和低 batch capacity；系统应该更好地分配、分页、共享和调度 KV memory。

- **PagedAttention / vLLM**：受操作系统分页启发，把 KV cache 分块管理，减少 fragmentation 和冗余复制，从而提升 serving throughput。它不挑战模型是否应该产生完整 KV，而是更有效地管理完整 KV。参考：[paper](https://huggingface.co/papers/2309.06180)。

- **vAttention**：保留连续 virtual memory，通过底层 demand paging 实现动态 physical memory allocation，避免 PagedAttention 对 kernel 和 serving framework 的复杂改动。它同样是系统 memory management，不是算法结构改变。参考：[paper](https://huggingface.co/papers/2405.04437)、[Microsoft Research](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/)。

- **RingAttention / distributed long-context attention**：把长序列 attention 和 KV 分布到多设备上，解决单卡无法承载超长上下文的问题。它通常默认 attention / KV 总量很大，只是通过并行和通信策略承载它。

这条线和我们的关系是反向的：它们默认巨大 KV cache 是既成事实，然后优化系统管理；我们认为系统瓶颈也许说明算法结构本身不合理，模型不应该在每一层保存所有 token-level KV。

### 1.5 KV quantization：压缩 KV 的 bitwidth，而不是 sequence length

代表工作：KIVI 以及后续 KV quantization 方法。

这条线的动机是：KV cache 是显存和带宽瓶颈，因此可以降低每个 KV entry 的存储精度。它压缩的是 value dimension / bitwidth，而不是 token sequence length。

- **KIVI**：分析 KV cache 的数值分布，提出 key 用 per-channel quantization、value 用 per-token quantization 的 2-bit tuning-free 方法，在较小精度损失下显著降低 KV memory 并提升吞吐。参考：[paper](https://huggingface.co/papers/2402.02750)。

这条线和我们是互补关系：KV quantization 减少每个 KV entry 的 bytes；我们减少 KV entry 的数量。两者理论上可以叠加。

### 1.6 调研后的初步定位

| 方向 | 核心动机 | 训练时/推理时 | 主要优化对象 | 和我们的关系 |
|---|---|---|---|---|
| Sparse Transformer / Longformer / BigBird | 降低 $O(n^2)$ attention compute | 多为训练时架构 | attention compute / full forward memory | fixed sparse pattern 已有；但不是为 KV discard 设计 |
| Reformer / Routing / MInference | 利用动态稀疏性加速长上下文 | 训练时或推理时 | prefill / attention compute | 关注 sparse compute，不直接训练可丢弃 KV layout |
| H2O / Scissorhands / SnapKV / StreamingLLM | 推理时 KV cache 太大 | 推理时 | KV cache capacity / throughput | 目标接近，但多是 post-hoc selection |
| PyramidKV | 高层信息 funnel 到更少 token | 推理时 | layer-wise KV budget | 直觉最接近，但不训练模型适配 anchor topology |
| PagedAttention / vAttention | KV cache 动态管理困难 | 系统推理时 | fragmentation / serving throughput | 管理 full KV，不挑战算法结构 |
| KIVI 等量化 | 每个 KV entry 太占显存 | 推理时 | KV bitwidth / bandwidth | 和 sequence compression 互补 |

当前我们的方法应被理解为：

> 不是发明一种新的 sparse attention pattern，而是用训练时 sparse causal mask 主动塑造一种推理时可删除非 anchor KV 的 memory layout。

这个定位仍需进一步验证：模型是否真的把非 anchor 信息压入 anchor，以及这种训练出来的 cache topology 是否能在长程检索、精确复制和代码引用任务上保持能力。

## 2. 当下科研 idea 还需要验证什么

当前实验已经支持：

- KV sequence 维度可以被压缩；
- stride-based anchor-only KV decode 在推理形态下基本成立；
- 更激进压缩比例仍然可训练，但开始出现可控能力代价；
- stride anchor 位置没有明显系统性退化；
- 普通任务没有系统性崩坏。

接下来需要验证：

- **长程检索能力**  
  测试 needle-in-a-haystack、key-value retrieval、multi-needle retrieval、多事实组合查询。核心问题是：远处信息是否真的能通过 anchor KV 被访问。

- **精确信息恢复能力**  
  测试 exact copy、exact span retrieval、rare entity recall、数字/URL/低频 token 复制。核心问题是：anchor KV 是否会丢失无法由语义摘要恢复的 token-level 细节。

- **代码长程引用能力**  
  测试 variable reference tracking、function/class definition lookup、identifier copy、long-file dependency retrieval。核心问题是：压缩 KV 是否会破坏代码中的精确绑定关系。

- **压缩比例上限**  
  比较 dense baseline、`unet-4`、`unet-4-8-4` 和更激进 stride variants。核心问题是：长程任务上的 KV cache saving 与能力损失曲线是什么。

- **信息是否真的压进 anchor**  
  需要证明 stride 内非 anchor token 的信息被融合进最近 anchor，而不是模型通过其他路径绕过。可以考虑 block ablation、anchor representation probing、删除/扰动 anchor vs non-anchor 的对比、controlled retrieval。

## 3. 变成论文还需要补什么

### 3.1 原理分析

- 明确区分我们的目标和传统 sparse attention：传统 sparse attention 多优化 attention compute；我们优化 decode-time KV cache sequence length。
- 明确区分我们的目标和 post-hoc KV eviction：它们通常在训练后选择重要 token；我们是在训练时约束模型适配可丢弃 KV topology。
- 形式化 cache-discard invariant：如果未来 token 的 mask 永远不会访问某些位置，那么这些位置的 KV 在当前 step 后可以安全删除。
- 分析 layer-wise stride schedule 为什么对应层次化 memory，而不是普通随机稀疏。
- 分析这种方法的理论收益是常数级 KV cache reduction，而不是 sub-quadratic attention complexity；但该常数收益在常见 serving context length 下仍然有实际意义。

### 3.2 实验设计

- 训练曲线：比较 baseline、`unet-4`、`unet-4-8-4` 和更激进 variants。
- 推理 correctness：比较 full KV decode 与 anchor-only KV decode 的 loss、logits diff、top-1 match。
- 位置诊断：按 `i mod stride` 统计 token loss，确认 anchor/non-anchor 位置是否系统性退化。
- 普通能力：常识问答、多选、普通 LM validation。
- 长程能力：retrieval、copy、code reference、多事实组合。
- 机制验证：证明 anchor 表示确实承载了局部压缩信息。
- 对比实验：和 sparse attention、post-hoc KV compression、KV quantization 等代表方法比较。

### 3.3 系统设计

- 当前推理代码主要用于 correctness，不是高效实现。
- 真正的系统实现需要 compact anchor-only KV cache layout。
- decode 阶段不应每层每步重新构造 4D mask；mask/访问规则应当固定或隐式表达。
- 单 token decode 应该直接按 stride 规则访问 compact KV，而不是依赖完整 additive mask。
- 需要测量端到端 memory footprint、decode latency、throughput、batch size capacity。
- 需要区分 prefill 收益和 decode 收益：当前方法主要面向 decode-time KV memory 和 bandwidth。
- 最终系统结果必须证明：丢弃 KV 不只是逻辑上可行，而且带来真实显存和时延收益。

