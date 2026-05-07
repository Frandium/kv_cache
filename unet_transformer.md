# U-Net Transformer: 一个系统瓶颈驱动的算法假设

Date: 2026-05-07

## 1. 核心问题

现代大模型推理中的 KV cache 瓶颈，表面上是系统问题：长上下文 decode 时需要持续读取大量历史 K/V，导致 HBM 容量、访存带宽、跨设备传输和推理延迟都变成瓶颈。

但这个系统瓶颈可能暴露的是更深层的算法设计问题。

标准 Transformer 在所有层都维持 token 级别的序列长度：

$$
H_\ell \in \mathbb{R}^{N \times d}, \quad \ell = 1,\ldots,L
$$

这意味着不管模型在中间层是否已经形成了更抽象的语义表示，每一层仍然保留 $N$ 个位置，并为这些位置产生 KV cache。于是 KV cache 的规模近似为：

$$
O(L \cdot N \cdot d)
$$

这里隐含了一个非常强的结构假设：

> 模型在所有深度上都需要以 token 分辨率保存和访问历史信息。

U-Net Transformer 想挑战的正是这个假设。

如果语言理解过程本身是层次化、组合式的，那么模型内部表示的 sequence length 也许不应该在所有层保持不变。浅层需要 token 分辨率，但中间层可能只需要少量高语义密度的 latent units。也就是说，模型结构可以从：

$$
N \rightarrow N \rightarrow N \rightarrow \cdots \rightarrow N
$$

变成：

$$
N \rightarrow M_1 \rightarrow M_2 \rightarrow M_b \rightarrow M_2 \rightarrow M_1 \rightarrow N
$$

其中：

$$
N > M_1 > M_2 > M_b
$$

例如输入两端仍然是 10,000 个 token，但中间 bottleneck 可能只有 100 个 latent tokens。

这个想法的目标不是在系统层面更聪明地管理一个已经巨大的 flat KV cache，而是让算法从一开始就不产生那么多中间层 token-level KV。

## 2. 语言数据 feature 空间假设

这里所谓 “hierarchical and compositional 的人类语言数据 feature 假设”，不是一句泛泛的“语言有层次”。它至少包含三层含义。

第一，语言中的低层特征会组合成高层特征。

token 不是孤立发挥作用的。字符、subword、词、短语、句子、段落、章节、文件、对话历史之间存在组合关系。一个高层语义对象通常由多个低层对象组成，例如：

$$
\text{tokens}
\rightarrow
\text{phrases}
\rightarrow
\text{entities/events}
\rightarrow
\text{claims/plans}
\rightarrow
\text{global task state}
$$

这种组合不是简单相邻拼接。一个实体可能在多处被指代，一个论证可能跨段落展开，一个代码功能可能分布在多个文件里。因此真正重要的是 semantic composition，而不只是 positional pooling。

第二，高层 feature 的数量通常远少于低层 token 的数量。

一段 10,000 token 的文本，并不一定包含 10,000 个彼此独立、同等重要的语义对象。它可能包含的是几十到几百个更稳定的结构：人物、变量、函数、约束、事件、主题、证据、任务步骤、因果关系等。

因此，随着表示从词法层走向语义层，feature 空间可能天然变得更稀疏：

$$
\#\text{semantic units} \ll \#\text{tokens}
$$

这就是 U-Net Transformer 中间层可以变短的核心依据。不是因为我们希望压缩，而是因为假设真实语言数据在高层语义空间中本来就有更低的有效维度和更少的 active units。

第三，高层 feature 应当保留可组合、可检索、可展开的结构。

中间 bottleneck 不能只是一个全局平均向量，也不能只是普通压缩码。它应该是一组 latent semantic units，每个 unit 对应某种可被后续推理使用的结构，例如一个实体簇、一个事件链、一个代码模块、一个约束集合、一个论证子图或一个任务状态。

这组 units 应该满足三个性质：

- **compositional**：高层 unit 由低层 tokens/phrases 组合而来，并能继续组合成更高层结构。
- **sparse**：当前任务或上下文中真正活跃的高层 units 数量远小于 token 数。
- **recoverable**：高层 unit 不必无损保存所有 token 细节，但必须能通过 skip/residual/retrieval 路径找回必要细节。

所以这个假设可以更形式化地写成：

> 对于自然语言或代码上下文，存在一个多层 latent feature hierarchy，使得较深层的有效语义状态可以由远少于 $N$ 的 latent units 表示；这些 latent units 保留推理所需的组合结构，并能在需要时重新关联到底层 token 证据。

如果这个假设不成立，U-Net Transformer 就只是人为压缩，会损害模型能力。如果这个假设成立，那么标准 Transformer 在所有层保留 token-length sequence 就是结构性浪费。

## 3. 架构直觉

### 3.1 原始 U-Net 直觉

U-Net Transformer 的直觉是把 Transformer 的深度方向改成“先收缩，再展开”。

输入和输出两端仍然是 token 级别，因为语言建模最终需要精确 next-token prediction：

$$
x_{1:N} \rightarrow y_{1:N}
$$

但模型中间不必始终保持长度 $N$。它可以先通过 downsampling encoder 把 token 序列变成更短的 semantic sequence：

$$
H^{fine} \in \mathbb{R}^{N \times d}
\rightarrow
H^{coarse} \in \mathbb{R}^{M \times d}, \quad M \ll N
$$

一种抽象写法是学习一个 assignment matrix：

$$
A \in \mathbb{R}^{M \times N}
$$

并通过：

$$
H^{coarse} = A H^{fine}
$$

得到 coarse semantic units。

这里的 $A$ 不能理解成固定 pooling 矩阵。它更像一个 learned semantic router：模型要学习哪些 token、短语、引用、变量、段落或事件应该被组合到同一个 latent unit 中。

在 bottleneck 处，模型只在少量 latent units 上做全局推理：

$$
H_b \in \mathbb{R}^{M_b \times d}, \quad M_b \ll N
$$

如果 $N=10000$ 且 $M_b=100$，那么中间层的 attention/KV 规模会大幅下降。系统收益来自这里：昂贵的长程推理不再发生在全 token 分辨率上。

然后模型通过 upsampling decoder 回到 token 分辨率：

$$
H^{coarse} \rightarrow \tilde{H}^{fine} \in \mathbb{R}^{N \times d}
$$

可以抽象成：

$$
\tilde{H}^{fine} = B H^{coarse}, \quad B \in \mathbb{R}^{N \times M}
$$

但仅靠 bottleneck 展开一定不够。很多局部细节、位置边界、rare token、代码符号、精确引用不应该被迫穿过 100 个 latent tokens。因此 U-Net 结构需要 skip/residual paths：

$$
H^{up}_\ell =
\operatorname{Fuse}
(
\operatorname{Upsample}(H^{coarse}),
H^{skip}_\ell
)
$$

架构直觉因此是：

- downsampling path 负责把 token 组织成层次化语义对象；
- bottleneck 负责低成本的全局推理和长期状态整合；
- upsampling path 负责把全局语义重新分配回 token 位置；
- skip/residual path 负责保留不适合被压缩的精确信息。

这不是简单为了省 KV 而压缩 sequence length。更准确地说，它假设“语义抽象层的有效序列长度本来就应该更短”，然后用这个结构性假设带来系统收益。

### 3.2 物理 U-Net 迁移到语言模型的核心困难

如果直接把这个原始 U-Net 直觉实现成物理的 sequence length 变化，会立刻遇到几个技术问题。

第一，压缩比例和语义切分并不是最本质的困难。我们可以先固定一个 schedule，例如：

$$
N \rightarrow N/4 \rightarrow N/16 \rightarrow N/4 \rightarrow N
$$

也可以先不要求 downsampling 严格对齐实体、事件、段落或代码结构。语言虽然不像图像那样强局部连续，但局部 token window 仍然有足够的短语、句法和块结构；远处相关信息也可以通过 attention 继续访问。因此，第一版不必把问题变成“如何找到完美语义边界”。

第二，残差连接本身也不是不可解决的问题。经典 U-Net 的 skip connection 是同分辨率相连；如果物理 sequence length 确实变成 $N, N/4, N/16, N/4, N$，那么可以在 decoder 恢复到相同长度时再连接 encoder 侧对应表示。问题不在于 residual 能不能连，而在于这种物理降采样/升采样是否适合语言模型。

真正困难的是第三点：语言模型是 causal prefix-to-next-token 映射，而不是视觉 U-Net 中固定长度、非因果的 dense-to-dense 映射。

如果真的把 4 个 token fuse 成 1 个 latent token，那么训练时会遇到未来泄露：预测第 2 个 token 时，不能使用由第 1-4 个 token 共同形成的 latent token。推理时也会遇到动态边界：当 prefix 长度为 40 时，中间层可能有 3 个 latent tokens；当 prefix 长度变成 41 时，最后一个 group 又变成 partial state。于是模型必须回答：如何处理未满 4 个 token 的 group？如何 upsample 回变长 prefix？如何保证 batch training 中每个位置的计算图与逐 token decode 一致？

也就是说，原始物理 U-Net 的核心技术问题不是“能否压缩”，而是：

> 如何在不破坏 causal next-token prediction 和批训练并行性的前提下，实现一种等价于中间低分辨率表示的结构？

下面的 mask-based 方案正是为了解决这个问题。

### 3.3 通过 attention mask 实现 U-Net 式多尺度结构

进一步讨论后，一个更可实现的版本是：不在训练时真的改变 hidden states 的 sequence length，而是通过不同层的 attention mask 设计，强迫模型形成多尺度的信息访问结构。

也就是说，所有层在训练时仍然保持：

$$
H_\ell \in \mathbb{R}^{N \times d}
$$

但不同层使用不同稀疏度的 causal attention mask。

在普通 causal attention 中，第 $i$ 个 token 可以看到所有历史位置：

$$
\{1,2,3,\ldots,i\}
$$

在 stride-$s$ 的稀疏层中，第 $i$ 个 token 只允许看到：

$$
\{s,2s,3s,\ldots,\lfloor i/s \rfloor s\} \cup \{i\}
$$

例如 stride 为 4 时，第 $i$ 个 token 只能看到：

$$
\{4,8,12,\ldots,i\}
$$

stride 为 16 时，第 $i$ 个 token 只能看到：

$$
\{16,32,48,\ldots,i\}
$$

这里的 $i$ 表示当前 token 自己。这个 self position 保证当前 token 在该层仍然能完成自己的前向计算；但如果 $i$ 不是该层的 anchor position，例如 $i$ 不是 4 的倍数或 16 的倍数，那么它的 K/V 在推理结束后不需要长期缓存，因为未来 token 的 mask 不会再访问它。

因此，U-Net 的“降采样/升采样”不再通过物理改变 tensor shape 实现，而是通过 attention mask 的稀疏度调度实现。

一个 15 层模型可以采用如下 U-shaped mask schedule：

$$
\begin{aligned}
&\text{layers } 1-3: && \text{stride } 1 \\
&\text{layers } 4-6: && \text{stride } 4 \\
&\text{layers } 7-9: && \text{stride } 16 \\
&\text{layers } 10-12: && \text{stride } 4 \\
&\text{layers } 13-15: && \text{stride } 1
\end{aligned}
$$

这对应原始 U-Net 直觉中的：

$$
N \rightarrow N/4 \rightarrow N/16 \rightarrow N/4 \rightarrow N
$$

但实际训练时每层仍然是长度 $N$。变化的是：中间层的 attention matrix 被强制稀疏化，只能访问更粗粒度的 anchor tokens。

这个设计可以解释为：

- stride-1 层保留完整 token-level 访问能力；
- stride-4 层强迫模型把局部 4-token 范围的信息逐渐 fuse 到 anchor positions；
- stride-16 层只访问更稀疏的 anchor positions，从而承担更高层、更低分辨率的语义整合；
- 后续 stride-4 和 stride-1 层恢复更细粒度的 token-level 表达能力。

这个方案解决了直接迁移视觉 U-Net 时遇到的几个核心技术问题：

- **避免动态 upsampling**：模型不需要从 100 个 bottleneck tokens 物理生成 10,000 个 token states。所有层始终有 $N$ 个位置，因此输出长度天然等于输入 prefix 长度。所谓升维，只是 attention mask 从 stride-16 放宽回 stride-4，再放宽回 stride-1。
- **天然适配 batch training**：训练时每个样本仍然是标准的长度 $N$ causal LM 训练。区别只是不同层使用不同 attention mask。模型仍然可以一次性并行计算所有位置的 next-token loss，不需要为每个 prefix 构造不同长度的 partial bottleneck，也不需要处理动态 upsample 出几个 token 的问题。
- **不破坏残差连接**：因为所有层的 hidden states 都是 $N \times d$，所以标准 Transformer residual 可以照常使用：

$$
H_{\ell+1} = H_\ell + \operatorname{Block}_\ell(H_\ell)
$$

- **保留 long skip 的实验自由度**：也可以选择额外加入类似 U-Net 的 long skip connection，例如 encoder 侧 stride-1 或 stride-4 层连到 decoder 侧相同 stride 的层。由于 shape 始终相同，这些 residual/skip 设计都变成可做 ablation 的自由度，而不是结构障碍。
- **直接支持推理阶段的 KV cache 压缩**：以 stride-4 层为例，未来 token 只会访问 $4,8,12,16,\ldots$，因此该层中非 anchor positions 的 KV 在当前 token 前向计算结束后可以直接丢弃。类似地，stride-16 层只需要长期缓存 $16,32,48,\ldots$。

这意味着训练阶段可以保留完整序列并使用稀疏 mask；推理阶段则根据 mask 规则只缓存未来会被访问的 anchor KV。

例如输入长度为 40、需要预测第 41 个 token 时：

$$
\begin{aligned}
&\text{stride-1 layers:} && \{1,2,\ldots,40\} \\
&\text{stride-4 layers:} && \{4,8,12,\ldots,40\} \cup \{40\} \\
&\text{stride-16 layers:} && \{16,32\} \cup \{40\}
\end{aligned}
$$

其中 40 是当前 frontier token。若某一层的 stride 为 16，40 可以参与当前预测，但它不是长期 anchor；未来如果 mask 不再访问 40，则该层的 40 号 KV 可以被丢弃。等生成到 48 时，48 才成为 stride-16 的长期 anchor。

因此，这个方案的核心可以概括为：

> 训练时通过 U-shaped sparse causal attention mask 强迫模型在中间层只访问粗粒度 anchor tokens；推理时利用同一 mask 规则，只保留未来会被访问的 anchor KV，从而实现中间层 KV cache 压缩。

这不是传统意义上的物理 U-Net，而是一个 mask-based U-Net Transformer，或者说 U-shaped sparse attention schedule Transformer。

## 4. 尚存技术问题

经过 3.3 的转向后，原始物理 U-Net 中最麻烦的动态 upsampling、变长 bottleneck、batch training 不一致等问题，被 attention mask 方案解决。剩下值得单独实验的问题主要是 residual 和 long skip 的设计。

### 4.1 Residual 和 long skip 的 ablation

在 mask-based 方案中，所有层长度都是 $N$，所以标准 Transformer residual 一定可以照常使用。

但是否加入类似 U-Net 的长 skip connection 仍然值得实验：

- **standard residual only**：只使用普通逐层 residual；
- **gated long skip**：从 encoder 侧相同 stride 的层连到 decoder 侧；
- **concat/projection skip**：把浅层表示 concat 或 project 后融合；
- **no long skip vs. long skip**：观察模型是否绕过中间稀疏层。

这里的关键风险是：长 skip 可能帮助保留 token 细节，也可能让模型不再依赖中间层 anchor hierarchy。因此它应该作为 ablation，而不是第一版架构的必要条件。
