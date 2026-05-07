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

## 4. 尚存技术问题

目前这个 idea 需要区分两类问题。

U-Net 迁移到 Transformer 的核心挑战可以概括为：

> U-Net 在视觉任务中处理的是固定长度、非因果的全局到全局映射，例如用当前 256 个像素预测下一步去噪后的 256 个像素；而语言模型处理的是变长、因果的 prefix 到 next-token 映射，例如用当前 256 个 token 预测第 257 个 token，因此 token 间因果约束和序列长度动态增长使得视觉 U-Net 的下采样/上采样逻辑无法直接迁移到 Transformer。

第一类问题包括压缩比例、局部残差、以及 downsampling 是否必须严格按语义边界切分。这些问题重要，但可以先用一个固定设计绕开，不必一开始追求最优。

第二类问题才是当前最关键的：这个 U-Net 形态如何适配 predict-next-token 的训练范式，并保证训练和推理行为一致。尤其是 causal constraint、padding/boundary、以及 decode 阶段的增量更新问题。

### 4.1 压缩与残差

#### 4.1.1 压缩比例
压缩比例可以先不做动态学习，而采用一个固定 schedule。

例如，一个 15 层模型可以设计为：

$$
\begin{aligned}
&\text{layers } 1-3: && N \\
&\text{layers } 4-6: && N/4 \\
&\text{layers } 7-9: && N/16 \\
&\text{layers } 10-12: && N/4 \\
&\text{layers } 13-15: && N
\end{aligned}
$$

也可以更简单地描述为：每 3 层 downsample 一次，每次 sequence length 变为原来的 $1/4$；到第 7-9 层达到最短 bottleneck；然后从第 10 层开始每 3 层 upsample 一次，逐步回到 token 长度。

这样做的好处是先把问题离散化：不纠结不同数据是否需要不同压缩率，而是先验证“中间层 sequence length 大幅缩短后，模型是否仍能完成 next-token prediction 和长上下文任务”。

固定压缩比例不一定是最终方案。它只是第一版实验设计。只要固定 schedule 能跑通，后面再讨论动态压缩、内容自适应压缩、或者任务相关压缩才有意义。

#### 4.1.2不强求显式语义切分

一个容易陷入的误区是：既然我们说中间层代表高层语义，那么 downsampling 就必须严格按照实体、事件、段落、函数、论证结构等语义边界切分。

这未必必要。

语言不像图像那样具有强局部连续性，但也不是完全离散的随机序列。局部 token window 往往仍然包含较强的短语、句法、段落和代码块结构。因此，第一版 downsampling 可以先使用相对简单的局部聚合，例如每 4 个 token fuse 成 1 个 latent token。

如果远处有相关语义，模型仍然可以通过 attention 去访问远处位置。也就是说，我们不必要求“同一个语义对象必须在 downsampling 时被放进同一个窗口”。downsampling 的作用不是完成所有 semantic grounding，而是降低中间层的序列分辨率，让后续 attention 在更短的 latent sequence 上完成跨位置的信息整合。

因此，第一版可以采用更弱的假设：

> 局部 fuse 不需要精确对齐完整语义对象；它只需要提供一个较粗粒度的局部表示，后续层可以通过 attention 继续组合远程相关信息。

这样可以避免一开始就把问题变成“如何发现完美语义边界”。真正需要检验的是：即使用简单局部 fuse，模型是否会在训练中自动学出足够有用的中间表示。

#### 4.1.3 残差传播

残差传播也可以先采用一个固定、对称的 U-Net 设计。

相邻 3 层之间正常传递 residual；encoder 侧的高分辨率表示通过 skip connection 传给 decoder 侧对应分辨率。例如：

$$
\text{layer } 3 \rightarrow \text{layer } 13 \quad \text{layer } 6 \rightarrow \text{layer } 10
$$

也就是说，第 3 层输出的 token-level residual 可以传递到第 13 层，用于恢复 token 级别细节。类似地，中间分辨率的 residual 也可以在 U-Net 的对称位置相连。

这个设计的直觉是：

- bottleneck 不负责无损保存所有局部细节；
- skip path 负责把低层精确信息送到上采样阶段；
- 中间短序列负责全局语义整合和长程依赖；
- 每个尺度内部保留 3 层连续计算，避免每一层都改变 sequence length 导致训练过于不稳定。

这里仍然有一个系统问题需要后续计算：skip path 本身会不会重新引入大量 token-level memory。如果 skip residual 只在一次 forward 内使用，而不作为 decode 阶段每层都要长期保存和反复访问的 KV cache，那么它的系统代价和标准 Transformer 的 full-layer KV cache 不同。这个区别需要在实现里明确。

### 4.2 核心问题：如何适配现有范式

#### 4.2.1 如何适配 predict-next-token 训练

U-Net 在图像任务里通常能看到完整输入。但语言模型自回归生成时不能泄露未来。

这个问题比“如何语义切分”更关键。因为 U-Net Transformer 仍然要使用标准语言模型训练目标：

$$
\mathcal{L}
= - \sum_t \log p(x_{t+1} \mid x_{\le t})
$$

因此，第 $t$ 个位置的预测只能依赖 $x_{\le t}$，不能因为 fuse/downsample/upsample 操作看到未来 token。

如果每 4 个 token fuse 一次，一个直接问题是：训练时第 1-4 个 token 可以形成一个 latent token，但预测第 2 个 token 时不能使用第 3、4 个 token 的信息。也就是说，naive non-causal pooling 会破坏 next-token prediction。

需要澄清：

- fuse 操作是否必须是 causal fuse；
- 一个 coarse token 在时间上代表哪个 causal frontier；
- 第 $t$ 个 fine token 能访问哪些 coarse tokens；
- upsampling 后的 token 表示如何保持 causal mask；
- 训练时 chunk 内未来 token 是否会泄露给当前位置。

这本质上是在问：U-Net 结构如何在不破坏自回归因果性的前提下，嵌入 decoder-only language model。

#### 4.2.2 训练和推理行为如何保持一致

另一个关键问题是训练和推理的一致性。

训练时通常一次给定完整序列，模型可以并行计算所有位置。但推理时是逐 token 生成：

$$
x_1, x_2, \ldots, x_t
\rightarrow
x_{t+1}
$$

如果 downsampling 规则是“每 4 个 token fuse 一次”，那么推理时会出现边界状态：

- 当前已经生成 1 个 token，还不够组成一个完整 4-token group；
- 当前已经生成 2 个 token，group 仍然不完整；
- 当前已经生成 3 个 token，还差一个 token 才能 fuse；
- 当前生成到第 4 个 token，才形成一个完整 coarse token。

训练时如果总是使用完整 4-token group，而推理时最后一个 group 经常是不完整的，就会产生 train-test mismatch。

这也是 padding 问题的本质：一条数据末尾如果空出了 1-3 个位置，到底应该如何 fuse？这些 padding 在训练中是否参与形成 coarse token？如果训练时用 padding 补齐，而推理时没有真实未来 token，也会造成行为不一致。

需要解决：

- 不完整 group 如何表示；
- padding 是否参与 downsampling；
- coarse token 是否在 group 未满时就产生一个 partial state；
- partial state 在新 token 到来后如何更新；
- 训练时是否要模拟推理中的 partial group 状态；
- decode 阶段是否需要增量维护每个尺度的 latent cache。

一种可能方向是把 fuse 设计成 causal streaming operator，而不是静态 pooling。也就是说，每个尺度维护自己的增量状态；新 token 到来后，只更新受影响的局部 latent units，而不是重算整个 U-Net。

但这会带来新的问题：如果 coarse unit 可以随着第 1、2、3、4 个 token 逐步更新，那么第 1 个 token 的预测阶段看到的是 partial coarse unit，第 4 个 token 之后看到的是 completed coarse unit。训练时也必须复现这种状态演化，否则推理和训练仍然不一致。

所以当前最核心的技术问题可以表述为：

> 如何定义一个 causal, streaming-compatible 的 downsample/upsample 机制，使得模型既能用标准 next-token prediction 训练，又能在 decode 阶段增量更新，并且训练时看到的计算图与推理时一致？
