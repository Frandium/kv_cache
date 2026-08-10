---
card_id: 15_07_layerwise_conditional_fine_discriminant_directions
owner: researcher
status: AI_DRAFT_AWAITING_HUMAN_CONFIRMATION
created: 2026-08-10
updated: 2026-08-10
---

# Thinking Card：逐层 conditional-fine 判别方向

## 研究者当前判断

频谱 band 的审核先暂停。A15_04 已经说明，本层参数秩上的后移不能自动解释为进入一个跨层共享的 broad tail；A15_05 也说明，固定 covariance band 改变分发并不等于获得功能收益。因此，下一步不再先问信息位于 head、middle 还是 tail，而应先问：**同一父类内部真正区分多个细类的信息，是否集中在少量、跨表达可复现、逐层分别定义的方向中。**

研究者已经明确授权把这一问题写成独立、自包含的 Anchor；A15_03 middle-band audit 暂停。这个授权不等于批准 Protocol 或运行实验。

## 候选机制

如果一个方向对细类有用，那么不同细类在该方向上的中心应相互分开，而同一细类换模板、事实表达或措辞后不应剧烈漂移。因此候选方向不应由总体方差或参数秩决定，而应最大化“类间差异 / 类内表达波动”。

对应的候选方程是

$$
B_\ell v=\lambda(W_\ell+\rho I)v,
$$

其中 $B_\ell$ 是同一父类内部不同细类中心之间的协方差，$W_\ell$ 是同一细类跨表达的协方差，$\rho I$ 是防止小样本协方差奇异并抑制伪低噪声方向的正则项。$\lambda$ 只是构造数据上的判别信噪比，不是留出准确率、信息量或 Router 收益。

## 最强竞争解释

1. **小样本过拟合：**维度为 4,096，而单个构造 split 只有 640 条表达，$W_\ell$ 的秩仍远低于维度；若没有正则化和跨表达留出，最大的 $\lambda$ 可以来自几乎零类内方差的偶然方向。
2. **表达捷径：**方向区分的是模板、词汇或事实包，而不是父类内部的细类身份。
3. **分布式或非线性编码：**细类可能线性可读但不集中在小秩空间，或者只能被非线性边界读取；低秩广义判别方向失败不能推出细类信息不存在。
4. **对象错位：**当前可复用数据对象是 Qwen3-8B 的实际 attention-induced MLP-input increment $\Delta n_\ell$。它不是 MoE 的实际 Router 输入，因此即使通过也只是候选语义坐标证书，不是 Router 功能证书。

## 什么证据会改变判断

支持候选机制所需的最小证据是：方向只用 TRAIN 或 DEVELOPMENT 表达构造，在未参与任何方向或超参数选择的 CONFIRMATION 表达上仍提高同父类细类 balanced accuracy；优势超过同秩 Haar 随机空间和父类内样本—child 归属置乱；TRAIN 与 DEVELOPMENT 分别构造的子空间重合超过同秩随机基线；这些结论在预注册层和分组重采样下稳定。

若构造集 $\lambda$ 很大而留出优势不超过随机，判断更新为“表观判别方向主要是表达/小样本过拟合”。若全空间线性读出有效而低秩方向无优势，更新为“细类线性信息存在但没有获得低秩集中证书”。若全空间线性能力本身不成立，则实验只能判为能力不足，不能判断线性或非线性结构。

## 待研究者确认的判断

以下均为 `AI_PROPOSAL`，尚未得到研究者确认：

1. 主表征使用十层冻结的 Qwen3-8B 实际 $\Delta n_\ell$，而不是 residual state、attention output 或未来 MoE 的实际 Router 输入。
2. 唯一主语义对象是 conditional-fine：在每个父类内区分八个 child；不再做 coarse/fine 主比较。
3. 使用 A15_02_07 的 1,920 条平衡 TAX 文本：每个 child 有 10 TRAIN、10 DEVELOPMENT、10 CONFIRMATION 表达；只使用 complex/conditional-fine 任务。它们已经用于旧指标，因此 CONFIRMATION 只是对新方法冻结后的分析留出，不声称为全新总体复制。
4. 候选秩 $r\in\{1,2,4,8,16,32,56\}$，无量纲正则系数 $\alpha\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$，每个构造集用 $\rho=\alpha\operatorname{tr}(W_\ell)/4096$ 得到实际收缩强度；$r,\alpha$ 仅在 TRAIN 内部选择。
5. 主指标是九个非终端采样层的留出判别优势曲线 $G_\ell$，对全部层同时校正；只准入同时置信下界大于零的具体层。第 36 层只展示，不参与主裁定。
6. 本 Anchor 到“逐层低秩语义方向证书”即停止。即使 Pass，也必须在同一个实际 MoE Router 输入和独立功能目标上另做 admission，才可设计 Router 训练。

## 唯一下一决策

研究者审核并确认或修改以上六项。只有这些对象和判据冻结后，才把同一问题写成 `DRAFT_NOT_EXECUTABLE` Protocol；Protocol 再获明确执行批准之前，不实施、不 smoke、不 full run。
