---
anchor_id: 15_07_layerwise_conditional_fine_discriminant_directions
status: draft_human_review_required
canonical_companion: 15_07_layerwise_conditional_fine_discriminant_directions_anchor.md
thinking_card: 15_07_layerwise_conditional_fine_discriminant_directions_thinking_card_cn.md
parent_line: 15_spectral_representation_and_functional_routing
created: 2026-08-10
updated: 2026-08-10
---

# A15_07：逐层 conditional-fine 判别方向

研究者判断见 [Thinking Card](15_07_layerwise_conditional_fine_discriminant_directions_thinking_card_cn.md)。本 Anchor 是候选研究设计，等待研究者审核；当前不存在 Protocol，也未批准任何实现或运行。

## 1. Problem Definition

A15_02 观察到语义方差随层和本地参数秩重排，但没有得到稳定的 fine-specific 后移规律。A15_04 又表明，本地后秩重排没有进入注册的跨层共享 F9--F16 broad tail。A15_05 的功能实验进一步表明，固定 covariance band 即使改变 token-to-expert 分发，也不自动改善功能目标。因此 band 位置现在不是候选语义坐标的选择规则。

本 Anchor 暂停 A15_03 的 middle-band 审核，把唯一语义对象固定为 **conditional-fine**：已知父类后，只区分该父类内部的 child。它不再比较 coarse 与 fine，也不预设信息应位于 head、middle 或 tail。

`AI_PROPOSAL` 的表征对象是冻结 Qwen3-8B 十个采样层中的实际 attention-induced MLP-input increment：

$$
x_\ell=\Delta n_\ell
=RMSNorm_\ell(h_\ell+a_\ell)-RMSNorm_\ell(h_\ell).
$$

实际计算可使用其在完整 4,096 维正交参数基中的坐标，因为完整正交变换不会改变本 Anchor 的广义判别问题。采样层固定为 1/5/9/13/17/21/25/29/33/36；第 36 层是 terminal boundary，只描述、不进入主裁定。

**唯一决策问题：**

> 在固定的 conditional-fine 数据和实际 $\Delta n_\ell$ 上，只用一组表达学习出的逐层低秩方向，能否在另一组表达上继续区分同父类 child，并同时超过同秩 Haar 随机空间和父类内标签置乱，形成一个跨表达可复现的逐层线性语义坐标证书？

这一步只决定是否存在值得进入下一次功能 admission 的候选方向。它不判断 MLP 是否使用这些方向，不判断专家收益，也不判断 Router 训练是否获益。

## 2. Physical Priors

1. **判别信号与表达噪声必须分开。**真正的 conditional-fine 方向应使不同 child 的中心分开，同时使同一 child 换模板、事实包或措辞后的波动较小。
2. **高维小样本需要显式收缩。**当前维度为 4,096，而每个 child 只有少量表达；未正则化的类内协方差必然奇异，最大广义特征值可能只是偶然的近零噪声方向。
3. **方向必须在未参与构造的表达上验证。**构造集上的特征值、分类准确率或可视化都不能证明方向复现。
4. **每层可以有自己的方向。**$S_\ell$ 与 $S_m$ 不必共享向量身份；本 Anchor 测逐层候选坐标，不把相同秩或相同 band 当作跨层同一方向。
5. **语义可分不等于功能可用。**即使一个方向稳定区分 child，也未证明它能预测哪个 expert 对 token 更有益。

## 3. Falsifiable Hypotheses

**H1 — 稳定的低秩 conditional-fine 方向。**在 TRAIN 或 DEVELOPMENT 表达上分别求出的前 $r$ 个广义判别方向，在 CONFIRMATION 表达上仍提高同父类 child 的 balanced accuracy，超过同秩 Haar 和父类内样本—child 归属置乱；两组构造表达得到的子空间重合也超过同秩随机基线。

**R1 — 表达捷径或小样本过拟合。**构造集的广义特征值或准确率很高，但留出优势不超过随机，或者交换表达半后结论反向。此时不能把方向解释为稳定细类语义。

**R2 — 没有低秩线性集中证书。**全空间正则线性读出能够区分 child，但候选低秩子空间不超过随机。这表示线性信息可能分布在更高维空间；若全空间线性能力本身也不成立，则只能判为任务能力不足。两种情况都不能推出“细类信息不存在”或“必须使用非线性 Router”。

**Pass：**至少一个预注册层的 $G_\ell$ 分组重采样 95% 同时置信下界大于零；该层的 TRAIN-built 与 DEVELOPMENT-built 结果均为正；方向子空间重合超过同秩 Haar q95；全空间能力和数据可靠性 guards 全部通过。Pass 只准入满足全部条件的具体层。

**Fail：**全空间线性能力通过，但低秩留出优势稳定不超过零，或构造/确认发生稳定反转。Fail 只关闭当前 $B/W$ 定义和候选秩范围下的低秩线性证书。

**Insufficient：**全空间能力不足、表达拆分不独立、标签或缓存不可靠、置信区间跨零、或者 $r/\alpha$ 选择发生确认集泄漏。

## 4. Mathematical Model

### 4.1 $B_\ell$ 与 $W_\ell$ 的明确对象

令 $x_{pce}^{(\ell)}\in\mathbb R^{4096}$ 表示第 $\ell$ 层、父类 $p$、child $c$、表达 $e$ 的实际 $\Delta n_\ell$。每个父类有 $C_p=8$ 个 child。只在当前构造半上定义

$$
\mu_{pc}^{(\ell)}=\frac{1}{E_{pc}}\sum_e x_{pce}^{(\ell)},
\qquad
\mu_p^{(\ell)}=\frac{1}{C_p}\sum_c\mu_{pc}^{(\ell)}.
$$

同父类 child 中心之间的协方差为

$$
B_\ell=
\frac{1}{P}\sum_{p=1}^{P}\frac{1}{C_p}
\sum_{c=1}^{C_p}
(\mu_{pc}^{(\ell)}-\mu_p^{(\ell)})
(\mu_{pc}^{(\ell)}-\mu_p^{(\ell)})^\top.
$$

同一 child 跨表达的协方差为

$$
W_\ell=
\frac{1}{P}\sum_{p=1}^{P}\frac{1}{C_p}
\sum_{c=1}^{C_p}\frac{1}{E_{pc}}
\sum_e
(x_{pce}^{(\ell)}-\mu_{pc}^{(\ell)})
(x_{pce}^{(\ell)}-\mu_{pc}^{(\ell)})^\top.
$$

$B_\ell$ 的对象是 child 身份差异，$W_\ell$ 的对象是同一 child 的表达变化。父类共同差异已被 $\mu_{pc}-\mu_p$ 消去，因此两者都不是 coarse/fine 混合量。

### 4.2 广义特征方程从哪里来

对任意方向 $v$，定义判别比

$$
J_\ell(v)=
\frac{v^\top B_\ell v}
{v^\top(W_\ell+\rho I)v}.
$$

分子是在该方向上不同 child 中心的方差，分母是同一 child 跨表达的方差加收缩项。固定分母为 1，最大化分子：

$$
\max_v v^\top B_\ell v
\quad\text{s.t.}\quad
v^\top(W_\ell+\rho I)v=1.
$$

拉格朗日函数为

$$
\mathcal L(v,\lambda)=
v^\top B_\ell v-
\lambda\left[v^\top(W_\ell+\rho I)v-1\right].
$$

令 $\nabla_v\mathcal L=0$，得到

$$
B_\ell v=\lambda(W_\ell+\rho I)v.
$$

所以它不是凭经验写下的拟合式，而是最大化“细类差异 / 表达噪声”这一 Rayleigh quotient 的一阶最优条件。对解向量有 $\lambda=J_\ell(v)$；$\lambda$ 越大，只表示构造数据上这个比值越大。

等价地，令

$$
C_\ell=(W_\ell+\rho I)^{-1/2}
B_\ell(W_\ell+\rho I)^{-1/2},
$$

先解普通特征方程 $C_\ell u=\lambda u$，再令 $v=(W_\ell+\rho I)^{-1/2}u$。这表明该方法先把类内表达波动白化，再在白化空间找 child 中心变化最大的方向。

$\rho>0$ 有两个作用：使 $W_\ell+\rho I$ 可逆；防止模型偏爱构造样本中方差偶然接近零的方向。$\rho$ 不能用确认表达调节。对于 8 个父类、每个 8 个 child，$\operatorname{rank}(B_\ell)\le 8(8-1)=56$，所以超过 56 个非零判别方向没有当前类别结构上的意义。

### 4.3 唯一主指标

令 $S_\ell^{T}(r,\alpha)$ 只用 TRAIN 构造，$r,\alpha$ 只在 TRAIN 内部的预冻结 5/5 表达交叉验证中选择；随后用相同 $r,\alpha$ 从 DEVELOPMENT 独立构造 $S_\ell^{D}$。每个构造集分别使用 $\rho=\alpha\operatorname{tr}(W_\ell)/4096$。令 $V_\ell$ 的列满足 $V_\ell^\top(W_\ell+\rho I)V_\ell=I$，分类坐标为 $z=V_\ell^\top x$。两个分类器都只使用各自构造集的父类内 child 中心，在投影空间对 CONFIRMATION 样本做同父类最近中心预测。

balanced accuracy 是先对每个 child 计算正确率，再对 child 和父类等权平均。令 $BA_\ell^{disc}$ 是 TRAIN-built 与 DEVELOPMENT-built 两个子空间在 CONFIRMATION 上 balanced accuracy 的平均。对每个层和选定秩，用完全相同的数据拆分与分类器计算 512 个同秩 Haar 空间和 512 个父类内样本—child 归属置乱的结果。定义

$$
G_\ell=BA_\ell^{disc}
-\max\left\{q_{0.95}(BA_\ell^{Haar}),
q_{0.95}(BA_\ell^{perm})\right\},
$$

令 $LCB_{0.95}^{sim}(G_\ell)$ 是对九个非终端采样层同时校正的分组重采样下界，并定义可准入层集合

$$
\mathcal L^*=\left\{\ell\in\mathcal L_{dec}:\,
LCB_{0.95}^{sim}(G_\ell)>0\right\}.
$$

逐层 $G_\ell$ 曲线是本 Anchor 的**唯一主指标**，单位是 balanced-accuracy 绝对百分点；$\mathcal L^*$ 只是由同一指标产生的决策集合。这样不会用一个跨层平均掩盖真正的 layerwise 差异。`AI_PROPOSAL`：$\mathcal L^*\ne\varnothing$ 才允许 Pass，并且只准入集合内的层。跨层中位数和 late-minus-early 变化只作解释，不能替换 $G_\ell$。

由 TRAIN、DEVELOPMENT 分别构造的广义特征向量先做 Euclidean QR 正交化，再得到同秩投影 $P_\ell^T,P_\ell^D$。其重合

$$
O_\ell=\frac{\operatorname{tr}(P_\ell^TP_\ell^D)}{r}
$$

是稳定方向 guard。它必须超过同秩独立 Haar 的 q95，但不是第二主指标。$\lambda$、构造准确率、band 能量和跨层重合都不能替换 $G_\ell$。

## 5. Computational Realization

1. 冻结 `/data/share/Qwen3-8B`、A15_02_07 的十层、tokenization、最后 `Classification:` readout 和完整 4,096 维实际 $\Delta n_\ell$ 坐标。
2. 冻结 A15_02_07 的 TAX 数据，SHA-256 为 `ce91dbbd3c5071e17beeccf0d86a280dc8a3e48e0fdbf2178868da45eea18af4`：父类固定为 mathematics、physics、chemistry、biology、computer science、economics、medicine、linguistics；每个父类 8 个 child，每个 child 10 TRAIN + 10 DEVELOPMENT + 10 CONFIRMATION 表达，共 1,920 条文本。只使用八个 complex/conditional-fine 任务；simple 条件完全排除。父类和 child 等权，chance balanced accuracy 为 $1/8=12.5\%$。
3. `AI_PROPOSAL`：TRAIN 内按冻结表达 ID 分为 5/5 两半，双向内部交叉验证选择一组共享的 $r,\alpha$；随后分别用全部 TRAIN 和全部 DEVELOPMENT 构造 $S_\ell^T,S_\ell^D$，两者都只在 CONFIRMATION 上裁定。
4. `AI_PROPOSAL`：$r\in\{1,2,4,8,16,32,56\}$；$\rho=\alpha\operatorname{tr}(W_\ell)/4096$，$\alpha\in\{10^{-4},10^{-3},10^{-2},10^{-1},1\}$。平分时优先更小的 $r$ 和更大的 $\alpha$。
5. 所有中心、$B_\ell$、$W_\ell$、$r$、$\alpha/\rho$ 和方向只从 TRAIN/DEVELOPMENT 产生；CONFIRMATION 不能决定符号、秩、正则、层、样本或图形范围。现有 CONFIRMATION 已被旧实验查看，因此这里只是新方法的 analysis-heldout split，不是全新总体确认。
6. 512 个 Haar 和 512 个归属置乱对所有候选使用冻结种子。置乱在每个父类内把单条 TRAIN/DEVELOPMENT 表达重新分配到 child bin，同时保持每个 child 样本数；它不是不会改变 $B_\ell$ 的简单标签改名。置乱对照使用与真方向相同的冻结秩并重新执行 $B/W$ 构造和 CONFIRMATION 评估。
7. 全空间能力 guard 使用同一 TRAIN-only 超参数纪律下的 4,096 维正则多类线性读出，并在 CONFIRMATION 上裁定。某层只有在 $BA_\ell^{full}$ 减去其父类内归属置乱 q95 的同时置信下界大于 0 时才通过；它回答全空间线性任务是否可做，不能替代低秩 $G_\ell$。
8. 不必显式形成 4,096×4,096 稠密逆矩阵；Protocol 应要求在样本张成空间中完成稳定的 ridge-whitened 求解，并验证残差、正交性和直接/低秩实现一致。
9. 不画或裁定 head/middle/tail。可在主 verdict 冻结后描述 $S_\ell$ 在参数秩上的能量分布，但它不能改变结论或事后选 band。
10. 分组重采样以 8 个父类为外层不确定性单位，并在每个父类内嵌套重采样表达；每次重采样必须重新构造方向和分类中心。九个裁定层使用同一次 max-statistic 同时校正。

## 6. Minimal Falsification Tests

| 结果类型 | 可检查规则 | 问题更新 |
| --- | --- | --- |
| `STABLE_LOW_RANK_LINEAR_DIRECTIONS` | 全空间能力通过；$\mathcal L^*$ 非空；集合内每层的 TRAIN-built 与 DEVELOPMENT-built 结果均为正且子空间重合超过 Haar q95；父类/表达来源分组结果无主导性反转 | 仅集合内层获得 conditional-fine 跨表达低秩候选坐标；可进入独立功能 admission，但尚不能进入 Router 训练 |
| `EXPRESSION_OR_SMALL_SAMPLE_OVERFIT` | 构造 $\lambda$ 或准确率高，但全部层的 $G_\ell$ 不超过随机，TRAIN-built 与 DEVELOPMENT-built 结果反转，或子空间重合不超过 Haar | 当前方向主要反映表达捷径、采样噪声或不稳定估计；不能按这些方向设计 Router |
| `DISTRIBUTED_LINEAR_SIGNAL` | 若干层的全空间正则线性读出稳定超过 chance/Haar，但这些层没有低秩 $G_\ell$ 优势 | 细类线性信息存在，但未集中到注册的低秩方向；下一问题应是高维/稀疏/局部判别结构，而不是 band 位置 |
| `INSUFFICIENT_TASK_CAPABILITY` | 全空间能力本身不稳定超过 chance，或模型/数据/缓存 guard 失败 | 当前实验没有资格判断低秩方向；先修复任务或表征能力，不转向非线性 Router |

**最小反例：**在构造半人为加入一个只与 child 标签共同变化、但在确认半独立翻转的模板偏置。它会产生很大的构造 $\lambda$ 和训练准确率，却使确认 $G_\ell\le0$、双向折叠反转。这个反例说明必须用跨表达确认，而不能用广义特征值本身作为结论。

## 7. Current Evidence

1. [A15_02_05](../evidence/a15_02_05/summary.md) 表明 conditional-fine residual 可复现，但在实际 $\Delta n$ 中没有得到超出 coarse 的稳定额外本地秩后移；named case 的方向也不一致。它没有求解 $B/W$ 判别方向。
2. [A15_02_07 TAX](../evidence/a15_02_07/summary.md) 在相同十层和完整谱上没有得到稳定 fine-specific rank relocation；已有线性可读量不等于低秩、跨表达方向证书。
3. [A15_04](../evidence/a15_04/summary.md) 的全局 shared F9--F16 verdict 为 Fail：本地后移没有进入一个可跨层复用的 broad tail。这是停止用 band 位置替代方向身份的直接依据。
4. [A15_05_04](../evidence/a15_05_04/summary.md) 与 [A15_05_05](../evidence/a15_05_05/summary.md) 没有准入固定 M/T/N band；改变高维邻域或分发不是功能证据。
5. 当前尚未计算 $B_\ell/W_\ell$ 广义判别方向、跨表达 $G_\ell$ 或子空间重合。A15_07 的全部结果状态均为 `NOT RUN`。

## 8. Claim Boundary And Next Decision

本 Anchor 最多证明：在一个冻结模型、一个已经检查过的 1,920 条英文 TAX bank、一个明确表征位置和注册的线性正则方法下，conditional-fine 差异能否被压缩为逐层、跨表达复现的小秩线性子空间。

即使 Pass，也不能声称这些方向是模型自然使用的功能模块、包含 attention 新增信息、跨模型共享、对应固定频谱 band、预测 expert 身份、降低 expert 冲突或改善 Router NLL。要推进 layerwise Router，下一次独立功能 admission 必须在**同一个实际 MoE 的 Router 输入**上冻结这些方向，并检验它们是否预测 expert-specific utility 或已验证的 same-expert compatibility，且超过 native-score、同秩 Haar 和 wrong-layer controls。

**唯一下一决策：**研究者确认或修改六个设计点：主表征是否为实际 $\Delta n$；conditional-fine 是否为唯一对象；TRAIN/DEVELOPMENT/CONFIRMATION 的复用边界；$G_\ell$ 是否为逐层主指标；$r/\alpha$ 的选择空间与 $\rho$ 缩放；以及 Pass 是否只开放功能 admission 而不开放 Router 训练。

**完成标准：**以上六点被明确记录，且研究者接受 A15_03 继续 parked、A15_07 不审 band、不把语义可分性升级成功能性。

**恢复动作：**确认后，用本 Anchor 写唯一一份 `DRAFT_NOT_EXECUTABLE` Protocol；完成 Protocol 审核并获得单独执行批准后，才能实施 smoke。
