---
anchor_id: 15_08_01_layerwise_long_range_compositional_innovation
status: AWAITING_HUMAN_BLOCK_B_RECONFIRMATION
canonical_companion: 15_08_01_layerwise_long_range_compositional_innovation_anchor.md
thinking_card: 15_08_01_layerwise_long_range_compositional_innovation_thinking_card_cn.md
parent_anchor: ../15_08_target_conditioned_layer_innovation_anchor_cn.md
parent_line: 15_spectral_representation_and_functional_routing
execution_authority: human_approved_2026_08_11
created: 2026-08-11
updated: 2026-08-11
---

# A15_08_01：逐层长程新增可访问性与表征秩

研究者判断见 [Thinking Card](15_08_01_layerwise_long_range_compositional_innovation_thinking_card_cn.md)。父定义见 [A15_08 目标条件层创新](../15_08_target_conditioned_layer_innovation_anchor_cn.md)。英文正式版见 [Anchor](15_08_01_layerwise_long_range_compositional_innovation_anchor.md)。执行合同见[已批准 Protocol](../../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/protocol.md)。

研究者已经确认研究方向、attention-only 表征位置、完整逐层增益曲线、表征秩而非任务秩、暂不进入 Router，以及 TRAIN-only、逐层 trace-normalized、三十六层等权公共基。用户指定的 Reserved 镜像与当前容器环境已经通过验证。精确 tokenizer 预检发现原始长实体字符串无法装入注册 far slot 之前；候选修订 A4 只把实体表面编码改为紧凑且全局不重合的标识，并通过完整 320-world 预检，但 Block B 仍需研究者重新确认。此前任务均不具证据资格，当前没有科学结果。

## 1. Problem Definition

A15_08_E04 已在第 25 个 block 的一个受控两跳任务上建立局部事实：不能由旧状态线性预测的归一化 attention 更新，在目标独立同预算对照之外增加了留出目标可访问性。E04 没有回答这种增益如何随深度变化，也没有回答与长程必要信息对应的更新表征是否具有低表征秩。

本 Anchor 的唯一主决策问题是：

> 在冻结 Qwen3-8B 和匹配的一跳/两跳 × 近程/远程关系任务上，扣除一般位置移动和一般两跳难度后，深层 attention 写入是否比浅层增加更多与远程必要桥接事实有关的留出目标可访问性？

本轮同时注册一个不改变主 verdict 的表征问题：该长程交互更新在 36 个 block 上的完整协方差谱、有效秩和公共数据基分布如何变化？这里的“表征秩”是更新激活自身的协方差秩，不是八分类目标天然不超过七的任务读出秩。

术语定义如下：

- **源实体（source entity）**：查询起点；**桥接实体（bridge entity）**：连接源实体和最终答案的中间实体。
- **桥接事实（bridge fact）**：源实体到桥接实体的事实；两跳查询必须使用，一跳查询不需要使用。
- **终点事实（terminal fact）**：桥接实体到最终答案的事实。
- **虚构关系世界（world/episode）**：一套随机关系映射、全部目标和四个匹配条件；它是统计重采样的独立单位。
- **决策位置（decision token）**：模型尚未输出答案时的最终查询 token。
- **新增可访问性（added accessibility）**：在冻结旧状态读出器后，加入当前层 residual update 所带来的留出交叉熵下降；它不是 Shannon 信息创造。
- **有效秩（effective rank）**：根据更新协方差的完整特征值分布估计其能量实际分布在多少维；它不能证明这些维度具有目标功能。

36 个 block 全部逐层测量并绘制趋势。先前十个相邻层对只是在有限测量预算下采样局部斜率；完整曲线成为必需交付后，它们不再定义正式采样。相邻层差分只作描述，不是独立重复，也不把跨层向量相减定义为新知识。

## 2. Physical Priors

1. **同一层的匹配 MLP-input 坐标允许精确相减。**在同一 token、block 和 post-attention RMSNorm 下，无 attention 写入与有 attention 写入的反事实/实际 MLP 输入共享同一隐藏坐标。token、hook 或 normalization 不匹配时，该先验失效。
2. **新增可访问性与谱能量是两个证据层。**留出 ridge 增益回答“该层新增了多少可读目标信息”；协方差谱回答“对应更新激活的能量如何分布”。谱集中不能替代增益，增益为正也不自动证明低表征秩。
3. **必要性 × 距离的差中差隔离长程机制。**移动桥接事实会改变一跳和两跳的物理位置，但只有两跳需要使用它；因此两种远近差再相减，可以去掉可加的一般位置和一般 hop 效应。

## 3. Falsifiable Hypotheses

**主 H2-LC——深度选择性的长程新增可访问性。**在冻结模型具备任务能力且桥接依赖通过反事实检查时，远程必要桥接对应的 residual-update gain 交互在深层为正，并且深层整体大于浅层；远程两跳的绝对增益也必须增加，避免仅由 control 条件恶化制造正交互。

**最强竞争解释 R1——一般位置检索 + 一般两跳难度。**距离对一跳和两跳产生相似影响，两跳仅增加一个与距离无关的难度。R1 允许 distance 和 hop 主效应，却不预测随深度增强的必要性 × 距离交互。

其他具名竞争解释是：旧信息冗余重编码、困难条件有更大旧状态交叉熵余量、浅层已经具有该交互、中层局部峰值被误写成深层规律、模板或 code 捷径，以及冻结模型不能可靠完成远程两跳。

**次要 H1-REP——长程交互更新的表征谱紧致。**研究者要求报告每层完整谱、熵有效秩和 80% 方差秩，并在一个冻结公共数据基中比较跨层能量曲线。当前尚未批准把何种绝对 effective-rank 阈值称为“低秩 Pass”，所以本实验描述并比较 H1-REP，不给未经注册的 Pass/Fail。任何 $2\ldots7$ 目标读出秩、广义特征方向或 Router 均不属于本轮。

主 falsifier 是：所有有效性 guards 通过时，深层减浅层的注册增益交互 $T_{depth}$ 的 95% 上界不大于零。此时必须拒绝当前 deep-emergence 表述，不能用谱图挽救。

## 4. Mathematical Model

### 4.1 匹配的关系世界

对世界 $w$，随机采样两个类型匹配的双射：

$$
\phi_w:\mathcal S\rightarrow\mathcal B,
\qquad
\psi_w:\mathcal B\rightarrow\mathcal Y.
$$

对源实体 $S_i$，桥接实体和最终答案为：

$$
B_i=\phi_w(S_i),
\qquad
Y_i=\psi_w(B_i).
$$

一跳查询直接给出 $B_i$，只需终点事实 $\psi(B_i)=Y_i$；两跳查询只给出 $S_i$，必须先使用桥接事实 $\phi(S_i)=B_i$。终点事实固定靠近查询，只把桥接事实与同关系类型、同 token 长度的匹配干扰事实交换到近程或远程位置。

四个条件为：一跳近程 $1N$、一跳远程 $1F$、两跳近程 $2N$、两跳远程 $2F$。同一 $(w,i)$ 的四条输入保持世界、事实多重集合、答案、模板族和总 token 长度一致。

### 4.2 层状态、原始 attention 写入与归一化更新

在答案前决策位置，令：

- $h_{\ell,w,i,c}\in\mathbb R^d$：block $\ell$ 的实际输入 residual；对 $\ell>1$，它是上一 block 的实际输出；
- $a_{\ell,w,i,c}\in\mathbb R^d$：attention 输出投影后、将要加到 residual stream 的原始写入；
- $H_{\ell,w,i,c}\in\mathbb R^d$：完整 block $\ell$ 的实际输出；
- $N_\ell$：block $\ell$ 的 post-attention RMSNorm。

定义无写入反事实和实际 MLP 输入：

$$
X_{\ell,c}=N_\ell(h_{\ell,c}),
\qquad
Z_{\ell,c}=N_\ell(h_{\ell,c}+a_{\ell,c}),
$$

以及精确归一化 attention update：

$$
U_{\ell,c}=Z_{\ell,c}-X_{\ell,c}.
$$

$a_\ell$ 已经是 residual 坐标中的加性写入，所以不定义 $a_\ell-h_\ell$；两者角色不同，该差没有层新增信息的物理意义。$a_\ell$ 只作为原始写入谱诊断保存，$U_\ell$ 是主功能对象。$H_\ell$ 的逐层状态读出只描述累计可读性，不替代层内增量。

### 4.3 条件 residual update 与逐层增益

只用 TRAIN worlds，并按完整 world 分组交叉拟合：

$$
\widehat m^U_{\ell,c}(X)
\approx\mathbb E_{lin}[U_{\ell,c}\mid X_{\ell,c}],
$$

$$
R_{U,\ell,c}
=U_{\ell,c}-\widehat m^U_{\ell,c}(X_{\ell,c}).
$$

$R_U$ 是注册线性预测器无法从当前层旧状态恢复的计算更新部分，不是统计独立量或知识矩阵。

在 TRAIN 上从 $X_{\ell,c}$ 拟合基础 ridge 读出器 $b_{\ell,c}$，从 $R_{U,\ell,c}$ 拟合同预算附加修正器 $q_{\ell,c}$；DEVELOPMENT 只选正则，CONFIRMATION 只评价冻结对象。对一条 confirmation 样本：

$$
g_{\ell,c}
=CE\!\left(b_{\ell,c}(X_{\ell,c})\right)
-CE\!\left(b_{\ell,c}(X_{\ell,c})
+q_{\ell,c}(R_{U,\ell,c})\right).
$$

按 world 等权的均值为：

$$
G_{\ell,c}=\mathbb E_{conf}[g_{\ell,c}]
\quad\text{nats/example}.
$$

每层必要性 × 距离交互为：

$$
I_\ell
=(G_{\ell,2F}-G_{\ell,2N})
-(G_{\ell,1F}-G_{\ell,1N}).
$$

AI_PROPOSAL：将 36 层预先等分为 early $\mathcal L_E=\{1,\ldots,12\}$、middle $\mathcal L_M=\{13,\ldots,24\}$ 和 deep $\mathcal L_D=\{25,\ldots,36\}$。唯一主指标为：

$$
T_{depth}
=\operatorname{median}_{\ell\in\mathcal L_D}I_\ell
-\operatorname{median}_{\ell\in\mathcal L_E}I_\ell.
$$

完整 $G_{\ell,c}$ 与 $I_\ell$ 折线承担趋势解释。相邻差分 $I_{\ell+1}-I_\ell$ 只描述局部斜率，不被当作独立检验。

### 4.4 表征秩与跨层公共数据基

在同一世界和目标内构造长程必要性交互更新：

$$
D_{\ell,w,i}
=(R_{U,\ell,2F}-R_{U,\ell,2N})
-(R_{U,\ell,1F}-R_{U,\ell,1N}).
$$

先在每个 world 内对八个目标中心化，得到 $\widetilde D_\ell$，再计算 TRAIN covariance：

$$
\Sigma_{D,\ell}^{tr}
=\frac{1}{n-1}\widetilde D_{\ell,tr}^{\top}\widetilde D_{\ell,tr}.
$$

设其非负特征值为 $\mu_{\ell,1}\ge\mu_{\ell,2}\ge\cdots$，并令 $p_{\ell,j}=\mu_{\ell,j}/\sum_k\mu_{\ell,k}$。表征秩报告：

$$
r_{eff,\ell}
=\exp\!\left(-\sum_jp_{\ell,j}\log p_{\ell,j}\right),
$$

$$
r_{80,\ell}^{var}
=\min\left\{r:
\frac{\sum_{j=1}^{r}\mu_{\ell,j}}
{\sum_j\mu_{\ell,j}}\ge0.8
\right\}.
$$

它们衡量更新能量的表征秩，不使用目标标签，不受八分类任务秩七的定义限制。还必须报告相对样本可识别最大秩的归一化值以及 TRAIN/DEVELOPMENT/CONFIRMATION 的复现情况，避免把有限样本秩上限误写成模型低秩。

为在同一方向坐标中比较 36 层，AI_PROPOSAL 使用每层等权、总方差归一化的 TRAIN pooled covariance：

$$
\Sigma_{common}
=\frac1{36}\sum_{\ell=1}^{36}
\frac{\Sigma_{D,\ell}^{tr}}
{\operatorname{tr}(\Sigma_{D,\ell}^{tr})},
$$

$$
\Sigma_{common}
=V_{common}\Lambda_{common}V_{common}^{\top}.
$$

该基由跨层候选长程交互更新产生，不使用参数矩阵和 confirmation。trace normalization 防止高总能量层垄断公共方向；等层权使深浅层对基的定义对称。若任一层 trace 数值近零，该层谱为无效而不是强行归一化。

在冻结公共基中，每层第 $k$ 个公共方向的归一化能量为：

$$
e_{\ell,k}
=\frac{v_k^{\top}\Sigma_{D,\ell}v_k}
{\operatorname{tr}(\Sigma_{D,\ell})},
\qquad
F_\ell(r)=\sum_{k=1}^{r}e_{\ell,k}.
$$

$F_\ell(r)$ 是公共基中的累计候选更新能量，不是按方向分解的目标可访问量。可访问性只由 $G$、$I$ 和 $T_{depth}$ 承担。

## 5. Computational Realization

### 5.1 数据、任务与监督边界

AI_PROPOSAL：冻结 Qwen3-8B 和 tokenizer；使用八个平衡 terminal code；生成 128/64/128 个完全不重叠的 TRAIN/DEVELOPMENT/CONFIRMATION worlds。每个 world 含八个目标和四个匹配条件。源实体、桥接实体、映射、模板实例和完整文本在 split 间不重叠；共享答案字母表是有意重合。

每条输入长度、终点事实位置和匹配干扰事实严格控制。模型必须通过远程两跳 restricted-choice 能力和 bridge-swap 反事实 guard，否则表征 verdict 为 Insufficient。目标 code 在模型作答前不作为输入捷径出现。

ridge 的学习任务是：使用 $X_{\ell,c}$ 产生基础八分类 logits，并使用 $R_{U,\ell,c}$ 产生加性 logits 修正。所有拟合只使用 TRAIN，正则只由 DEVELOPMENT 选择，CONFIRMATION 标签不能选择层、图范围、ridge、公共基或任何谱阈值。

### 5.2 提取身份和全层覆盖

一次冻结模型前向必须为 36 个 block 保存同一个决策位置的 $h_\ell$、$a_\ell$、$X_\ell$、$Z_\ell$、$U_\ell$ 和完整 block 输出 $H_\ell$。每层验证 $Z_\ell-X_\ell-U_\ell=0$、attention output-projection identity、token identity 和 replay identity。

不计算 $a_\ell-h_\ell$、$Z_b-X_a$ 或两个层各自 PCA rank 的相减。原始 $a_\ell$ 谱、完整状态 $H_\ell$ 的读出 CE 和相邻层斜率都是辅助轨迹；只有 $R_U$ 的条件增益进入 H2-LC。

### 5.3 冻结公共基与表征谱

$D_\ell$、每层 covariance、$r_{eff,\ell}$、$r_{80,\ell}^{var}$ 和 $V_{common}$ 都只用 TRAIN 表征冻结。DEVELOPMENT 与 CONFIRMATION 只投影到这个基，并分别报告同一指标；不能根据 confirmation 曲线重排公共 rank、删层、换 normalization 或选择更好看的谱定义。

当前设计不构造目标条件矩阵、广义特征方向或任务读出 rank，不投影到逐层 MLP 参数 eigensystem，也不训练 Router。

### 5.4 两张必需主图

| 图 | 必须回答的问题 | 坐标与聚合 | 允许结论 | 限制 |
| --- | --- | --- | --- | --- |
| 逐层新增增益图 | $G_{\ell,c}$ 和必要性×距离交互是否随深度变化？ | 横轴 block 1--36；纵轴 nats/example；四个条件和 $I_\ell$ 分面；world 等权、paired interval | 描述完整层间趋势并承担 H2-LC 直接证据 | 不能定位向量方向或证明低秩 |
| 公共基表征谱图 | 长程交互更新的表征秩和公共方向能量如何随层变化？ | 一面为 block 对 $r_{eff}$/$r_{80}^{var}$；一面为 common rank 对 $F_\ell(r)$，36 条曲线按深度由浅到深着色 | 描述表征秩轨迹和同一数据基中的谱重排 | 不能把谱能量称为目标可访问量或 Router 功能 |

相邻层差分可以叠加为弱线或单独表格，但不产生第三个主 verdict。两张图均须在 Protocol 冻结轴、单位、曲线身份、区间和允许结论。

## 6. Minimal Falsification Tests

### 6.1 H2-LC 正式 verdict

在能力、桥接依赖、身份、数据和记录 guards 全部有效时：

**Pass** 要求 $T_{depth}$、深层交互中位数、远程两跳绝对深浅增益、旧状态余量匹配后的 $T_{depth}$，以及相对目标独立同预算 mismatch-bank q95 的对比，其 paired 95% 下界均严格大于零。

**Fail** 要求 guards 有效，且 $T_{depth}$ 的 95% 上界不大于零，或具名绝对/余量/容量条款精确否定预期解释。Fail 必须映射到一般距离、一般两跳、浅层已可用、control 恶化、余量或同预算容量等具名 rival。

**Insufficient** 适用于任一决定性区间跨零、能力或 bridge-swap guard 失败、confirmation 泄漏、缺失完整 world-level 数组，或任何层 hook 身份不一致。

middle 局部峰值、谱低秩或某个相邻层大跳跃都不能挽救全局 deep-vs-early Fail。

### 6.2 表征秩报告规则

表征秩必须在 36 层完整报告，不能只展示低秩层或 best split。以下模式只产生有边界的描述：

| 观察 | 允许更新 | 不能声称 |
| --- | --- | --- |
| $r_{eff}$ 和 $r_{80}^{var}$ 在各 split 稳定且相对可识别秩较小 | 长程交互更新的能量在该数据对象上谱集中 | 目标信息低秩、模型原生使用或 Router 可读出 |
| effective rank 随深度下降 | 深层候选更新几何更紧致 | 深层新增可访问性更大；仍需 $T_{depth}$ |
| effective rank 随深度上升或谱变宽 | 深层候选更新使用更分散的表示能量 | H2-LC 必然失败或信息不存在 |
| TRAIN 紧致但 DEV/CONF 不复现 | 当前低秩几何不稳定 | 不能给 H1-REP 正结论 |
| $D_\ell$ trace 近零 | 该层没有可解释的交互更新能量 | 不能把归一化后噪声曲线解释为低秩 |

因为绝对低秩阈值尚未人工批准，本 Anchor 不注册 H1-REP Pass/Fail。Protocol 只能如实报告这些量，不得事后从曲线选择阈值。

## 7. Current Evidence

1. 学长的[方差区间报告](../../../../../../../daily_research_reports/0810/docs/DAILY_SUMMARY_ADVISOR_VARIANCE_INTERVAL_20260810.md)显示，variance-growth 选择的 160 维区间没有稳定超过同维随机方向。它关闭了“参数方向中的方差增长即可定义新增信息”，并促使本 Anchor 改用留出增益和激活数据公共基。
2. [A15_02_07 TAX](../../../../experiments/A15/15_02_layerwise_representation_spectral_atlas/A15_02_07_E01_matched_taxonomy_full/summary.md)观察到 update-only readability，却没有旧状态之外的 conditional novelty。它证明非零 update 谱或单独可读不能替代条件增益；TAX 没有必要桥接和距离干预。
3. [A15_08_E04](../../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/summary.md)在 block 25 得到合格的完整 $R_U$ 新增可访问性，$G_{true}=0.767207$ nats/example，95% 区间 $[0.751296,0.785146]$。它验证局部 ridge 度量，但不验证完整深度趋势、长程特异机制或表征秩规律。
4. 当前实现通过 37 项模型无关测试；候选紧凑编码也已用精确 tokenizer 完成全部 320 worlds、10,240 records 的构造预检。在 A4 获得确认前不得提交新的合格任务，当前不存在 36 层科学数组或 verdict。

## 8. Claim Boundary And Next Decision

若 H2-LC Pass，最强允许结论是：

> 对一个冻结 Qwen3-8B、一个匹配合成关系族、一个答案前决策位置和注册线性读出器族，深层 attention 写入比浅层增加更多对远程必要桥接事实特异的留出目标可访问性。

表征谱可以同时说明该匹配长程交互更新在 36 层中的 effective-rank 和公共方向能量轨迹，但在没有人工批准的绝对 rank 阈值时不产生全局低秩 verdict。

即使 H2-LC Pass 且谱紧致，也不能建立：深层状态包含全部浅层信息、Shannon 信息创造、事实知识存储、任务信息或 Router 在相同低秩子空间中、MLP 原生使用、专家效用、自然语言普适性，或 Router 改善 NLL/负载均衡。

**唯一下一决策：**批准或拒绝候选数据修订 A4：把原始长实体字符串替换为六字符、全局不重合的 source/bridge 标识，同时保持注册任务结构不变。

**完成标准：**四份 rank-local identity 回执、数据/提取/选择冻结、未触碰的 CONFIRMATION 评估、五项决策区间、全部 guard、秩表、公共基曲线和两组注册主图均完整且通过谱系审核。

**恢复动作：**A4 获批后，使用指定镜像、Reserved 配额和全新 `_r2` root 提交唯一任务；获得合格证据后，再写 canonical Summary 与 Detailed。
