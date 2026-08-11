---
card_id: 15_08_01_layerwise_long_range_compositional_innovation
owner: researcher
status: HUMAN_JUDGMENT_CONFIRMED_PROTOCOL_DRAFT_PENDING_REVIEW
parent_anchor: ../15_08_target_conditioned_layer_innovation_anchor_cn.md
created: 2026-08-11
updated: 2026-08-11
---

# Thinking Card：逐层长程组合新增可访问性与表征秩

## 研究者已确认的判断

1. 保留两个研究先验：H1 是“新增表征对象自身具有低表征秩”，H2 是“模型深层在浅层基础上增加了新的信息”。H1 的秩不是八分类任务天然不超过七的任务读出秩。
2. 当前沿路线 B 推进：先用同一度量逐层比较新增可访问性，再描述对应更新表征在一个跨层公共数据基中的谱分布。暂不训练 Router，也暂不检验目标条件方向的 $r=2\ldots7$ 读出充分性。
3. 使用同一虚构关系世界、同一事实集合和同一答案构造一跳/两跳与近程/远程的 $2\times2$ 配对。只有两跳查询必须使用被移动的桥接事实，因此差中差能够扣除一般位置移动与一般两跳难度。
4. 当前机制判断是：相对浅层，深层 attention 写入更可能增加与远程、必要桥接事实有关的组合目标可访问性。
5. 三十六个 block 必须逐层计算增益，并绘制完整深度折线。先前十个相邻层对只是在有限测量预算下采样局部斜率；当完整逐层曲线成为必需交付后，它们不再决定采样。所有相邻层差分只作曲线斜率诊断，不是独立重复。
6. 当前主表征对象是 attention 写入对本层实际 MLP 输入造成的变化。attention 输出 $a_\ell$ 已经是加到 residual stream 的原始写入，不能再用 $a_\ell-h_\ell$ 定义新增量；同时保存 $a_\ell$ 作为原始写入谱的辅助诊断。
7. 公共基必须来自跨层共享的激活坐标，而不是各层不同的参数矩阵特征 rank。表征有效秩使用完整协方差谱；不把 $2\ldots7$ 当作表征秩范围。

## 机制与可观察量

每个关系世界包含：

$$
S\xrightarrow{\phi}B\xrightarrow{\psi}Y,
$$

其中 $S$ 是源实体，$B$ 是桥接实体，$Y$ 是最终答案。桥接事实是 $\phi(S)=B$；终点事实是 $\psi(B)=Y$。一跳查询直接给出 $B$，两跳查询只给出 $S$。终点事实始终靠近查询，只把桥接事实与同类型、同长度的干扰事实交换到近程或远程位置。

在每个 block $\ell$ 的答案前决策位置，令 $h_\ell$ 为该 block 的输入 residual，$a_\ell$ 为 attention 原始写入，$N_\ell$ 为该 block 的 post-attention RMSNorm：

$$
X_\ell=N_\ell(h_\ell),\qquad
Z_\ell=N_\ell(h_\ell+a_\ell),\qquad
U_\ell=Z_\ell-X_\ell.
$$

$h_\ell$ 对 $\ell>1$ 是上一 block 的实际输出；$X_\ell$ 则是把它放入当前层 post-attention RMSNorm 后得到的无写入反事实 MLP 输入。$U_\ell$ 精确测量当前 attention 写入对本层 MLP 输入的影响，而不是完整 block 更新。

再去掉旧状态可线性预测的更新：

$$
R_{U,\ell}=U_\ell-\widehat{\mathbb E}_{lin}[U_\ell\mid X_\ell].
$$

真正承担“新增可访问性”结论的是：在全新留出世界上，$R_{U,\ell}$ 给冻结旧状态目标读出器带来的交叉熵下降。逐层增益、必要性×距离交互和深浅对比承担 H2-LC；原始写入或 residual update 的非零谱不承担该结论。

## 表征秩与公共基判断

表征秩只分析激活更新自身的协方差谱。对每个层的匹配 residual-update 交互：

$$
D_\ell=
(R_{U,\ell,2F}-R_{U,\ell,2N})
-(R_{U,\ell,1F}-R_{U,\ell,1N}),
$$

计算完整特征值谱、熵有效秩和 80% 方差秩。该分析不使用目标标签，也不受八分类任务秩上限七的定义约束。

研究者要求不同层的谱位于同一公共坐标，并已确认从 TRAIN 的 $D_\ell$ 构造每层总方差归一化 covariance，再对三十六层等权平均并求特征基。该基是“跨层长程交互更新公共数据基”，不是参数基。它用于绘制每层 normalized spectral-energy 曲线，颜色随层深加深；它不把谱能量重新命名为目标可访问量。

## 最强竞争解释

最强替代解释是可加的“一般位置检索 + 一般两跳难度”：事实变远对一跳和两跳产生相近影响，两跳只增加与距离无关的难度。因此可以出现 distance 或 hop 主效应，但不会出现随深度增强的必要性×距离交互。

其他必须隔离的解释包括：旧信息冗余重编码、困难条件拥有更大交叉熵余量、模板或位置捷径、模型不能可靠完成远程两跳，以及只有中层局部峰值却被事后写成深层规律。

对表征秩，最强限制是：低 effective rank 可能由一个高方差但与任务无关的 nuisance 方向产生；高 effective rank 也可能只是各向同性噪声。因此表征秩曲线只回答更新几何是否紧致，不证明其中的任务信息、模型原生使用或 Router 功能。

## 什么证据会改变判断

支持窄化 H2 所需的最小证据是：在全新世界、相同 ridge 预算和配对重采样下，深层必要性×距离增益交互稳定大于浅层；深层交互本身为正；远程两跳的绝对增益也增加；结果超过目标独立同预算对照并在旧状态余量匹配后保持同号。

逐层 $G_{\ell,c}$ 与 $I_\ell$ 必须完整呈现，不能只报告 early/deep 汇总。层对差分只能帮助识别局部转折。

表征秩必须报告每层完整谱、effective rank、80% 方差秩和跨 split 稳定性。研究者尚未批准一个将 H1 判为 Pass/Fail 的绝对 rank 阈值，因此首份 Protocol 不得把“曲线看起来低”写成 H1 verdict。

## 已落实的补充约束

1. 主图一绘制所有三十六层的四个 $G_{\ell,c}$ 和 $I_\ell$ 折线；主图二绘制每层 effective rank/80% 方差秩轨迹，以及公共基中的 normalized cumulative-energy 曲线，层越深颜色越深。
2. 先前十个层对从正式采样合同中移除。对 $\ell=1,\ldots,35$ 的相邻差分 $I_{\ell+1}-I_\ell$ 只生成描述性轨迹，不进入主 verdict，也不被当作独立样本。
3. 当前不构造目标条件广义特征方向，不比较任务读出秩，不投影到逐层参数特征基，不运行 Router。

## 已确认的公共基、Protocol 与执行

公共基采用 TRAIN-only、trace-normalized $D_\ell$ covariance 的三十六层等权 pooled eigensystem；DEVELOPMENT 和 CONFIRMATION 只投影到冻结基。该判断及精确 world 数、距离区间、能力阈值、读出/cross-fitting、对照预算、bootstrap、图面板、资源上限和一次四卡运行均已由研究者确认，详见[Protocol](../../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/protocol.md)。指定镜像/环境已通过验证，但候选紧凑实体编码 A4 等待 Block B 重新确认；尚无实验结果。

## 唯一下一决策

先批准或拒绝候选数据修订 A4；若批准，再用指定镜像、Reserved 配额和全新 `_r2` root 获得合格证据并确定注册 verdict。两张主图的职责、取消十层对正式采样、表征秩只作描述、公共基及暂不进入任务秩/Router 均保持冻结。
