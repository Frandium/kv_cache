# MTP 如何更高效地学习长程语义变量

## 0. 执行摘要

**研究问题：** 多词元预测目标（MTP）是否比下一词元预测目标（NTP）更直接、更高效地让当前隐藏状态 $h_T$ 学会一个未来才需要的语义变量 $Z$。

**机制解释：**

1. 本研究把“长程语义”定义为：变量 $Z$ 已经出现在前缀中，最近的未来词不依赖它，直到第 $\tau$ 个未来词才必须用它决定正确内容分支。

$$
I(Z;Y_j\mid Y_{<j})=0\quad(j<\tau),
\qquad
Y_\tau=S_Z,
\qquad
I(Z;Y_\tau\mid Y_{<\tau})=H(Z).
$$

2. 当预测范围覆盖第一个需要 $Z$ 的未来位置时，低该位置损失会强迫当前隐藏状态保留 $Z$。

$$
\mathbb E[-\log q_\tau(S_Z\mid h_T)]\le\varepsilon
\quad\Longrightarrow\quad
I(h_T;Z)\ge H(Z)-\varepsilon.
$$

3. 在直接优化隐藏状态、固定语义读出方向的模型中，语义边距每一步的增长精确等于该步的语义速度。

$$
M_K(t+1)-M_K(t)=\eta G_K^{hidden}(t).
$$

4. 在 Transformer 参数训练中，目标给出的语义方向还要经过编码器参数化，并受到其他位置损失的干扰。

$$
G_\theta
=\frac1{m^2}v^\top\Theta v
-\frac1m v^\top\Theta
\left(c+\nabla_HL_{background}\right).
$$

**主要结果：**

1. 在受控长程语义构造中，只要 MTP 覆盖第一个需要 $Z$ 的未来词，低该位置损失就能保证当前 $h_T$ 含有 $Z$；NTP 在同一前缀上没有这条直接约束。
2. 在固定读出方向、直接优化隐藏状态的模型中，这条直接约束会累积为有限步语义恢复；多个未来位置是否加速由其语义方向是否对齐决定，而不是由 $K$ 变大本身决定。
3. 在受控全位置 Transformer 训练中，直接语义项产生持续的正语义速度，并可靠形成模型自身的远期预测；非直接损失虽然能让 $Z$ 被外部读出，却没有形成同样的模型自身预测，因此当前证据支持“直接路径更强”，但尚未证明普遍的参数空间或自然语言效率优势。

**证据说明：**

1. 对应结果 1：
   - 互信息定理给出低有信息位置损失到 $I(h_T;Z)$ 的表示下界。
   - 在 $\tau=3$ 的单决策前缀实验中，K1/K2 不覆盖 $S_Z$ 且保持随机恢复，K3 覆盖后恢复明显增强。
2. 对应结果 2：
   - K=2 最小模型中，NTP 的一阶语义速度为 0，MTP 的一阶语义速度严格为正。
   - 一般 K 实验中，K3 首次覆盖有信息位置后速度从 0 变为 `0.003816`；新增共享 H4 无直接增量，对齐 H4 增强到 `0.015262`，冲突 H4 削弱到 `0.000954`。
   - 新的隐藏状态定理给出精确有限步累积式；在规则单纯形输出几何下，还能给出 K=2 的显式恢复步数上界。
3. 对应结果 3：
   - 全位置训练中，直接条件的模型自身第三位置预测和最终 $Q$ 都达到 `1.0`；K2 和去掉直接项的 K3 最终 $Q$ 仍为 `0.75/0.70`，但模型自身第三位置预测为 `0`，语义边距接近 `0`。
   - 有限步审计中，直接有信息条件在早期窗口 `5/5` 个随机种子保持正语义速度；但参数空间曲率修正后的命中时间证书经常非正或极松。
   - 所有全位置条件的局部 Y1/Y2 准确率最终均为 `1.0`，直接语义恢复不是以牺牲局部预测为代价。

**结论边界：** 当前已经证明受控隐藏状态模型中的表示保证和有限步效率，并得到受控 Transformer 实验证据；尚未证明 Transformer 参数空间中的统一命中时间界、自然语言样本效率、MTP 普遍优于 NTP，或 $K$ 越大越好。

**执行动作：** 下一步只审核并正式化 A11_11 的 K=2 语义切向核传递问题；完成标准是判定参数空间语义速度能否由 $v^\top\Theta v$、背景干扰和非线性余项联合解释。

## 1. 术语解释

| 术语 | 中文含义 | 具体对象或计算方式 | 单位或公式 | 为什么影响当前判断 | 不能证明什么 |
|---|---|---|---|---|---|
| 长程语义变量 $Z$ | 前缀中已经给出、局部未来词暂时不用、较远未来词才必须使用的内容分支变量 | 当前实验中的离散分支身份 | $I(Z;Y_j\mid Y_{<j})=0$ 对 $j<\tau$，而 $Y_\tau=S_Z$ | 给出“长程语义”的可操作定义 | 自然语言全部语义 |
| 长程 | 当前状态与第一个需要 $Z$ 的目标之间存在预测距离 | 第一个有信息未来位置 $\tau$ | 未来位置偏移 | 决定 MTP 是否提前看到有用监督 | 序列必须在绝对长度上很长 |
| 语义 | 决定互斥内容分支、并改变远期正确词元的变量 | $Z\mapsto S_Z$ 一一对应 | 离散变量 | 区分内容相关信息与共享局部模式 | 真实语料中的开放式语义 |
| 多词元预测（MTP） | 同一个当前状态预测多个未来词 | 同时预测 $Y_1,\ldots,Y_K$ | $L_K=\sum_{j=1}^K\lambda_j\operatorname{CE}(q_j,Y_j)$ | 可能覆盖较远的有信息位置 | 任意增大 K 都有益 |
| 下一词元预测（NTP） | 当前状态只预测下一个词 | K=1 对照 | $L_1=\operatorname{CE}(q_1,Y_1)$ | 当前前缀上没有较远直接项 | NTP 最终学不到 $Z$ |
| 第一个有信息未来位置 $\tau$ | 第一个必须使用 $Z$ 的未来偏移 | $\tau=\min\{j:I(Z;Y_j\mid Y_{<j})>0\}$ | 未来位置 | 判断 K 是否覆盖直接语义目标 | 真实文本只有一个固定 $\tau$ |
| 直接语义监督 | 当前 $h_T$ 直接预测 $S_Z$ | $\operatorname{CE}(q_\tau(\cdot\mid h_T),S_Z)$ | 交叉熵 | 缩短当前状态到语义目标的监督路径 | 全位置训练中的唯一学习通路 |
| 读出有效语义边距 $M_K$ | 当前状态沿模型输出头可用语义方向移动的程度 | $M_K=m^{-1}\sum_zv_z^\top h_z$ | 标量 | 比外部探针更接近模型自身使用 | 下游任务收益 |
| 隐藏状态语义速度 $G_K^{hidden}$ | 一步损失梯度对语义边距的推动 | $m^{-1}\sum_zv_z^\top(-\nabla_{h_z}L_K)$ | 每步边距变化率 | 判断目标是否直接推动语义方向 | 参数空间长期收敛 |
| 保守恢复分数 $Q$ | 三种恢复检测中最弱的一项 | $Q=\min\{A_{decoder},A_{probe},C_{swap}\}$ | 0 到 1 | 防止只依赖一个外部探针 | 直接监督来源 |
| 模型自身第三位置准确率 | 模型从当前 $h_T$ 直接预测 $S_Z$ 的准确率 | $\Pr[\arg\max q_3(\cdot\mid h_T)=S_Z]$ | 0 到 1 | 区分模型自身读出与间接可读性 | 自然语言质量 |
| 语义切向核 $\Theta$ | 编码器参数能把多少语义梯度传到当前状态 | $\Theta=JJ^\top$，$J=\partial H/\partial\theta$ | 半正定矩阵 | 连接隐藏状态定理与参数训练 | 全局非线性训练行为 |

## 2. 机制解释与建模

**核心模型：**

1. 最小 K=2 数据把“下一词不需要 $Z$、第二个词需要 $Z$”隔离出来。

$$
Y_1=A_{shared},
\qquad
Y_2=S_Z.
$$

对应的 $\tau=3$ 实验把第一个有信息位置后移：

$$
Y_1=A_{shared},
\qquad
Y_2=C_{shared},
\qquad
Y_3=S_Z.
$$

2. 表示定理回答“低损失是否保证 $h_T$ 含有 $Z$”。

$$
\mathbb E[-\log q_\tau(S_Z\mid h_T)]\le\varepsilon
\quad\Longrightarrow\quad
I(h_T;Z)\ge H(Z)-\varepsilon.
$$

3. K=2 一阶定理回答“直接项是否立即推动正确语义方向”。令 $u_z$ 为 $S_Z$ 的输出向量，$\bar u=m^{-1}\sum_zu_z$，则：

$$
G_1^{margin}=0,
\qquad
G_2^{margin}
=\frac{\lambda_2}{m^2}
\sum_z\|u_z-\bar u\|^2>0.
$$

4. 一般 K 的效果由有信息位置的方向向量和决定，而不是由 K 的数值决定。

$$
v_z^{(K)}
=\sum_{j\in\mathcal I_K}
\lambda_j(u_{j,z}-\bar u_j),
\qquad
G_K=\frac1{m^2}\sum_z\|v_z^{(K)}\|^2.
$$

新增有信息位置 $r$ 的增量为：

$$
G_{K\cup r}-G_K
=\frac1{m^2}\sum_z
\left(2v_z^{(K)\top}a_{r,z}+\|a_{r,z}\|^2\right).
$$

5. 隐藏状态有限步定理回答“一阶速度能否累积”。

$$
M_K(t+1)-M_K(t)=\eta G_K^{hidden}(t).
$$

若达到阈值前 $G_K^{hidden}(t)\ge g_K>0$，则：

$$
T_\gamma(K)
\le
\left\lceil
\frac{\gamma-M_K(0)}{\eta g_K}
\right\rceil.
$$

6. 参数空间分解回答“隐藏状态优势为什么还不能直接写成 Transformer 定理”。

$$
G_\theta
=\frac1{m^2}v^\top\Theta v
-\frac1m v^\top\Theta
\left(c+\nabla_HL_{background}\right).
$$

**符号含义：**

- $Z$：决定远期内容分支的受控变量。
- $h_T$：模型读完当前决策前缀后的隐藏状态。
- $S_Z$：与 $Z$ 一一对应的未来语义词元。
- $v$：所有被覆盖有信息位置的中心化输出方向之和。
- $\Theta$：编码器参数对隐藏状态变化的局部传递矩阵。
- $c+\nabla_HL_{background}$：共享目标和其他位置损失产生的背景更新。

**机制链条：**

1. MTP 覆盖 $S_Z$ 后，当前状态的目标函数中出现分支相关梯度。
2. 该梯度首先增加读出有效语义边距，并在受控隐藏状态模型中累积为有限步恢复。
3. 在 Transformer 中，这条梯度必须经过编码器参数化；是否真正更快取决于语义切向核、背景干扰和非线性变化。

**当前问题分解：**

- 已闭合：低损失表示保证、K=2 一阶速度、一般 K 方向向量和、受控隐藏状态有限步恢复。
- 实验支持：直接语义速度在早期持续，并对应更强模型自身预测。
- 尚未闭合：参数空间统一下界与真实自然语言样本效率。
- 最危险的替代解释：MTP 虽提供了正确隐藏状态方向，但编码器参数化或其他位置损失可能抵消这条方向。

## 3. 主要结果

**一句话总结：** MTP 在受控问题中的确定优势，是把远期语义目标提前变成当前隐藏状态的直接监督，并在理想隐藏状态模型中形成可证明的有限步恢复；Transformer 实验支持这条路径更强，但参数空间普遍效率仍未闭合。

### 结果 1：覆盖第一个需要 $Z$ 的未来位置，会给当前隐藏状态一个可证明的表示保证。

**机制对应：** 对应第 2 节的长程语义定义和互信息下界。

**指标或条件定义：** $Y_\tau=S_Z$ 一一编码 $Z$，且第 $\tau$ 位置交叉熵不超过 $\varepsilon$。

**证据说明：**

- 理论上，交叉熵上界给出 $I(h_T;Z)\ge H(Z)-\varepsilon$。
- 在 $\tau=3$ 决策前缀实验中，K1/K2 最终 $Q=0.25$、达到率为 `0`；K3 在相同分支初始化下最终 $Q=0.80$、达到率为 `0.60`。

**解释：** K1/K2 的当前目标只要求预测共享未来词；K3 第一次要求当前状态区分 $S_Z$，因此低损失不能由完全不含 $Z$ 的状态实现。

**边界：** 该结论只保证受控变量被编码，不说明自然语言中的全部语义，也不单独说明训练速度。

**来源：** 11_04 anchor、A11_04 summary/detailed、当前 story 第 8.1 节。

### 结果 2：直接语义监督在受控隐藏状态模型中形成有限步效率，但 K 增大本身不是原因。

**机制对应：** 对应第 2 节的一阶速度、方向向量和与有限步累积式。

**指标或条件定义：** $G_K^{hidden}$ 测量一步梯度对语义边距的推动；$T_\gamma$ 是语义边距首次达到阈值 $\gamma$ 的步数。

**证据说明：**

- K=2 最小理论给出 $G_1^{margin}=0$、$G_2^{margin}>0$。
- A11_06 的 $\tau=3$ 一阶结果为 K1 `7.28e-12`、K2 `-1.16e-11`、K3 `0.003816`。
- A11_09 中，新增共享 H4 对一步语义速度无增量；对齐信息 H4 将速度从 `0.003816` 增强到 `0.015262`，冲突信息 H4 将其削弱到 `0.000954`。
- 隐藏状态空间中，$M_K(t+1)-M_K(t)=\eta G_K^{hidden}(t)$ 精确成立；规则单纯形 K=2 模型还给出显式有限步恢复上界。

**解释：** MTP 的效率对象是“被覆盖有信息目标的合成方向”，而不是预测位置数量。方向对齐时信号叠加，方向冲突时额外目标可以抵消已有信号。

**边界：** 显式有限步定理依赖固定输出方向和受控语义几何，不能直接替代 Transformer 参数训练定理。

**来源：** 11_06、11_08、11_09 anchors；对应 summary/detailed；当前 story 第 8.5 至 8.8 节。

### 结果 3：全位置训练支持直接语义路径更强，但外部可读性不能代替模型自身读出。

**机制对应：** 对应第 2 节的参数空间分解和背景损失项。

**指标或条件定义：** $Q$ 测量 $Z$ 是否可读；模型自身第三位置准确率和 $M_Z^{ref}$ 测量当前状态是否形成 $h_T\to S_Z$ 的直接可用读出。

**证据说明：**

- K2 与去掉直接项的 K3 最终 $Q=0.75/0.70$，但模型自身第三位置准确率均为 `0`，最终 $M_Z^{ref}=0/0.004$。
- 完整 K3 与 K2 加直接 H3 项最终 $Q=1.0$，模型自身第三位置准确率均为 `1.0`，最终 $M_Z^{ref}=2.430/3.766$。
- A11_10 有限步审计中，直接有信息条件在 `5/5` 个随机种子保持正早期速度，但参数空间命中时间证书仍过松。

![直接语义读出与外部可读性分离](figures/mtp_a11_10_direct_native_margin_split.png)

图中蓝柱是最终 $Q$，橙柱是模型自身第三位置准确率，绿线和紫线分别是最终与早期语义边距。前两个非直接条件保留非零 $Q$，但模型自身预测和语义边距接近 0；后两个直接条件三类指标同时提高。这张图说明 $Q$ 不能单独识别直接监督，不证明自然语言收益。

**解释：** 全位置训练通过共享参数可以让重新计算的 $h_T$ 对 $Z$ 可读，但只有当前决策状态上的直接 $S_Z$ 损失稳定形成模型自身的远期预测方向。

**边界：** 当前实验支持直接路径是更强机制组件，不证明间接迁移无用，也不证明 Transformer 参数空间存在统一有限步优势。

**来源：** 11_10 全位置 anchor/summary/detailed；11_10 有限步 anchor/summary/detailed。

## 4. 执行动作

**下一步动作：** 先审核 A11_11 的 K=2 语义切向核传递判定卡，审核后再写正式 anchor 和 protocol。

**目的：** 区分三种可能：目标提供了语义信号但编码器无法传递、背景损失抵消了直接信号、或线性预测被参数训练的非线性变化破坏。

**完成判据：** 固定数据、输出方向和损失权重，只改变分支敏感参数通道强度后，能够分别测出并核对：

$$
\frac1{m^2}v^\top\Theta v,
\qquad
\frac1m v^\top\Theta
\left(c+\nabla_HL_{background}\right),
\qquad
B.
$$

若三项能够解释真实参数更新后的语义边距变化，则进入全位置参数分解；若不能，则把效率结论收缩为“目标函数提供直接隐藏状态语义方向，但模型未必有效吸收”。

## 5. 证据索引

| 结果 | 证据 | 主要指标或图 | 支持什么 | 不能证明什么 | 来源 |
|---|---|---|---|---|---|
| 结果 1 | 表示定理与 $\tau=3$ 覆盖实验 | $I(h_T;Z)$ 下界、$Q$、达到率 | 覆盖有信息位置给当前状态表示约束 | 自然语言效率 | 11_04 anchor；A11_04 summary/detailed；story |
| 结果 2 | K=2/K=3 一阶定理、一般 K 方向实验、隐藏状态有限步定理 | $G_K^{hidden}$、$T_\gamma$、方向增量 | 直接语义方向产生受控有限步效率，K 大小不是原因 | Transformer 参数空间统一效率 | 11_06/11_08/11_09 anchors 与结果；story |
| 结果 3 | 全位置直接项加减与有限步审计 | $Q$、模型自身第三位置准确率、$M_Z^{ref}$、中心图 | 直接路径更强，$Q$ 不能单独证明直接监督 | 自然语言和 MoE 收益 | 两个 11_10 anchors 与结果 |

## 6. 来源索引

**Story：**

- `Projects/from-attention-to-search/main/stories/11_long_horizon_mtp_objective/story_cn.md`

**Anchor：**

- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_04_k3_first_informative_horizon_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_06_readout_effective_margin_efficiency_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_08_general_k_readout_margin_dynamics_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_08b_output_geometry_dynamics_audit_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_09_next_k_inclusion_law_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_10_all_position_indirect_transfer_dynamics_anchor.md`
- `Projects/from-attention-to-search/main/problem_anchors/11_long_horizon_mtp_objective/11_10_finite_step_semantic_efficiency_anchor.md`

**Protocol：**

- `Projects/from-attention-to-search/main/experiments/A11/A11_09_next_k_inclusion_law/protocol.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/protocol.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/protocol.md`

**Summary：**

- `Projects/from-attention-to-search/main/experiments/A11/A11_04_k3_first_informative_horizon/summary.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_06_readout_effective_margin_efficiency/summary.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_08_general_k_readout_margin_dynamics/summary.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_09_next_k_inclusion_law/summary.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/summary.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/summary.md`

**Detailed：**

- `Projects/from-attention-to-search/main/experiments/A11/A11_04_k3_first_informative_horizon/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_06_readout_effective_margin_efficiency/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_08_general_k_readout_margin_dynamics/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_09_next_k_inclusion_law/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_all_position_indirect_transfer_dynamics/detailed.md`
- `Projects/from-attention-to-search/main/experiments/A11/A11_10_finite_step_semantic_efficiency/detailed.md`

**Figures：**

- `daily_research_reports/0710/meetings/figures/mtp_a11_10_direct_native_margin_split.png`
