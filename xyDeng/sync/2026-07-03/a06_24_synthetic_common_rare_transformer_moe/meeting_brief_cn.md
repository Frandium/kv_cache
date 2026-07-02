# 会议简报：公共方向剔除不能单独产生公共/稀有特征专家分工

## 0. 执行摘要

**研究问题：** 在高频公共特征和低频稀有特征同时存在时，只剔除全局公共方向，是否足以让不同特征进入不同专家？

**机制解释：** 公共方向剔除只能削弱路由分数里的共享偏置项；它不会自动选出任务相关的隐藏状态样本池，也不会自动形成稀有特征中心，因此它更像负载修复，而不是特征分离方法。

**主要结果：** A06_24_synthetic 在无位置编码的一层 Transformer 加一层 MoE 合成任务上支持该边界：全位置公共方向剔除不能替代任务位置中心选择；训练目标学会以后，路由仍可能不是干净的特征分工。

**证据说明：** 联合分离分数同时检查“稀有特征彼此分开”和“公共特征与稀有特征分开”。Step 0 时，全位置公共方向剔除的联合分离分数为 `0.405`，任务位置残差中心和 oracle 中心为 `0.637`；训练后，全位置公共方向剔除仍只有 `0.432`，且稀有路由低尾 margin 为 `-5.427`。

1. 全位置公共方向剔除没有补上任务相关样本池选择，因此不能作为可靠的稀有特征分专家方法。
2. 所有条件都能把合成目标预测到 `1.0` 准确率，但路由分离质量差异仍然很大，因此任务准确率不是专家分工证据。
3. 在已有正确中心的条件下，路由行投影比残差输入更好地保持完整的 common/rare 分区。

**结论边界：** 当前结论只覆盖无位置编码的合成 Transformer-MoE 机制桥；不能推出真实 DCLM、语义专家、专家功能价值或所有公共方向方法都失败。

**执行动作：** 下一步最小动作是建立一个方法 anchor：优先测试“无标签任务相关状态选择器”或“路由行投影式保持更新”，完成判据必须包含联合分离分数、稀有路由 margin、位置泄漏 guard 和任务 loss guard。

## 1. 术语解释

| 术语 | 中文含义 | 具体对象或计算方式 | 单位或公式 | 为什么影响当前判断 | 不能证明什么 |
|---|---|---|---|---|---|
| 特征级专家分工（feature-level expert specialization） | 不同特征稳定进入不同专家，且这种分配不是负载假象 | 本实验先只审计 feature id 与 routed expert 的一致性和 margin | 路由指标，不是 loss | 当前问题的目标概念 | 专家功能价值或语义专家 |
| 路由器（router） | 决定 token 进入哪个专家的线性打分模块 | 对每个 expert row $w_e$ 计算分数 $z_e=w_e^\top h$，top-1 选择专家 | logit 分数 | 所有分离指标都来自 router assignment | 被选专家一定有功能价值 |
| 隐藏状态（hidden state） | Transformer 在某个 token 位置产生的向量表征 | 记为 $h_t$ | 向量 | router 只能根据 hidden state 分专家 | hidden state 中每个方向都有语义 |
| 公共特征（common feature） | 高频出现的背景性特征 | 合成任务中的 feature id 0，校准和训练中约占 `70%` | 采样概率 | 检查路由是否被高频结构吸走 | 真实语言里的公共语义 |
| 稀有特征（rare features） | 低频出现、需要彼此区分的特征 | 合成任务中的 feature ids 1--3，总计约占 `30%` | feature id | 检查稀有特征是否分别进入不同专家 | 真实稀有语义 |
| 任务位置（route position） | 真正用于路由审计和预测目标的位置 | repeated feature slot 的最后一个 token | 序列位置 | 保证审计对象是任务相关隐藏状态 | 真实语言中的任务位置发现 |
| 全位置样本池（all-position pool） | 所有 token 位置的隐藏状态集合 | 包含任务位置、背景 token、slot 内非末尾位置 | hidden-state 集合 | 容易混入非任务状态，导致聚类中心不对应特征 | 一定会失败 |
| 公共方向剔除（common subtraction） | 从路由输入里减去公共均值或公共方向 | 用校准 hidden states 估计 $c$，再用 $h-c$ 作为路由输入或聚类输入 | 向量操作 | 可削弱公共偏置和负载集中 | 自动发现任务相关特征中心 |
| 归一化互信息（normalized mutual information, NMI） | 两个离散分组的一致程度 | 比较 feature label、common/rare label、slot start 与 routed expert | `0--1` | 衡量路由是否跟目标分组一致 | 专家功能价值 |
| oracle 中心 | 带标签的正控制中心 | 直接按真实 feature id 计算隐藏状态中心 | 正控制条件 | 显示“如果中心正确，路由是否可达/可保持” | 无标签方法已经解决 |
| 稀有特征一致性（rare-feature NMI） | 稀有 feature id 与专家分配的一致程度 | 在稀有样本上计算 NMI(feature id, routed expert) | `0--1` | 判断 rare-rare separation | common-vs-rare 是否分开、专家是否有功能价值 |
| common/rare 一致性 | 公共特征和稀有特征是否被路由区分 | NMI(feature is rare, routed expert) | `0--1` | 防止稀有特征彼此分开但公共特征混入稀有专家 | 稀有特征彼此分开 |
| 联合分离分数（joint feature score） | 同时检查 rare-rare 和 common-vs-rare 分离 | `rare_feature_NMI * common_rare_NMI` | 无量纲 | 避免只看单一 NMI 后过度声称 specialization | 专家功能价值 |
| 稀有路由 margin 低尾（rare margin p05） | 最脆弱 5% 稀有样本的路由安全距离 | 匹配专家分数减最强竞争专家分数的第 5 百分位 | logit 差 | 判断稀有路由是否在稳定 basin 内 | 专家是否有用 |
| 位置泄漏指标（slot-start NMI） | 专家分配是否主要跟 slot 起点绑定 | NMI(slot_start, routed expert) | `0--1` | 排除“看位置而不是看特征”的替代解释 | 完全没有上下文长度效应 |
| 残差输入（residual input） | router 读取 $h-c$，专家仍接收原 hidden state | 训练和评估时对 router 输入做减公共操作 | 输入控制 | 检查只让 router 看 residual 是否保护稀有分离 | 完整 common/rare 分离 |
| 路由行投影（row projection） | 每步更新后去掉 router row 沿公共方向的分量 | $w_e \leftarrow w_e-\frac{w_e^\top c}{\|c\|^2}c$ | 参数更新约束 | 防止 router row 重新吸收公共方向 | 最优方法或真实 DCLM 可迁移 |

## 2. 机制解释与建模

**核心模型：**

$$
h_t = c + r_{f(t)} + n_t,\qquad z_e(t)=w_e^\top h_t.
$$

$$
\hat\mu_k(\mathcal P)=\operatorname{kmeans}\{h_t-c:t\in \mathcal P\},\qquad
\mathcal P\in\{\mathcal P_{\mathrm{all}},\mathcal P_{\mathrm{route}}\}.
$$

$$
w_e \leftarrow w_e-\frac{w_e^\top c}{\|c\|^2}c.
$$

**符号含义：**

- $h_t$：第 $t$ 个 token 的隐藏状态。
- $c$：从校准 hidden states 中估计出的公共方向或公共均值。
- $r_{f(t)}$：与 feature id 相关的残差成分。
- $n_t$：位置、背景 token、非任务状态等 nuisance 成分。
- $w_e$：第 $e$ 个专家的 router row。
- $z_e(t)$：token $t$ 选择专家 $e$ 的路由分数。
- $\mathcal P_{\mathrm{all}}$：所有 token 位置组成的样本池。
- $\mathcal P_{\mathrm{route}}$：只包含任务位置 hidden states 的样本池。
- $\hat\mu_k(\mathcal P)$：在指定样本池上得到的第 $k$ 个聚类中心。

**机制链条：**

1. 路由分数 $z_e(t)=w_e^\top c+w_e^\top r_{f(t)}+w_e^\top n_t$ 同时包含公共偏置、特征残差和非任务扰动。
2. 公共方向剔除主要削弱 $w_e^\top c$，所以它可以改变负载集中或 common/rare 二分，但它不负责选择 $\mathcal P_{\mathrm{route}}$。
3. 全位置样本池 $\mathcal P_{\mathrm{all}}$ 混入背景和非任务位置，聚类中心可能对应位置/背景/混合状态，而不是 feature center。
4. 任务位置中心或 oracle 中心直接使用更接近 $r_{f(t)}$ 的样本池，因此更容易形成稀有特征 basin。
5. 训练阶段还可能把公共方向重新写回 router row；路由行投影直接约束 $w_e$ 的公共分量，因此比单纯残差输入更适合保持完整 common/rare 分区。

**当前问题分解：**

- 需要判定的机制环节：公共方向剔除是否同时完成“公共偏置削弱”和“任务相关特征中心选择”。
- 最危险的替代解释：结果来自位置 shortcut、负载更均匀、目标预测学会、或只完成 common/rare 二分而没有 rare-rare 分离。
- 当前实验能区分：公共方向剔除、任务位置中心、oracle 中心、残差输入、路由行投影在同一无位置 synthetic surface 上的差异。
- 当前实验不能区分：真实 DCLM 中哪个 common operator 最稳定，也不能证明专家功能价值。

## 3. 主要结果

**一句话总结：** A06_24_synthetic 把“公共方向剔除作为主方法”降级为负载/偏置控制，把下一步推进到任务相关状态选择或路由行投影保持。

### 结果 1：全位置公共方向剔除没有补上任务相关样本池选择。

**机制对应：** 该结果对应第 2 节中的 $\mathcal P_{\mathrm{all}}$ 与 $\mathcal P_{\mathrm{route}}$ 区分：剔除 $c$ 只能改变输入坐标，不能保证聚类样本来自任务相关状态。

**指标或条件定义：** 全位置公共方向剔除是在所有 token hidden states 上减公共方向后聚类；任务位置残差中心只在 repeated feature slot 的最后一个 token 上减公共方向后聚类。联合分离分数越高，说明 common/rare 和 rare-rare 两个分离要求越同时成立。

**证据：** Step 0 时，全位置公共方向剔除的 rare-feature NMI 为 `0.690`、联合分离分数为 `0.405`、稀有路由 margin 低尾为 `-2.759`；任务位置残差中心的 rare-feature NMI 为 `1.000`、联合分离分数为 `0.637`、稀有路由 margin 低尾为 `11.657`；oracle raw 中心的联合分离分数也是 `0.637`。

![Step-0 联合分离分数](experiment/figures/step0_joint_feature_score_by_condition.png)

**解释：** 如果公共方向剔除本身就是特征分离器，全位置公共方向剔除应接近任务位置或 oracle 中心。实际结果显示它仍保留负 margin 和较低联合分离分数，说明缺失环节是任务相关样本池选择，而不是只缺一个公共方向校正。

**边界：** 这只削弱简单全局公共方向剔除；不排除带任务目标、梯度信号或稳定性约束的更强 common operator。

**来源：** anchor、protocol、`summary.md`、`detailed.md`、`figures/step0_joint_feature_score_by_condition.png`。

### 结果 2：任务预测准确率不能作为专家分工证据。

**机制对应：** 该结果对应“目标学习”和“专家分工”分离：模型可以通过共享表征或不干净的路由完成预测，而不形成稳定 feature-to-expert mapping。

**指标或条件定义：** target accuracy 是任务位置预测 feature-specific target token 的准确率；联合分离分数是 common/rare 分离和 rare-rare 分离的乘积；稀有路由 margin 低尾检查最脆弱稀有样本是否仍在正确 basin 内。

**证据：** 训练到 step 160 后，所有条件 target accuracy 都达到 `1.0`。但全位置公共方向剔除的 final 联合分离分数为 `0.432`，稀有路由 margin 低尾为 `-5.427`；任务位置 raw 的 final 联合分离分数为 `0.620`，稀有路由 margin 低尾为 `5.227`。

![Final 联合分离分数](experiment/figures/final_joint_feature_score_by_condition.png)

**解释：** 合成目标足够简单，模型可以学会预测而不保持干净路由。准确率只能说明任务可学，不说明路由专家分工成立。

**边界：** 该结果不能直接外推到真实语言模型 loss；真实 LM 中 loss、routing 和 feature utility 的关系需要单独审计。

**来源：** `summary.md`、`detailed.md`、`tables/condition_aggregate_final.csv`、`figures/final_joint_feature_score_by_condition.png`。

### 结果 3：路由行投影比残差输入更适合作为完整 common/rare 分区的保持候选。

**机制对应：** 残差输入改变 router 读到的 hidden state；路由行投影直接限制 router row 沿公共方向增长。当前目标是同时保持 rare-rare 和 common-vs-rare 分离，因此要看联合分离分数，而不能只看 rare-feature NMI。

**指标或条件定义：** oracle residual input 条件表示用 oracle 中心初始化，并让 router 读 $h-c$；oracle row-projected 条件表示用 oracle 中心初始化，并在训练更新后投掉 router row 的公共方向分量。

**证据：** 训练后，oracle residual input 的 rare-feature NMI 为 `1.000`、稀有路由 margin 低尾为 `8.924`，但 common/rare NMI 降到 `0.376`、联合分离分数为 `0.376`；oracle row-projected 的 rare-feature NMI 为 `1.000`、common/rare NMI 为 `0.636`、联合分离分数为 `0.636`、稀有路由 margin 低尾为 `8.646`。

![Final rare margin p05](experiment/figures/final_rare_margin_p05_by_condition.png)

**解释：** residual input 条件能保住稀有特征彼此分开和正 margin，但公共特征仍可能重新混入稀有专家。row projection 直接限制 router row 的公共方向回流，因此更符合“完整 common/rare partition preservation”的方法目标。

**边界：** 这不是 row projection 最优性证明，也不是真实 DCLM 可迁移证明；它只给出下一步方法 anchor 的优先候选。

**来源：** `summary.md`、`detailed.md`、`tables/condition_aggregate_final.csv`、`figures/final_rare_margin_p05_by_condition.png`。

## 4. 执行动作

**下一步动作：** 建立 `A06_25` 方法 anchor，优先选择“路由行投影式保持更新”；另列“无标签任务相关状态选择器”为后续或并行候选。

**目的：** 在已知简单公共方向剔除不能产生分离之后，验证一个真正面向 solution space 的方法：给定可用中心后，约束 router row 的可更新空间，是否能稳定保持 common/rare 与 rare-rare 分区。

**完成判据：** 新 anchor 必须在同一合成桥上先复现 A06_24_synthetic 的正负对照，再在更难条件中报告联合分离分数、rare-feature NMI、common/rare NMI、稀有路由 margin 低尾、sign-flip rate、slot-start NMI 和 task loss；只有联合分离和 margin 同时保持，才算 preservation 方法成立。

## 5. 证据索引

| 结果 | 证据 | 主要指标或图 | 支持什么 | 不能证明什么 | 来源 |
|---|---|---|---|---|---|
| 结果 1 | Step-0 全位置公共方向剔除 vs 任务位置/Oracle 中心 | `step0_joint_feature_score_by_condition.png` | 公共方向剔除不能替代任务相关样本池选择 | 所有 common operator 都失败 | A06_24_synthetic anchor / summary / detailed |
| 结果 2 | 训练后任务准确率与路由分离分数解耦 | `final_joint_feature_score_by_condition.png` | 准确率不是专家分工指标 | 真实 LM loss 与专家分工的关系 | A06_24_synthetic summary / detailed |
| 结果 3 | oracle residual input vs oracle row-projected | `final_rare_margin_p05_by_condition.png` 和 final aggregate table | row projection 是更强的 preservation candidate | row projection 最优或可直接迁移真实 DCLM | A06_24_synthetic summary / detailed |

## 6. 来源索引

**Anchor：**

- `anchors/06_24_synthetic_common_rare_transformer_moe_anchor.md`
- `anchors/06_24_synthetic_common_rare_transformer_moe_anchor_cn.md`
- `anchors/06_24_toy_common_rare_residual_proxy_anchor_cn.md`

**Protocol：**

- `experiment/protocol.md`
- `experiment/protocol_cn.md`

**Summary：**

- `experiment/summary.md`

**Detailed：**

- `experiment/detailed.md`

**Figures：**

- `experiment/figures/step0_joint_feature_score_by_condition.png`
- `experiment/figures/final_joint_feature_score_by_condition.png`
- `experiment/figures/final_rare_margin_p05_by_condition.png`
- `experiment/figures/step0_slot_start_nmi_guard.png`

**Tables and logs：**

- `experiment/tables/condition_aggregate_step0.csv`
- `experiment/tables/condition_aggregate_final.csv`
- `experiment/tables/training_trajectory.csv`
- raw ACP log is excluded from this sync package; job id and completeness checks are recorded below.

## 7. 补充材料

### 实验完整性检查

4 卡 full run 已完成并产出完整结果：

- job id: `pt-hb9swzcm`
- run name: `a06_24_synthetic_full_4gpu_20260702_172558`
- detailed 记录状态：`SUCCEEDED`
- rank 覆盖：runtime log 中 `rank 0--3` 均启动，`world_size=4`
- seed / slot cell 覆盖：`8` 个 seed × `4` 个 slot length = `32` cells，全部有 `cell_done`
- `step0_discovery.csv`：`256` 行
- `training_trajectory.csv`：`1280` 行
- `condition_aggregate_step0.csv`：`8` 行
- `condition_aggregate_final.csv`：`8` 行
- final step：`160`

### 数据构造

合成序列由中性背景 token、一个 repeated feature slot、以及 slot 后的 feature-specific target token 组成。公共特征在训练和校准中高频出现，稀有特征低频出现；评估集保持 balanced，使指标衡量路由分离而不是频率先验。模型没有 learned 或 sinusoidal position embedding；slot length 为 `1/2/4/8`，slot start 平衡采样，用于排除位置 shortcut。

### 位置 guard

Step-0 slot-start NMI 最大均值为 `0.024`。因此主要结论不是显式 slot-start shortcut；但这不等于完全排除所有上下文长度或 causal attention 影响。
