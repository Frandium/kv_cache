# 第 25 层 attention 写入是否增加了新的目标可访问性？

```text
type: daily_research_report
status: HUMAN_CONFIRMED
human_audit_scope: local_added_accessibility_only
unaudited_auxiliary_scope: target_conditioned_generalized_eigen_directions_and_rank_sufficiency
date: 2026-08-10
evidence_completed_utc: 2026-08-11
human_audit_completed_utc: 2026-08-11
topic: layer25_conditional_innovation
canonical_anchor: Projects/from-attention-to-search/main/problem_anchors/15_spectral_representation_and_functional_routing/15_08_target_conditioned_layer_innovation/15_08_target_conditioned_layer_innovation_anchor.md
frozen_protocol: Projects/from-attention-to-search/main/experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/protocol.md
canonical_summary: Projects/from-attention-to-search/main/experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/summary.md
canonical_detailed: Projects/from-attention-to-search/main/experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/detailed.md
```

> **人工审计边界：**本报告只确认第 25 层完整剩余更新 $R_U$ 的局部新增
> 可访问性。广义特征方程、目标条件方向、二维充分性和后续秩阶梯虽然出现在
> canonical 实验记录中，但尚未进行本轮人工审计，也不直接回答当前“怎样度量
> 层写入新增信息”的问题。因此，它们不进入本报告的认识更新或下一决策。

## 1. 一句话直觉

> “状态发生变化”不等于“出现了新增信息”。只有当写入中旧状态线性预测不了
> 的部分，在全新数据上进一步提高正确答案概率，并超过目标独立、预算相同的
> 对照时，我们才称它增加了**局部线性可访问性**。

## 2. 研究问题与当前结论

本轮把组会中的一般问题收缩成一个可审计实例：

> 在冻结 Qwen3-8B 的受控两跳任务中，第 25 层 attention 写入里不能由写入前
> 状态线性预测的部分，是否让终端答案在全新数据上更容易读出？

当前结论是：

1. **完整剩余更新增加了留出目标可访问性。**正确答案交叉熵降低
   $0.767207$ nats/example，配对 95% 区间为 $[0.751296,0.785146]$。
2. **该增益不是任意同预算附加支路的自然结果。**相对 64 个目标独立错配
   对照第 95 百分位的余量为 $0.766839$，95% 区间为
   $[0.750643,0.784307]$。
3. **结论只到“局部线性可访问性增加”。**它不证明信息论意义的新信息创造、
   整个表征低秩、模型原生使用该信号或 Router 能利用该信号。

因此，本轮经人工审计的认识更新是：

> 对这个模型、任务、token 和第 25 层写入，精确差分经旧状态线性残差化后，
> 确实含有旧状态读出之外、并超过注册目标独立对照的终端答案可访问性。

## 3. 任务、数据与比较对象

模型是冻结的 Qwen3-8B。“冻结”表示实验不更新大模型参数，只拟合离线线性
读出器。每个 **episode** 是一组共享两张随机映射表、并覆盖 A--H 八个终端
答案码的查询；它是统计独立单位。主条件 `complex` 必须连续查询两张映射表
完成两跳组合，配套 `simple` 条件用于能力检查。

我们只读取提示末尾 `Answer:` 前一个决策 token 在第 25 层 attention 写入
前后的状态。数据用途严格分开：

这里 **token** 是模型处理的离散文本单元；**attention** 是从上下文聚合信息
的子模块；**residual 状态**是 Transformer 子模块共同读写的 token 向量；
**RMSNorm** 是按向量均方根缩放状态的归一化。

| 划分 | 白话含义 | 允许做什么 | 规模 |
| --- | --- | --- | ---: |
| TRAIN | 拟合数据 | 拟合旧状态读出器、更新残差模型和附加修正器 | 每条件 1,024 records |
| DEVELOPMENT | 开发数据 | 选择 ridge 正则强度，不能报告最终效果 | 每条件 512 records |
| CONFIRMATION | 最终确认数据 | 只评价已经冻结的对象 | 每条件 128 episodes / 1,024 records |

E04 没有重新训练或选择对象，只把 E02 已冻结的模型应用到全新 CONFIRMATION。
E03 不满足自己的冻结记录合同，因此不承担本报告结论。

## 4. 怎样把“写入变化”变成“新增可访问性”

### 4.1 同层精确差分

令 $d=4096$ 为隐藏维度。对一条样本，定义：

$$
X=N_{25}(h),\qquad
Z=N_{25}(h+a),\qquad
U=Z-X.
$$

| 符号 | 形状 | 白话含义 | 决策角色 |
| --- | ---: | --- | --- |
| $h$ | $4096$ | 第 25 层 attention 写入前的 residual 状态 | 原始旧状态 |
| $a$ | $4096$ | attention 输出投影后、加回 residual 前的写入 | 实际写入量 |
| $N_{25}$ | 映射 | 写入前后共同使用的 post-attention RMSNorm | 保证比较位置和归一化规则一致 |
| $X$ | $4096$ | 假设不加入本层 attention 写入时的 MLP 输入 | 旧信息基线 |
| $Z$ | $4096$ | 实际加入写入后的 MLP 输入 | 同维新状态对照 |
| $U$ | $4096$ | 本次写入造成的精确归一化变化 | 合法计算差分，不自动等于新增信息 |

若直接比较协方差，存在：

$$
\Sigma_Z-\Sigma_X
=\Sigma_U+\operatorname{Cov}(X,U)+\operatorname{Cov}(U,X).
$$

$\Sigma_A$ 表示对象 $A$ 的协方差，$\operatorname{Cov}(A,B)$ 表示交叉协方差。
式中两个交叉项说明：协方差差不是“新增知识协方差”；即使 $\Sigma_U$ 也只
描述更新能量，不说明它是否在旧状态之外对目标有用。

### 4.2 去掉旧状态可线性预测的写入

在 TRAIN 上拟合 ridge（岭回归：在线性拟合中加入 $L_2$ 平方惩罚）模型，
从 $X$ 预测 $U$：

$$
\widehat m^U
=\arg\min_m
\frac1n\|U-m(X)\|_F^2+\lambda\|m\|_F^2,
$$

$$
R_U=U-\widehat m^U(X).
$$

这里 $n$ 是 TRAIN 记录数，$\|\cdot\|_F$ 是 Frobenius 范数，$\lambda$ 是由
DEVELOPMENT 选择的正则强度，$\widehat m^U(X)$ 是旧状态线性预测出的更新，
$R_U$ 是未被该模型重构的剩余更新。TRAIN 按完整 episode 交叉拟合，避免一条
训练记录的残差由见过同一 episode 的模型产生。

$R_U$ 仍不是知识矩阵：它只排除了注册线性模型能解释的部分，没有排除所有
非线性冗余。是否“新增可访问”，还必须看留出目标增益。

### 4.3 用留出目标增益作最终判定

从 $X$ 拟合基础八分类读出器 $f_X$，从 $R_U$ 拟合附加修正器 $g_{full}$，
并单独从 $Z$ 拟合同维状态读出器 $f_Z$。三者均只在 TRAIN 拟合、由
DEVELOPMENT 选择正则，并在 CONFIRMATION 前冻结。

$$
G_{true}
=CE_{conf}(f_X(X))
-CE_{conf}\!\left(f_X(X)+g_{full}(R_U)\right),
$$

$$
G_{state}
=CE_{conf}(f_X(X))-CE_{conf}(f_Z(Z)).
$$

| 符号 | 具体含义 |
| --- | --- |
| $f_X(X)\in\mathbb R^8$ | 只看旧状态得到的八个答案 logits |
| $g_{full}(R_U)\in\mathbb R^8$ | 完整剩余更新提供的附加答案 logits |
| $f_Z(Z)\in\mathbb R^8$ | 只看同维新状态得到的八个答案 logits |
| $CE_{conf}$ | CONFIRMATION 上正确答案的平均交叉熵；越低越好 |
| $G_{true}$ | 加入完整 $R_U$ 后降低了多少交叉熵 |
| $G_{state}$ | 同维新状态相对旧状态降低了多少交叉熵 |

`logit` 是 softmax 前的答案分数。因为交叉熵是正确答案概率的负对数，
$G_{true}>0$ 表示加入 $R_U$ 后正确答案概率提高。单位 `nats/example` 表示使用
自然对数后的每样本平均损失差。

最后构造 64 个**平衡错配对照**：保持 $R_U$ 输入维度、附加线性结构、数据
预算和正则网格不变，只打破更新与正确目标的对应关系。令 $q_{0.95}^{mis}$ 为
这些错配增益在每次 bootstrap 内的第 95 百分位，则：

$$
T_{cap}=G_{true}-q_{0.95}^{mis}.
$$

$T_{cap}>0$ 表示真实对应关系的增益超过几乎所有同预算目标独立支路。H2 Pass
要求 $G_{true}$、$G_{state}$ 和 $T_{cap}$ 的配对 95% 区间下界都大于零。
区间以完整 episode 为单位做 2,000 次配对 bootstrap。

## 5. 直接结果

模型先通过八选一能力检查：`simple` 准确率为 99.32%，`complex` 为 77.64%，
分别超过冻结的 80% 和 60% 门槛。承担局部新增可访问性结论的三项结果是：

| 判断量 | 估计值与配对 95% 区间（nats/example） | 直接读法 |
| --- | ---: | --- |
| $G_{true}$ | $0.767207\ [0.751296,0.785146]$ | 完整 $R_U$ 稳定改善留出预测 |
| $G_{state}$ | $0.754508\ [0.738226,0.773069]$ | 同维新状态相对旧状态同号改善 |
| $T_{cap}$ | $0.766839\ [0.750643,0.784307]$ | 目标独立、同预算支路不能解释该增益 |

三项区间下界均大于零，因此注册的局部 H2 通过。这个结果支持“本次写入增加了
局部线性可访问性”，不支持把 $R_U$ 本身称为新知识实体。

> **未纳入本轮审计的结果：**canonical 记录还包含广义特征方向、二维压缩和
> 秩充分性指标。它们不进入上表，不改变局部 H2 结论，也不作为本报告的下一步。

## 6. 认识更新、边界与唯一下一决策

### 确切认识更新

此前只知道层状态和 attention 输出在谱上发生变化；这种几何变化可能只是旧
信息搬运。现在我们得到一个更严格的局部度量：同层精确差分 $U$ 经旧状态线性
残差化得到 $R_U$，只有当 $R_U$ 在全新数据上提高正确答案概率并超过目标独立
同预算对照时，才称为“新增可访问性”。第 25 层这个实例满足该定义。

### 不能推出什么

- 不能推广到任意相邻层、其他 token、模型或自然语言任务。
- 不能证明 $Z$ 保留 $X$ 的全部信息，也不能证明非线性意义上的唯一新增内容。
- 不能把 $U$、$R_U$、协方差或谱方向直接称为知识。
- 不能裁定整个表征低秩，也不审计广义特征方向或二维/多维充分性。
- 不能证明模型原生使用该信号，更不能推出专家效用或 Router 收益。

### 唯一下一决策

把同一新增可访问性度量扩展到全部 36 层，并在匹配的一跳/两跳 × 近程/远程
任务上，用必要性 × 距离的差中差比较深层与浅层。只有这个逐层结果返回后，
才决定“深层是否增加更多长程组合目标可访问性”；不由未审计方向实验推进。

## 7. 证据与来源

- **问题来源：**0810 私有组会记录（未纳入外部同步包）
- **前序负向证据：**[方差区间报告](source_records/advisor/variance_interval_report.md)
- **正式问题边界：**[A15_08 中文 Anchor 快照](source_records/anchor/anchor_cn.md)
- **冻结实验合同：**[E04 Protocol 快照](source_records/e04/protocol.md)
- **结论入口：**[E04 Summary 快照](source_records/e04/summary.md)
- **完整证据账本：**保留在 Research_System canonical 项目中，未纳入同步包
- **同步包导航：**[README](README.md)
