---
story_id: A15_linear_gate_spectral_access_and_learning_dynamics
status: final_synthesis
updated: 2026-07-31
canonical_language: zh-CN
companion_en: 01_linear_gate_spectral_access_and_learning_dynamics.md
---

# 从频谱访问到线性 MoE Gate 的学习动力学

*Router 看见什么、为何可能先学大方差方向，以及为何固定频带还不是合格的分发
坐标。*

状态：2026-07-31 最终综合稿；受控机制得到支持，真实联合训练迁移仍未解决。

## 0. 术语约定

- **实际 Router 输入：**线性 Gate 直接接收的表征，不是随后送入专家网络的
  表征。
- **covariance 谱：**中心化实际 Router 输入的协方差特征值与特征方向。
  head、middle、tail 分别表示高、中、低方差的特征值排序区间，不表示 token
  频率、语义重要性或功能价值。
- **等能 Gate 增益：**去除 covariance 特征值对输入能量的放大后，每个方向
  得到的 Gate 敏感度。
- **当前使用：**冻结检查点上的原生 logit 响应、路由翻转率或路由间隔依赖。
  当前使用不等于训练收益。
- **有限时间速度偏置：**某个方向用更少更新达到自身目标的相同比例；它不表示
  其他方向无法表达，也不表示最终解只能使用 head。
- **共同训练兼容性：**两个独立 token 组分别对同一专家做一步更新时，彼此的
  交叉损失是下降还是上升。它是局部功能准入目标，不是长期联合训练结果。
- **Pass / Fail / Insufficient：**有效性护栏通过后，Pass 或 Fail 才裁定注册的
  科学假设；Insufficient 表示前提失效或所需证据没有形成。

## 1. 摘要

本报告研究 covariance 信息能否成为线性混合专家模型（Mixture-of-Experts，
MoE）Gate 的有效归纳偏置。现有证据支持的结论比“Router 只看谱头”更窄也更
准确：在 Gate 真正接收的表征上，训练后检查点的等能增益明显偏向
covariance head；但 middle 与 tail 的访问和原生路由作用并不为零。最早可用的
10k 检查点已经最强偏头，随后到 30k 相对展宽，因此已保存的检查点
不支持“训练一直把 Gate 锐化到 head”。

受控 E03-S 找到了这种现象的一个因果来源。当各方向的功能目标匹配、输入总
能量固定、线性 Gate 使用纯随机梯度下降（SGD）训练时，较大的 covariance
特征值会缩短相应模式达到同等学习进度所需的时间。平谱消除顺序；谱隙增大使
顺序增强；白化再次消除顺序；只把目标放在 tail 时，tail 仍然可以学会。因此，
covariance 各向异性能够造成“先学 head”，但不推出“只能学习 head”。

这一机制能否迁移到真实联合训练仍未解决。E03-R 的三个六层 DCLM 训练都在
形成时间可被有效判定前触发注册的负载守卫。负载崩溃后的频谱变化可以描述，
但没有科学裁定资格。因此 E03-R 是证据不足，不是真实迁移的负结果。

功能边界同样是负面的，但范围更窄。固定 middle、tail 和 non-head 频带确实
改变静态邻域，却没有在 LB 与 decommon 两条谱系上稳定增加留出数据的一步
同专家兼容性预测，也没有同时超过同维随机和错误层对照，所以没有获得匹配训练
资格。另一个浅层 head 预实验停得更早：head 与随机 64 维线性探针都能完美解码
受控粗变量，测量发生饱和，因而没有进入兼容性或训练收益阶段。

最终认识是：covariance 各向异性已被证明是受控系统中的有限时间学习速度偏置；
但现有证据既没有建立真实 Router--Expert 的形成路径，也没有把固定 covariance
频带认证为功能分发坐标。

## 2. 研究问题

Q1 的精确问题是：

> 排除输入能量对原始 logit 的机械放大后，训练后的线性 Gate 能访问哪些
> covariance 频带？covariance 各向异性本身能否使高方差方向更早学会？

必须另行回答三个下游问题：

1. 负载稳定的真实 MoE 轨迹是否形成同一签名；
2. middle 或 tail 是否在线性 Router 分数之外提供功能增量；
3. 任何频谱分发规则是否降低匹配计算量下的留出损失。

目前只有 Q1 的受控机制条款得到因果回答。在三个下游问题中，真实轨迹条款
证据不足，注册的固定频带功能门失败，匹配训练没有运行。

## 3. 为什么这个问题容易混淆

下面四种效应都可能被口头概括为“Router 跟着 head”：

1. **原始能量：**固定权重在大方差输入方向上机械地产生更大 logit；
2. **Gate 方向：**训练后的专家相对行空间本身对大方差方向分配更多增益；
3. **学习速度：**相同功能目标沿大方差方向更快被学会；
4. **功能目的：**专家损失关系确实要求 Router 使用该方向。

终点对齐既不能确定来源，也不能证明价值。净更新本身可以偏 head，却仍然稀释
一个更强偏 head 的旧 Gate。静态频带可以产生新划分，却不能保证把适合共同
训练的 token 放在一起。实验链必须围绕这些区分组织。

## 4. 数据与训练布局

| 证据模块 | 模型与数据 | 干预或比较 | 回答范围 |
| --- | --- | --- | --- |
| E01 | 两条 12 层 H768 top-1 DCLM 谱系；30k/40k/80k | 实际 Gate 输入、粗细频带、完整 Gate×基底交叉 | 终点访问、当前使用、晚期净区间分配 |
| E02 | LB 与 batch-gradient 12 层谱系；10k/20k/30k | 把同一实际输入审计移到更早检查点 | 最早可见对齐与 10k--30k 展宽 |
| E03-S | 768→8 线性 Gate；固定 Gaussian 表征；八个随机种子 | flat、4:2:1、16:4:1、whitened、tail-only；纯 SGD | 受控 covariance 速度因果性 |
| E03-R | 六层 H768 top-1 DCLM；随机种子 17/29/43 | 从初始化训练，仅语言模型损失，无负载均衡辅助损失，密集诊断 | 真实形成签名及负载有效性 |
| A15_02_01_E01 | LB/decommon 80k；层 1/6/12；新留出文档 | 一步同专家兼容性、同维随机与错误层对照 | 固定频带功能准入 |
| 浅层预实验 | 四层受控 top-1 MoE；两任务、五个随机种子 | 兼容性/训练前比较第二层 head 与 256 个随机 64 维线性探针 | 浅层变量是否特异地集中在 head |

所有检查点频谱审计都使用 Gate 实际接收的表征。E01/E02 用固定训练序列估计
基底，用独立留出文档测响应和路由诊断。E03-S 的训练、轨迹评估和最终留出
数据流相互独立。E03-R 在重诊断点保存有序校准张量，使基底自助法重采样、
随机方向零假设对照与 $W_s\times U_t$ 分解能够复核。

## 5. 物理直觉

covariance 不能凭空制造路由目标。只有不同专家处理同一个 token 的损失存在
差异，Gate 才有功能性学习信号。一旦这种专家优势存在，较大的输入方差会放大
它与 Gate 更新之间的交叉协方差，使高方差模式更早进入 Gate。

该机制是条件性的：

$$
\text{专家优势信号}
\times
\text{输入 covariance}
\times
\text{优化器响应}
\longrightarrow
\text{Gate 模式学习速度}.
$$

若专家优势不存在、只位于 tail、被预条件器抵消，或随快速旋转的表征基底一起
变化，非平谱都不保证最终 head alignment。

## 6. 定义与指标

令中心化实际 Router 输入为 $x$，
$\Sigma=U\Lambda U^\top$；令
$C_E=I-\mathbf1\mathbf1^\top/E$ 去除所有专家共同的 logit 分量。对投影矩阵
$P_B$、基底 $U_B$ 和维数 $d_B$ 的频带 $B$：

$$
V_B=\mathbb E\|C_EWP_Bx\|_2^2
$$

表示实际 token 上的专家相对 logit 响应，包含输入特征值的放大。单位频带能量
响应为

$$
S_B=\frac{V_B}{\mathbb E\|P_Bx\|_2^2}.
$$

$S_B$ 控制整段总能量，但在宽频带内部仍按实际方差加权。严格的等能 Gate 增益
是

$$
G_B=\frac1{d_B}\|C_EWU_B\|_F^2.
$$

终点对比为

$$
B_{H:M}=\log(G_H/G_M),\qquad B_{H:T}=\log(G_H/G_T).
$$

它们测 Gate 方向，不测功能收益。E03-S 使用 $T_B(0.5)$：频带 $B$ 第一次持续
消除自身初始目标误差一半时的优化器 step，并比较 log 学习时间比。E03-R 只有
在两组 head 对比超过匹配随机方向对照、基底自助法重采样稳定、连续保持且负载
护栏通过时，才允许定义形成时间。

功能准入量是

$$
\Delta R_S^2
=R^2(C\mid X_{native},\phi_S)-R^2(C\mid X_{native}),
$$

其中 $C$ 是一步双向兼容性，$X_{native}$ 包含原生分数与干扰变量对照，
$\phi_S$ 是两个固定频带 pair features。它最多授予训练资格，不能证明训练终点
收益。

## 7. 数学模型

令冻结专家损失向量为 $\ell(x)$，定义中心化专家优势

$$
a(x)=-C_E\ell(x).
$$

对软路由损失 $L=\mathbb E[p(Wx)^\top\ell(x)]$，精确梯度为

$$
\nabla_WL
=\mathbb E[(\operatorname{Diag}(p)-pp^\top)\ell(x)x^\top].
$$

在平衡初始化 $W=0$ 处，

$$
\dot{\bar W}(0)=\frac1E\mathbb E[a(x)x^\top].
$$

若 $a(x)=Ax+\varepsilon(x)$ 且
$\mathbb E[\varepsilon(x)x^\top]=0$，则

$$
\dot{\bar W}(0)u_i=\frac{\lambda_i}{E}Au_i.
$$

因此 covariance 只能缩放已有功能信号，不能创建功能信号。局部可解二次模型为

$$
\dot w_i=-(\kappa\lambda_i+\beta)w_i+\kappa\lambda_i a_i,
$$

达到相对进度 $\rho$ 的学习时间为

$$
T_i(\rho)=\frac{-\log(1-\rho)}{\kappa\lambda_i+\beta}.
$$

## 8. 定理状态

**各向同性定理：**若 $\Sigma=\lambda I$，covariance 给所有方向相同的时间
常数。单个有限 run 选中的方向来自目标、初始化或采样，而不是 covariance 预先
定义的 head。

**条件性各向异性定理：**若目标专家优势在 covariance 方向上的范数可比，
初始化平衡，且优化器不抵消尺度，则较大的 $\lambda_i$ 给出更短的有限学习时间。
无正则时，较慢方向最终可以追上；各向同性 $L_2$ 正则可以保留稳态偏置。

**适用边界：**定理不直接覆盖硬 top-1 边界、AdamW 自适应预条件、随时间
变化的专家优势或表征基底。它也不推出与 head 对齐的 Gate 具有更集中的奇异值：
右子空间对齐与 Gate 奇异值集中是不同对象。

## 9. 机制分解

1. **专家近似同质：**功能优势接近零；covariance 不能单独启动有用专业化。
2. **微小优势形成：**若优势方向可比，高方差模式更早进入 Gate。
3. **Router--Expert 反馈：**路由改变专家数据，从而改变优势谱；反馈可以强化
   head、向 middle/tail 展宽，也可以破坏负载稳定。
4. **路由间隔饱和或分区锁定：**softmax 敏感度下降，容量和负载干预可能超过
   原始速度效应。

E03-S 用固定目标检验阶段 2。E03-R 原本要联合观察阶段 1--4，但注册轨迹在
阶段 3 因负载集中而失效。

## 10. Anchor 证据链

| 问题 | 直接结果 | 裁定 | 最安全结论 |
| --- | --- | --- | --- |
| 去除能量后，训练 Gate 是否仍偏 head？ | 40k/80k 的 $G_H/G_M=4.03$--$6.36$，$G_H/G_T=14.61$--$25.36$ | Pass | 终点 head alignment 不是纯原始能量现象 |
| Gate 能否看到 middle/tail？ | 增益、路由翻转率、路由间隔非零；10k--30k 相对访问增加 | 测量层面 Pass | middle/tail 较弱，但不是不可见 |
| 保存区间是否持续锐化 head？ | 所有被审计谱系与注册区间的固定基底 H:M 效应均为负 | 持续锐化 Fail | 净更新偏 head 不等于终点偏置增加 |
| 受控条件下 covariance 是否造成速度顺序？ | flat 重合；4:2:1 约 1:2:4；16:4:1 约 1:4:16；whitening 重合 | Pass | 受控有限时间速度因果性成立 |
| 真实 DCLM 是否形成同源签名？ | 三个随机种子都在有效形成前触发 0.8 负载守卫 | Insufficient | 真实迁移仍未解决 |
| 固定 non-head 频带能否预测兼容性？ | 没有 M/T/N 同时通过 LB、decommon、随机与错误层门 | Fail | 固定频带不获匹配训练资格 |
| 浅层 head 是否指导深层训练？ | head 与随机 64D 线性探针都为 1.0；后续阶段未运行 | Insufficient | Stage-A 特异性测试饱和，H2 未测试 |

## 11. 实验证据整合

### 受控学习时间

| 条件 | $T_H$ 中位数 | $T_M$ 中位数 | $T_T$ 中位数 | 解释 |
| --- | ---: | ---: | ---: | --- |
| Flat 1:1:1 | 140.82 | 140.83 | 140.80 | covariance 不指定顺序 |
| Moderate 4:2:1 | 55.76 | 111.46 | 223.01 | 约 1:2:4 |
| Strong 16:4:1 | 28.63 | 114.38 | 457.39 | 约 1:4:16 |
| Strong-whitened | 140.78 | 140.82 | 140.88 | 顺序消失 |

![受控 covariance 各向异性分离学习时间](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/figures/e03_s_crossing_times.png)

八个随机种子的各向异性中位数均超过由 2,048 个划分组成的平谱旋转零假设
对照；剂量差区间为正；每个 tail-only 随机种子的独立留出 KL 降幅都超过注册的 0.5
门槛。

### 真实运行的有效性边界

| Seed | 首个失败 20-step 窗口 | 首次失败份额 | Step-100 滚动最大值 |
| ---: | --- | ---: | ---: |
| 17 | 56--75 | 0.80208 | 0.99045 |
| 29 | 53--72 | 0.80246 | 0.99110 |
| 43 | 60--79 | 0.81781 | 0.98916 |

![负载崩溃先于有效形成裁定](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/figures/e03_r_load_collapse_and_contrasts.png)

step 50 及以前的已选有效点都没有同时超过两组随机方向对照。step 120 时两个随机种子
超过两组 null，但已经接近单专家负载，不能定义形成时间。

## 12. 反例与竞争解释

- **纯能量解释：**已被终点审计削弱，因为 $G$ 去除特征值放大后仍远超匹配
  随机方向对照。
- **任意方向巧合：**E03-S 的平谱旋转零假设对照和检查点的保持奇异值
  随机方向对照都削弱该解释。
- **真实专家目标本来偏 head：**E03-S 已控制，但真实联合训练中的专家优势会
  变化，因此仍未排除。
- **纯表征对齐：**不足以解释全部终点；固定基底 Gate 效应可测，但基底
  运动也很重要，有时方向相反。
- **自适应优化器：**迁移尚未测试；AdamW 可能拉平、保留或重塑 covariance
  时间常数。
- **负载崩溃伪影：**对 E03-R 的有效性是决定性的，阻止解释后期频谱运动。
- **不同几何就是功能：**被兼容性门否定；随机高维视图也会产生新邻域。
- **浅层 head 无用：**没有测试。浅层预实验因 head 与随机线性探针同时达到准确率
  上限而失去分辨率。

## 13. 结论边界

**已经建立：**在被审计检查点谱系中，实际输入的 Gate 方向在等能后仍
强烈偏 head；middle/tail 访问非零；10k 后持续锐化 head 不受支持；在注册的
固定目标纯 SGD 构造中，covariance 各向异性因果性地改变有限时间模式学习速度；
固定 M/T/N 频带没有通过注册的跨谱系局部功能准入门。

**尚未建立：**现有 DCLM 终点的确切形成原因；有效的真实 Router--Expert
head 形成轨迹；正向专家反馈；covariance ranks 的语义；middle/tail 功能缺失；
浅层 head 训练收益；或匹配 FLOP 下验证损失改善。

**不得推出：**“线性 Gate 只能看 head”“任何非平谱都迫使最终 head alignment”
“负载崩溃后的 contrast 增长就是形成”或“固定频带失败排除了所有频谱方法与
功能对齐方法”。

## 14. Anchor 分解

- **A15_00：**终点访问与保存检查点分配已有记录；
- **A15_00_01：**受控 covariance 速度条款 Pass；真实迁移为
  `insufficient_load_guard`；可训练专家 S2 尚未执行；
- **A15_01 / A15_01_01：**浅层到深层机制仍开放；当前 Stage-A 特异性操作化
  没有分辨力；
- **A15_02 / A15_02_01：**固定 M/T/N 频带局部兼容性准入 Fail，条件匹配训练
  继续阻塞。

这些分支回答不同因果层级。静态对齐不能替代兼容性，兼容性也不能替代匹配训练
收益。

## 15. 与主线的关系

A15 是机制与准入研究线，支持但不替代 A06 功能专业化主线。A06 要求受控路由
下的留出专家功能收益。A15 解释 Gate 为何可能形成频谱偏置，并检验固定
频谱坐标是否值得投入训练计算。当前 A15 证据既没有建立有用专家专业化，也没有
形成可部署 Router 设计。

## 16. 唯一下一决策

在 Q1 动力学范围内，决定是否批准一份新的 E03-R Protocol。它必须冻结并独立
验证不反传的负载稳定机制、归因边界和小型正式运行前稳定性门。完成标准是：三个
随机种子都越过本次失败窗口，同时任何连续 20 步的最大专家份额不超过 0.8，且不
触发 dead-expert 守卫；只有这样才允许新的 2B-token 运行。

现有 E03-R 轨迹不得在改变规则后续跑；E03-S S2 不得自动启动；A15_02 匹配
训练继续阻塞。

## 来源索引

- [A15 研究线索引](../../problem_anchors/15_linear_gate_spectral_training_bias/README.md)
- [A15_00 anchor](../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor_cn.md)
- [E03 动力学 subanchor](../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor_cn.md)
- [E01 结果](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)
- [E02 结果](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)
- [E03-S 结果](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
- [E03-R 结果](../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
- [浅层预实验结果](../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md)
- [兼容性门结果](../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md)
- [受控理论主文](../../../daily_research_reports/0731/router_spectral_learning_dynamics_theory_package/01_理论论文_线性MoE_Router的频谱学习动力学.md)
