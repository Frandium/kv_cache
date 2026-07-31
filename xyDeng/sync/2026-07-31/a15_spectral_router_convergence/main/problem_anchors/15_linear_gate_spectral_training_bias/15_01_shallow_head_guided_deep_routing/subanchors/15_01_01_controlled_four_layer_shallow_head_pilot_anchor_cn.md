---
anchor_id: 15_01_01_controlled_four_layer_shallow_head_pilot
parent_anchor: 15_01_shallow_head_guided_deep_routing
status: insufficient_stage_a_capability
canonical_language: en
canonical_file: 15_01_01_controlled_four_layer_shallow_head_pilot_anchor.md
updated: 2026-07-30
---

# A15_01_01 四层浅层 head 指导受控 Pilot


## 1. Problem Definition

本 subanchor 只测试 A15_01 的一个受控条款：

> 在 capable 四层 MoE 中，通过独立兼容性门并匹配负载、容量、token 和 FLOPs
> 后，把第二层 head 坐标提供给第三、四层 Gate，是否比真正普通的 native 四层
> 模型以及参数匹配的 random/token-shuffled side channel 更快降低 held-out loss？

主训练指标是在冻结累计 FLOP 预算下的配对 held-out NLL 差：

$$
\Delta L_{H-C}=L_H-L_C,
\qquad C\in\{\text{native},\text{random},\text{shuffled}\}.
$$

单位为 nat/token，负值支持 shallow head。它不能证明大规模或自然语言效率。

## 2. Physical Priors

1. 第二层 head 只有在预测深层更新兼容性时，才可能减少深层路由搜索；
2. 辅助读出本身增加容量，所以主比较必须同架构、同计算；
3. informative 与 nuisance 两任务缺一不可，否则可把期望答案直接写进生成器。

## 3. Falsifiable Hypotheses

**H1：**第二层 head 通过 held-out 兼容性门，在 informative 任务上对 native
模型和两个匹配旁路对照都满足 $\Delta L_{H-C}<0$，且 nuisance 任务不出现同样
的虚假优势。

**最强 rival：**额外参数、特征尺度、token identity 或负载重分配解释收益；
累计计算匹配后，native 或 random/shuffled 同样有效。

**Pass：**能力和兼容性护栏通过；head 对 native 和两个匹配旁路对照的配对区间
都低于零；负载与容量匹配；nuisance 不复现同样效应。

**Fail：**全部护栏通过但 head 不能同时超过三个对照，或计算/负载匹配后收益
消失。

**Insufficient：**第二层注册变量未稳定进入 head、基础任务未学会、兼容性估计
不确定或路由/计算护栏失败。

## 4. Mathematical Model

对 $\ell\in\{3,4\}$，

$$
z_\ell
=W_\ell g_\ell+A_\ell c_{2,H},
\qquad
c_{2,H}=U_{2,H}^\top(g_2-\mu_2).
$$

$A_\ell$ 零初始化，使全部条件从 native score 开始。random 使用 $g_2$ 的
冻结同维 Haar 子空间；shuffled 使用同 batch 另一 token 的正确 $c_{2,H}$。
三者的 $A_\ell$ 形状和运算相同。

训练前，独立 token 一步交叉更新目标必须满足

$$
\Delta_{\rm comp}^{H}
>
\max(\Delta_{\rm comp}^{random},
\Delta_{\rm comp}^{shuffled})
$$

并通过注册不确定性判断。它是硬准入，不替代 NLL endpoint。

## 5. Computational Realization

受控生成器包含一个人为设为高方差的粗变量；在 informative 条件中它预测深层
专家兼容性，在 nuisance 条件中与目标操作独立。四层、8-expert、top-1 模型先
把第 1--2 层训练到 capability/capture 门并冻结；第 3--4 层随后完成一个所有
条件共用、看不到 treatment 的 native calibration warmup。兼容性审计和
native/head/random/shuffled 四臂都从该 checkpoint 克隆启动。训练不使用
load-balance auxiliary loss；四臂共享同一不反传 expert-score bias 规则。

## 6. Minimal Falsification Tests

1. 独立数据上的能力门与第二层 head capture 门；
2. 相对 native、范数/离群值、random、shuffled、batch-resampling 的独立 token
   兼容性准入；
3. informative/nuisance 上 native/head/random/shuffled 的 paired seeds 和相同
   数据顺序匹配计算训练；
4. margin、flip、load、capacity drop、专家更新冲突与专家功能重复诊断。

## 7. Current Evidence

实验记录：[Protocol](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/protocol_cn.md)、
[结果摘要](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/summary.md)与
[完整证据](../../../../experiments/A15/15_01_shallow_head_guided_deep_routing/A15_01_01_E01_controlled_four_layer_shallow_head_pilot/detailed.md)。

**直接观测：**修复后的 smoke 通过 11/11 个工程护栏。获批 full run 随后在
informative 与 nuisance 两任务、五个 seed 上完成 Stage A，并以注册状态
`insufficient_stage_a_capability` 终止。10 个 task-seed 状态的粗变量准确率、
内容保留、head probe 绝对准确率和 split-half 基底稳定性全部通过；但 head
probe 与 256 个同维 random-subspace probe 的 q95 在每个状态都等于 1.0。因此
严格 specificity gap 全部为零，head 严格超过 random q95 的状态为 0/10。
B0、Stage 0 兼容性和 B1 均未运行，三者记录数都为零。

**解释：**受控粗变量确实可从 head 读出，但当前 64 维、以最终准确率判断的
probe 已经饱和，不能证明这种可读性是 covariance head 特有的。这削弱的是
Stage-A 操作化，而不是 shallow-head 机制本身。

**仍未解决：**非饱和 held-out specificity 检验能否识别 head 中更集中的信息，
以及该信息能否预测兼容性或改变匹配 FLOP 训练，均无证据。

## 8. Claim Boundary And Next Decision

**已支持：**正式 Stage-A 执行有效；目标与内容 proxy 已学会；估计的 head 稳定；
注册的 head-vs-random specificity 检验因两者都达到满准确率而失去区分力；
fail-closed 阶段门正确工作。

**被削弱：**当前 Stage-A capture 操作化不能在本设置中把第二层 head 认证为
具有特异性的 treatment variable。

**仍未解决：**H2 兼容性、相对 random/shuffled 的比较，以及任何
Router--Expert 学习路径或 matched-FLOP 效果都未测试。不得把该结果写成 H2
fail。

**不能声称：**shallow-head 信息不存在、兼容性失败、random 等价、训练收益或
损害、从初始化即获益、online PCA、DCLM 迁移或大规模效率。

**唯一下一决策：**是否批准一份使用非饱和 held-out Stage-A specificity 标准的
新 Protocol，再决定是否允许任何 B1 训练。不得绕过或续跑本次失败的门。
