# 线性 MoE Router 的频谱学习动力学

状态：定理核心已由研究者批准；主文已完成语言与严谨性复核并定稿；E03-S
固定目标受控实验已通过注册判据。本文是 2026-07-31 理论最终稿，不等价于
E03 Protocol，也不授权新的训练。

本包回答：输入 covariance 的各向异性何时会让线性 Gate 更快学习高方差方向，
各向同性时为什么不存在由 covariance 指定的方向偏好，以及专家共同训练会在
哪些附加条件下放大或抵消这种倾向。

## 认识更新

这篇理论主文把“Router 为什么偏向大奇异值方向”拆成了三个不同问题：

1. **信号从哪里来：**covariance 不能凭空创造专家差异。Gate 的任务梯度首先
   需要“哪个专家对这个 token 更好”的相对损失信号；covariance 只会放大已经
   存在的输入—专家优势关联。
2. **训练偏置是什么：**当不同谱方向承载的专家优势可比时，大方差方向拥有更
   短的有限时间学习常数，所以 Router 会先学会 head。这个结论描述“先后顺序”，
   不是“middle/tail 永远学不会”。没有正则时，中低方差方向最终可以追上。
3. **真实联合训练还缺什么：**专家也在变化，优化器也可能改变方向尺度，上游
   表征基底还会旋转。因此现有定理只给出一个受控因果根；真实模型是否形成
   Router—Expert 正反馈，必须由 E03-S 的联合专家条件和 E03-R 的逐步轨迹
   分别检验。

导师可直接复述的一句话是：

> 大方差不是 Router 的功能目标，而是一个有条件的学习速度放大器：它让已有
> 的专家优势信号更早进入 Gate，却不能单独创造专家区分，也不能证明最终只看
> head。

形式化主结论是条件性的：

> 当专家优势信号在谱方向上可比、Gate 处于非饱和可学习区、优化器没有白化
> 输入且表征基底短时稳定时，较大 covariance 特征值给出更短的有限时间学习
> 常数。它产生早期 head alignment，但不推出任意各向异性输入最终都由 head
> 主导。

## 受控实证状态（2026-07-30）

E03-S 已在固定 Gaussian 表征、匹配 Gate-space 目标、trace-matched 谱和 pure
SGD 下验证理论的受控预测：flat 的三频带学习时间重合；4:2:1 谱约形成
1:2:4；16:4:1 谱约形成 1:4:16；白化后重新重合；tail-only 最终可学。
因此“covariance 是有条件的有限时间速度乘子”已同时具有定理和受控实验证据。
这没有验证可训练专家正反馈、AdamW 下的同型排序或真实 DCLM 的形成原因。
首个 E03-R 正式配置在三个 seeds 的 step100 前均触发负载守卫，因而只能判
`insufficient_load_guard`；它不支持也不反驳真实 head 形成命题。

[E03-S 完整结果](../../../main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)

阅读入口：

1. [主文：线性 MoE Router 的频谱学习动力学](01_理论论文_线性MoE_Router的频谱学习动力学.md)
2. [E03 动力学 subanchor](../../../main/problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor_cn.md)

本文状态严格区分：

- **已证明：**固定表征、固定或局部线性专家优势、线性 Gate 的精确初始梯度；
  二次局部模型中的各向同性与各向异性学习时间定理。
- **条件命题：**softmax 非饱和邻域和带正则训练的推广。
- **受控实验已支持：**匹配目标下的谱隙剂量关系、平谱旋转 null、白化回归
  和 tail-only 可学习性。
- **待实验：**可训练专家是否形成额外正反馈；以及在负载稳定、仍无 LB
  auxiliary loss 的有效 E03-R 配置中，AdamW 实际更新是否保留理论排序和
  真实 DCLM 的形成时间。
