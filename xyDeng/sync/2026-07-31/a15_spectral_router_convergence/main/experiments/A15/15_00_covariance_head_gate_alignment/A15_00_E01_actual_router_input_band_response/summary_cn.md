---
experiment_id: A15_00_E01_actual_router_input_band_response
status: completed_strict_h1_fail_typed_result
completed: 2026-07-30
canonical_summary: summary.md
---

# 结果摘要：实际 Router 输入上的频带访问与训练分配

主 anchor：[A15_00](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor_cn.md)  
Protocol：[已批准中文 Protocol](protocol_cn.md)  
完整证据账本：[detailed.md](detailed.md)

## 结论

**Q1 的 endpoint 部分通过。** 在 Gate 真正接收的表征上，LB 与 decommon 从
30k 起都明显偏向 covariance head；该结论不是输入方差机械放大的假象。40k
/80k 时，逐层 $G_H/G_M$ 中位数分别为 LB 5.41/6.36、decommon
4.03/4.27；$G_H/G_T$ 分别为 19.98/25.36、14.61/17.15。对应 log
contrast 远高于保持 Gate 奇异值不变的 Haar q95（约 0.04），配对的
calibration-sequence basis bootstrap 区间也都高于零。

**middle/tail 能看见，但弱得多，不能说“看不见”。** 它们的 $G$、实际
response、去带后的 route flip 和 native-margin effect 都非零。80k 的 coarse
head/middle/tail route-flip 中位数是 LB 0.741/0.126/0.018、decommon
0.645/0.089/0.013。因此准确表述是“当前访问和使用较弱”，不是线性 Gate 在
表达能力上无法读取中低方差方向。

**“30k→40k 与 40k→80k 都持续把 Gate 训得更偏 head”的强假设失败。** 两段
净 Gate 位移 $\Delta W$ 单独看都具有 head-oriented 的等能方向，且超过匹配
null；但固定表征基底后，它们没有一致增强已有 Gate 的 head contrast：
H:M 在两个区间、两个谱系都精确下降；H:T 只在 LB 两段增强，decommon
早段下降、晚段无法与零区分。

因此本实验的分型结论是：

> **30k 时已存在强 head-aligned endpoint；之后的净位移仍偏 head，但并未
> 持续增强固定基底下的相对 head 偏好。**

这解释了为什么必须同时看两个量：$\mathbf B^{update}$ 问“若把净更新本身当
成一个 Gate，它的能量朝哪里”；$\Delta_W\mathbf B$ 问“把它加到现有 Gate
后，相对对比是否真的增强”。已有 Gate 比更新向量更偏 head 时，一个仍然
head-oriented 的更新也会稀释 H:M；此外还存在带内 $W$ 与 $\Delta W$ 的有符号
交叉项。

## 主要数值

| 谱系 | step | $G_H/G_M$ | $G_H/G_T$ | endpoint 判定 |
| --- | ---: | ---: | ---: | --- |
| LB | 30k / 40k / 80k | 5.38 / 5.41 / 6.36 | 19.60 / 19.98 / 25.36 | 全部超过零与匹配 Haar q95 |
| decommon | 30k / 40k / 80k | 4.32 / 4.03 / 4.27 | 16.63 / 14.61 / 17.15 | 全部超过零与匹配 Haar q95 |

| 谱系 | 区间 | $B^{update}_{H:M/H:T}$ | $\Delta_WB_{H:M/H:T}$ | 解释 |
| --- | --- | --- | --- | --- |
| LB | 30k→40k | 0.990 / 2.630 | -0.036 / +0.080 | 位移偏 head；只增强 H:T |
| LB | 40k→80k | 0.974 / 2.814 | -0.054 / +0.162 | 位移偏 head；只增强 H:T |
| decommon | 30k→40k | 0.293 / 1.511 | -0.067 / -0.015 | 位移偏 head，但相对偏好被稀释 |
| decommon | 40k→80k | 0.410 / 1.570 | -0.061 / +0.008（区间跨零） | H:M 稀释；H:T 不足 |

完整区间、逐层值与 null 见
[endpoint table](tables/endpoint_contrasts.csv) 和
[trajectory table](tables/trajectory_decomposition.csv)。

## 护栏与证据边界

- 六个 checkpoint、坐标一致性、实际 Gate 输入/no-op replay、基底与能量重建
  全部通过。
- 12 层在六个 endpoint 的 coarse half-split 基底均稳定；F1 在每个 endpoint
  都是最强的模型中位 fine band，没有事后挑峰造成 coarse 结论。
- 保持奇异值的方向 null 最大相对奇异值误差约 $1.9\times10^{-6}$。
- 错误层基底和专家输入 $h$ 基底的 head contrast 明显更弱；用 $h$ 重放原生
  top-1 仅约 0.49--0.51 一致，进一步确认本轮测量对象正确。

本实验能回答访问、当前使用和两个保存区间的宏观净分配；不能回答 head 偏好
是否有功能收益、middle/tail 是否有额外功能信息、偏头从何时形成、每步梯度
是否持续偏头，或频谱 Router 是否改善 loss/FLOP。

## 下一步

**主线下一决策仍是 Q2 的功能准入，而不是立刻训练频谱 Router。** 频谱必须
在独立 token 上，控制 native 线性分数、负载、容量和 token 数后，增量预测
一步共同训练兼容性，并超过随机同维和错误层基底。

若另立在线动力学 Protocol 并使用 8×5090 训练 4/6 层小模型，目的只能是回答
“head alignment 在何时形成、由 raw gradient、optimizer 预条件、$W$--update
交叉项还是 $U$ 漂移维持”。应从初始化密集保存 $W_t$、raw Gate gradient、
实际 optimizer update、固定 probe buffer 上的 $U_t$、各带 signed cross term、
margin、flip 和 load。它不能替代 Q2，也不能证明功能效用。

