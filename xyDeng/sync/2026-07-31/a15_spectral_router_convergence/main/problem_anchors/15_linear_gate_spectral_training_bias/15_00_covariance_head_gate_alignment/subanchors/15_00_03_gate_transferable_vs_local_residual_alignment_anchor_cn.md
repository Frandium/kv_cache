---
anchor_id: 15_00_03_gate_transferable_vs_local_residual_alignment
parent_anchor: 15_00_covariance_head_gate_alignment
status: full_execution_authorized
canonical_language: en
companion_language: zh
updated: 2026-07-31
---

# A15_00_03 Gate 对 pooled common 与局部 residual 的偏好


## 1. Problem Definition

A15_00 已证明线性 Gate 对 pooled covariance head 有很强的等能对齐，但没有
区分该 head 是跨数据公共候选还是某个 calibration 样本的偶然方向。

**唯一决策问题：**

> 给每个方向相同输入能量后，decommon Gate 是否对独立 pooled
> centered-common candidate 产生比同维 shard-local residual 更大的专家间
> logit 增益和 native route dependence；这种偏好在 30k/40k/80k 如何变化？

本 subanchor 的直接对象是“pooled candidate vs local residual”。只有
A15_00_02 独立确认前者稳定、后者较不稳定后，联合解释才可写成“Gate 偏向
稳定 common”。

**主指标：**

$$
B_{\ell,P:L}
=\log\frac{G_{\ell,P}+\epsilon}
{\operatorname{median}_s G_{\ell,L_s}+\epsilon},
$$

单位为 log equal-energy gain ratio。它回答 Gate 权重偏向，不回答功能价值。

## 2. Physical Priors

1. **重复方向积累 prior。** 固定线性 Gate 更容易沿跨数据重复出现的方向累计
   一致权重；随 shard 旋转的方向，其更新可能平均抵消。
2. **能量不是选择性。** pooled top directions 的真实输入能量天然更大，必须
   使用完全去掉 covariance eigenvalue 的 $G$，不能用 raw response 判偏好。
3. **endpoint 由 $W$ 与 basis 共同决定。** checkpoint 间变化必须用固定 basis
   crossing 分离 Gate 权重变化和表示漂移。

## 3. Falsifiable Hypotheses

**H1——Gate 偏 pooled common candidate。** decommon 在 80k 的
$B_{P:L}$ 为正并超过保持 Gate 奇异值的 orientation null；40k 方向复现。
去除 pooled candidate 后，原生 winner 的 margin 支撑下降也大于同维 local
residual；route flip 只作辅助。LB 判断该现象是否跨谱系复现。

**最强 rival R0——只有能量差。** raw response 偏 pooled，但 $G$ 与
$B_{P:L}$ 不偏 pooled。

**R1——pooled estimator 特权。** 任何 pooled 同维方向、错误层 basis 或
orientation null 都得到相同偏好。

**R2——只有表征漂移。** endpoint $B_{P:L}$ 变化，但固定 basis 后
$W_{30/40/80}$ 没有相应变化。

**R3——稳定且有用。** Gate 的 pooled 偏好真实存在，但该方向承担共享功能；
本实验不能把这种偏好定性为好或坏。

**Pass：** decommon 主 endpoint 的 $B_{P:L}$ 和 native-margin 支撑对比都
支持 H1，40k 复现且 null/错误层护栏通过。

**Fail：** 有效精确结果支持 R0、R1，或 native route 对 local residual 的依赖
不弱于 pooled candidate。

**Insufficient：** A15_00_02 的稳定性结论可独立为 insufficient；本 anchor
仍可裁定 pooled-vs-local preference，但不得使用“稳定 common”联合解释。
若 actual-input、basis、route replay、checkpoint crossing 或 precision 护栏
失败，则本 anchor 自身 insufficient。

## 4. Mathematical Model

令 $\bar W_\ell=C_EW_\ell$，其中
$C_E=I-\mathbf1\mathbf1^\top/E$ 去除所有专家共有的 logit 平移。

在只用于 basis fit 的独立 pooled documents 上得到 $U_{\ell,P}\in
\mathbb R^{768\times64}$。对每个 shard $s$，先去掉 pooled projector，再在
剩余空间拟合局部 top-64：

$$
Y_{\ell,s}=X_{\ell,s}(I-U_{\ell,P}U_{\ell,P}^{\top}),
\qquad
U_{\ell,L_s}=\operatorname{TopSV}_{64}(Y_{\ell,s}).
$$

对任意 64 维 basis $U$，定义每方向等能 Gate 增益

$$
G_\ell(W,U)=\frac1{64}\|\bar W_\ell U\|_F^2.
$$

主指标 $B_{\ell,P:L}$ 比较 pooled candidate 与所有 local residual bases 的
中位增益。实际使用以 64 维去带后的 native-winner margin support 为主，
top-1 flip 只作辅助。

对 $a,b\in\{30k,40k,80k\}$ 计算 $W_a\times U_b$ crossing；固定 basis 的
Gate-weight effect 只说明保存区间的净变化，不等于逐步梯度因果。

## 5. Computational Realization

[已批准的 E01 中文 Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_03_E01_gate_transferable_vs_local_residual_alignment/protocol_cn.md)
复用 A15_00_02 的 frozen actual-input cache、文档 split 和 64 维 bases，但
独立计算 Gate gain、route effect、orientation null 与 checkpoint crossing。

模型为现有 12-layer H768 decommon 与 LB 的 30k/40k/80k checkpoints；80k 是
主 endpoint，40k 是复现，30k 只帮助描述宏观形成状态。所有 12 层报告。

两份 E01 共享 S0 provenance 与一次激活提取，可并行执行分析；任何一个失败都
不会事后改变另一份的 estimator、数据或 metric。

## 6. Minimal Falsification Tests

1. 对 pooled candidate、每个 local residual 和 full/complement Haar-64 使用
   相同的 $G$、flip、margin 实现。
2. 以 256 个保持 Gate 非零奇异值的 orientation null 判断方向偏好；错误层
   pooled basis 排除任意层共享几何。
3. 在相同 held-out documents 上比较 equal-rank band ablation，禁止用 total
   energy 差解释 route effect。
4. 用完整 $3\times3$ checkpoint crossing 分离 $W$ 与 basis 漂移。
5. LB/decommon 分开裁定；二者不是 center/LB 的单变量因果对照。

## 7. Current Evidence

A15_00 E01/E02 已证明 pooled covariance head 上的 $G$ 显著高于 middle/tail，
但当时没有构造 cross-shard pooled/local residual 对比。

A15_02_01 E01 证明 non-head 有额外静态划分，却没有稳定的一步功能增量。这与
“Gate 偏 pooled common”相容，但不能证明它，也不能说明这种偏好有害。

E01 Protocol 与完整冻结运行已于 2026-07-31 获得授权。

## 8. Claim Boundary And Next Decision

通过最多支持：注册 checkpoint 的 Gate 对 pooled centered-common candidate
比对同维 shard-local residual 有更强等能偏好和 native 使用。

只有 A15_00_02 同时通过，才能将 pooled/local 分别解释为“跨文档稳定 common”
和“较不稳定 residual”。即使两者都通过，也不能声称稳定性导致 loss 改善、
residual 无语义或 decommon 失败。

**唯一下一决策：** 完成已授权的冻结审计，并与 A15_00_02 合并 typed verdict。
联合 Pass 才可打开匹配稳定-vs-局部干预的设计决策；否则关闭或收窄“残差
不稳定导致 Gate 回到 common”的机制解释。
