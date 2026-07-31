---
anchor_id: 15_02_01_cross_update_compatibility_gate
parent_anchor: 15_02_middle_tail_functional_resolution
status: completed_fail
created: 2026-07-30
updated: 2026-07-30
canonical_language: en
canonical_file: 15_02_01_cross_update_compatibility_gate_anchor.md
---

# A15_02_01 频带对共同训练兼容性的增量预测

父 anchor：[A15_02](../15_02_middle_tail_functional_resolution_anchor_cn.md)。

## 1. Problem Definition

本 subanchor 继承父问题，只裁定一个准入条款：

> 在控制 native Router logits、margin、原生专家、负载、容量、token 数、
> 表示范数、token loss、文档和 batch 后，actual Router input 的 middle、
> long-tail 或二者联合，是否在独立 DCLM 文档上额外预测两组 token 更新同一
> 专家时的交叉 loss，并超过同维随机与错误层基底？

它区分三个概念：

1. **静态新颖度：** 频带产生 native logits 之外的新邻域；
2. **局部功能兼容性：** 一组 token 的专家更新帮助或伤害另一独立组；
3. **长期训练收益：** 匹配联合训练后的 held-out loss/FLOP。

只有第二项是本 subanchor 的决策对象。第一项只作解释，第三项属于父 anchor
的条件性 E02。

### 核心指标合同

| 指标 | 普通含义 | 具体计算 / 单位 | 为什么测 / 能回答多少 | 不能回答 |
| --- | --- | --- | --- | --- |
| 残差邻域新颖度 $N_S$ | 控制 native scores 后，频带找到多少不同邻居 | held-out kNN 新邻居比例，无量纲 | 频带是否提供不同划分 | 划分是否有用 |
| 双向交叉更新兼容性 $C_e(A,B)$ | A 更新专家后 B 是否变好，反向再测一次 | nat/token | 本地、专家条件下的共同训练关系 | 长期优化收益 |
| 梯度余弦 | A/B 专家梯度是否同向 | $[-1,1]$ | 解释 $C$ 是否来自一阶梯度对齐 | 实际 loss 一定下降 |
| held-out $\Delta R_S^2$ | 加入频带后，对未见文档的兼容性多解释多少 | $R^2$ 差，无量纲 | **本 subanchor 主指标；决定训练准入** | treatment 训练后一定更好 |
| random gap | 真频带比同维随机方向多多少 $\Delta R^2$ | $\Delta R^2$ 差 | 排除“维度多就有效” | 层特异性 |
| wrong-layer gap | 本层频带比错误层同 rank 基底多多少 | $\Delta R^2$ 差 | 排除任意层通用几何 | 长期因果机制 |

## 2. Physical Priors

1. 线性 Gate 只保留少量 logit 坐标；若 middle / long-tail 含有与专家梯度
   关系相关的结构，则在 native score 条件内仍应预测 $C_e(A,B)$。
2. 任意高维子空间都能制造新近邻，因此静态新颖度不能单独支持功能 claim；
   只有 held-out $\Delta R^2$ 超过同维随机和错误层才区分 strongest rival。
3. 兼容性是局部量。固定路由和小步长用于隔离目标专家更新；它们提高内部
   效度，但也限制对真实联合训练的外推。

## 3. Falsifiable Hypotheses

**H1——non-head compatibility signal。** 至少一个
$S\in\{M,T,N=M\cup T\}$ 的标准化 pair features 在 native controls 之外
提供正的 held-out $\Delta R_S^2$，并超过 256 个同维随机 bases 的 q95 和
错误层；该方向在 12-layer 两个谱系与 4-layer transfer checkpoint 上复现。

**最强 rival R0——geometry-only。** $N_S$ 高，但 $\Delta R_S^2\le0$ 或不超过
同维随机。此结果足以阻止训练，即使频带产生明显 route 或邻域差异。

**R1——norm / outlier / difficulty。** 增量由 hidden norm、band energy、
token NLL、gradient norm 或极端样本解释；加入这些 controls 后消失。

**R2——document / batch leakage。** A/B 共享文档、context 或 batch 使兼容性
虚高；按文档隔离与 batch 重采样后消失。

**R3——step-size artifact。** 只有一个过大更新步长产生信号；半步长下符号或
排名不稳定，或 self-update 不能降低自身 loss。

**Pass：** 一个预注册 candidate 在 final held-out documents 上
$\Delta R^2>0$，document-bootstrap 下界高于零，同时高于同维 random q95 与
wrong-layer；所有 operational guards 通过，并在 4-layer branch checkpoint
复现。

**Fail：** 指标有效且区间足够窄，但所有 $M,T,N$ 均不高于零或不超过关键
controls。

**Insufficient：** 路由重放、step-size、自身 loss、basis stability、pair
independence、文档级统计精度或 4-layer transfer guard 失败。

## 4. Mathematical Model

在独立 calibration set 上，对实际 Gate input 定义

$$
\Sigma_\ell=U_\ell\Lambda_\ell U_\ell^\top,
\quad
M=U_{65:320},\quad T=U_{321:768},\quad N=U_{65:768}.
$$

对 token $i$ 和 band $S$，使用方向归一的坐标

$$
q_{i,S}=\frac{U_{\ell,S}^\top(r_{i,\ell}-\mu_\ell)}
{\|U_{\ell,S}^\top(r_{i,\ell}-\mu_\ell)\|_2+\epsilon}.
$$

band energy 另作 nuisance control，因此 pair similarity 不会只由输入幅度决定。
对固定大小 token group $A$，$\bar q_{A,S}=|A|^{-1}\sum_{i\in A}q_{i,S}$；
注册 pair features 为 cosine 与 squared distance，所有 bands 使用相同两个统计量。

令 $\theta_{\ell,e}$ 为目标层专家参数。A/B 来自不同文档、原生进入同一专家，
且 native controls 匹配。固定并重放 native routes，只更新该专家：

$$
\Delta_{A\rightarrow B}
=L_B(\theta_{\ell,e}-\eta\nabla_{\theta_{\ell,e}}L_A)
-L_B(\theta_{\ell,e}),
$$

$$
C_e(A,B)=-\frac12
\left(\Delta_{A\rightarrow B}+\Delta_{B\rightarrow A}\right).
$$

$C>0$ 表示互助，$C<0$ 表示冲突。主指标为

$$
\Delta R_S^2
=R^2_{test}(C\mid X_{native},\phi_S)
-R^2_{test}(C\mid X_{native}),
$$

其中 $X_{native}$ 包含完整 logits、margin、expert stratum、load、token NLL、
hidden/band norm、gradient norm、position、document 与 batch controls；
$\phi_S$ 是固定的 band pair features。

## 5. Computational Realization

**频谱：** 复用 Q1 的 32×256 calibration token ids 和已验证 actual-input
bases；任何兼容性 fit/test 数据都不得重估 $U,\Lambda,\mu$。Q2 使用新的 DCLM
holdout documents，并按 document 分成 fit、validation、final test。

**12-layer evidence：** LB 与 decommon 的 80k 为主 checkpoint、40k 为复现；
静态 Q2-A 跑全部 12 层，一步 Q2-B 预注册 layers 1/6/12，不按结果挑层。

**4-layer transfer：** 在已通过 8×5090 fast-warmup / resume smoke 的 H768
4-layer checkpoint-800（约 0.629B nominal tokens）上，对全部 4 层重复同一
compatibility gate。父 anchor 的 E02 只有在这里复现后才可能解锁。

**Pair ledger：** 每组固定 32 个 routed tokens；A 与 B 无 token、document 或
batch 重叠；同 native expert，并在完整 logits、margin、loss、norm 和 position
上匹配。ledger 只用 native controls 构造，随后对 $M,T,N$ 和所有随机 bases
复用，避免 treatment-specific sampling。

**一步更新：** 全模型参数冻结；缓存并重放全部 native MoE routes；每次从同一
专家参数快照开始，只用 masked LM loss 更新目标专家一次，测另一组更新前后
loss，恢复参数后反向重复。$\eta$ 只在 calibration pairs 上选择；要求 self-loss
下降且 $\eta/2$ 下兼容性符号和排序稳定。

**随机 controls：** 对 $k=256,448,704$，用固定 seed 的 Gaussian matrix QR
生成 256 个 full-space Haar subspaces；$M/T$ 另有 non-head 704-space 内同维
随机 bases。$N$ 已占满 non-head，故使用 full-space Haar-704 与 wrong-layer
ranks 65--768。所有随机条件复用同一 $C$，不重复更新模型。

## 6. Minimal Falsification Tests

1. **Measurement smoke：** known-positive self/split pairs、shuffled pair
   known-bad、high-norm confusing cases；目的：证明 $C$ 可测。它只能验证指标，
   不能支持频带 claim。
2. **Static Q2-A：** 测残差邻域新颖度与跨 document 稳定性；目的：说明频带
   是否产生新划分。它无训练准入权。
3. **Functional Q2-B：** 在同一 pair ledger 上比较 baseline、+M、+T、+N；
   目的：直接检验 native score 之外的 compatibility 增量。
4. **Direction controls：** 同维 Haar、non-head random（可定义时）与 wrong-layer；
   目的：分别排除维度效应、任意 non-head 方向和任意层几何。
5. **Nuisance controls：** norm、outlier、difficulty、document、batch 和 gradient
   norm；目的：排除非功能 shortcuts。
6. **4-layer transfer：** 只决定能否进入预设计 E02；不能把 4-layer 结果外推
   为 12-layer 训练收益。

## 7. Current Evidence

[A15_02_01_E01](../../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md)
已经完成，运行护栏 Pass，科学裁定 Fail。注册层和谱系中，真实 M/T/N 的残差
邻域新颖度为 0.732--0.902，同维固定随机参考已达到 0.714--0.877。因此 non-head
确实改变静态划分，但新颖度很大一部分是高维子空间普遍效应。

Validation 的模型级三层中位兼容性增量，LB/decommon 分别为：M
$-7.35\times10^{-5}/-4.30\times10^{-5}$；T
$+2.24\times10^{-4}/-5.20\times10^{-5}$；N
$-5.90\times10^{-5}/-4.29\times10^{-5}$。T 只在 LB 通过点估计门；没有候选在
两条谱系同时为正并超过同维随机和错误层。

Fit/Validation 共 3,072 对全部完成；self-loss 通过率 1.0，专家参数精确恢复，
主步长与半步长 Spearman 为 0.87--1.00。候选集为空，因此按注册规则在 Final、
40k、4-layer transfer 和 E02 前停止。

## 8. Claim Boundary And Next Decision

本 subanchor 现在建立一个有边界的负结果：在注册 80k 模型、层、native-routed
expert 总体、局部步长、DCLM 文档、两个方向 pair features 和低容量 ridge 范围
内，固定 covariance-rank M/T/N geometry 没有获得超过 native controls、随机
方向和错误层的 same-expert compatibility 准入。

它不能证明：频带 routing 的长期收益、专家专业化、语义相似性、所有层普遍
成立，或一步梯度关系就是训练动力学。

**唯一下一决策：** 关闭固定 covariance M/T/N 直接作为 dispatch 坐标的路线，
或另立新 anchor，从 expert gradients / cross-update residuals 定义功能对齐子空间，
再考虑 matched training。当前 E02 继续阻塞。
