---
experiment_id: A15_00_02_E01_centered_common_subspace_stability
status: approved_for_full_execution
created: 2026-07-31
approval_date: 2026-07-31
primary_anchor: 15_00_02_centered_common_subspace_stability
canonical_protocol: protocol.md
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: local-2xh100-remote-8x5090-fallback
---

# Protocol 中文伴随版：去均值后的公共子空间是否跨文档稳定

## 0. Approval Snapshot

**审批状态：** 研究者已于 2026-07-31 批准科学合同，并授权实现、smoke 与
完整冻结运行；英文 canonical [Protocol](protocol.md) 已生成。未授权新训练。

**目的：** 判断 decommon 去掉均值后，actual Router input 是否仍包含跨独立
文档组复现的 centered-common top-64，以及移除这部分后，同维局部 residual
是否明显更不稳定。

**Primary anchor：**
[A15_00_02](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_02_centered_common_subspace_stability_anchor_cn.md)。

**Anchor 唯一问题：** 一个文档组拟合的去均值 top-64，能否解释另一未见文档
组的中心化变化；正交剩余中的 local top-64 是否不能同样迁移？

**检验的物理先验：** 固定减均值只删除 DC；可由统一 Gate 累积的方向应跨数据
复现，而真正组特异 residual 应更不稳定。

**核心模型项：**
$g=\mu+U_*a+\epsilon_s$ 中的 $U_*$ 与 $\epsilon_s$。

**主证伪：** 原 top-64 不超过同维 Haar/错误层，或 residual top-64 与原
top-64 同样稳定。

**实验角色：** 冻结 root-cause / metric audit；不训练 Router 或专家。

**主指标：** held-out cross-capture 超过同维 Haar q95 的差值
$\Gamma_{64}$，单位为 activation-energy fraction。

**最小设置：**

- 现有 12-layer H768、8-expert、top-1 LB 与 decommon；
- 80k 主 endpoint，40k 复现，30k 只作宏观支持；
- 128 个独立 pooled-basis 文档；
- 512 个独立 confirmation 文档，hash 固定成 8 个 shard；
- 每文档固定前 256 个有效 token；
- 所有文档与 Q1/Q2 manifests 去重；
- actual Gate input 全 12 层。

**必须运行：** centered top-64、pooled-top-64 移除后的 residual top-64、
Haar-64、错误层、8/16/32 文档 sample-size、decommon $g$/$g-c$ invariance。

**Pass：** decommon 80k 的 model-level median $\Gamma_{64}$ 文档级 95% 区间
下界大于 0，且
$\Gamma_{64}-\Gamma_{64}^{res}$ 下界大于 0；40k 点估计同向。LB 单独报告，
用于判定是否跨谱系复现。

**Fail：** 测量有效且精确，但原 top-64 不超过 null；或 residual transfer
不弱，支持“稳定维度更宽”而非“局部 residual 不稳定”。

**Insufficient：** 文档不独立、actual-input replay、center invariance、
sample-size convergence、rank、数值或 bootstrap 精度护栏失败。

**不能声称：** 稳定方向就是语义 common、residual 没有功能、稳定性改善训练，
或 decommon 没有收益由此造成。

**审批决策：** 已授权实现、smoke，并与 A15_00_03 E01 并行完整执行冻结审计。
本地 2×H100 为主执行面，远程 8×5090 只作时间/容量回退；不提交新训练。

## 1. Terminology / Definitions

| 术语 / 指标 | 普通含义 | 具体计算 | 单位 | 为什么测 | 不能证明 |
| --- | --- | --- | --- | --- | --- |
| 上游表征 $g$ | center 变换前的 Router reference | 直接 hook center 输入 | activation | 验证减均值对象 | Gate 最终接收对象 |
| actual input $r$ | Gate 真正收到的输入 | LB：$g$；decommon：$g-c$ | activation | 唯一主分析表征 | 专家输入几何 |
| shard mean $\mu_s$ | 一个文档组的平均 Router 状态 | fit 文档 token 均值 | activation | 测 DC 稳定性 | centered 子空间稳定 |
| centered top-64 | 去组均值后的最大变化方向 | source-fit covariance top-64 | 64 directions | 候选 centered-common | 语义 common |
| pooled top-64 $U_P$ | 独立文档池拟合的 top-64 | pooled-basis documents only | 64 directions | 定义可重复移除项 | 已经稳定或有用 |
| local residual-64 | 移除 $U_P$ 后的组内最大方向 | shard-fit residual covariance top-64 | 64 directions | 候选组特异 residual | 语义特异 |
| cross-capture $E_{s\to t}$ | source basis 能解释多少 target 变化 | $\|X_tU_s\|_F^2/\|X_t\|_F^2$ | energy fraction | 跨文档迁移 | 功能收益 |
| $\Gamma_{64}$ | cross-capture 超过随机方向多少 | $E_{s\to t}-q_{.95}(E_{R\to t})$ 的跨组中位 | energy fraction | **主裁定** | 因果学习机制 |
| projector overlap | 两组 64 维空间有多重合 | $\|U_s^\top U_t\|_F^2/64$ | $[0,1]$ | 辅助解释旋转 | target energy capture |
| mean dispersion | 各组均值离 pooled mean 多远 | $\|\mu_s-\mu_P\|/\|\mu_P\|$ | ratio | 防止只看高余弦 | centered stability |

## 2. Anchor Alignment

- **Decision question：** centered top-64 是否跨文档稳定，residual top-64 是否
  更不稳定。
- **Physical prior：** 平移只删除 DC；统一 Gate 只能稳定累计跨数据复现方向。
- **Core term：** $U_*$ 与 $\epsilon_s$ 的 held-out transfer contrast。
- **Falsifier：** top/residual transfer 不可区分或都落入 null。
- **Claim boundary：** 只裁定几何稳定性，不裁定语义、效用或训练收益。

## 3. Tested Hypothesis

**H1：** 去均值后的 top-64 在 decommon 80k 跨文档迁移，且其 null-relative
transfer 高于移除 pooled top-64 后的 local residual-64。

预期模式：

1. raw shard means 高对齐，但该量只确认 DC；
2. centered top-64 的 $\Gamma_{64}>0$；
3. residual $\Gamma_{64}^{res}$ 更低或不可区分于 0；
4. 40k 同向，LB 至少给出描述性复现或明确谱系差异。

## 4. Rival Explanations

| Rival | 预测 | 区分方法 | 指标最多能回答 |
| --- | --- | --- | --- |
| R0 有限样本 PCA | 文档少时看似不同，样本增加后收敛 | 8/16/32 fit-doc curve、half split | 排除注册样本量噪声 |
| R1 稳定空间更宽 | top 与 residual 都超过 null | 同维 residual cross-capture | 说明 64 维切分太窄 |
| R2 只是大特征值 | top capture 高是 target 谱陡 | 同维 Haar q95、projector overlap | 排除随机同维能量捕获 |
| R3 任意层共享 | 错误层也同样迁移 | layer+$6$ wrong-layer basis | 检查本层特异性 |
| R4 文档主题混合 | hash shard 恰好主题不同 | 多组交叉、文档 bootstrap、实际 minibatch sensitivity | 限制单一分组偶然性 |
| R5 稳定但无功能 | 几何 Pass，Q2 仍 Fail | 本实验不能区分 | 保留为下一层 rival |

## 5. Data / Model / Algorithm / Objective

### 5.1 模型与 checkpoints

| Lineage | actual input | Checkpoints | 作用 |
| --- | --- | --- | --- |
| decommon | $r=g-c$，冻结 running center | 30k/40k/80k | 主机制对象 |
| LB | $r=g$，center off、LB 训练 | 30k/40k/80k | 跨谱系描述性对照 |

checkpoint roots、Gate shape、expert ordering 与 A15_00 E01 一致。执行前重新记录
sha256/config；LB 与 decommon 不是 center/LB 的单变量因果对照。

### 5.2 文档分离

从 DCLM held-out document store 固定选择 640 个已批准的新文档，每篇前 256 个
有效 token：

| Split | 文档数 | 用途 | 禁止用途 |
| --- | ---: | --- | --- |
| pooled-basis | 128 | 估计 $\mu_P,U_P$ | confirmation verdict |
| confirmation | 512 | 8 shards × 64 docs | 调整 estimator |

选择仅由 document hash 决定，并与 Q1 calibration/evaluation、Q2
operationalization/fit/validation/final manifests 去重。每个 confirmation
shard 再 hash 分为 32 fit + 32 evaluation documents。

若没有 640 个合格且不重叠文档，必须在任何 activation metric 前提交 amendment；
不得静默复用 Q2 final-test 或减少 shard。

### 5.3 中心化与 cross-fit

- source basis 只读取 source-fit documents；
- target capture 只读取 target-evaluation documents；
- target evaluation 用 target-fit mean 中心化；
- pooled projector 只由 pooled-basis split 构造；
- 同一 token 不同时承担 basis fit 与 verdict evaluation。

## 6. Conditions, Seeds, And Checkpoints

| Item | Anchor clause | Rival | 为什么需要 | 证据级别 | Pass | Fail | Insufficient | 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actual-input replay | 对象正确 | hook 错位 | 保证测 Gate 输入 | hard guard | logits/top1 重放 | mismatch | 未定位 | replay table |
| $g$ vs $g-c$ | 平移不改 covariance | center 污染 | 审计 decommon 指标 | hard guard | 误差在数值容差 | 系统差异 | center 非固定 | invariance table |
| centered top-64 | H1 | R0/R2 | 主候选 | primary | $\Gamma>0$ | $\Gamma\le0$ | 区间宽 | transfer heatmap |
| residual top-64 | residual 不稳定 | R1 | 主对比 | primary | 明显更低 | 同样稳定 | rank 不足 | paired contrast |
| Haar-64 ×256 | 方向零假设 | R2 | 排除同维随机 capture | hard control | 真 basis 超过 | 不超过 | null 数值失败 | null table |
| wrong layer +6 | 本层性 | R3 | 排除任意层 | control | 真层更高 | 错层相同 | 坐标失败 | layer table |
| 8/16/32 docs | 样本收敛 | R0 | 排除局部 PCA 噪声 | guard | 方向稳定 | 反转 | 无收敛 | curve |
| logical-batch regrouping | batch composition | R4 | 对应研究者最初 batch 假设 | secondary | 与主结论相容或给出条件差异 | 方向相反则限制边界 | batch manifest 失败 | sensitivity table |
| 30/40/80k | 状态复现 | checkpoint 特异 | 描述宏观变化 | support | 40/80 同向 | 反向 | 不兼容 | checkpoint table |

随机 seed、document hashes 与 Haar seeds 必须在读取主结果前写入 manifest。
checkpoint 不作为独立 seed；12 层也不伪装成 12 个独立重复。

## 7. Primary Metric

对 target shard $t$ 和 source shard $s$：

$$
E_{\ell,s\rightarrow t,64}
=\frac{\|X_{\ell,t}^{eval}U_{\ell,s,64}\|_F^2}
{\|X_{\ell,t}^{eval}\|_F^2}.
$$

同一 target 上生成 full-space Haar-64；residual 条件在 $U_P^\perp$ 内生成
Haar-64。定义

$$
\Gamma_{\ell,64}
=\operatorname{median}_{s\ne t}
\left[E_{\ell,s\rightarrow t,64}
-q_{0.95}(E_{\ell,R\rightarrow t,64})\right].
$$

模型级主摘要是 12 层 $\Gamma_{\ell,64}$ 的中位数，但完整逐层结果必须交付。
配对文档-block bootstrap 同时重采样 source/target documents，报告
$\Gamma_{64}$ 与
$\Gamma_{64}-\Gamma_{64}^{res}$ 的 95% 区间。

**为什么能裁定：** 一个方向只有在未见文档上捕获超过随机方向的 centered
variation，才能称为跨文档稳定坐标。

**假阳性代价：** 把有限样本 PCA 当稳定 common，会错误启动稳定性干预训练。
因此使用 q95、sample-size 与独立 pooled split。

## 8. Secondary Metrics

1. 分别报告上游 $g$ 与 actual input $r=g-c$ 的 shard-mean cosine、
   $\delta\mu_s=\mu_s-\mu_P$ cosine 与 mean dispersion；
2. projector overlap $O_{s,t,64}$；
3. $k\in\{16,32,128,256\}$ transfer profile；
4. pooled basis 对每个 shard 的 energy capture；
5. 将同一文档集合按确定性逻辑 dataloader batch 重新分组，作为训练时
   batch-composition sensitivity；它不替代 IID hash-shard 主 verdict；
6. fine eigenvalue spectrum，仅解释稳定维度，不替代 transfer。

## 9. Known Good / Known Bad / Known Confusing Cases

- **Known good：** source 与 target 使用同一 fit/eval 分布时，within-shard
  half-split 应明显高于 Haar。
- **Known bad：** 独立 Haar source basis 不应系统超过自身 null。
- **Known confusing：** 相邻 eigenvalues 近简并时单根 eigenvector 会旋转；
  因此只裁定 projector/capture，不裁定向量符号或顺序。
- **Known confusing：** raw batch-mean cosine 可因巨大 $\mu_P$ 接近 1；必须
  同时报告 mean dispersion 和 centered subspace。
- **Known confusing：** residual capture 分母是 residual energy，不与 raw
  total-energy response混用。

## 10. Stage-Level Profiling Plan

| Stage | 局部问题 | 输入 | Pass / fail / unclear | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S0 | provenance 可比吗 | checkpoints/doc hashes | compatible / amend / stop | manifests | S1 |
| S1 | hook 与中心正确吗 | $g,r,z$ | replay + invariance / stop | replay log | shared cache |
| S2 | 文档数足够吗 | pooled/shard fits | within-shard curve stable / unclear | sample curve | S3 |
| S3 | top/residual 能迁移吗 | cross-capture | typed H1/R0/R1 | transfer tensors | S4 |
| S4 | 超过方向/错误层吗 | nulls | pass/fail/insufficient | null ledger | S5 |
| S5 | 结论可复述吗 | all evidence | one typed verdict | figures/tables | result record |

S1 的只读 activation cache 同时交给 A15_00_03；S2--S5 两个分析进程可并行。

**资源预案：** 若批准后调用 8×5090，六个 model×checkpoint endpoint 最多各占
一张卡并行做冻结前向，剩余卡不强行占用；basis/null/bootstrap 在共享只读
cache 上并行分析。若实际卡数更少，只降低并行度，不减少文档、层或条件。本
阶段没有 optimizer、backward 或新训练。

## 11. Algorithm Specification

**input：** frozen checkpoints、640-document manifest、actual-input hooks。

**parameters：** $k=64$ primary；Haar orientations 256；fit docs
$\{8,16,32\}$；document bootstrap 2000。

**steps：**

1. 验证 checkpoints、数据去重与 actual-input no-op。
2. 提取所有 endpoint/layer 的 $g,r$；保存 token/document mapping。
3. 用 pooled-basis documents 拟合 $\mu_P,U_P$。
4. 每个 shard 用 fit documents 拟合 $\mu_s,U_{s,64}$。
5. 跨所有有序 $s\ne t$ 在 target-evaluation documents 上测 capture。
6. 对 $(I-P_P)x$ 重复 residual-64 流程。
7. 生成 full/complement Haar 与 wrong-layer 对照。
8. 运行 sample-size curve、bootstrap、聚合和图像审计。

**outputs：** checkpoint/data manifests、transfer tensor、null ledger、
mean/invariance table、layer/checkpoint tables、central figures、typed verdict。

**failure reasons：** object、data leakage、finite-sample、rank、numerical、
null、precision、checkpoint incompatibility。

### 11.1 Central figure contract

- **图名：** `centered_common_vs_residual_transfer.png`
- **Anchor / Protocol 问题：** 去均值 top-64 是否跨文档迁移，且强于同维
  residual top-64。
- **Metric / unit：** $\Gamma_{\ell,64}$ 与
  $\Gamma_{\ell,64}^{res}$；activation-energy fraction above Haar q95。
- **数据：** locked confirmation evaluation documents；basis 只来自 pooled
  或 source-fit documents。
- **聚合：** ordered shard pairs 中位数；document-block bootstrap 95% CI；
  12 层不作独立 seeds。
- **坐标：** x=layer 1--12；y=null-relative cross-capture；颜色=top/residual；
  facet=lineage×checkpoint。
- **H1 预期：** top 曲线高于 0 且高于 residual。
- **削弱预期：** 两者都近 0，或 residual 同样高。
- **它裁定：** centered-common/local-residual 几何稳定性分解。
- **Observed：** pending。
- **允许 claim：** 仅跨文档几何迁移。
- **不能证明：** 语义、Gate 使用、功能或训练收益。

## 12. Success / Failure / Insufficient Evidence

- **Pass — stable-centered/common-local-residual split：** 注册的 top-64 transfer
  和 paired top-vs-residual gap 均通过。
- **Fail-R0 — no stable centered top：** 原 top-64 不超过 null。
- **Fail-R1 — stable structure broader than 64：** 原 top 与 residual 都稳定，
  paired gap 不成立。
- **Lineage-conditioned：** decommon 与 LB 方向不同；只分别报告。
- **Insufficient：** 任一 hard guard 失败或区间不能区分上述类别。

不设置 25%、10% 或层数硬门槛；q95 与 bootstrap 只判断能否区别 null/zero，
逐层效应大小全部报告。

## 13. What This Cannot Claim

本实验不能证明稳定方向：

1. 是语言语义或功能 common；
2. 被 Gate 使用；
3. 对专家训练有益；
4. 导致 decommon 的 loss/负载结果；
5. 应该被删除；
6. 在其他模型、尺度或数据上普遍存在。

Gate 是否使用由并行 A15_00_03 裁定；训练因果需要之后的新 anchor。

## 14. Review Notes And Protocol Changes

### 已批准的研究判断

1. 以“跨文档 held-out capture”而不是 batch-mean cosine 作为主指标。
2. top-64 为主维度，16/32/128/256 只作敏感性分析。
3. 使用 128 pooled + 512 confirmation 新文档及 8×64 shard 结构。
4. “原 top 稳定但 residual 也稳定”判为 R1，而不是强行支持 H1。
5. LB/decommon 分开裁定，不作 center/LB 因果比较。
6. 本 Protocol 完全冻结，不启动 8×5090 新训练。

研究者批准记录：2026-07-31 接受上述六点；没有修改科学条件。

批准后变更：生成英文 canonical `protocol.md`，并将本中文伴随版标记为已批准；
研究者随后于 2026-07-31 授权实现、smoke 与完整冻结运行。科学条件未改变。
