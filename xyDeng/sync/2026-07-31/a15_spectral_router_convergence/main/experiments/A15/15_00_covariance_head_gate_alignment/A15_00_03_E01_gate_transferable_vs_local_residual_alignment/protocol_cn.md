---
experiment_id: A15_00_03_E01_gate_transferable_vs_local_residual_alignment
status: approved_for_full_execution
created: 2026-07-31
approval_date: 2026-07-31
primary_anchor: 15_00_03_gate_transferable_vs_local_residual_alignment
canonical_protocol: protocol.md
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: local-2xh100-remote-8x5090-fallback
---

# Protocol 中文伴随版：Gate 偏 pooled common 还是 shard-local residual

## 0. Approval Snapshot

**审批状态：** 研究者已于 2026-07-31 批准科学合同，并授权实现、smoke 与
完整冻结运行；英文 canonical [Protocol](protocol.md) 已生成。未授权新训练。

**目的：** 在完全去除输入能量差异后，判断 decommon Gate 是否对独立 pooled
centered-common candidate 比对同维 shard-local residual 有更高权重增益和
native route dependence。

**Primary anchor：**
[A15_00_03](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_03_gate_transferable_vs_local_residual_alignment_anchor_cn.md)。

**Anchor 唯一问题：** Gate 在 pooled candidate 与 local residual 之间偏向
哪一个；30k/40k/80k 的保存状态是否支持这种偏好已形成或变化？

**检验的物理先验：** 单一线性 Gate 更容易累计跨数据重复方向；随 shard 旋转
的方向可能在更新中平均抵消。

**核心模型项：** $\bar W=C_EW$ 对 $U_P$ 与 $U_{L_s}$ 的 equal-energy gain。

**主证伪：** raw response 偏 pooled，但 equal-energy
$B_{P:L}$ 不偏 pooled；或 native route 对 local residual 的同维去带影响
不弱于 pooled。

**实验角色：** 冻结 root-cause audit + saved-checkpoint macro trajectory；
不等于逐步训练动力学。

**主指标：**

$$
B_{\ell,P:L}
=\log\frac{G_\ell(W,U_P)+\epsilon}
{\operatorname{median}_sG_\ell(W,U_{L_s})+\epsilon}.
$$

它是 log equal-energy gain ratio，无量纲。

**最小设置：**

- 与 A15_00_02 完全相同的 LB/decommon 30k/40k/80k checkpoints；
- 完全相同的 128 pooled + 512 confirmation DCLM documents；
- 完全相同的 actual-input cache、pooled $U_P$ 和 local residual bases；
- 80k 主 endpoint、40k 复现、30k 宏观支持；
- 全 12 层、64 对 64 维公平比较。

**必须运行：** $G$、$B_{P:L}$、equal-rank native-margin support 与辅助
route flip、256 个保持 Gate 奇异值的 orientation null、full/complement
Haar、错误层、完整
$W_{30/40/80}\times U_{30/40/80}$ crossing。

**Pass：** decommon 80k 的 model-level median $B_{P:L}$ 文档/basis bootstrap
95% 下界大于 0，并超过 orientation-null q95；pooled-vs-local 的同维
native-margin support 差值下界也大于 0；40k 点估计同向。flip 只作辅助。

**Fail：** 测量有效且精确，但 $B_{P:L}$ 不偏 pooled、不过 null，或 route-use
对比不支持 pooled。

**Insufficient：** actual input、basis cross-fit、orientation null、
equal-rank ablation、checkpoint crossing、route replay 或 precision 护栏失败。

**不能声称：** pooled candidate 已经被证明稳定、偏好有益、local residual
无语义，或该几何解释了训练 loss。只有 A15_00_02 同时 Pass，才允许使用
“Gate 偏稳定 common”的联合语言。

**审批决策：** 已授权实现、smoke，并与 A15_00_02 E01 并行完整执行冻结审计。
本地 2×H100 为主执行面，远程 8×5090 只作时间/容量回退；不启动新训练。

## 1. Terminology / Definitions

| 术语 / 指标 | 普通含义 | 具体计算 | 单位 | 为什么测 | 不能证明 |
| --- | --- | --- | --- | --- | --- |
| expert-relative Gate $\bar W$ | 只保留决定专家比较的权重 | $(I-\mathbf1\mathbf1^\top/E)W$ | weight | 去掉共同 logit 平移 | 功能效用 |
| pooled candidate $U_P$ | 独立 pooled 文档的 centered top-64 | A15_00_02 pooled split | 64 directions | 候选 common | 已跨文档稳定 |
| local residual $U_{L_s}$ | 去掉 $U_P$ 后 shard-fit top-64 | cross-fitted residual PCA | 64 directions | 候选局部 residual | 语义特异 |
| equal-energy gain $G$ | 单位方向能量造成的专家分数差强度 | $\|\bar WU\|_F^2/64$ | logit²/activation² | 去掉 eigenvalue 放大 | token 实际使用 |
| $B_{P:L}$ | pooled gain 相对 local gain | $\log(G_P/\operatorname{median}G_L)$ | log ratio | **主 Gate 偏好** | 训练收益 |
| route flip | 去掉该 64 维后 token 是否换专家 | winner identity change rate | token fraction | 直观辅助 | 换路由有益 |
| margin support | 该 64 维对原 winner 优势的贡献 | native margin before-after ablation | logit/token | **route-use 主量** | 专家正确 |
| orientation null | 保持 Gate 奇异值但随机转方向 | Haar rotate right singular space | null distribution | 排除 Gate 谱本身 | 所有训练 rival |
| fixed-basis Gate effect | 固定 basis 换 checkpoint $W$ | $B(W_b,U_a)-B(W_a,U_a)$ | log-ratio change | 宏观权重变化 | 每步 gradient |

## 2. Anchor Alignment

- **Decision question：** equal-energy Gate 是否偏 pooled candidate 而非 local
  residual。
- **Physical prior：** 跨数据重复方向提供一致累计，局部旋转方向平均抵消。
- **Core term：** $G(W,U_P)$ 与 $G(W,U_{L_s})$。
- **Falsifier：** $B_{P:L}$ 不过 zero/null，或 route-use 不支持。
- **Claim boundary：** Gate 偏好与 route use，不是稳定性、功能或训练收益。

## 3. Tested Hypothesis

**H1：** decommon 80k Gate 在同维、等能条件下偏 pooled candidate；40k 同向。
pooled candidate 的 equal-rank ablation 移除更多 native-winner margin
support；flip 同向只作辅助。LB 若同向，说明现象不只存在于 running-center
谱系。

checkpoint 轨迹只回答：保存的 Gate 权重和 basis 如何共同改变
$B_{P:L}$。本 Protocol 不要求单调增强，因为 Q1 已显示 head preference
可能早期形成后被稀释。

## 4. Rival Explanations

| Rival | 预测 | 区分方法 | 指标最多能回答 |
| --- | --- | --- | --- |
| R0 只有输入能量 | raw response 高，$G/B$ 不高 | equal-energy $G$ | 排除 eigenvalue 放大 |
| R1 Gate 本身各向异性 | 任意方向都可得到极端比值 | singular-value-preserving null | 判断与方向随机的差别 |
| R2 pooled estimator 特权 | Haar/wrong-layer pooled 也同样高 | full/complement Haar、wrong layer | 排除注册 estimator 捷径 |
| R3 只有 basis 漂移 | endpoint 变，固定 basis 的 $W$ 效应不变 | $3\times3$ crossing | 保存区间分解 |
| R4 只在 logit 不在 route | $B>0$，margin support 不更强 | equal-rank ablation | 区分权重几何与当前使用 |
| R5 pooled 方向功能有益 | 所有本实验指标可 Pass | 本实验不能区分 | 保留功能 rival |

## 5. Data / Model / Algorithm / Objective

### 5.1 共享数据与 basis 合同

本实验不得另选 documents 或重估另一套 basis。它只读
A15_00_02 E01 在查看主结果前冻结的：

- 128 pooled-basis documents；
- 8×64 confirmation documents；
- 每 shard 的 32 fit / 32 evaluation split；
- actual-input cache；
- 每 checkpoint/layer 的 $U_P$ 与 $U_{L_s}$；
- checkpoint/data/basis hashes。

若 A15_00_02 的 cache/basis hard guard 失败，本实验也停止；但 A15_00_02 的
科学 verdict 不需要先产生。

### 5.2 模型与 checkpoints

| Lineage | Checkpoints | 主次 | 能回答多少 |
| --- | --- | --- | --- |
| decommon | 30k/40k/80k | 80k 主、40k 复现、30k support | running-center 谱系的 pooled/local 偏好 |
| LB | 30k/40k/80k | 同上 | 跨谱系描述，不是 center/LB 因果 |

### 5.3 actual-input 和 DC

所有 band contribution 使用
$x=r-\mu^{fit}$；DC 项 $C_EW\mu^{fit}$ 单独报告，不进入任何 64 维 band。
decommon 同时捕获 $g$ 与 $r=g-c$，复用 A15_00_02 invariance guard。

## 6. Conditions, Seeds, And Checkpoints

| Item | Anchor clause | Rival | 为什么需要 | 证据级别 | Pass | Fail | Insufficient | 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled $G_P$ | H1 | R0 | 主候选增益 | primary | 高于 local/null | 不高 | basis 失败 | gain table |
| local $G_{L_s}$ | H1 | R0/R2 | 同维对手 | primary | pooled gap 正 | gap 非正 | local rank 不足 | shard table |
| $B_{P:L}$ | H1 | R0/R1 | 主裁定 | primary | CI>0 且过 q95 | 不过门 | CI 宽 | layer heatmap |
| pooled ablation | native use | R4 | 看实际 margin 支撑 | primary support | margin support 更大 | 不更大 | replay 失败 | route table |
| local ablation | native use | R4 | 公平 64 维比较 | primary support | margin paired gap 正 | 非正 | rank/route 失败 | paired plot |
| orientation null ×256 | 方向特异 | R1 | 保持 Gate 奇异值 | hard control | 真值过 q95 | 不过 | SVD 重建失败 | null ledger |
| full/complement Haar | estimator 对照 | R2 | 排除任意 64 维 | control | 真 basis 更高 | 随机相同 | seed/null 失败 | null table |
| wrong layer +6 | 本层性 | R2 | 排除共享错误层 | control | target 更高 | 错层相同 | 坐标失败 | layer table |
| 30/40/80 crossing | 权重 vs basis | R3 | 宏观分解 | secondary | typed effect | rival | incompatible | crossing matrix |

Haar/null seeds、basis hashes 和 checkpoint ordering 在读取 $B$ 前冻结。层不是
独立 seed，checkpoint 也不是独立 seed。

## 7. Primary Metric

对每个 layer、checkpoint：

$$
G_\ell(W,U)=\frac1{64}\|C_EW_\ell U\|_F^2,
$$

$$
B_{\ell,P:L}
=\log\frac{G_\ell(W,U_P)+\epsilon}
{\operatorname{median}_{s=1}^{8}G_\ell(W,U_{L_s})+\epsilon}.
$$

令 $P_A=U_AU_A^\top$，并只移除 centered contribution：

$$
z_{\ell}^{(-A)}
=z_\ell-C_EW_\ell P_Ax_\ell,
$$

$$
D_{\ell,A}
=\mathbb E\!\left[
m_{\mathrm{native}}(z_\ell)
-m_{\mathrm{native}}(z_\ell^{(-A)})
\right],
\qquad
\Delta D_{\ell,P:L}
=D_{\ell,P}-\operatorname{median}_sD_{\ell,L_s}.
$$

$D$ 的单位是 logit/token；$\Delta D>0$ 表示 pooled candidate 对原生 winner
的 margin 支撑更强。它不表示该 winner 对语言建模更好。

模型级摘要是 12 层中位数；basis/document block bootstrap 同时重采样 pooled
documents 与 shard-fit documents。orientation null 固定 Gate 非零奇异值，
随机旋转其 input right-singular directions，每个 null 重算完整 $B_{P:L}$。

**为什么能裁定：** $G$ 完全不含 covariance eigenvalue，因此
$B_{P:L}>0$ 表示 Gate 权重对 pooled candidate 每方向施加更高 expert-relative
平方增益。

**假阳性代价：** 若只看 raw response，会把“输入更大”误写成“Gate 学到”；
若只看 $B$ 不看 route ablation，会把不影响 winner 的几何偏好写成当前使用。

## 8. Secondary Metrics

1. pooled/local raw response $V$：只显示真实 token 总贡献；
2. equal-rank native-winner margin support 与辅助 top-1 flip；
3. DC expert-bias norm $\|C_EW\mu\|$；
4. 30k/40k/80k $W\times U$ crossing；
5. 固定 basis 的 $\Delta_WB_{P:L}$ 与固定 Gate 的 $\Delta_UB_{P:L}$；
6. pooled/local fine gain profile，仅作定位；
7. 使用 A15_00_02 的 logical-batch regrouping 重算 local residual
   $B_{P:L}$，只作 batch-composition sensitivity；
8. LB/decommon 差值只作描述，不作 center 因果量。

## 9. Known Good / Known Bad / Known Confusing Cases

- **Known good：** 用 $U$ 替换为 Gate right-singular top directions 时，$G$
  应显著高于其正交方向，验证实现敏感。
- **Known bad：** orientation-null percentile 应近似均匀，SVD 重建相对误差
  必须在数值容差内。
- **Known confusing：** $G_P>G_L$ 不保证 $V_P>V_L$；前者是选择性，后者同时
  含输入能量。
- **Known confusing：** 多个 band 的 route effects 不可相加，因为 winner
  是非线性 argmax。
- **Known confusing：** A15_00_02 Fail 不会使 $B_{P:L}$ 无法计算，只会禁止
  把 pooled/local 命名为稳定/不稳定。

## 10. Stage-Level Profiling Plan

| Stage | 局部问题 | 输入 | Pass / fail / unclear | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S0 | 共享对象一致吗 | A02 manifests/cache | hashes match / stop | shared manifest | S1 |
| S1 | $G/B$ 实现正确吗 | known good/bad | sensitive + null valid / stop | unit ledger | S2 |
| S2 | endpoint 偏 pooled 吗 | $G,B$ | H1/R0/R1 typed | gain tensors | S3 |
| S3 | native route 使用吗 | margin support + flip | H1/R4 typed | route ledger | S4 |
| S4 | 变化来自 $W$ 还是 $U$ | crossings | weight/basis/mixed | crossing table | S5 |
| S5 | 联合语言是否合法 | A02 verdict + own verdict | combined / separate | verdict.json | result record |

S2--S4 可与 A15_00_02 的 stability analysis 并行；S5 才读取另一实验的 typed
verdict。

**资源预案：** 本实验不重复模型前向，直接读取 A15_00_02 S1 的共享
activation/Gate cache。批准后可与其 S2--S5 同时运行；矩阵、null 和 route
ablation 可分配到空闲 5090，但不需要新的 8 卡训练作业。

## 11. Algorithm Specification

**input：** shared frozen caches/bases、Gate matrices、held-out evaluation docs。

**parameters：** 64 dimensions；256 orientation nulls；2000 paired
document/basis bootstraps；$\epsilon$ 只防止数值零并在运行前冻结。

**steps：**

1. 校验 shared manifests、basis orthogonality 和 Gate shapes。
2. 对 pooled、8 个 local residual、Haar、wrong-layer 统一计算 $G$。
3. 计算逐层 $B_{P:L}$ 与 orientation-null percentile。
4. 在相同 evaluation tokens 上做 centered equal-rank ablation。
5. 计算 flip、margin、raw response 和 DC。
6. 对 30/40/80 完成全部 $W\times U$ crossings。
7. bootstrap、聚合、生成 central figure 和 typed verdict。

**outputs：** gain/route/null/crossing tables、layer heatmap、checkpoint
trajectory、typed standalone verdict、conditional joint verdict。

**failure reasons：** shared-object、basis、SVD/null、route replay、precision、
checkpoint coordinate、bootstrap precision。

### 11.1 Central figure contract

- **图名：** `pooled_vs_local_gate_preference_and_use.png`
- **Anchor / Protocol 问题：** Gate 是否在等能后偏 pooled candidate，且
  native route 是否也更依赖它。
- **Metric / unit：** 左面板 $B_{\ell,P:L}$（log ratio）；右面板
  pooled-minus-local margin-support difference（logit/token）；flip difference
  （token fraction）只作辅助。
- **数据：** locked shared bases、Gate matrices 与 confirmation evaluation
  documents。
- **聚合：** 8 个 local bases 中位数；basis/document bootstrap 95% CI；
  model-level median 另列，不把层当 seed。
- **坐标：** x=layer；y=metric；颜色=checkpoint；facet=lineage 与 metric。
- **H1 预期：** $B>0$ 且 margin-support difference $>0$。
- **削弱预期：** 只有 raw response 高、$B\le0$；或 $B>0$ 但
  margin-support difference 不为正。
- **它裁定：** pooled-vs-local Gate preference 与当前使用。
- **Observed：** pending。
- **允许 claim：** 等能 Gate 几何偏好及 native dependence。
- **不能证明：** pooled 稳定、语义、功能或训练收益。

## 12. Success / Failure / Insufficient Evidence

- **Pass — pooled alignment with native use：** $B$ 与 margin-support 两道门
  都过。
- **Alignment-only：** $B$ 过门但 margin-support 不过；H1 整体 Fail，只保留权重
  几何偏好。
- **Energy-only：** raw response 偏 pooled，$B$ 不过门。
- **Local-not-weaker：** local residual 的 $G$ 或 margin-support 不弱于
  pooled。
- **Lineage-conditioned：** LB/decommon 方向不同，分别报告。
- **Insufficient：** hard guard 或精度失败。

只有本实验 Pass 且 A15_00_02 Pass，联合结论才是：
“decommon Gate 更偏向跨文档稳定 centered common，而不是较不稳定 local
residual。”

## 13. What This Cannot Claim

本实验不能证明：

1. pooled candidate 一定跨文档稳定；
2. pooled preference 对 loss 有益或有害；
3. local residual 不含语义；
4. residual instability 导致优化失败；
5. decommon 与 LB 差异由 center 单独造成；
6. checkpoint crossing 等于在线 gradient dynamics。

如果两个 subanchor 同时 Pass，下一份新 anchor 才能讨论匹配的稳定性干预训练；
不能直接复活固定 M/T/N dispatch。

## 14. Review Notes And Protocol Changes

### 已批准的研究判断

1. pooled/local 都固定为 64 维，先回答方向偏好而非覆盖全部 residual。
2. 使用 $B_{P:L}$ + equal-rank route effect 两道门；只过 $B$ 不算完整 Pass。
3. orientation null 保持 Gate 奇异值，Haar basis 另测 estimator。
4. 80k 为主、40k 复现、30k 只作 macro trajectory。
5. A15_00_02 只控制联合命名，不阻塞本实验并行计算。
6. 两份 Protocol 都是冻结审计，不预先启动 8×5090 训练。

研究者批准记录：2026-07-31 接受上述六点；没有修改科学条件。

批准后变更：生成英文 canonical `protocol.md`，并将本中文伴随版标记为已批准；
研究者随后于 2026-07-31 授权实现、smoke 与完整冻结运行。科学条件未改变。
