---
experiment_id: A15_00_E02_early_head_alignment_onset
status: approved_for_implementation_smoke_and_full_frozen_audit
execution_status: completed_early_onset_pass_progressive_strengthening_fail
result_summary: summary_cn.md
created: 2026-07-30
updated: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
canonical_protocol: protocol.md
execution_scope: existing_checkpoint_frozen_audit_only
---

# Protocol：A15_00_E02 线性 Gate 的 head 偏置何时形成

## 0. 审批快照

**审批状态：** 研究者于 2026-07-30 批准实现、smoke 和完整冻结审计；不需要、
也不授权新训练。

**唯一决策问题：** 在 LB 与 batch-gradient 两条训练谱系中，去掉输入奇异值/
方差增幅后，线性 Gate 是否在 10k 已经明显更对齐 head；10k--20k 与
20k--30k 的 Gate 变化是在继续增强、维持还是稀释这种偏置？

本实验只把 E01 的观察窗口前移。不再处理 decommon，也不测试只用
middle/tail 分发的功能效果。

## 1. 术语和指标先解释

| 术语 | 直白含义 | 判断作用 | 不能证明 |
| --- | --- | --- | --- |
| 实际 Gate 输入 $r_\ell$ | 第 $\ell$ 层直接送入 `mlp.gate` 的张量 | 只用它拟合 covariance 基底 | expert 输入的几何或语义 |
| head $H$ | $r_\ell$ 中方差最大的 1--64 维 | 与 middle/tail 比较 | “通用语义” |
| middle $M$ | 第 65--320 维 | 中段对照 | 功能价值 |
| tail $T$ | 第 321--768 维 | 尾段对照 | 没有功能 |
| 细频带 $F_j$ | 连续 64 个 covariance eigenvectors | 查看 coarse 平均是否隐藏结构 | 任务定义的“频率” |
| 等能 Gate 增益 $G_A$ | 给子空间 $A$ 同样的输入能量时，Gate 产生多少专家相对 logit 差异 | 去掉大方差机械加成 | token 上真实使用量或 loss 收益 |
| endpoint 对比 $B$ | head 增益与 middle/tail 增益的 log 比值 | 判断训练好的 Gate 看哪一段 | 偏置何时形成 |
| 更新方向 $B^{update}$ | 两个保存点之间净 Gate 位移朝向哪个谱带 | 描述净权重位移 | endpoint 是否变得更 head、逐步梯度 |
| 固定基底 Gate 效应 $\Delta_WB$ | 不让表征基底变化，只换 Gate 权重后，head 对比改变多少 | 判断 Gate 训练是否增强/稀释 head 偏置 | 优化器因果机制 |
| 基底效应 $\Delta_UB$ | 不换 Gate 权重，只换表征基底后的变化 | 排除 representation drift | 表征漂移原因 |

## 2. 与 Q1 的关系

Q1 的核心不是“大方差方向是否产生更大的 raw logit”，而是：把各方向输入
能量拉平以后，训练出来的线性 Gate 本身是否更朝向大方差方向。E01 已在
30k/40k/80k 看到 endpoint head alignment，但不能判断它何时形成。E02 查看
两个谱系最早共同保存的 10k/20k/30k。

**物理先验：** 大方差方向若不被目标或优化抵消，会提供更强训练信号，所以
head 偏置可能在训练早期迅速形成，此后基本固定。

**最强竞争解释：** 30k 的 head alignment 是更晚才由 Gate 权重或表征基底
漂移形成；或者早期 Gate 更新根本没有持续偏向 head。

**证伪条件：** 10k 的 head 对比无法区别于零/随机方向，或两个早期区间的
固定基底 Gate 效应并不一致增强 head。

## 3. 假设与可报告结果类型

### H1-early：10k 前已经形成 head alignment

对每条谱系，10k 的 $B_{H:M}$、$B_{H:T}$ 都为正，其 paired basis bootstrap
下界高于零，观测中位数也超过保持 Gate 奇异值不变的随机方向 q95。若成立，
只能把形成时间界定为 **10k 之前**，不能知道具体 step。

### H1-progressive：10k 后的 Gate 更新继续增强 head alignment

在 10k--20k 和 20k--30k 两段中，$\Delta_WB_{H:M}$ 与
$\Delta_WB_{H:T}$ 的区间均高于零。$B^{update}$ 必须报告，但它只说明净位移
朝向哪里，不能替代 $\Delta_WB$。

### Rival R-maintain/dilute：偏置更早形成，随后维持或稀释

10k 已偏 head，但后两段 $\Delta_WB$ 接近零、两个对比方向不一致，或为负。
这说明训练倾向在更早阶段形成，并非 10k 后持续加强。

### Rival R-drift：主要是表征基底变化

endpoint 变化主要对应 $\Delta_UB$，而 $\Delta_WB$ 无法确定或方向相反。

### 谱系依赖

LB 与 batch-gradient 分开裁定。如果两者不同，报告 lineage-conditioned，
不能平均成“所有 Router”的规律。

## 4. 模型和 checkpoint

| 谱系 | 训练配置 | 冻结 checkpoint |
| --- | --- | --- |
| LB | 不做中心化；线性 Gate；$\lambda_{LB}=0.01$ | 10k、20k、30k |
| batch-gradient | running input center；`batch_only` center gradient；无 LB | 10k、20k、30k |

路径：

- LB：`/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-linear_nocenter_lb001_8gpu-center_off-gate_off-acp_off-lb_0.01-linear/checkpoints`
- batch-gradient：`/mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-moe_linear_running_center_batchgrad_8gpu-center_running-gate_off-acp_off-lb_0-linear/checkpoints`

两者均为 12 层 MoE、hidden width 768、8 experts、top-1。

**重要边界：** batch-gradient 不是“只改梯度”的纯因果对照。训练时
`batch_only` 的可微 batch center 也进入 forward center。因此两条谱系若不同，
只能说真实训练配置不同；不能把差异单独归因于是否对 center 求梯度。冻结审计
仍然直接 hook Gate pre-input，所以每条谱系内部的频带测量对象是正确的。

## 5. 数据和双分辨率

完全复用 E01 的确定性数据合同：

- 32 个互不重叠的 DCLM 训练 sequences，每个 256 tokens，用于拟合基底；
- 64 个 held-out DCLM documents，每个 256 tokens，用于实际响应和 route
  ablation；
- 六个 endpoint 使用完全相同的 token IDs、顺序和 mask。

令实际 Gate 输入 covariance 的 eigenvectors 按 eigenvalue 从大到小排序：

$$
F_j=[64(j-1)+1,64j],\quad j=1,\ldots,12,
$$

$$
H=F_1,\qquad M=F_2\cup F_3\cup F_4\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

读结果后不能重新挑频带。

## 6. 主指标

先去掉所有专家共同增加的 logit，因为共同平移不会改变路由：

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad \bar W_\ell=C_EW_\ell.
$$

对维数为 $d_A$ 的频带 $A$，

$$
G_{\ell,A}(W,U)=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2.
$$

单位是 `logit² / activation² / direction`。它表示每个方向获得同样能量时的
Gate 选择性，因此去掉了“head 输入方差更大”的机械加成；但它不表示该频带
在真实 token 上更常被用，也不表示更有功能。

coarse endpoint 对比为：

$$
B_{\ell,H:M}=\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},\qquad
B_{\ell,H:T}=\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon},
\quad\epsilon=10^{-12}.
$$

$B>0$ 表示等能后仍更偏 head；$\exp(B)$ 是 head/middle 或 head/tail 的增益
倍数。除 coarse 外必须交付十二个 $G_{F_j}$。

对区间 $a\to b$，令 $\Delta W=W_b-W_a$：

$$
B^{update}_{a\to b}
=\frac12[B(\Delta W,U_a)+B(\Delta W,U_b)].
$$

它只说明净位移朝向，不等于 endpoint 变强。真正判断增强/稀释的是：

$$
\Delta_WB_{a\to b}=\frac12\{[B(W_b,U_a)-B(W_a,U_a)]
+[B(W_b,U_b)-B(W_a,U_b)]\}.
$$

基底漂移项为：

$$
\Delta_UB_{a\to b}=\frac12\{[B(W_a,U_b)-B(W_a,U_a)]
+[B(W_b,U_b)-B(W_b,U_a)]\}.
$$

必须计算 $s,t\in\{10k,20k,30k\}$ 的完整 $3\times3$ crossing
$B(W_s,U_t)$，不能只由三个对角点近似。

## 7. 次指标

在 held-out token 上同时报告：

$$
V_A^\perp=\mathbb E\|C_EWP_Ax\|_2^2,
\qquad
S_A^\perp=\frac{V_A^\perp}{\mathbb E\|P_Ax\|_2^2}.
$$

$V$ 是某频带在真实 token 上产生的总专家相对 logit 响应；$S$ 去掉该组总
能量，但在一个宽频带内部仍按 eigenvalue 加权。因此回答“等能方向选择性”时
仍以 $G$ 为主。

逐频带去除后还报告：

- route flip：top-1 expert 改变的 token 比例；
- margin support：去除该带后，native top-1 与第二名 logit 间距降低多少。

二者只说明当前路由依赖，不能说明 loss 收益或共同训练兼容性。

## 8. 护栏与不确定性

1. **checkpoint provenance：** 指标前冻结路径、大小、SHA-256、Gate shape、
   模型配置和 checkpoint identity。
2. **实际输入/no-op：** hook `mlp.gate` pre-input；离线 replay native logits
   的相对 Frobenius error 必须 $\le10^{-5}$，top-1 agreement 必须 1.0。
3. **基底有效性：** orthogonality error $\le10^{-4}$，eigenvalues 单调递减，
   768 ranks 完整覆盖。
4. **基底稳定性：** calibration half-split projector overlap 必须超过随机同维
   overlap null；不稳定 layer/contrast 不进入汇总。
5. **方向 null：** 保持 $C_EW$ 或 $C_E\Delta W$ 所有非零奇异值，只随机化
   相对 covariance basis 的右方向；256 次 Haar-Stiefel samples。
6. **bootstrap：** 200 次 paired calibration-sequence basis bootstrap；token
   指标使用 2,000 次 paired held-out-document bootstrap。
7. **聚合：** 以 eligible layers 的中位数作为模型级结果，同时保留所有逐层
   数值；不能把 12 层当作 12 个独立 seed。
8. **没有 practical 硬门槛：** 报告连续比值、零点、区间和 matched null；
   不加入 10%/25% 效应线，也不在看结果后规定“至少几层”。

## 9. Pass、Fail 与 Insufficient

先检查 hard guards。护栏有效后，每条谱系分开给出：

- **early-present：** 10k 的两个 endpoint 对比 bootstrap 下界均高于零，且
  观测中位数均超过 matched orientation-null q95；
- **progressive-strengthening：** early-present，且两个区间、两个对比的
  $\Delta_WB$ 区间均高于零；
- **early-present-maintained/mixed：** 10k 已形成，但后续 fixed-basis 效应
  接近零或两个对比方向不一致；
- **early-present-diluted：** 10k 已形成，且至少一段的两个 fixed-basis 对比
  均精确为负；
- **late-emerging：** 10k 不支持，但后续 endpoint 支持，且之前的
  $\Delta_WB$ 为正；
- **no-head-specificity：** endpoint 对比精确非正或落在随机方向范围内；
- **insufficient：** hard guard 失败、没有 eligible layers，或区间不能区分
  正/零/负。

如果两条谱系类型不同，总结为 **lineage-conditioned**。这些是证据分类规则，
不是“达到多少才有实际价值”的硬阈值。

## 10. 核心图与表合同

1. **决定性图：** 两行对应 LB/batch-gradient；展示 endpoint
   $B_{H:M}/B_{H:T}$ 和区间 $\Delta_WB$。保留 layer traces，叠加模型中位数、
   paired 95% 区间、零点和 matched null q95。它必须让读者判断 10k 前是否已
   形成，以及 10k 后是否继续增强。
2. **fine heatmap：** 六个 endpoint 上十二带的 $G$、$V$、$S$、route flip、
   margin support；只能解释几何/当前依赖，不能解释功能收益。
3. **crossing/decomposition 图：** 完整 $W_s\times U_t$ 矩阵，以及两个区间的
   $B^{update}/\Delta_WB/\Delta_UB$。
4. **紧凑决策表：** 每条谱系的 endpoint gain ratio 与 fixed-basis effect，
   包括 uncertainty。

所有图在归档前必须实际打开审核。

## 11. 执行阶段

1. 冻结 Protocol、checkpoint 和数据 manifests。
2. 参数化已验证的 E01 worker，并保持 E01 默认输出不变。
3. 单元测试、数据 hash 对比、preflight；每条谱系各跑一个 smoke endpoint。
4. 完成六个 endpoint extraction；有条件时两个谱系各用一张 GPU。
5. 每条谱系运行 basis bootstrap 与 orientation null。
6. 生成合并表、图和 typed verdict。
7. 写 `summary.md` 与 `detailed.md`；主线目录只保存紧凑证据和图，raw artifact
   留在 worker run directory。

## 12. 结论边界

E02 可以回答：

- 10k 时等能 head alignment 是否已经出现；
- 10k--20k、20k--30k 的净 Gate 位移是否偏 head；
- 固定表征基底后，Gate 权重变化是在增强、维持还是稀释 endpoint 偏置；
- LB 与 batch-gradient 是否给出相同描述。

E02 不能回答：

- 10k 之前的确切形成 step；
- 每一步 gradient dynamics 或优化器因果机制；
- `batch_only` 单独造成了谱系差异；
- middle/tail 分发的功能好坏；
- 专家形成、训练效率或所有模型的普遍规律。

## 13. 审批与执行合同

- 已批准：实现、smoke、完整 E02 frozen audit。
- 不包含：decommon、新训练、middle/tail-only Router、Q2/Q3、graph、root
  sync、commit、push。
- 若读取主指标前发现未注册的模型/坐标不兼容，停止并写 amendment；不得在
  看到结果后替换模型或 step。
