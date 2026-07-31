---
anchor_id: 15_02_middle_tail_functional_resolution
status: blocked_by_compatibility_fail
created: 2026-07-30
updated: 2026-07-30
canonical_language: en
canonical_file: 15_02_middle_tail_functional_resolution_anchor.md
depends_on: 15_00_covariance_head_gate_alignment
---

# A15_02 middle / long-tail 频谱的功能分辨率


## 1. Problem Definition

Q1 已证明：训练后的线性 Gate 在实际 Router 输入上主要与 covariance head
对齐，但 middle 和 long-tail 仍可访问并参与部分 native 路由。尚未回答的是：
这些较弱频带是否保留了线性 logits 没有利用的功能关系。

**父问题：** 频谱信息能否补充线性 Gate 的功能分辨率，并在匹配计算下改善
Router--Expert 联合训练？

**唯一决策问题：** 若一个 middle、long-tail 或 middle+long-tail treatment
首先通过独立 token 组的一步兼容性准入，把同一冻结频带用于 4-layer DCLM
MoE 路由，是否能在相同累计 FLOPs 下得到更低的 held-out next-token NLL，且
优于 native Router 与同维随机子空间？

### 术语与指标合同

| 术语 / 指标 | 普通含义 | 具体计算与单位 | 为什么测 / 能回答什么 | 不能回答什么 |
| --- | --- | --- | --- | --- |
| 实际 Router 输入 $r_\ell$ | Gate 真正收到的表示 | 直接 hook `mlp.gate` pre-input | 保证频谱对象与部署对象一致 | 专家输入几何 |
| Middle $M$ | 中方差方向 | eigen-ranks 65--320，256 维 | 检查 head 之外的中方差功能信息 | 语义类别 |
| Long-tail $T$ | 低方差方向 | eigen-ranks 321--768，448 维 | 检查低能量方向的功能信息 | 稀有词或长尾数据 |
| Non-head $N$ | middle 与 long-tail 联合 | ranks 65--768，704 维 | 检查完整非 head 信息 | middle 与 tail 各自贡献 |
| 一步兼容性准入 | 两组 token 更新同一专家是否互助 | 子 anchor `15_02_01` 的 held-out $\Delta R^2$ | 决定是否值得进入联合训练 | 长期训练收益 |
| held-out NLL | 未参与训练文档上的平均 next-token 负对数似然 | nat/token | 直接衡量语言建模质量 | 原因或专家专业化 |
| matched-FLOP NLL 差 | 相同累计 FLOPs 下 treatment 与 baseline 的 held-out NLL 差 | nat/token | **父 anchor 主指标；回答训练效率是否改善** | 跨模型尺度或跨数据普遍性 |

## 2. Physical Priors

1. **线性压缩 prior。** 8 个 Gate logits 是 768 维 Router 输入的低维线性压缩；
   Q1 又观察到强 head alignment，因此 middle / long-tail 可能保留 native logits
   未表达的 token--token 功能关系。若控制 native logits 后没有 held-out
   兼容性增量，该 prior 在本任务上被削弱。
2. **功能而非几何 prior。** 新邻域、route flip 或更均衡负载本身不是收益；
   候选频带必须先预测同一专家的交叉更新 loss，再由联合训练 NLL 裁定。
3. **共同动力学 prior。** 即使局部兼容性为正，Router、专家、负载和表示共同
   演化仍可能抵消它。因此一步兼容性是必要的准入证据，不是充分的终点证据。

## 3. Falsifiable Hypotheses

**H1——频谱功能分辨率。** 至少一个 $S\in\{M,T,N\}$ 在子 anchor 中提供
native controls 之外的 held-out 兼容性增量，并超过同维随机与错误层；将该
$S^*$ 用于匹配训练后，在相同 FLOPs 下的 held-out NLL 低于 native 与随机
projector。

**H1-M——兼容性机制。** 若收益来自 E01 所测的共同训练关系，则 $S^*$ arm
的专家内更新冲突应下降。H1-M 是解释性子假设，不是 H1 的替代主指标：
NLL Pass 而冲突不降时，只能保留训练收益、放弃兼容性机制解释。

**最强 rival R0——只有额外几何。** 频带产生新划分，但 compatibility
$\Delta R^2$ 不超过随机或错误层；此时不得启动联合训练。

**R1——局部代理不能迁移到长期训练。** compatibility gate 通过，但
matched-FLOP NLL 不优于 native / random。此时否定的是“一步代理足以选择
有效训练 treatment”，不是频谱信息在所有设计中无用。

**R2——负载或容量混杂。** treatment 的表面 NLL 差来自专家负载、overflow、
token dropping、参数数或实际 FLOPs 不匹配，而不是频带方向。

**Pass：** 子 anchor 先通过；随后 $S^*$ 相对 native 和随机的 paired
matched-FLOP NLL 差在注册的 4-layer DCLM 范围内稳定小于零，且负载、容量、
token、参数、数据顺序和 FLOPs 护栏通过。

**Fail：** compatibility gate 有效且精确为非正；或 gate 通过后，匹配训练
精确显示 $S^*$ 不优于 native / random。

**Insufficient：** 兼容性 operationalization、4-layer transfer、基底稳定性、
训练稳定性、load/capacity/FLOP 匹配或 paired-seed 精度失败。

## 4. Mathematical Model

对 actual Gate input $r_\ell$，在独立 calibration set 上定义

$$
x_\ell=r_\ell-\mu_\ell,
\qquad
\Sigma_\ell=\mathbb E[x_\ell x_\ell^\top]
=U_\ell\Lambda_\ell U_\ell^\top.
$$

令 $P_{\ell,S}=U_{\ell,S}U_{\ell,S}^\top$。训练 treatment 使用 branch
checkpoint 上冻结的基底与均值：

$$
r_{\ell,S}=\mu_{\ell,*}+P_{\ell,S}(r_\ell-\mu_{\ell,*}),
\qquad
z_{\ell,S}=W_\ell r_{\ell,S}+b_{\ell,S}.
$$

$b_{\ell,S}$ 是 calibration-only 的冻结 load-matching offset；它只校准 branch
时刻的专家份额，不允许根据 held-out loss 调整。所有 arms 使用同一 projector
kernel、同形状 Gate 和同一 frozen-offset slot，以保持参数与执行路径可比。

子 anchor 从 $M,T,N$ 中只选一个通过者 $S^*$ 进入训练；同维随机 projector
$P_{\ell,R^*}$ 是方向 rival。父 anchor 主指标为注册累计 FLOPs $F^*$ 下的

$$
\Delta L_{S^*:B}(F^*)
=L_{S^*}^{heldout}(F^*)-L_B^{heldout}(F^*),
\qquad B\in\{native,R^*\}.
$$

$\Delta L<0$ 表示同等计算下更低的验证损失。它能回答当前 treatment 是否改善
训练效率，不能单独说明改善来自兼容性、负载或专家分化；这些由次级动力学
指标解释。

## 5. Computational Realization

### Stage 1：局部功能准入

[子 anchor `15_02_01`](subanchors/15_02_01_cross_update_compatibility_gate_anchor_cn.md)
在现有 12-layer LB / decommon checkpoints 上比较 middle、long-tail、non-head，
并在预训练 4-layer branch checkpoint 上做 transfer gate。它使用同一批 A/B
token-group pairs 和真实一步交叉更新 loss；静态邻域只作诊断。

### Stage 2：条件性 8×5090 匹配训练

[E02 中文审核稿](../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_E02_matched_spectral_dispatch_training/protocol_cn.md)
预注册但被 Stage 1 阻塞。执行面复用已验证的 H768、4-layer、8-expert +
1 shared expert、top-1、DCLM、8×5090 环境。共同 burn-in 到约 0.63B tokens
后估计每层实际 Router-input 频谱，再从完全相同的模型、optimizer、数据游标和
RNG 状态分叉 native、$S^*$、$R^*$ 三个 arms。

8 卡只是每个 arm 的资源配置，不代表并行获得独立种子。pilot 为一组 paired
seed 到 1B total tokens；只有稳定性通过后，正式证据才使用 3 个 paired seeds
到 2B total tokens。

## 6. Minimal Falsification Tests

1. **兼容性准入：** $S^*$ 的 held-out $\Delta R^2$ 必须高于零、同维随机 q95
   和错误层；目的：排除“新几何就是功能”的 rival。它最多授予训练资格。
2. **4-layer transfer guard：** 同一 $S^*$ 必须在 E02 的 4-layer branch
   checkpoint 上复现；目的：避免把 12-layer 信号直接假定迁移到 4-layer。
3. **三 arm 匹配训练：** native、$S^*$、$R^*$ 从同一 branch state 开始；
   目的：同时区分 native baseline 与随机方向 rival。
4. **终点裁定：** 以 2B total-token 附近的 matched-FLOP NLL 为主；目的：回答
   实际训练效率。loss--FLOP AUC、margin、flip、load、专家更新冲突和功能重复
   只解释路径，不能替代主指标。
5. **失效护栏：** 若 fixed spectral projector 与当前 covariance band 的 overlap
   退化到随机范围，只能把结果解释为 fixed-subspace treatment，不得称为持续
   middle / tail routing。

## 7. Current Evidence

[A15_00 E01](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)
和
[A15_00 E02](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)
证明 actual-input head alignment 在 10k 已强，middle / tail 的 access 和 native
route effect 较弱但非零，并在 10k--30k 相对增加。它们没有测 compatibility、
forced same-expert update、matched training 或 held-out loss/FLOP。

[A15_02_01_E01](../../../experiments/A15/15_02_middle_tail_functional_resolution/A15_02_01_E01_cross_update_compatibility_gate/summary_cn.md)
现已把静态分辨率和功能分辨率分开：M/T/N 的残差邻域新颖度很高
（0.732--0.902），但同维随机参考也很高（0.714--0.877）。没有一个频带在 LB
与 decommon 上同时得到正的、超过随机和错误层的模型级兼容性增量，子 anchor
裁定为 Fail。

因此没有 $S^*$，E02 的条件性授权没有生效；没有提交 8×5090 作业，本父 anchor
也没有 matched-training 收益或伤害证据。

## 8. Claim Boundary And Next Decision

本父路线被必要的局部功能门阻塞。因为匹配训练按规则未运行，它既没有建立
matched-FLOP 收益，也没有建立 matched-training 伤害。

它不能证明：自然语义专家、所有层或尺度通用、固定频带长期保持当前谱身份、
频谱是唯一原因、Q1 的 head alignment 有害，或该方法优于所有 Router 设计。

**唯一下一决策：** 关闭固定 covariance band 作为 dispatch treatment 的父路线，
或仅通过新的、已批准的功能对齐子空间 anchor 替换 treatment。当前 M/T/N 定义下
不得恢复 E02。
