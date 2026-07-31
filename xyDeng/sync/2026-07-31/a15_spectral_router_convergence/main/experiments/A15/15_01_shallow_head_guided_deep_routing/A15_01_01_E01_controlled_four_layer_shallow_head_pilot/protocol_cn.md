---
experiment_id: A15_01_01_E01_controlled_four_layer_shallow_head_pilot
status: approved_for_full_execution
canonical_protocol: protocol.md
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol：四层浅层 head 指导深层 Router Pilot

## 0. Approval Snapshot

研究者已批准四层 pilot、实现与 smoke，并要求新增一个**真正正常的四层对照**。
因此正式比较包含：普通 native 四层模型 N4，以及参数匹配的 head/random/shuffled
三种 side-channel 模型。所有条件不使用 load-balance auxiliary loss；统一使用
不反传的 auxiliary-loss-free expert bias。修复后的 smoke 通过全部 11 项工程守卫
后，研究者于 2026-07-30 明确授权 full run。

- **唯一问题：**在第二层 head 已通过独立共同训练兼容性门后，把它提供给第
  3--4 层 Gate，能否比普通四层训练和同维随机/打乱旁路更快降低 held-out NLL？
- **实验角色：**兼容性准入后的受控训练 pilot；不是 DCLM 方法结论。
- **主指标：**固定累计训练 FLOPs 处的 paired held-out NLL 差，nat/token。
- **批准资源：**ACP、单节点、闲时 8×5090；profile `5090-8-spot`。

Primary anchor：[A15_01_01 四层浅层 head pilot](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_01_shallow_head_guided_deep_routing/subanchors/15_01_01_controlled_four_layer_shallow_head_pilot_anchor_cn.md)。

## 1. Terminology / Definitions

| 术语 | 具体对象 | 单位 / 公式 | 判断作用 | 不能证明 |
| --- | --- | --- | --- | --- |
| native 四层对照 N4 | 4 个标准 top-1 MoE blocks，深层 Gate 只读本层输入 | NLL/FLOP | 回答方法是否胜过正常训练 | 排除额外参数解释 |
| layer-2 head $c_{2,H}$ | 第二层实际 Gate 输入的 covariance ranks 1--64 系数 | 64 维 | 候选浅层信号 | 深层功能价值 |
| H2 | 第 3--4 层 Gate 额外读同 token 的 $c_{2,H}$ | condition | 主干预 | 自然语言收益 |
| R2 | 读同 token、同维 frozen Haar-random coefficients | condition | 排除旁路容量 | token identity |
| SH2 | 读 batch 内另一 token 的真实 head coefficients | condition | 排除分布/尺度 | 同 token 信息 |
| 一步交叉更新兼容性 $K$ | 在独立 A 组更新专家一次后，B 组 loss 的改善，做对称平均 | nat/token | side channel 的硬准入 | 长期训练收益 |
| 增量 held-out $R^2$ | 加入候选频谱特征后，对 $K$ 的 out-of-sample $R^2$ 提升 | 无量纲 | 线性分数外残差增益 | 因果训练效率 |
| matched-FLOP NLL | 在相同累计训练 FLOPs 处插值的 held-out cross entropy | nat/token | 主训练比较 | 大规模部署成本 |

## 2. Anchor Alignment And Decision Question

本 pilot 检查竞争机制 2：浅层已形成的 head/common 坐标是否能减少深层 Router
搜索。它不检查“本层 middle/tail 指导本层分发”，也不能由 A15_00 的 head
alignment 自动推出。

实验必须先回答“第二层 head 是否预测哪些 token 共同更新同一深层专家更兼容”，
再回答“这种信号是否转化为 matched-FLOP 训练收益”。若兼容性门失败，训练
比较不得启动。

## 3. Hypotheses And Rival Explanations

**H1（兼容性增量）：**informative 任务上，base controls 加 $c_{2,H}$ 相似度
对一步交叉更新 $K$ 的 held-out 增量 $R^2$ 同时超过 random、shuffled、错误层
和 batch-resampling null。

**H2（训练收益）：**H1 通过后，H2 在冻结 FLOP 预算 $F_\star$ 处的 held-out
NLL 同时低于 N4、R2、SH2；nuisance 任务不复现同样排序。

**最强 rival R1（额外容量）：**任意 64 维旁路都会改善；R2/SH2 与 H2 参数和
计算相同，直接排除。

**R2（普通模型更省计算）：**H2 只是多算；与 N4 在相同累计 FLOPs 而非相同步
比较。

**R3（负载变化）：**head 只改变负载或 dead experts。所有条件共享无辅助损失
bias 规则，并注册负载匹配 guard。

**R4（coarse shortcut）：**生成器把答案直接写入高方差变量。nuisance 保持
完全相同的 coarse 边际和方差，但让真正 transformation family 独立。

## 4. Controlled Data And Splits

每个样本视为一个 token。生成器固定以下潜变量：

- coarse identity $c\in\{0,\ldots,7\}$；
- transformation family $r\in\{0,\ldots,7\}$；
- content $v\sim\mathcal N(0,I_{32})$；
- position/nuisance id $p\in\{0,\ldots,31\}$。

输入 $x\in\mathbb R^{256}$ 由三个相互正交、预先随机的编码器组成：64 维高方差
coarse code、128 维 content code、64 维低方差 family/position code。所有任务
保持 $x$ 的总 trace、各块边际、label 数和噪声相同。

- **informative：**$r=c$；label 为
  $y=\arg\max_{k\le16}(M_rv)_k$；
- **nuisance：**$r$ 与 $c$ 独立，$r$ 仍由低方差 family code 提供给网络；label
  由同一组 $M_r$ 生成。

$M_r\in\mathbb R^{16\times32}$ 在所有条件和 seeds 中由注册 generator 生成后
冻结。train、Stage-A validation、compatibility-fit、compatibility-validation、
compatibility-test 和 B1 held-out evaluation 使用独立 RNG streams，禁止复用
token group。full paired seeds 为 `3101, 3102, 3103, 3104, 3105`。

## 5. Four-Layer Model And Router

模型是四个 residual top-1 MoE blocks，hidden width 256；每层 8 experts，expert
MLP width 512，GELU；无 shared expert、无硬 capacity limit。每层 native Gate
为 `Linear(256,8,bias=False)`，实际 Gate 输入就是该 block 的归一化 residual
state。

第 3--4 层的旁路形式为

$$
z_\ell=W_\ell g_\ell+A_\ell s_2,
\qquad \ell\in\{3,4\},
$$

其中 $s_2$ 分别为 H2 的 $c_{2,H}$、R2 的 random coefficients 或 SH2 的
batch-permuted $c_{2,H}$。$A_3,A_4\in\mathbb R^{8\times64}$ 零初始化。N4 没有
$A_3,A_4$，保持真正普通的四层结构。

所有 Gate score 另加与 E03-R 同型的 non-gradient bias，step size $10^{-3}$、
clip $[-0.1,0.1]$、逐专家零均值；由每步全 batch load 更新。`lambda_lb=0`。

## 6. Training Stages And Algorithm

### Stage A：形成并冻结两层浅层表征

训练 layers 1--2、coarse proxy head 和 content reconstruction head：

$$
L_A=\operatorname{CE}(\hat c,c)+0.1\|\hat v-v\|_2^2.
$$

达到 capability 后，用独立 calibration set 在第二层实际 Gate input $g_2$ 上
拟合 $\mu_2,U_{2,H}$；随后冻结 layers 1--2、$\mu_2,U_{2,H}$。

Stage-A full guards：coarse held-out accuracy $\ge0.90$；content explained
variance $\ge0.80$；head-only linear probe accuracy $\ge0.85$ 且超过 256 个同维
random-subspace q95；两个 calibration half 的 head projector overlap $\ge0.80$。

full 固定优化器：AdamW 500 steps，batch 512，constant LR $3\times10^{-4}$，
betas (0.9,0.95)，weight decay 0.01。capability validation 为 4,096 samples；
两个 projector calibration halves 各 2,048；head probe fit/test 为
2,048/4,096；q95 固定由 256 个 64 维 Haar-random probes 给出。

### Stage B0：共同 native calibration

附加 layers 3--4 和 16-class output head，所有条件都只使用 native Gate，训练
300 steps。要求 held-out accuracy 在 0.25--0.75、NLL 比 uniform 至少低 0.10
nat、任层最大 expert share $\le0.60$，以保证任务已可学但未饱和。每个
seed/task 冻结一个共同 B0 checkpoint。
B0 固定使用 AdamW、batch 512、constant LR $10^{-4}$、betas (0.9,0.95)、
weight decay 0.01；held-out 为 4,096 samples。

### Stage 0：独立一步交叉更新兼容性门

在 B0 上构造独立 token-group pairs $(A,B)$。只对 A 当前 native route 所指向的
一个深层 expert 做一次 SGD 更新，step size 为 B1 expert LR；router、trunk 与
其他 experts 固定。定义对称兼容性

$$
K(A,B)=-\frac12\Big[
L_B(\theta-\eta\nabla L_A)-L_B(\theta)
+L_A(\theta-\eta\nabla L_B)-L_A(\theta)
\Big].
$$

$K>0$ 表示两组 token 放入同一 expert 更新时相互帮助。base ridge predictor
包含 native deep score 相似度、native route equality、expert-load stratum、
两组表征 norm、Mahalanobis outlier score 和 position。候选增量分别加入
layer-2 head、同维 random、token-shuffled head、layer-1 wrong-basis similarity。
pair split 为 60% fit / 20% hyperparameter selection / 20% untouched test；
全部标准化参数只在 fit split 学习。
每个 seed/task 固定 256 个不相交 pairs，group size 4，来自 32,768-token pool。
ridge alpha 由 validation MSE 从
$\{10^{-4},10^{-3},10^{-2},10^{-1},1,10,100\}$ 选择；paired seed-group
bootstrap 为 2,000 次，batch-label resampling null 保持 1,000 次。

### Stage B1：四臂 matched-compute 训练

只有 informative 的 Stage 0 通过，才从同一 B0 克隆 N4/H2/R2/SH2。每 arm
最多 2,000 optimizer steps，batch 512；AdamW LR $3\times10^{-4}$、betas
(0.9,0.95)、weight decay 0.01，cosine decay，无额外 loss。相同 seed/task 的
数据顺序与可共享初始化完全配对。
NLL/FLOPs/margin/route flip/load/bias 在 step 0、每 50 steps 和 step 2,000
评估；昂贵的 expert conflict/redundancy 每 200 steps 及 final 评估。

## 7. Conditions And Matching

| 条件 | 深层 Gate 输入 | 参数关系 | 判断角色 |
| --- | --- | --- | --- |
| N4 | $W_\ell g_\ell$ | 无旁路 | 真正普通四层主对照 |
| H2 | native + 同 token layer-2 head | +2×8×64 参数 | 主干预 |
| R2 | native + 同 token random | 与 H2 完全相同 | 容量/计算对照 |
| SH2 | native + 另一 token head | 与 H2 完全相同 | 尺度/分布对照 |

H2/R2/SH2 精确匹配参数和 FLOPs；N4 参数更少，因此按累计 FLOPs 比较。所有
条件匹配：B0、optimizer、tokens、batch、数据顺序、bias 规则、无容量 drop、
评估集与 early-stop 禁用。

## 8. Primary Metrics And Decision Rules

### Stage 0 admission metric

$$
\Delta R^2_X=R^2(\text{base}+X)-R^2(\text{base})
$$

在 untouched compatibility-test pairs 上计算，无量纲。H2 必须满足：

1. paired group bootstrap 95% lower bound $>0$；
2. $\Delta R^2_H-\Delta R^2_C$ 对每个
   $C\in\{R2,SH2,wrong\ layer\}$ 的 95% lower bound $>0$；
3. 超过 1,000 次 batch-label resampling null q95。

不设任意增益百分比门；若 CI 太宽则 insufficient。

### Stage B1 primary metric

定义

$$
F_\star=F_{N4}(\text{2,000 B1 optimizer steps}),
$$

其中 $F$ 是在运行前由固定 analytic MAC counter 计算的 forward+backward+update
累计 FLOPs。每条曲线在 $F_\star$ 处线性插值，得到

$$
\Delta L_{H-C}(F_\star)=L_H(F_\star)-L_C(F_\star),
\quad C\in\{N4,R2,SH2\}.
$$

单位 nat/token；负值支持 H2。主统计为五个 paired seeds 的 exact paired
permutation 95% 区间。H2 必须对三个 C 的区间都低于 0。

## 9. Secondary Metrics

- 达到注册 NLL levels 所需 FLOPs 与 tokens；
- Router margin、route flip、每层 load、dead experts、bias norm/saturation；
- expert update norm、同 expert 内 token-group gradient cosine/conflict；
- expert functional redundancy：在固定 probe 上不同 experts 输出 residual 的
  centered kernel alignment；
- layer-2 head/random/wrong-layer capture 与 basis stability；
- 参数数、analytic FLOPs、measured step time 和 peak memory。

这些指标解释路径，不能替代 matched-FLOP NLL。

## 10. Known Cases And Debug Controls

- H2/R2/SH2 在 B1 step 0 的 logits/routes/outputs 必须与 N4 一致；
- SH2 permutation 不得出现 fixed point；inverse permutation replay 应恢复 H2；
- random basis 正交且维数 64；wrong-layer basis 不能被误拟合到 $g_2$；
- informative 中 oracle $c$ grouping 应有正 compatibility；nuisance 中同 $c$
  grouping 不应超过同 $r$ grouping；
- 把 side coefficients 全置零时，三旁路模型必须与 N4 数值一致。

## 11. Logging And Figure Contract

每 run 保留 config/hash/seed/task/arm、Stage A/B0 guards、compatibility split
hash、每步 NLL/FLOPs/load/bias/margin、定点评估和 artifact manifest。

**中心图 1：**informative 与 nuisance 两面板；横轴累计训练 FLOPs，纵轴
held-out NLL，显示四臂 paired-seed 均值/区间和 $F_\star$。允许结论为受控
matched-compute 训练差；不能外推 DCLM。

**中心图 2：**Stage 0 的 $\Delta R^2$，显示 head、random、shuffled、wrong
layer 与 resampling q95。允许决定训练准入；不能替代训练结果。

**中心表：**三组 $\Delta L_{H-C}$、paired interval、负载/参数/FLOP guard 和
informative/nuisance typed verdict。

## 12. Execution Contract

### Smoke（当前已授权）

- 单个 8×5090 ACP job；8 ranks 分别运行 2 tasks ×4 arms；
- 每 task/arm 独立但确定性地重建同一 Stage-A/B0 起点，并核对 checkpoint hash；
- 工程缩减：Stage A 32 steps、B0 16 steps、compatibility 8 pairs、B1 8 steps；
- 保持 4 layers/width256/8E/top-1/64-dim side channels/N4/no-LB/bias 规则不变；
- smoke pass：八臂结束、同 task B0 hash 一致、step-0 equivalence、数据 split
  无泄漏、cross-update 有限、FLOP counter 区分 N4 与旁路、bias 无梯度且 cadence
  正确、manifest 完整；
- smoke 不要求 capability、compatibility 或 NLL 科学门通过。

### Full（已授权）

5 paired seeds ×2 tasks ×4 arms；Stage 0 失败则不得启动 B1。任何 generator、
guard、bias、模型或 $F_\star$ 定义改变都要修订 Protocol。
五个 seeds 必须先完成 Stage A、B0、Stage 0，再形成全局 informative admission。
任一 Stage-A/B0/data/gradient guard 失败均为 insufficient 并阻断 B1；有效但
未通过的 informative Stage 0 属于 scientific fail，也阻断 B1。

## 13. Pass / Fail / Insufficient

**Pass：**Stage A/B0 全部有效；informative Stage 0 通过；informative 中 H2
在 $F_\star$ 对 N4/R2/SH2 的三个 paired intervals 均低于 0；负载/计算 guards
通过；nuisance 不出现对三个对照同样的 H2 优势。

**Fail：**能力和匹配有效但 Stage 0 不通过；或 Stage 0 通过后 H2 未同时超过
三个对照；或优势由负载/FLOP/额外容量解释。Stage 0 fail 时不运行 B1，也属于
本 pilot 的科学 fail，而非工程失败。

**Insufficient：**Stage A capture、B0 capability、compatibility 精度、五个
paired seeds、数据独立性、load/bias、FLOP counter 或复现任一关键 guard 失败。

## 14. Claim Boundary And Next Decision

即使通过，也只支持：在注册的受控四层任务、冻结两层 trunk 和固定 layer-2
basis 条件下，浅层 head 在独立兼容性准入后改善 matched-compute 深层训练。

不能声称：从初始化端到端收益、online PCA、DCLM/自然语言收益、middle/tail
无用、E03 covariance 定理成立，或大规模训练效率改善。

**唯一下一决策：**五个 paired seeds 的 Stage A/B0、Stage-0 准入，以及仅在准入
后运行的 matched-FLOP B1 全部完成后，按注册规则裁定科学 verdict。

**full 前 Protocol 冻结说明：**上述 Stage-A/B0 预算和 optimizer、held-out/
calibration 数量、compatibility pairs、ridge grid 与评估 cadence 都在 full 前
仅用 capability preflight 固定。B0 LR 设为 $10^{-4}$，因为 seed-3101 的 native
calibration 落在预注册能力窗内；没有运行或查看 H2/R2/SH2 的 B1 效果。preflight
提示 random probe 可能追平 head probe，但 generator、随机对照数、阈值与
fail-closed Stage-A 规则均不改变。
