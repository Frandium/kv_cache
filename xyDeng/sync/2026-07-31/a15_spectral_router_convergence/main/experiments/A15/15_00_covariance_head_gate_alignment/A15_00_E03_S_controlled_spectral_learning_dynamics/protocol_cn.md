---
experiment_id: A15_00_E03_S_controlled_spectral_learning_dynamics
status: approved_for_full_execution
canonical_protocol: protocol.md
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol：E03-S 受控频谱学习动力学

## 0. Approval Snapshot

研究者已批准原 Snapshot，并在 2026-07-30 授权补全 Protocol、实现与 smoke。
注册 smoke 的全部工程守卫通过后，研究者于同日进一步明确授权 full run。

- **唯一问题：**专家优势信号在谱方向上可比时，输入 covariance 各向异性是否
  因果性地缩短高方差 Gate 模式的学习时间？
- **实验角色：**受控根因审计，不是 Router 方法实验。
- **主指标：**middle/head 与 tail/head 的 50% 目标拟合时间比。
- **主要证伪：**输入谱改变后学习时间没有改变，或白化后排序仍存在，或
  tail-only 目标学不会。
- **批准资源：**ACP、单节点、闲时 8×5090；profile `5090-8-spot`。

Primary anchor：[A15_00_01 频谱学习动力学](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor_cn.md)。

## 1. Terminology / Definitions

| 术语 | 具体定义 | 单位 / 公式 | 判断作用 | 不能证明 |
| --- | --- | --- | --- | --- |
| 实际输入方向 | 线性 Gate 接收向量的正交坐标 | activation | 固定受控对象 | 自然语言语义 |
| head / middle / tail | ranks 1--64 / 65--320 / 321--768 | 64 / 256 / 448 维 | 粗粒度速度比较 | 功能重要性 |
| fine band | 连续 64 维，共 12 段 | 64 维/段 | 检查组内单调性 | 任务频率 |
| 原始专家优势 $A_{raw}$ | 目标专家相对 score 对输入的未缩放线性系数，列均中心化且等范数 | score / activation | 匹配功能信号 | 可训练专家形成机制 |
| Gate-space 目标 $A_{gate}$ | 真正与 Gate 权重比较的系数，$A_{gate}=\tau A_{raw}$，$\tau=0.25$ | logit / activation | 使拟合量与 softmax 目标一致 | 功能收益 |
| 目标拟合比例 $F_B(t)$ | 频带 $B$ 上相对初始误差被消除的比例 | 0--1 | 定义学习时间 | held-out loss 收益 |
| $T_B(0.5)$ | $F_B$ 首次达到并保持 0.5 的 optimizer step | step | 主速度量 | 最终性能 |
| flat-rotation null | 平谱下随机旋转预注册频带得到的时间比分布 | 无量纲 | 排除有限样本方向破缺 | 真实训练机制 |

令 $C_E=I_E-\mathbf1\mathbf1^\top/E$，$\bar W=C_EW$。频带拟合比例为

$$
F_B(t)=1-
\frac{\|(\bar W_t-A_{gate})U_B\|_F^2}
{\|(\bar W_0-A_{gate})U_B\|_F^2}.
$$

若某次评估越过 0.5，且随后两个注册评估点仍不低于 0.5，则用相邻两点线性
插值得到 $T_B(0.5)$。未越过记为右删失，不得填成最大预算。

## 2. Anchor Alignment And Decision Question

理论主文证明：固定表征、局部线性专家优势和未白化梯度流下，模式时间常数
随 covariance 特征值增大而缩短。E03-S 将“专家优势方向”“输入谱”和“优化器”
分开，使该关系可被直接证伪。

本实验只决定受控因果根是否成立。即使通过，也不表示真实 DCLM 中一定由此
形成 head alignment；该外部签名由 E03-R 单独检查。

## 3. Hypotheses And Rival Explanations

**H1（条件性谱加速）：**在列等范数的同一 $A_{raw}$ 与 $A_{gate}$ 下，moderate 与 strong
各向异性都满足

$$
R_{M:H}=\frac{T_M(0.5)}{T_H(0.5)}>1,
\qquad
R_{T:H}=\frac{T_T(0.5)}{T_H(0.5)}>1,
$$

且 strong 的配对 log-ratio 大于 moderate；白化后回到 flat null。

**最强 rival R1（任务方向）：**$A_{raw}$ 本身偏 head，而非 covariance 加速。通过
逐方向等范数、跨条件共用 $A_{raw}$、tail-only 和随机旋转排除。

**R2（总能量 / 数值尺度）：**strong 条件只是输入总方差更大或有效学习率更高。
通过所有谱的 $\operatorname{tr}(\Sigma)/d=1$、相同 optimizer 和 S0 闭式正对照
排除。

**R3（表达失能）：**线性 Gate 根本不能使用 tail。tail-only 若在共同预算内
达到注册拟合比例，则否定该解释。

**R4（专家反馈）：**trainable experts 才产生观察到的排序。S1 固定目标是主
因果测试；S2 只测 joint-minus-frozen 的附加反馈，不可反写 S1 定理。

## 4. Data And Splits

- $d=768$，$E=8$；每个 seed 生成一个固定 Haar 基底 $U$。
- 潜变量 $s\sim\mathcal N(0,I)$；$x=U\Lambda^{1/2}s$。
- 频带为 $H=1{:}64$、$M=65{:}320$、$T=321{:}768$；fine 为 12×64。
- 原始谱比为 flat $1{:}1{:}1$、moderate $4{:}2{:}1$、strong
  $16{:}4{:}1$。每档都按维数加权后缩放，使 $\operatorname{tr}(\Sigma)/d=1$。
- 每个 seed 的 $A_{raw}\in\mathbb R^{8\times768}$ 先逐列做
  expert-centering，再逐列归一到相同范数；定义
  $A_{gate}=0.25A_{raw}$。所有谱条件共享二者、初始化与潜变量流。
- train、trajectory-eval 和 final-held-out 使用相互独立的随机流；eval 不回传。
- full seeds 固定为 `20260730`--`20260737`，不得看到结果后替换。

S1 的固定目标分布为

$$
q(x)=\operatorname{softmax}(A_{gate}x)
=\operatorname{softmax}(\tau A_{raw}x),\qquad \tau=0.25,
$$

并最小化 $-\mathbb E[q^\top\log p_W]$。$\tau$ 预先固定在非饱和区；held-out
平均目标熵低于 $0.35\log E$ 时，该 seed 判为无效而不是调温重跑。

strong-whitened 使用同一 strong 潜变量和同一目标 $q(x)$，但 Gate 接收
$\Lambda^{-1/2}U^\top x$；因此功能目标不变，输入时间常数被拉平。

## 5. Model And Optimizer

### S0：精确二次正对照

直接积分

$$
\dot w_i=-(\kappa\lambda_i+\beta)w_i+\kappa\lambda_i a_i
$$

并与闭式解比较。主值 $\kappa=1$、$\beta=0$、$w_i(0)=0$。

### S1：固定目标 softmax Gate

- 线性 Gate `Linear(768, 8, bias=False)`；只优化 $C_EW$ 等价类；
- pure SGD，learning rate `0.02`，momentum 0，weight decay 0；
- batch size 4096 / seed；最多 8,000 optimizer steps；
- 每 10 steps 评估到 step 400，此后每 50 steps；
- float64 用于 S0 和判定统计，S1 训练允许 float32，但指标累计为 float64。

### S2：可训练专家条件

仅在 full S1 已通过时启动。输入、Gate 与 S1 相同；八个两层 MLP experts
共同学习一个固定 teacher 输出。teacher 对各谱方向的目标能量仍匹配。
S2 同时运行 frozen-expert 与 trainable-expert，唯一报告量为同 seed 的
`joint minus frozen` 学习时间差及专家优势谱 $\|A_tu_i\|^2$ 的变化。

## 6. Conditions

| Stage | 条件 | 唯一改变 | 作用 |
| --- | --- | --- | --- |
| S0 | flat / moderate / strong | covariance 谱 | 验证闭式与统计代码 |
| S1 | flat / moderate / strong | covariance 谱 | 主因果比较 |
| S1-W | strong-whitened | Gate 输入预条件 | 排除非谱解释 |
| S1-T | strong + tail-only $A_{raw}$ | 功能目标位置 | 排除 tail 表达失能 |
| S2 | frozen / trainable experts | 专家是否更新 | 测额外反馈 |

tail-only 中 head/middle 的 $A_{raw}$ 与 $A_{gate}$ 列严格为零；其 capability 用 tail 的 held-out
KL 和 $F_T$ 判断，不定义无目标频带的 $T_B$。

## 7. Matched Variables And Guards

跨谱条件锁定：$A_{raw}$、$A_{gate}$、$U$、Gate 初始化、潜变量、batch 顺序、optimizer、batch、
更新预算、评估点、数值精度和分析代码。只改变注册的谱、whitening 或目标位置。

必须通过：

1. trace 匹配相对误差 $\le10^{-6}$；经验 eigenvalue 顺序正确；
2. $A_{raw}$ 与 $A_{gate}$ 的逐列范数最大相对偏差 $\le10^{-6}$，且
   $C_EA_{gate}=A_{gate}$；
3. S0 数值轨迹相对闭式最大误差 $\le10^{-5}$；
4. flat 随机旋转不产生预注册方向的系统优势；
5. loss、gradient、$F_B$ 全部有限；无越界后补 seed；
6. tail-only held-out KL 相对初始化至少下降 50%，否则为 capability failure。

## 8. Primary Metric And Decision Rule

每 seed 计算

$$
D_{M:H}=\log T_M(0.5)-\log T_H(0.5),\qquad
D_{T:H}=\log T_T(0.5)-\log T_H(0.5).
$$

单位为无量纲 log step ratio。$D>0$ 表示 middle/tail 达到相同比例目标所需更新
更多。主判定使用 seed-paired 中位数；不设任意“至少快 20%”门槛。

flat-rotation null：每个 flat seed 用 256 个与 H/M/T 同维的 Haar 旋转分割，
重算 $D$；合并配对 null 的 q95。moderate/strong 的两个 $D$ 都必须高于各自
匹配 q95。剂量关系使用 paired
$D^{strong}-D^{moderate}$ 的 exact sign/permutation 95% 区间。

该指标只判断有限时间学习速度，不判断专家功能收益、最终 NLL 或训练效率。

## 9. Secondary Metrics

- 12 个 fine-band 的 $T_{F_j}(0.5)$ 与 $F_{F_j}(t)$；
- 每个频带的 Gate 增益 $G_B=\|C_EWU_B\|_F^2/d_B$；
- held-out KL、gradient norm、最大 logit、目标/预测熵；
- S2 的专家优势谱、专家更新范数、专家内梯度冲突和 route load；
- wall time 与显存仅作工程信息，不进入因果判定。

## 10. Known Cases And Debug Controls

- closed-form S0 是必过正对照；
- flat + 任意正交旋转是必过无偏好对照；
- strong-whitened 应回到 flat envelope；
- tail-only 应学会 tail，但不要求比 head 条件更快；
- 把 $A_{raw}$ 列范数故意设为 head-only 的 debug case 必须被目标能量 guard 拒绝。

## 11. Profiling, Logging And Figure Contract

每次 run 写入：完整 config、代码 SHA-256、seed、设备与包版本、每个评估点的
loss/$F/G$/梯度、首次 crossing、删失状态和 guard 结果。

**中心图：**横轴 optimizer steps（log scale），纵轴 $F_B(t)$（0--1），颜色为
H/M/T，面板为 flat/moderate/strong/whitened；seed 细线加中位数粗线。允许结论
仅为“不同谱下达到同一目标比例的先后顺序”。它不能证明真实 MoE 或 loss/FLOP
收益。

**中心表：**每条件的 $T_H,T_M,T_T,R_{M:H},R_{T:H}$、null q95、删失数与
guard。S2 必须另表，不得与 S1 合并。

## 12. Execution Contract

### Smoke（当前已授权）

- 单个 ACP job，`5090-8-spot`，8 个进程各占一张 GPU；
- 每进程运行一个固定 seed；S0 全条件，S1 每条件 128 steps，S2 仅 16 steps
  验证 forward/backward/logging；
- smoke 可缩短预算但不得改变 $d/E$/频带/谱/目标定义；
- smoke pass：8 ranks 正常结束、S0 误差门通过、所有条件产生有限指标、
  tail-only 有非零梯度、whitening covariance guard 通过、结果 manifest 完整；
- smoke 不应用科学 H1 的 pass/fail，也不触发 S2 full。

### Full（已授权）

8 seeds、完整 8,000-step 上限与 256-rotation null。S1 结果冻结并判定后，S2
是否启动仍受本 Protocol 的 S1 gate 约束，不得因工程运行成功自动启动。任何
超参数修改都需要 Protocol 版本更新和重新批准。

## 13. Pass / Fail / Insufficient

**Pass：**S0 和全部 guards 通过；moderate/strong 的两个主 $D$ 均超过 flat
rotation q95；strong-minus-moderate 为正；strong-whitened 落回 flat envelope；
tail-only 通过 capability。

**Fail：**S0、能力和数值 guards 均有效，但 anisotropic $D$ 不超过 null，或
whitening 后排序不消失并可由任务方向解释。

**Insufficient：**任一关键正对照、目标匹配、tail capability、删失、seed 数、
数值或复现 guard 失败。S1 通过而 S2 失败，只能判联合专家反馈 insufficient/fail，
不得撤销固定目标因果结果。

## 14. Claim Boundary And Next Decision

通过最多支持：在注册的线性 Gate、匹配专家优势、SGD 和固定基底条件下，
covariance 各向异性因果性地改变有限时间模式学习速度。

不能声称：AdamW 等价、真实 DCLM 的形成原因、所有 Router 必然偏 head、
middle/tail 无功能、专家正反馈、验证损失/FLOP 改善。

**唯一下一决策：**完整 trajectories、rotation null、whitening return 与 tail
capability guards 完成后，按注册规则裁定 S1 的 pass / fail / insufficient。

**full 前定义澄清：**smoke 的目标分布始终使用 $\tau=0.25$，但原书面
$F_B$ 公式漏写了这一缩放。交叉熵的最优 Gate 是
$W=A_{gate}=\tau A_{raw}$，因此所有拟合比例与 crossing time 均比较
$W$ 和 $A_{gate}$。whitened 条件先把 Gate 权重映回原谱坐标再比较。数据、目标
分布、阈值、条件、seed 与训练超参数均未改变。
