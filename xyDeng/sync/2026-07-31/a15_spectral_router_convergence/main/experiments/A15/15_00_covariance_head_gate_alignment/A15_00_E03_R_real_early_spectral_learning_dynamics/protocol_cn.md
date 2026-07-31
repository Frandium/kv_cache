---
experiment_id: A15_00_E03_R_real_early_spectral_learning_dynamics
status: approved_for_full_execution
canonical_protocol: protocol.md
approval_date: 2026-07-30
implementation_authorized: true
smoke_authorized: true
full_run_authorized: true
resource_profile: 5090-8-spot
---

# Protocol：E03-R 真实 DCLM 早期 Router 频谱轨迹

## 0. Approval Snapshot

研究者已批准 E03-R 并授权 Protocol、实现和 smoke。相对原 Snapshot 的唯一
配置修订是：**不使用 load-balance auxiliary loss**；改用所有层共享规则的
**auxiliary-loss-free expert bias** 防止工程性空载。该偏置不反传、单独记录，
且不计入频谱 Gate 权重。首次 smoke 暴露了混合精度回放误差；修复后的同批
GPU-fp32 回放 smoke 通过全部注册守卫后，研究者于 2026-07-30 明确授权 full。

- **唯一问题：**从初始化到 2B DCLM tokens，真实线性 Gate 是否出现与 E03-S
  有限时间谱加速同型的早期 head-formation 签名？
- **实验角色：**真实训练轨迹审计；不承担 covariance 因果证明。
- **主指标：**head alignment 首次形成时间 $T_{form}$，单位 training tokens。
- **批准资源：**ACP、单节点、闲时 8×5090；profile `5090-8-spot`。

Primary anchor：[A15_00_01 频谱学习动力学](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/subanchors/15_00_01_spectral_learning_dynamics_anchor_cn.md)。

## 1. Terminology / Definitions

| 术语 | 具体对象 | 单位 / 公式 | 判断作用 | 不能证明 |
| --- | --- | --- | --- | --- |
| actual Router input $g_{\ell,t}$ | 第 $\ell$ 层 `mlp.gate` 直接收到的 post-attention RMSNorm 表征 | activation | 唯一拟合 covariance 的表征 | expert input 几何 |
| auxiliary-loss-free bias $b_{\ell,e}$ | 加在专家 score 上、由负载计数更新且不反传的标量 | logit | 防止空载，不引入 LB 梯度 | 自然 Router 动力学 |
| equal-energy gain $G_{\ell,B}$ | $\|C_EW_{\ell}U_{\ell,B}\|_F^2/d_B$ | logit²/activation²/direction | 权重方向选择性 | 功能收益 |
| head contrast $B$ | $\log(G_H/G_M)$、$\log(G_H/G_T)$ | 无量纲 | 判断 head alignment | 形成原因 |
| raw Gate gradient | optimizer 预处理前、完整 accumulation 后的 $\nabla_WL_{LM}$ | gradient norm / band | 任务梯度指向 | 实际参数位移 |
| applied update | optimizer step 前后 $\Delta W=W_{t+1}-W_t$ | weight / band | AdamW 实际施加方向 | 单独的 loss 原因 |
| $W_s\times U_t$ crossing | 保存时刻 $s$ 的 Gate 权重在时刻 $t$ 的表征基底上测量 | $B$ | 分开权重与基底运动 | covariance 因果性 |

频带固定为 H=ranks 1--64、M=65--320、T=321--768；fine resolution 为
12×64。所有 covariance 均来自实际 $g_{\ell,t}$，不读取专家输入替代。

## 2. Anchor Alignment And Decision Question

E01/E02 只看到 10k step 之后的宏观保存点；最早一点已约 7.86B tokens，无法
定位形成窗口。E03-R 从 step 0 密集记录 Gate 梯度、AdamW 位移和表征基底，
检查真实模型是否出现“先快速偏 head，随后相对展宽”的外部签名。

它只检查与 E03-S 相容的真实轨迹。即使轨迹通过，也不能从相关性反推
covariance 是唯一原因；E03-S 才是受控因果测试。

## 3. Hypotheses And Rival Explanations

**H1（真实早期签名）：**至少 2/3 seeds 在 2B tokens 前得到有限 $T_{form}$；
形成窗口中，固定 $U$ 的 Gate-weight contribution 为正，raw gradient 与
applied update 至少一种显示 head 优先，随后允许 middle/tail 相对追上。

**最强 rival R1（专家优势本来偏 head）：**观测轨迹可由任务/专家优势谱解释，
不是 covariance 因果。E03-R 只能报告这一未决解释，不能排除；需与 E03-S
联合裁定。

**R2（表征基底漂移）：**$W$ 没有向 head 学习，只是 $U_t$ 旋转到已有 $W$。
完整 $W_s\times U_t$ crossing 与固定基底分解直接检验。

**R3（AdamW 改写）：**raw gradient 偏 head，但自适应预条件或 weight decay
消除/逆转排序。逐步 raw/applied 对照检验。

**R4（负载控制伪影）：**score bias 改变分发并间接改变梯度。报告 weight-only、
weight-plus-bias 路由和 bias update；结论明确限定在该无辅助损失负载护栏下。

## 4. Data And Splits

- 训练：DCLM binary stream
  `/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10`；
- sequence length 1024；global batch 768 sequences；每 optimizer step 名义
  786,432 tokens；首个不低于 2B 的 step 为 2,544；
- calibration：32 个固定、与训练 batch 不重叠的 256-token sequences；
- probe：64 个固定 held-out DCLM documents ×256 tokens；
- trajectory basis 只由 calibration 拟合；所有 $V/S$/route/margin 只在 probe
  计算；两个 buffer 的 token hash 在三个 seeds 间相同；
- full seeds：`17, 29, 43`，每个从 step 0 独立初始化。

## 5. Model, Router And Optimizer

- 6 decoder layers，hidden size 768，6 attention heads，3 KV heads；
- 每层 8 sparse experts + 1 shared expert，top-1；expert intermediate 1536；
- linear Gate `Linear(768, 8, bias=False)`；Router input 不中心化；
- LM loss only；`lambda_lb=0`，没有任何 load-balance loss gradient；
- AdamW，LR $10^{-4}$，betas (0.9,0.95)，epsilon $10^{-8}$，weight decay 0.01；
- linear warmup 1,000 optimizer steps，随后按 100B-token horizon cosine decay；
- bf16 forward/backward，诊断累计与 eigendecomposition 使用 float32/float64；
- activation checkpointing 与 exact-resume 开启；checkpoint 含 optimizer、
  scheduler、sampler、RNG、score bias 和诊断状态。

实际 Router score 为

$$
z_{\ell,e}=W_{\ell,e}g_\ell+b_{\ell,e}.
$$

每个 optimizer step 汇总全卡、全 accumulation 的 top-1 counts $c_{\ell,e}$，
并在 step 后执行

$$
b_{\ell,e}\leftarrow
\operatorname{clip}\left[
b_{\ell,e}+10^{-3}
\frac{\bar c_\ell-c_{\ell,e}}{\bar c_\ell+10^{-6}},
-0.1,0.1
\right],
$$

随后减去 expert 均值使 $\sum_e b_{\ell,e}=0$。$b$ 是 buffer、无梯度、零初始化；
其规则和超参数不得按 seed 调整。

## 6. Conditions

科学 full 只有一个注册训练条件：上述 native linear Gate + 无辅助损失 bias。
不在 E03-R 中重复 flat、whitening 或中心化干预。所有 seed 使用完全相同的训练
和诊断配置，只改变 seed。

为审计 bias，每个 probe snapshot 同时重放：

1. 实际 route：$Wg+b$；
2. weight-only counterfactual：$Wg$；
3. bias-only route：$b$（诊断，不作为可部署或功能对照）。

## 7. Matched Variables And Guards

锁定模型几何、初始化算法、数据顺序规则、batch、optimizer、scheduler、bias
规则、probe tokens、保存点和分析代码。

关键 guards：

1. actual-input hook 重放 Gate logits，相对误差 $\le10^{-5}$，top-1 一致率 1；
2. calibration/probe hash 与 split 无泄漏；
3. basis 正交误差 $\le10^{-4}$，eigenvalues 降序且覆盖 768 维；
4. raw gradient 不被诊断调用改写；有/无诊断的一步参数与 loss 一致；
5. $b$ 无 `.grad`、不进入 optimizer，更新计数恰为每 optimizer step 一次；
6. checkpoint-resume 后下一步 batch、loss、$W$、$b$ 与 uninterrupted replay
   在注册容差内一致；
7. loss/gradient/update/eigenvalues 全部有限，无少于 6 个有效层；
8. 若任层连续 20 steps 的最大负载份额 $>0.8$ 或有 4 个以上 dead experts，
   标为 load guard failure，不临时改变 bias 强度。

## 8. Primary Metric

令每个 heavy snapshot 的模型级 contrast 为六层 eligible $B$ 的中位数。方向
null 保持每层 $C_EW$ 的非零奇异值，只随机化右奇异子空间；每 snapshot/seed
使用 256 个 Haar-Stiefel samples 得到联合 q95。

$T_{form}$ 是最早的 heavy snapshot token count，满足：

1. $B_{H:M}$ 与 $B_{H:T}$ 同时超过匹配 null q95；
2. paired calibration bootstrap 的两个 95% lower bounds 均大于 0；
3. 六层中至少四层的两个 contrast 同为正；
4. 随后两个注册 heavy snapshots 仍满足 1--3。

单位为 training tokens。它定位可辨认的形成时刻，不是形成的因果效应量。

## 9. Dynamics Decomposition And Secondary Metrics

对相邻 heavy snapshots $a\to b$，计算完整 crossing $B(W_s,U_t)$，并报告
固定基底 Gate effect $\Delta_WB$ 与固定 Gate basis effect $\Delta_UB$。

每 optimizer step，在最近一次注册 $U$ 上记录：

$$
G_B(\nabla W),\qquad G_B(\Delta W),
$$

及有符号交叉项

$$
C_B=\frac{2}{d_B}
\langle C_EWU_B,C_E\Delta WU_B\rangle_F,
\quad
Q_B=\frac1{d_B}\|C_E\Delta WU_B\|_F^2,
$$

使 $\Delta G_B=C_B+Q_B$ 可被核对。

另报：fine 12×64；$V_B^\perp$、$S_B^\perp$；route flip、margin、实际与
weight-only load、dead experts、capacity drop（本实现无硬容量上限，应为 0）；
bias norm/update；专家参数更新范数、专家内梯度冲突；$C_EW$ singular values
与 stable rank。所有这些均为解释指标，不替代 $T_{form}$。

## 10. Known Cases And Debug Controls

- 随机初始化的 orientation null 必须覆盖随机方向波动；
- 对 Gate 右乘随机正交矩阵、保持奇异值时，分析代码不得保留原 head 结论；
- 人工令 $U_t=U_0$ 的 debug replay 必须使 $\Delta_UB=0$；
- 人工冻结 $W$ 的 replay 必须使 $\Delta_WB=0$；
- bias-only 不得改变 $G_B(W,U)$，否则频谱实现错误。

## 11. Logging And Figure Contract

轻日志每 optimizer step 写：tokens/LR/loss、raw Gate gradient、applied update、
signed cross term、clip、margin、load、dead experts、bias、step time。

heavy snapshots：step 0；1--100 每步；101--1000 每 10 steps；之后在
1,100/1,250/1,500/1,750/2,000/2,250/2,544 保存。每个 heavy snapshot 写
$U/\lambda/G/V/S$、fine bands、probe routes、$W$、$b$ 和 checkpoint pointer。
主线只保留紧凑表与图，不复制 raw checkpoints。

**中心图 1：**tokens 对 $B_{H:M},B_{H:T}$，显示 null q95、三个 seeds 与
$T_{form}$。允许结论为形成时间与 seed 稳定性；不能证明因果。

**中心图 2：**形成窗口的 stacked decomposition：raw gradient、applied
update、$\Delta_WB$、$\Delta_UB$。允许区分权重/基底/optimizer；不能排除专家
优势谱。

## 12. Execution Contract

### Smoke（当前已授权）

- 一个 8×5090 ACP job、DDP 8 processes、seed 17；
- 与 full 相同的 6-layer/768/8E/top-1/no-LB/bias 实现；
- 工程缩减为 seq len 256、global batch 16、fresh 24 optimizer steps，保存
  step 24 后 exact-resume 到 step 26；
- heavy snapshots 0/1/2/4/8/16/24/26，calibration/probe 各 8 sequences；
- smoke pass：8 GPUs/NCCL、fresh+resume、DCLM loader、actual-input replay、
  raw/applied update恒等式、basis、bias 无梯度与一次更新、manifest 全部通过；
- smoke 不计算科学 $T_{form}$ verdict，也不允许用 26 steps 的方向趋势作结论。

### Full（已授权）

3 seeds，各到 2,544 optimizer steps / 2.000683008B 名义 tokens。任意训练配置、
bias 规则或保存密度修改，都必须先修订 Protocol 并重新批准。

## 13. Pass / Fail / Insufficient

**Pass：**3 个有效 seeds 中至少 2 个在 2B 内得到有限 $T_{form}$；形成窗口
$\Delta_WB$ 为正且不由 $\Delta_UB$ 单独解释；raw gradient/applied update 的
差异完整报告。结论限定为“真实签名存在于至少 2/3 seeds”。

**Fail：**3 个有效 seeds 中至少 2 个没有有限 $T_{form}$，或可辨认的形成主要
由 basis drift 产生而 Gate-weight contribution 不支持。Fail 不否定 E03-S。

**Insufficient：**少于 3 个有效 seeds，或训练、负载、probe、actual-input、
gradient/update、保存密度、checkpoint/resume 任一关键 guard 失败。

## 14. Claim Boundary And Next Decision

通过最多说明：在注册的 6-layer DCLM 模型、AdamW 与 auxiliary-loss-free bias
条件下，真实 Gate 在 2B tokens 内出现与受控谱加速相容的早期形成签名。

不能声称：covariance 是唯一原因、无 bias 的 native 训练相同、middle/tail
没有功能、Gate 奇异值集中等于 head alignment、频谱 Router 改善训练效率。

**唯一下一决策：**三个 seed 的完整轨迹与分解守卫完成后，按注册规则裁定
full run 的 pass / fail / insufficient。
