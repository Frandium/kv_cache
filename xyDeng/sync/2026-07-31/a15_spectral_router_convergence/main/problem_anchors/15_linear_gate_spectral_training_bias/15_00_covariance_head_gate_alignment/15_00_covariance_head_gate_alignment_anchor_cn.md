---
anchor_id: A15_00_covariance_head_gate_alignment
parent_node: rq.expert_specialization
status: controlled_pass_real_insufficient_load_guard
created: 2026-07-30
updated: 2026-07-31
canonical_language: zh-CN-companion
canonical_anchor: 15_00_covariance_head_gate_alignment_anchor.md
---

# A15_00 实际 Router 输入上的频带访问与训练分配


研究者已在 2026-07-30 批准并执行 E01--E03。E01 支持 30k 之后明显偏 head
的等能访问，但否定持续的晚期固定基底强化；E02 表明该对齐在 10k 已更强，
随后到 30k 相对减弱；E03-S 通过受控 covariance 速度条款，E03-R 则因真实
训练负载守卫先于有效形成而证据不足。见
[E01 结果](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)、
[E02 早期形成结果](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)与
[E03 动力学 subanchor](subanchors/15_00_01_spectral_learning_dynamics_anchor_cn.md)。

## 1. 问题定义

**父问题：** 线性 MoE Router 是否能够使用实际输入中的中低方差信息，而不是
主要沿 covariance 谱头形成专家间 logit 差异？

“Router 看见某个频带”必须拆成四个不同命题：

1. **可访问：** Gate 的专家对比权重在该子空间上有多大等能增益；
2. **当前使用：** 真实 token 在该频带上产生多少 logit 变化，以及去除该带
   是否改变 margin 或 top-1 路由；
3. **训练分配：** checkpoint 间净 Gate 位移是否给该频带更大等能增益，并
   是否增强 endpoint 偏好；
4. **功能效用：** 用该频带分发 token 是否改善 held-out loss 或共同训练
   兼容性。

前三项是 E01/E02 的冻结审计对象；第四项不是。冻结专家是在 native Router 下形成
的，直接用 middle/tail 重新分发会混入专家不匹配、负载和容量变化，不能由 E01
替代功能实验。

**唯一决策问题：** 在被审计的线性 Gate 谱系中，去掉输入方向能量差异后，
训练后的 Gate 是否仍以 head 为主；现有 checkpoint 最迟能把形成时间界定到
哪里；分离表征基底漂移后，保存区间的 Gate 权重变化是在增强还是稀释该偏好？

**主指标：** 粗粒度等能增益对比向量

$$
\mathbf B_\ell^{coarse}(W,U)
=
\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right).
$$

完整的 fine 12×64 $G_{\ell,b}$ profile 是不可省略的定位证据。

**主证伪条件：** 谱头实际响应 $V_H^\perp$ 高，但等能后 head 不高于
middle 或 tail；或者两个训练区间的净 Gate 位移不持续偏向 head，且 endpoint
变化可由表征基底漂移解释。

### 术语与定义

| 术语 | 具体对象 / 计算 | 单位 | 判断作用 | 不能证明 |
| --- | --- | --- | --- | --- |
| 实际 Router 表征 | 直接 hook 到的 mlp.gate 输入及其上游 Router reference | activation | 唯一允许的 covariance 对象 | 专家输入几何 |
| coarse head | eigen-ranks 1--64 | 64 directions | 最大方差组 | 语义 common |
| coarse middle | eigen-ranks 65--320 | 256 directions | 中方差组 | 功能价值 |
| coarse tail | eigen-ranks 321--768 | 448 directions | 低方差组 | 功能价值 |
| fine band | 每 64 个相邻 eigen-ranks 一组，共 12 组 | 64 directions | 显示组内峰值和非单调结构 | 语义频率 |
| 等能增益 $G_A$ | $\|C_EWU_A\|_F^2/d_A$ | logit²/activation² | Gate 对每个单位方向的访问强度 | token 上的实际使用或效用 |
| 实际响应 $V_A^\perp$ | $\mathbb E\|C_EWP_Ax\|^2$ | logit²/token | 当前数据中该组总共推动多少专家相对 logits | Gate 主动偏好 |
| 当前路由使用 | 去除该组后的 top-1 flip 和 native margin change | token fraction / logit | 当前决策是否依赖该组 | loss 改善 |
| 训练分配 | 净 Gate 位移的频带增益及固定基底 endpoint 变化 | log ratio | checkpoint 区间内权重变化偏向哪里 | 每步梯度或功能效用 |
| 功能效用 | 频带分发对 held-out loss 或更新兼容性的影响 | loss / compatibility | 后续准入判断 | E01/E02 不测 |

## 2. 物理先验

**P1——输入能量会机械放大实际响应。** 即使 Gate 对所有方向的等能增益相同，
大 $\lambda_i$ 仍会让谱头 $V^\perp$ 更高。因此 raw 响应不能单独证明训练
偏好；$G$ 才去掉 covariance 特征值放大。

**P2——优化可能把更多 Gate 增益分配给高方差方向。** 在误差信号相关程度
相近时，高方差坐标可提供更大、更稳定的 Gate 梯度；低方差方向若要产生相同
logit 方差，通常需要更大权重。因此训练后的 $\mathbf B^{coarse}$ 和净更新
对比可能偏向 head。若 middle/tail 的等能增益或净更新不弱，该 prior 被削弱。

**P3——Router 权重与 Router 表征共同演化。** endpoint $W_t$ 与 covariance
基底 $U_t$ 的对齐变化可来自 $W$、$U$ 或二者交互。必须交叉计算
$W_{30/40/80}\times U_{30/40/80}$，不能把 endpoint 差异直接归因于 Gate
训练。

## 3. 可证伪假设

**H1——持续的谱头训练分配。**

1. 在 40k 与 80k，$\mathbf B^{coarse}$ 的两个分量都稳定为正并超过保持 Gate
   奇异值的方向零假设；
2. middle/tail 可以有非零 $G$、$V$ 或 route effect，但相对 head 较弱；
3. 30k--40k 与 40k--80k 两段净 Gate 位移都偏向 head，且固定基底下都增强
   endpoint 的 head-vs-middle 和 head-vs-tail 对比。

**E02 follow-up H1-early——10k 时已经形成。** 在最早共同 checkpoint，两个
head 对比都高于零和保持奇异值的匹配方向 null。

**E02 follow-up H1-progressive——10k--30k 继续强化。** 10k--20k 与
20k--30k 的固定基底 Gate 效应在 H:M、H:T 上都为正。注册 rival 是：偏置更早
形成，之后维持或向中低频带扩展。

**最强 rival R0——只有输入能量支配。** $V_H^\perp$ 高来自
$\lambda_H$ 大；$\mathbf B^{coarse}$ 不偏 head，净 Gate 位移也不偏 head。

**Rival R1——只有表征漂移。** endpoint 对比随 checkpoint 改变，但主要由
$U_t$ 的变化解释；固定基底的 Gate 权重效应接近零。

**Rival R2——阶段特异或非单调训练。** 只有一个区间偏 head，或两个区间方向
相反。此时可以报告 early-only、late-only、saturation 或 reversal，不能称为
持续训练倾向。

| 证据 | H1 | R0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 谱头 $V_H^\perp$ | 高 | 高 | 可能高 | 不限定 |
| $\mathbf B^{coarse}$ | 两分量正 | 平坦/非正 | endpoint 可正 | 随阶段变 |
| 净更新 $\mathbf B^{update}$ | 两区间偏 head | 不偏 head | 不稳定/可假象 | 区间不一致 |
| 固定基底 $\Delta_W\mathbf B$ | 两区间正 | 近零 | 近零 | 区间不一致 |
| 固定 Gate $\Delta_U\mathbf B$ | 不限定 | 不限定 | 解释主要变化 | 不限定 |

**Pass：** 两个谱系都支持 H1。

**Fail：** 有效测量精确支持 energy-only、middle/tail 不弱于 head，或两个
区间都不显示 head-directed Gate 权重变化。

**Insufficient：** 对象/no-op、checkpoint、基底稳定性或不确定性护栏失败，
或区间无法区分正、零和负方向。

没有 25%、10% 或 8/12 层的 practical 硬门槛。Haar q95 和 bootstrap 区间
仅用于判断是否能与随机方向或零区分；完整效应大小和逐层结果必须报告。

## 4. 数学模型

### 4.1 实际 Router 输入与 logit 分解

令 $r_\ell$ 为直接 hook 到的 Gate-effective input。对 LB，$r_\ell=g_\ell$；
对 decommon，$r_\ell=g_\ell-c_\ell$。covariance 基底始终在 Router 真正
送入线性 Gate 的 $r_\ell$ 上定义；上游 $g_\ell$ 只用于验证变换：

$$
x_\ell=r_\ell-\mu_\ell^{(r)},\qquad
\Sigma_\ell=\mathbb E[x_\ell x_\ell^\top]
=U_\ell\Lambda_\ell U_\ell^\top.
$$

若 calibration 矩阵奇异值为 $s_i$，则 $\lambda_i=s_i^2/N$。Router logits
可写成中心/DC 项和频带项之和：

$$
z_\ell=W_\ell r_\ell
=W_\ell\mu_\ell^{(r)}+\sum_AW_\ell P_{\ell,A}x_\ell.
$$

top-1 只依赖专家间差异，因此定义

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad
\bar W_\ell=C_EW_\ell.
$$

### 4.2 双分辨率频带

fine 频带为

$$
F_j=\{64(j-1)+1,\ldots,64j\},\qquad j=1,\ldots,12.
$$

coarse 频带为

$$
H=F_1,\qquad
M=F_2\cup F_3\cup F_4\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

对应维度为 $d_H=64$、$d_M=256$、$d_T=448$。所有跨 coarse 组的增益比较
必须除以方向数；实际总响应另行报告，不能因 tail 维度更大而误读。

### 4.3 可访问、实际响应与部分归一

对任意 coarse 或 fine 方向集合 $A$，

$$
G_{\ell,A}(W,U)
=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2,
$$

$$
V^\perp_{\ell,A}
=\mathbb E\|\bar W_\ell P_{\ell,A}x_\ell\|_2^2,
\qquad
v^\perp_{\ell,A}=\frac{V^\perp_{\ell,A}}{d_A},
$$

$$
S^\perp_{\ell,A}
=\frac{V^\perp_{\ell,A}}
{\mathbb E\|P_{\ell,A}x_\ell\|_2^2}.
$$

$G$ 完全去掉 $\lambda_i$；$S$ 只去掉组总能量，在宽 coarse 组内仍按
$\lambda_i$ 加权。$V$ 是总实际响应，$v$ 是每方向实际响应。

主指标是

$$
\mathbf B_\ell^{coarse}
=\left(B_{\ell,H:M},B_{\ell,H:T}\right)
=\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right).
$$

### 4.4 当前路由使用

令

$$
z_\ell^{(-A)}=z_\ell-\bar W_\ell P_{\ell,A}x_\ell.
$$

定义

$$
F_{\ell,A}
=\Pr[\arg\max z_\ell\ne\arg\max z_\ell^{(-A)}],
$$

$$
D_{\ell,A}
=\mathbb E[m_{\rm native}(z_\ell)-m_{\rm native}(z_\ell^{(-A)})].
$$

$F$ 和 $D$ 说明当前 native 决策是否依赖该组，不说明用该组重新分发会改善
loss。

### 4.5 两段训练分配与 $W/U$ 分解

对预注册的三个 checkpoint 集合 $\mathcal T$，对每个 $s,t\in\mathcal T$
计算完整 $3\times3$ crossing：

$$
\mathbf B_{\ell;s,t}
=\mathbf B_\ell^{coarse}(W_{\ell,s},U_{\ell,t}).
$$

对每个相邻注册区间 $a\rightarrow b$，令
$\Delta W_\ell^{a\to b}=W_{\ell,b}-W_{\ell,a}$。净位移方向为

$$
\mathbf B_{\ell}^{update,a\to b}
=\frac12\left[
\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,a})
+\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,b})
\right].
$$

固定基底的 Gate 主效应为

$$
\Delta_W\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;b,a}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;a,b})
\right],
$$

固定 Gate 的基底主效应为

$$
\Delta_U\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;a,b}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;b,a})
\right].
$$

$\mathbf B^{update}$ 是方向量，可能在极小 $\Delta W$ 上仍很大；必须与
$\|\bar{\Delta W}\|_F$ 和 $\Delta_W\mathbf B$ 一起解释。两个区间长度不同，
只能比较方向与 endpoint 轨迹；未经 step/token 归一不得比较变化速率。

### 4.6 裁定流程

~~~mermaid
flowchart LR
  R["直接 hook 的 Router 表征"] --> C["coarse 3 带 + fine 12×64"]
  C --> V["实际响应 V 与每方向响应 v"]
  C --> G["等能访问 G 与 H:M/H:T 对比"]
  C --> U["去带后的 flip 与 margin"]
  G --> J1{"head 是否仍强于 middle/tail？"}
  V --> J1
  J1 -- "只有 V 偏头" --> E["energy-only"]
  J1 -- "G 也偏头" --> Q["保存 checkpoint 的 W×U 分解"]
  Q --> J2{"何时已形成，后续 W 是否继续强化？"}
  J2 -- "否" --> S["阶段特异、非单调或表征漂移"]
  J2 -- "是" --> H["持续晚期谱头训练分配"]
  U --> B["只说明当前使用，不说明功能效用"]
~~~

## 5. 计算实现

**模型与 checkpoint：** E01 使用 LB/decommon 的 30k/40k/80k；E02 使用
LB/batch-gradient 的 10k/20k/30k。各谱系都是 12 层、width-768、8-expert、
top-1 线性 Gate 模型。所有主状态文件都已只读记录 hash、配置、Gate shape、
expert order、center state、tokenizer 和坐标签名。

**Router 表征：** 必须直接捕获 mlp.gate pre-input、native logits 和上游
Router reference。$h_\ell$ 只作为 known-bad 对照。离线重算必须 replay native
logits 和 top-1。

**数据：** 与训练顶层 shard 分离的 DCLM holdout；32 个固定的 256-token
训练序列拟合 $\mu,U$，64 个 evaluation 来源文档计算 held-out 响应、flip 和
margin。所有 checkpoint 使用相同 token ids 和顺序。训练 binary stream
没有来源文档边界。

**输出粒度：** 每个 model × checkpoint × layer 同时输出 coarse 三组与 fine
十二带；不允许只交付 head/rest 汇总。

**必要对照：** 保持 $\bar W$ 或 $\bar{\Delta W}$ 奇异值的随机输入方向 null；
错误层基底；calibration half-split；中心/DC 单列；频带重构与 native no-op。

**不在 E01/E02：** band-only 或 middle/tail-only 分发、forced-expert loss、一步
交叉更新兼容性、专家训练和 loss/FLOP。

## 6. 最小证伪实验

**决定性实验：** [E01 实际输入多分辨率 Protocol](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/protocol_cn.md)
和 [E02 早期形成 Protocol](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/protocol_cn.md)。

**Endpoint 比较：** 在 30k、40k、80k 给出完整 coarse/fine
$V^\perp/v^\perp/S^\perp/G/F/D$，以
$(B_{H:M},B_{H:T})$ 与零和方向 null 比较。

**训练比较：** 对 30k--40k 与 40k--80k 分别计算 fine
$G(\Delta W)$、coarse $\mathbf B^{update}$、$\Delta_W\mathbf B$ 和
$\Delta_U\mathbf B$。

**Evidence rule：** 不设 practical 效应硬门槛。coarse access 与 trajectory
主比较使用模型级 Haar q95 和配对 calibration-sequence basis bootstrap
区间；held-out use metrics 重采样 evaluation 文档。fine profile 完整报告，
并用 simultaneous envelope 防止从 12 个 band 中事后挑峰。

**Pass：** 两个谱系在 40k/80k 的 coarse head 都高于 middle 与 tail，且两个
区间的净 Gate 位移都偏 head 并在固定基底下增强 endpoint 对比。

**Fail：** 护栏通过且结果精确支持 energy-only、middle/tail 等能增益不弱、
或两个区间都没有 head-directed Gate 权重变化。

**Insufficient / 分型：** 仅一段为正时报告 early-only 或 late-only，不升级为
持续趋势；两段相反时报告 non-monotonic；若由 $U$ 主导则报告
representation-drift-only；护栏或精度不足则判 insufficient。

**允许 claim：** 被审计 Gate 对每个频带的等能访问、当前真实响应和 native
决策依赖，以及两个已保存训练区间的净权重分配方向。

**不能 claim：** middle/tail 的功能效用、用它们分发会改善或损害 loss、每步
梯度方向、从初始化开始的因果机制、所有模型普遍如此。

## 7. 当前证据

**直接结果——实际输入访问：** [E01 证据记录](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)
通过 checkpoint、native-input replay、基底重建/稳定性和保持奇异值方向 null
护栏。40k/80k 时，LB 的 $G_H/G_M$ 中位数为 5.41/6.36、decommon 为
4.03/4.27；$G_H/G_T$ 为 19.98/25.36 与 14.61/17.15。所有 log contrast
的配对 basis 区间都高于零并远超约 0.04 的匹配 Haar q95。六个 endpoint
均以 F1 为最强模型中位 fine band。

**直接结果——当前使用：** middle/tail 的 $G$、response、flip 和 margin
effect 非零但更弱。80k 的 head/middle/tail route-flip 中位数为 LB
0.741/0.126/0.018、decommon 0.645/0.089/0.013。

**直接结果——保存区间：** 每段净位移的 $\mathbf B^{update}$ 都为正并超过
匹配 null；但两个谱系、两个区间的 $\Delta_WB_{H:M}$ 都精确为负。
$\Delta_WB_{H:T}$ 在 LB 两段为正，在 decommon 30k--40k 为负，在
decommon 40k--80k 跨零。两谱系晚期 endpoint H:M 的正向变化包含正的
$U$ contribution，而固定基底 $W$ contribution 为负。

**解释：** energy-only endpoint rival 被否定；“两个 contrast 都持续在固定
基底下增强”的严格 H1 被否定。支持的分型是：endpoint 与净位移都偏 head，
但相对 head 偏好并未持续增强。$\mathbf B^{update}>0$ 不能替代
$\Delta_W\mathbf B$，因为把更新加到一个更强偏头的现有 $W$ 时，还存在有符号
交叉项。

**直接结果——最早可用形成边界：** [E02 证据记录](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)
通过全部护栏。10k 时，LB 的 $G_H/G_M=10.42$、$G_H/G_T=37.11$；
batch-gradient 为 9.19、42.73。对应 log contrast 远高于 0.034--0.048 的
匹配方向 null q95。因此 Router--表征系统在 10k 已形成强且非随机的 head 对齐。

**直接结果——10k--30k 扩展：** 到 30k，上述比值降为 LB 5.38/19.60、
batch-gradient 4.99/24.80。两谱系、两区间的固定基底
$\Delta_WB_{H:M}$ 都为负；batch-gradient 的 $\Delta_WB_{H:T}$ 也都为负，
LB 的 H:T Gate 效应略正，但负的 $\Delta_UB_{H:T}$ 在 endpoint 上超过它。
middle/tail 的 gain 和 route dependence 非零，并相对增加。

**更新解释：** 强 endpoint 对齐在最早可用的 10k 前已经形成，10k--30k 没有
继续锐化。10k 名义上约为 7.86B tokens，且没有初始化 checkpoint，因此 E02
不能定位确切形成点，也不能把 10k 前 Gate gradient 与表征共同适配分开。

**受控因果结果：**
[E03-S 结果](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
通过注册的固定目标检验。在匹配 Gate-space 目标、trace-normalized Gaussian
输入和纯 SGD 下，4:2:1 covariance 产生约 1:2:4 的 head/middle/tail 学习时间，
16:4:1 产生约 1:4:16；flat 与 whitened 条件使三者重新回到同一范围，tail-only
目标也能够学会。因此，covariance 各向异性是注册受控系统中的有限时间速度
因素；它不是功能偏好，也不是真实 DCLM 结论。

**真实轨迹结果：**
[E03-R 结果](../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
的科学裁定为 `insufficient_load_guard`。三个 seed 都在形成时间可被有效判定前，
出现单一专家占某层连续 20 步负载 80% 以上；到 step 100，滚动集中度接近
0.99。诊断链通过，但负载崩溃后的频谱变化不能作为正常 Router--Expert 形成
过程的证据。

## 8. Claim Boundary And Next Decision

**当前支持：** 综合 E01/E02，训练后的 Gate 对实际输入具有可复现、逐层特异
的 head-dominant 等能访问。middle/tail 可访问且当前有使用，但强度更弱。
该对齐在最早可用的 10k 最强；10k--30k 的 endpoint 比值随后下降但仍明显为正。

**当前削弱/否定：** head 的高 $V$ 不是纯输入能量假象；但 10k--30k 与
30k--80k 都不支持两个 head contrast 普遍、持续地在固定基底下增强。只读
$\mathbf B^{update}$ 并解释成“训练让 Gate 越来越偏 head”不成立。

**Q1 内仍未解决：**负载稳定的真实 DCLM 轨迹是否呈现受控实验中的 head-first
签名；raw Gate gradient 与 optimizer 实际 update 的区别；有符号
$W$--update 频带交叉项；表征基底共同演化；以及可训练专家会放大还是补偿
该偏置。新增的局部机制问题是：减均值后是否仍存在跨文档稳定的
centered-common 子空间，以及 Gate 是否优先使用它而非 shard-local residual。

**不能 claim：** 线性 Gate 表达上无法读取 middle/tail、covariance 是现有
DCLM 终点偏头的因果来源、偏头有益或有害、middle/tail 分发的 loss 效果、专家专业化
或 loss/FLOP 改善。

**唯一下一决策：**完成已授权的并行冻结实验：
[A15_00_02 centered-common 稳定性](subanchors/15_00_02_centered_common_subspace_stability_anchor_cn.md)
与
[A15_00_03 pooled-vs-local Gate 偏好](subanchors/15_00_03_gate_transferable_vs_local_residual_alignment_anchor_cn.md)
两份 E01 Protocol。两项实验共享一次激活提取，之后独立并行分析；只有
二者同时通过，才讨论新的匹配稳定性干预训练。该决策仍严格属于 Q1，不测
功能效用，也不授权新训练。
