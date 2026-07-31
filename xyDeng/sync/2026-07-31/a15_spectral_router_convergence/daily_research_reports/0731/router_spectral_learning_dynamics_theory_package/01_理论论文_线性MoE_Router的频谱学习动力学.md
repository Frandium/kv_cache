---
document_status: final_reviewed
theorem_core_status: human_approved
prepared: 2026-07-30
package_date: 2026-07-31
---

# 线性 MoE Router 的频谱学习动力学

## 从各向同性无偏好到各向异性的有限时间谱加速

## 术语约定

- **Router 输入：**线性 Gate 实际接收的表征，而不是送入专家网络的表征；
- **covariance 谱：**Router 输入协方差矩阵的特征值及其方向；
- **head / middle / tail：**按 covariance 特征值从大到小划分的高、中、低方差
  方向，不表示词频、语义重要性或功能价值；
- **专家优势：**同一个 token 交给不同专家时的相对损失差，它定义 Router 应当
  学习的功能区分；
- **有限时间谱加速：**在尚未完全收敛的训练窗口内，大方差方向更快达到相同
  相对学习进度，不表示最终只能使用这些方向。

## 摘要

本文研究线性 Mixture-of-Experts（MoE）Router 在不同输入 covariance 谱下的
学习动力学。核心问题并非大方差方向是否机械地产生更大的 logit，而是：当
各方向承载的专家优势信号可比时，训练是否会更快地让 Gate 学会区分大
covariance 特征值方向。

我们首先在专家固定时给出精确的梯度恒等式。Gate 的任务更新由 Router 输入
与**专家优势**之间的交叉协方差决定；这里的专家优势，是不同专家处理同一
token 时的相对损失差。如果所有专家完全相同，输入 covariance 无论多么不平，
都不能独自启动有功能的路由学习。只有当专家优势已随输入变化时，covariance
特征值才会作为乘子进入各方向的 Gate 更新。

随后，我们在一个与非饱和 softmax Gate 的局部行为一致的二次模型中给出闭式
动力学。若输入各向同性，所有方向具有相同时间常数，covariance 不指定任何优先方向；
单个随机种子仍可破缺对称性，但不能解释为谱头偏好。若输入各向异性，而且
专家优势信号在谱方向上可比，则大特征值方向具有更短学习时间：早期 Gate
增益按特征值更快增长。无正则时，中低方差方向最终可以追上；带各向同性
$L_2$ 正则时，大特征值方向还会获得更大的稳态系数。

这一结论有明确条件。若专家优势只存在于 tail、优化器有效白化了坐标、Gate
已经饱和，或表征基底快速旋转，各向异性都不再保证 head alignment。专家与
Router 共同训练既可能形成正反馈，也可能补偿甚至逆转谱偏置；在没有进一步
约束专家优势如何形成时，不能把正反馈写成定理。现有 A15 结果中，最早可见的
10k checkpoint 已强烈偏向 head，随后 middle/tail 相对追上。这与“早期谱
加速、后期收敛展宽”相容，但仍不是因果证明。

## 1. 为什么研究这个问题

线性 Gate 为每个 token 产生一组专家 logits。若 Router 输入在某些方向上的
方差更大，即使 Gate 对所有方向使用相同的权重尺度，这些方向也会机械地产生
更大的 raw logit 波动。因此必须区分三件事：

1. **输入能量效应：**大方差方向在固定权重下机械放大输出；
2. **学习速度效应：**同样的功能信号在大方差方向上是否更快进入 Gate；
3. **功能目的：**专家损失差是否真的要求 Router 使用该方向。

A15 E01/E02 已在 Gate 的实际输入上控制了第一种机械效应。给各方向相同能量后，
10k checkpoint 的 head 增益仍约为 middle 的 9--10 倍、tail 的 37--43 倍。
但从 10k 到 30k，相对 head 比值下降，middle/tail 在追上。这提出一个更窄的
理论问题：这种形状是否来自大特征值方向更短的有限时间学习常数，而不是一个
“训练永远把 Gate 推向 head”的普遍定律？

本文使用**谱各向异性**来指 covariance 特征值不全相等，不再使用“谱奇异”。
因为矩阵“奇异”通常表示不可逆或含零特征值，与本文讨论的“谱不平”不是同一
概念。

## 2. 形式化对象

### 2.1 Router 输入与 covariance

令中心化后的实际 Router 输入为

$$
x\in\mathbb R^d,
\qquad
\mathbb E[x]=0,
\qquad
\Sigma=\mathbb E[xx^\top]
=U\Lambda U^\top.
$$

$U=[u_1,\ldots,u_d]$，特征值按
$\lambda_1\ge\cdots\ge\lambda_d\ge0$ 排列。本文的 head、middle、tail
只表示特征值排序区间，不预设语义。

有 $E$ 个专家，线性 Gate 为

$$
z=Wx,
\qquad
p=\operatorname{softmax}(z),
\qquad
W\in\mathbb R^{E\times d}.
$$

所有专家共同增加的 logit 不改变路由。定义

$$
C_E=I_E-\frac1E\mathbf 1\mathbf 1^\top,
\qquad
\bar W=C_EW.
$$

理论与实验都只把 $\bar W$ 视为有效 Gate。

### 2.2 固定专家损失与专家优势

对 token $x$，令专家 $e$ 的冻结损失为 $\ell_e(x)$，并记

$$
\ell(x)=(\ell_1(x),\ldots,\ell_E(x))^\top.
$$

定义中心化专家优势

$$
a(x)=-C_E\ell(x).
$$

$a_e(x)>0$ 表示专家 $e$ 相对其他专家损失更低。专家优势是 Router 应当区分
token 的功能目标；covariance 只描述输入变化大小，不给出哪个专家更好。

### 2.3 频带 Gate 增益

对方向集合 $B$，令 $U_B$ 收集相应 eigenvectors，$d_B=|B|$。等能 Gate
增益为

$$
G_B(W,U)
=\frac1{d_B}\|\bar WU_B\|_F^2.
$$

它衡量每个方向获得相同输入能量时的平均专家相对 logit 增益。它不包含
$\lambda_i$ 的 raw response 放大，也不等于功能收益。

## 3. 固定专家下的精确 Gate 梯度

考虑 soft routing 的期望专家损失：

$$
L_{\rm soft}(W)
=\mathbb E_x[p(Wx)^\top\ell(x)].
$$

### 引理 1：精确梯度恒等式

对任意 $W$，

$$
\nabla_WL_{\rm soft}
=\mathbb E_x\left[
(\operatorname{Diag}(p)-pp^\top)\ell(x)x^\top
\right].
$$

**证明。** softmax Jacobian 为
$J_p=\operatorname{Diag}(p)-pp^\top$。对单个样本，

$$
\frac{\partial(p^\top\ell)}{\partial z}=J_p\ell,
\qquad
\frac{\partial z}{\partial W}=x.
$$

链式法则给出单样本梯度 $(J_p\ell)x^\top$，再对 $x$ 取期望即得。∎

在对称初始化 $W=0$ 时，$p=\mathbf1/E$，因此

$$
J_p=\frac1E C_E.
$$

梯度流 $\dot W=-\nabla_WL$ 满足

$$
\dot{\bar W}(0)
=\frac1E\mathbb E[a(x)x^\top].
$$

这给出第一个必要边界：

> 若所有专家对每个 token 的损失相同，则 $a(x)=0$，Gate 的任务梯度为零。
> covariance 再各向异性，也不能单独产生功能性专家区分。

### 推论 1：局部线性专家优势

假设

$$
a(x)=Ax+\varepsilon(x),
\qquad
\mathbb E[\varepsilon(x)x^\top]=0,
\qquad
C_EA=A.
$$

则

$$
\dot{\bar W}(0)=\frac1E A\Sigma.
$$

沿 covariance eigenvector $u_i$，有

$$
\dot{\bar W}(0)u_i
=\frac{\lambda_i}{E}Au_i.
$$

因此大特征值会放大**已有的专家优势信号**，但若 $Au_i=0$，该方向仍没有
功能更新。

## 4. 可完整求解的局部学习模型

上节给出 softmax Gate 在平衡点的精确初始梯度。为了描述完整学习时间，考虑
其局部二次 surrogate：

$$
L_q(\bar W)
=\frac\kappa2
\mathbb E\|\bar Wx-Ax\|_2^2
+\frac\beta2\|\bar W\|_F^2,
$$

其中 $\kappa>0$ 表示局部 softmax 曲率或学习率尺度，$\beta\ge0$ 是各向同性
$L_2$ 正则。该模型把固定专家优势对应的目标 score 写成 $Ax$。它是一个精确
可解的局部替代模型，只用于刻画模式学习时间，并不等同于完整的 top-1 MoE
训练。

梯度流为

$$
\dot{\bar W}
=-\kappa(\bar W-A)\Sigma-\beta\bar W.
$$

定义每个谱方向上的 Gate 和目标列向量：

$$
w_i(t)=\bar W(t)u_i,
\qquad
a_i=Au_i.
$$

于是各方向解耦：

$$
\dot w_i
=-(\kappa\lambda_i+\beta)w_i
+\kappa\lambda_i a_i.
$$

### 引理 2：闭式动力学

令

$$
w_i^\star
=\frac{\kappa\lambda_i}{\kappa\lambda_i+\beta}a_i.
$$

则

$$
w_i(t)
=w_i^\star
+e^{-(\kappa\lambda_i+\beta)t}
\bigl(w_i(0)-w_i^\star\bigr).
$$

**证明。** 这是常系数一阶线性微分方程。将 $w_i^\star$ 代入可验证其为固定
点；对 $w_i-w_i^\star$ 求导得到
$\frac d{dt}(w_i-w_i^\star)=-(\kappa\lambda_i+\beta)
(w_i-w_i^\star)$，积分即得。∎

若 $w_i(0)=0$，则

$$
w_i(t)=q_i(t)a_i,
$$

$$
q_i(t)
=\frac{\kappa\lambda_i}{\kappa\lambda_i+\beta}
\left(1-e^{-(\kappa\lambda_i+\beta)t}\right).
$$

相对于自身稳态达到比例 $\rho\in(0,1)$ 的时间为

$$
T_i(\rho)
=\frac{-\log(1-\rho)}{\kappa\lambda_i+\beta}.
$$

## 5. 各向同性谱：没有 covariance 指定的方向偏好

### 定理 1：各向同性时间常数相同

若 $\Sigma=\lambda I$，则所有方向共享相同系数 $q(t)$：

$$
\bar W(t)
=q(t)A
+e^{-(\kappa\lambda+\beta)t}\bar W(0).
$$

因此 covariance 不会把任何预指定正交子空间变成 head；Gate 的方向来自
$A$ 或初始化，而不是来自谱排序。

**证明。** 各方向 $\lambda_i=\lambda$，引理 2 中的 $q_i(t)$ 完全相同。将
$w_i(t)$ 按 $U$ 的各列重新组合即得。∎

如果 $A$ 和初始化的分布也旋转等变，则对任意正交矩阵 $Q$，旋转输入与目标
后 $\bar W(t)Q$ 的分布不变。因而对任意预先指定的 $r$ 维子空间 $P$，期望
Gate 能量占比只由维数决定：

$$
\mathbb E
\frac{\|\bar W(t)P\|_F^2}{\|\bar W(t)\|_F^2}
=\frac rd.
$$

这并不禁止单个有限样本或随机 seed 偶然选择某个方向；它只说明不能把这种
选择解释为 covariance 预先指定了 head。

## 6. 各向异性谱：有条件的有限时间谱加速

### 定义：谱方向上均衡的专家优势

若

$$
\|Au_i\|_2^2=c
$$

对所有注册方向相同，称目标专家优势在谱方向上均衡。更一般地，只需要不同
频带的 $\|Au_i\|^2$ 可比，不能强到抵消特征值差异。

### 定理 2：大特征值方向学习更快

假设 $w_i(0)=0$、专家优势谱均衡，并且
$\lambda_i>\lambda_j>0$。则对任意有限 $t>0$，

$$
q_i(t)>q_j(t),
\qquad
\|w_i(t)\|_2^2>\|w_j(t)\|_2^2,
$$

且对任意 $\rho\in(0,1)$，

$$
T_i(\rho)<T_j(\rho).
$$

**证明。** $\kappa\lambda/(\kappa\lambda+\beta)$ 和
$1-e^{-(\kappa\lambda+\beta)t}$ 都随 $\lambda$ 严格增加，故 $q_i(t)>q_j(t)$。
谱均衡给出 $\|a_i\|=\|a_j\|$，所以 Gate 能量同序。学习时间公式的分母也随
$\lambda$ 严格增加。∎

对任意频带 $B$，零初始化时

$$
G_B(t)
=\frac1{d_B}\sum_{i\in B}q_i(t)^2\|a_i\|_2^2.
$$

在谱均衡条件下，若 head 的每个特征值都大于 middle，middle 又大于 tail，
则任意有限时间都有

$$
G_H(t)>G_M(t)>G_T(t).
$$

### 6.1 早期极限

当 $t\downarrow0$，

$$
q_i(t)=\kappa\lambda_i t+O(t^2),
$$

所以

$$
\|w_i(t)\|_2^2
=\kappa^2\lambda_i^2t^2\|a_i\|_2^2+O(t^3).
$$

这说明早期 Gate 权重能量对谱差异非常敏感。

### 6.2 无正则的后期追赶

若 $\beta=0$，则

$$
q_i(t)=1-e^{-\kappa\lambda_i t}
\longrightarrow1.
$$

因此

$$
w_i(t)\longrightarrow a_i.
$$

若专家优势谱均衡，各方向的最终 Gate 能量相同。大方差方向只是更早学习，
不必永久占优。因此，理论自然给出“早期 head 强、随后 middle/tail 追上”的
可检验预测。

### 6.3 正则化的稳态偏置

若 $\beta>0$，

$$
w_i^\star
=\frac{\kappa\lambda_i}{\kappa\lambda_i+\beta}a_i.
$$

即使 $a_i$ 等强，大特征值方向也受到更少的相对收缩，因而可保留最终的 head
偏置。这一结论只直接适用于显式 $L_2$ 正则下的梯度流。真实 AdamW 同时包含
解耦的 weight decay 与自适应预条件，不能与上述结果直接等同。

## 7. 为什么“只要谱不平就必然对齐 head”不成立

定理 2 需要专家优势谱均衡。对一般 $A$，方向 $i$ 比方向 $j$ 更强的充要比较
是

$$
q_i(t)^2\|Au_i\|_2^2
>
q_j(t)^2\|Au_j\|_2^2.
$$

因此存在以下直接反例。

### 反例 1：功能信号只在 tail

若对全部 head 方向 $Au_i=0$，而某个 tail 方向 $Au_j\ne0$，则

$$
w_i(t)=0,
\qquad
w_j(t)\ne0.
$$

无论 $\lambda_i/\lambda_j$ 多大，Router 都只能学习具有专家优势信号的 tail。

### 反例 2：专家完全相同

若 $\ell_e(x)$ 对所有专家相同，则 $a(x)=0$。引理 1 给出
$\dot W(0)=0$。各向异性不能自动创建专家功能差异。

### 反例 3：白化或理想预条件

若 Gate 实际接收 $\Sigma^{-1/2}x$，其 covariance 为单位阵，定理 1 适用。
类似地，若优化器精确乘以 $\Sigma^{-1}$ 抵消梯度中的 $\Sigma$，方向时间常数
可被拉平。

因此，本文能够证明的最强结论是：

> 各向异性 covariance 是一个条件性优化偏置。它在专家优势方向可比且优化器
> 不消除尺度时，使大特征值方向在有限时间更快进入 Gate；它不是功能目的，
> 也不是最终 head alignment 的充分条件。

## 8. 可训练专家：分阶段动力学

真实 MoE 中专家参数 $\theta_e(t)$ 与 Gate 同时变化。定义时间相关专家优势

$$
a_t(x)=-C_E\ell(x;\theta_1(t),\ldots,\theta_E(t))
$$

及其输入交叉协方差

$$
M_t=\mathbb E[a_t(x)x^\top].
$$

在 Gate 接近平衡、softmax 未饱和时，引理 1 给出局部关系

$$
\dot{\bar W}_t
=\frac1E M_t+\text{higher-order terms}.
$$

若在 $\Sigma$ 的支撑上定义最佳线性专家优势系数

$$
A_t=M_t\Sigma^\dagger,
$$

则 $M_t=A_t\Sigma$，并有

$$
\dot{\bar W}_tu_i
\approx\frac{\lambda_i}{E}A_tu_i.
$$

这里 $\lambda_i$ 仍作为乘子出现，但 $A_t$ 已由专家学习与当前路由共同决定。
据此可把联合训练分为四个概念阶段。

### 阶段 0：同质专家

$A_t\approx0$，Gate 几乎没有任务区分信号。负载损失、初始化噪声或专家参数
差异可以触发对称破缺，但不能被称为 covariance 功能选择。

### 阶段 1：微小专家优势形成

若 $A_tu_i$ 在不同方向可比，则 $\lambda_i$ 让 head Gate 模式更快增长。这是
固定专家定理最接近真实训练的局部窗口。

### 阶段 2：Router—Expert 反馈

已有路由改变每个专家接收的 token，从而改变 $A_t$。若 head 划分让专家在
同一 head 方向上获得更强相对优势，则形成正反馈；若专家开始利用 middle/tail
减少残差，$A_t$ 也可向中低方差扩展。两种方向都允许。

### 阶段 3：margin 饱和或分区锁定

softmax Jacobian $\operatorname{Diag}(p)-pp^\top$ 在极端概率处变小，有限时间
谱加速可能停止。负载和容量约束也可能迫使 Router 使用原先较弱的方向。

因此，“可训练专家产生额外 head 正反馈”是 E03-S 的 S2 阶段假设，而不是
本文在无附加专家模型下已经证明的结论。

## 9. Head 对齐不等于 Gate 矩阵更奇异

$\bar W\in\mathbb R^{E\times d}$ 满足

$$
\operatorname{rank}(\bar W)\le E-1.
$$

head alignment 描述 $\bar W$ 的右方向相对 covariance 基底的位置；Gate
奇异值集中描述其至多 $E-1$ 个有效输出方向是否由少数方向主导。二者独立。

例如，令 $A$ 有七个等强正交右方向，且全部位于 64 维 head 中。定理 2 可使
Gate 强烈 head-aligned，但七个非零奇异值仍相等。反之，一个位于 tail 的
rank-1 $A$ 会产生高度奇异值集中的 Gate，却完全不偏 head。

因此，E03-R 必须分别报告：

$$
G_H/G_M,
\qquad
G_H/G_T,
$$

以及

$$
r_{\rm stable}(\bar W)
=\frac{\|\bar W\|_F^2}{\|\bar W\|_2^2},
$$

而不能用其中一个替代另一个。

## 10. 表征与 Gate 共同演化

真实 Router 输入写成 $x_t$，其 covariance 基底为 $U_t$。观测到的
$G_B(W_t,U_t)$ 变化同时包含：

1. Gate 权重 $W_t$ 的变化；
2. 表征基底 $U_t$ 的变化；
3. 两者的交互。

上述定理固定 $U$。真实实验必须交叉计算 $W_s\times U_t$，并分别记录 raw
Gate gradient、优化器实际施加的更新和有符号 $W$--$\Delta W$ 频带交叉项。
否则只能观察共同对齐，不能说 covariance 因果训练了 Gate。

## 11. 可检验预测

理论给出五条区分性预测：

1. **平谱：**在谱均衡专家优势下，不应对任意预指定方向产生系统偏好；
2. **剂量关系：**谱隙越大，head 与 middle/tail 的学习时间比越大；
3. **白化：**消除特征值差异后，速度排序应消失；
4. **tail-only：**若专家优势只在 tail，Router 应能学习 tail，否定表达失能；
5. **真实早期：**若 A15 的 head alignment 来自有限时间谱加速，它应在训练
   早期迅速形成，随后在无持续正反馈时被 middle/tail 相对追赶。

E03-S 负责前四条因果预测；E03-R 负责第五条真实轨迹预测。

## 12. 与现有实验的关系

A15 E01/E02 已直接支持：

- 训练后的 Gate 在实际 Router 输入上具有非随机等能 head alignment；
- middle/tail 的增益和当前 route effect 非零；
- 10k 时偏头最强，10k--30k 的 H:M 固定基底 Gate 效应为负；
- 晚期没有两个 contrast、两个谱系都持续增强的证据。

2026-07-30 完成的 E03-S 固定目标受控实验进一步给出注册范围内的因果支持：
flat 条件的 head/middle/tail 50% 拟合时间均约为 141 steps；4:2:1 谱约为
56/111/223；16:4:1 谱约为 29/114/457；强谱白化后重新回到约 141/141/141。
2,048 个 flat-rotation null、谱隙剂量关系和 tail-only capability guards 均通过。
因此本节前四条受控预测已获实验支持；该结果不验证 AdamW、可训练专家反馈或
真实 DCLM 的形成原因。[结果记录](../../../main/experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)

首个 E03-R 三-seed 正式配置在 step100（约 78.6M tokens）前均出现注册的
20-step 最大专家份额超过 0.8，实际窗口最大值约 0.99；因此轨迹按 Protocol
停止并判为 `insufficient_load_guard`。停止前各 seed 的两组 head contrast
尚未同时超过方向 null。该结果定位的是负载稳定 operationalization，不是定理
或真实形成命题的反证。

这些观察与定理 2 的早期加速，以及 $\beta=0$ 时的后期追赶相容。但 10k 已约为
7.86B 名义 tokens，且没有逐步 gradient/optimizer 记录。因此当前证据不能
区分：

- covariance 谱加速；
- 专家优势本来偏 head；
- AdamW 预条件或正则效应；
- Router 表征基底共同旋转；
- 早期 Router—Expert 正反馈。

## 13. 总结

本文没有证明“只要 covariance 不平，Router 最终必然对齐 head”。它证明的
是一个更精确、也更有限的关系：

$$
\text{专家优势信号}
\times
\text{covariance 特征值}
\times
\text{优化器响应}
\longrightarrow
\text{Gate 模式学习速度}.
$$

各向同性谱不会由 covariance 指定优先方向。各向异性谱只有在专家优势可比、
优化器不消除尺度差异时，才会让 head 更快进入 Gate；无正则时 middle/tail
可以在后期追上，而正则和专家反馈可能保留、放大或逆转最终偏置。

## 14. 可能推进方向

1. **E03-S 固定目标（已完成）：**学习时间、平谱旋转对称、谱隙剂量关系、
   whitening 与 tail-only 注册判据均通过；
2. **E03-S 联合专家：**比较 frozen 与 trainable experts，判断是否出现超出
   固定优势模型的反馈放大；
3. **E03-R（需修订负载条件）：**首个正式配置因 step100 前负载守卫失败而
   insufficient；若继续，应先冻结并验证仍无 LB auxiliary loss 的负载稳定规则，
   再从初始化到 2B tokens 记录 $W_t$、raw gradient、optimizer update、$U_t$、
   margin、load 和 Gate 有效秩；
4. **理论扩展：**处理 top-1 分段光滑边界、AdamW 对谱时间常数的改变，以及
   $U_t$ 随上游网络训练的共同动力学；
5. **功能分支：**单独检验浅层 head 是否提供深层共同训练兼容性的增量信息。
   该问题不由本文的学习速度定理回答。
