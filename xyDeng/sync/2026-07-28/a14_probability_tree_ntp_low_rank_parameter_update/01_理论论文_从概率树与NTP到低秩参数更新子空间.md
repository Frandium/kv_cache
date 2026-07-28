# 从概率树与 NTP 到低秩参数更新子空间

## 共享 mappings 如何经 pure SGD 形成 task spike，并在条件下主导完整权重

**文档状态：**自包含理论草案 v0.4；实验 Protocol 保持待审批
**主证明对象：**注册参数矩阵 $W_T=W_0+\Delta W_T$ 的 rank-$kr$ 外相对谱能量
**精确定理范围：**固定特征、共享正 margin、循环暴露的 contrast-NTP writer 与 pure SGD
**条件推广范围：**learned residual MLP 的 $W_2$；一层 Transformer 的 $W_O/W_2$
**不声称：**有限时刻的 $W_T$ 具有低代数秩，或完整 Transformer 参数元组低有效秩

---

## 摘要

本文研究：概率树中的共享组合结构，何时会通过 next-token prediction（NTP）和 pure SGD，使一个从 Gaussian 满秩初始化出发的参数矩阵形成低有效秩。

对每个 prefix-position prediction event，概率活动树调用一个全局 mapping 库。若高概率活动头部只含 $k$ 个跨上下文共享的 causal mappings，每个 mapping 传递至多 $r$ 个任务自由度，并且受控 writer 以固定坐标实现这些 mappings，则 writer 输入的任务部分严格位于维数

$$
\dim U_\star\le kr
$$

的空间 $U_\star$。这是概率树进入证明的第一处必要环节；对普通 learned MLP，“共享 mapping 自动产生固定内部坐标”并非定理，而是必须用干预验证的实现假设。

对 full-position NTP，所有 prefix-position 事件共享同一 writer。单事件梯度为

$$
\nabla_W\ell_{e,t}=\delta_{e,t}a_{e,t}^{\top}.
$$

因此只要 $a_{e,t}\in U_\star$，pure SGD 的中心化更新满足

$$
\operatorname{rank}(\Delta W_T)\le kr.
$$

为说明 NTP 不只是外积梯度的背景，我们进一步定义 mapping exposure。对有限的 contrast-NTP 事件集，若每个事件对同一 separator 具有正 margin，并在 pure SGD 中获得无限累计步长质量，则仅由参数路径长度和 logistic pressure 的正性即可推出 pressure-weighted exposure 与结构化 task spike 发散。精确受控情形中 $R_T=0$；近似情形若空间外偏差和 martingale 波动满足

$$
\frac{\|R_T\|_F}{\|S_T\|_F}
\xrightarrow{\mathbb P}0,
$$

则

$$
\Delta W_T=S_T+R_T,
\qquad
\operatorname{rank}(S_T)\le q_\star\le q_0=kr,
\qquad
\|S_T\|_F\to\infty.
$$

Gaussian 初始化 $W_0$ 几乎必然满秩、旋转不变；在高维极限中其奇异值形成有宽度的连续 bulk，而非严格相等的“平谱”。令

$$
W_T=S_T+N_T,
\qquad
N_T:=W_0+R_T.
$$

最佳 rank-$q_0$ 逼近恒等式与奇异值扰动界给出

$$
\tau_{kr}(W_T)
\le
\frac{\|N_T\|_F^2}
{\bigl(\|S_T\|_F-\|N_T\|_F\bigr)^2}
$$

以及 task-outlier 的算子范数门。故若

$$
\frac{\|W_0+R_T\|_F}{\|S_T\|_F}\to0,
$$

则

$$
\tau_{kr}(W_T)\to0.
$$

这证明：在固定特征、共享正 margin、循环暴露的 full-position contrast-NTP writer 中，pure SGD 从 Gaussian 满秩初始化出发，完整权重在有限时刻一般仍满秩，但其 rank-$kr$ 外相对谱能量渐近趋零。对 learned MLP 和 Transformer，概率树到稳定 activation envelope、以及指定 writer block 持续承担 margin 增长，仍是明确的条件而非已证事实。

**关键词：**概率活动树；next-token prediction；mapping exposure；pure SGD；task spike；Gaussian bulk；有效秩

---

# 1. 问题与证明对象

## 1.1 核心问题

本文回答：

> 概率树中跨 prefixes 和 contexts 复用的少量 causal mappings，能否通过 full-position NTP 的重复监督和 pure SGD，在 Gaussian 满秩基座上形成维数受 $kr$ 控制、最终主导参数能量的 task spike？

## 1.2 主指标与记号约定

令

$$
n:=\min(d,m),
\qquad
q_0:=kr\le n,
$$

其中 $q_0$ 是由 $k$ 个 mappings、每个至多 $r$ 个自由度给出的**注册秩预算**。T1 定义实际任务空间 $U_\star$ 后，记

$$
q_\star:=\dim U_\star\le q_0.
$$

对任意非零矩阵 $A$ 和整数 $0\le u<n$，定义 rank-$u$ 外相对谱能量

$$
\tau_u(A)
:=
\frac{\sum_{j>u}\sigma_j(A)^2}{\|A\|_F^2},
$$

以及保留至少 $1-\varepsilon$ 谱能量所需的最小秩

$$
r_{\mathrm{en}}(A;\varepsilon)
:=
\min\{u:\tau_u(A)\le\varepsilon\}.
$$

全文主指标固定为 $\tau_{q_0}=\tau_{kr}$；$r_{\mathrm{en}}$ 是其等价能量秩表述，$r_{\mathrm{ent}}$ 仅表示 entropy effective rank。logistic residual pressure 统一记为 $\pi_e(W)$；相干投影记为 $\mathsf C_T$；随机增长下界与其确定性比较尺度分别记为 $\Gamma_T$ 和 $\underline{\Gamma}_T$。

## 1.3 必须区分的五个对象

| 对象 | 本文结论 |
|---|---|
| $\operatorname{rank}(\Delta W_T)$ | 精确共享包络下至多为 $q_\star\le q_0$ |
| $\tau_{q_0}(\Delta W_T)$ | 近似共享下由 $\|R_T\|_F/\|S_T\|_F$ 控制 |
| $\operatorname{rank}(W_T)$ | 不声称低；Gaussian 基座几乎必然满秩 |
| $W_T$ 的 task outliers | task spike 超过算子范数 bulk 门时出现 |
| $\tau_{q_0}(W_T)$ | task spike 在 Frobenius 能量上主导时趋零 |

“出现离群奇异值”严格弱于“完整矩阵低有效秩”。

# 2. 概率活动树与 causal mapping

## 2.1 Prediction events

令

$$
e=(i,\ell)
$$

表示样本 $i$ 在 prefix $X_{i,1:\ell}$ 后预测 $Y_{i,\ell}=X_{i,\ell+1}$ 的事件。full-position NTP 的事件集为

$$
\mathcal E_{\mathrm{full}}
=
\{(i,\ell):1\le\ell<L_i\}.
$$

root-only 事件集 $\mathcal E_{\mathrm{root}}$ 是其预注册子集。

## 2.2 概率活动树

在概率空间 $(\Omega,\mathcal F,\mathbb P)$ 上，每个事件 $e$ 具有活动树

$$
\mathcal T_e=(\mathcal V_e,\mathcal E_e,J_e),
$$

其中

$$
J_e(u)\in\{1,\ldots,K\}
$$

指定节点 $u$ 调用的 mapping。节点状态递归为

$$
s_u
=
f_{J_e(u)}
\bigl(\{s_v:v\in\operatorname{ch}(u)\},z_u\bigr),
$$

其中 $z_u$ 是该节点新增的局部 task innovation。

活动 mapping 集为

$$
\mathcal J_e
=
\{J_e(u):u\in\mathcal V_e\}.
$$

## 2.3 mapping 的因果含义

令 $z_e^\star\in\mathbb R^V$ 为 teacher 的 next-token logits，$\Pi_V=I-\mathbf1\mathbf1^\top/V$ 为去除共同 logit shift 的 contrast 投影。mapping $j$ 只有在合法配对干预改变 teacher contrast logits 时才被视为 causal：

$$
\operatorname{ACE}_j
=
\mathbb E
\left\|
\Pi_V
\left[
z_e^\star(\operatorname{do}(f_j,c))
-
z_e^\star(\operatorname{do}(f_j,c_0))
\right]
\right\|_2^2
>0.
$$

出现频率、同名生成器 ID 或树形可视化均不能替代该条件。

## 2.4 共享头部与局部自由度

假设存在共享 mapping 头部

$$
\mathcal H_\star\subseteq\{1,\ldots,K\},
\qquad
|\mathcal H_\star|\le k,
$$

且头部 mapping $j$ 的 task coordinate 为

$$
c_{e,j}\in\mathbb R^{r_j},
\qquad
r_j\le r.
$$

单事件活动稀疏度可另外满足 $|\mathcal J_e|\le s$，但跨事件参数预算由 $kr$ 而非 $sr$ 控制。

---

# 3. 概率树实现定理

## 3.1 受控实现

对注册 writer 输入，令 mapping $j$ 使用固定注入

$$
E_j\in\mathbb R^{m\times r_j}.
$$

事件 $e$ 的任务激活为

$$
a_e^{\mathrm{task}}
=
\sum_{j\in\mathcal J_e\cap\mathcal H_\star}
E_jc_{e,j}.
$$

精确受控模型把注册 writer 的输入定义为

$$
a_e:=a_e^{\mathrm{task}}\in\mathbb R^m.
$$

非任务上下文可以决定活动树与 $c_{e,j}$，但不直接写入该注册矩阵；若 learned activation 还含背景分量，它必须进入第 6 节的空间外项 $R_T$，不能被隐去。

定义

$$
U_\star
=
\sum_{j\in\mathcal H_\star}\operatorname{Col}(E_j),
\qquad
P_\star=\operatorname{Proj}_{U_\star}.
$$

## 定理 T1：概率树实现定理

若受控模型按照上述固定 mapping injections 实现活动树，则

$$
a_e^{\mathrm{task}}\in U_\star
$$

且

$$
\boxed{
q_\star
:=
\dim U_\star
\le
\sum_{j\in\mathcal H_\star}r_j
\le kr
=q_0.
}
$$

### 证明

每个 $E_jc_{e,j}$ 位于 $\operatorname{Col}(E_j)$，故其和位于这些列空间之和。子空间维数次可加：

$$
\dim U_\star
\le
\sum_{j\in\mathcal H_\star}\operatorname{rank}(E_j)
\le
\sum_{j\in\mathcal H_\star}r_j
\le kr.
$$

证毕。

## 3.2 learned-realization 假设

对 learned MLP 或 Transformer，概率树只定义 teacher/task 结构，不自动指定内部坐标。必须显式假设并验证：

$$
a_{e,t}
=
P_\star a_{e,t}
+
\epsilon^a_{e,t},
\qquad
\operatorname{rank}(P_\star)\le kr,
$$

其中 $P_\star$ 跨 contexts 和训练路径固定，且累计空间外写入受控。

**状态：未证明。** 配套 Protocol 用 mapping interventions 在 calibration 上估计 $P_\star$，在 held-out contexts 上测量 capture 和 projector drift。

---

# 4. full-position NTP 与 mapping exposure

## 4.1 注册 writer

考虑固定特征线性 writer

$$
o_e(W)=Wa_e,
\qquad
W\in\mathbb R^{d\times m}.
$$

同一个 $W$ 被所有样本、prefixes 与 positions 复用；这正是不同 prediction events 能在同一参数矩阵中累积 exposure 的参数共享条件。

T2--T3 与 T6 的精确情形使用第 3 节的 $a_e=a_e^{\mathrm{task}}\in U_\star$；learned 情形允许空间外分量并由 T4 单独控制。

对一般 softmax CE，单事件梯度始终为

$$
\nabla_W\ell_{e,t}
=
\delta_{e,t}a_e^\top.
$$

因此概率树实现的 $U_\star$ 直接进入参数更新的右行空间。

## 4.2 精确 contrast-NTP 模型

正式增长定理先对二类 next-token contrast 给出。令固定输出 contrast $b_e\in\mathbb R^d$，标签符号 $y_e\in\{-1,+1\}$，并定义

$$
X_e
=
y_e b_ea_e^\top,
\qquad
m_e(W)
=
\langle W,X_e\rangle_F.
$$

logistic NTP loss 为

$$
\ell_e(W)
=
\log(1+\exp[-m_e(W)]).
$$

令 residual pressure

$$
\pi_e(W)
=
-\frac{d\ell_e}{dm_e}
=
\frac{1}{1+\exp[m_e(W)]}
>0.
$$

则

$$
-\nabla_W\ell_e(W)
=
\pi_e(W)X_e.
$$

二词表 NTP 与上述模型完全等价；一般 multiclass softmax 作为条件推广。

## 4.3 raw exposure 与 effective exposure

第 $t$ 步 batch 为 $\mathcal B_t$，事件权重 $\omega_{e,t}\ge0$，且和为 $1$。mapping $j$ 的 raw exposure 为

$$
\mathsf{Exp}^{\mathrm{raw}}_j(T)
=
\sum_{t<T}\eta_t
\sum_{e\in\mathcal B_t}
\omega_{e,t}
\mathbf1\{j\in\mathcal J_e\}.
$$

考虑 margin 饱和后的有效学习压力，定义

$$
\mathsf{Exp}^{\mathrm{eff}}_j(T)
=
\sum_{t<T}\eta_t
\sum_{e\in\mathcal B_t}
\omega_{e,t}
\mathbf1\{j\in\mathcal J_e\}
\pi_e(W_t).
$$

总 effective exposure 为

$$
\mathsf A_T
=
\sum_{t<T}\eta_t
\sum_{e\in\mathcal B_t}
\omega_{e,t}\pi_e(W_t).
$$

把各 mapping 的有效暴露相加：

$$
\mathsf E_T^{\mathrm{map}}
:=
\sum_{j\in\mathcal H_\star}
\mathsf{Exp}^{\mathrm{eff}}_j(T).
$$

若每个注册事件调用至少一个、至多 $s$ 个头部 mappings，则

$$
\mathsf A_T
\le
\mathsf E_T^{\mathrm{map}}
\le
s\mathsf A_T.
$$

full-position NTP 是否比 root-only 提供更强强化，由两者的 $\mathsf{Exp}^{\mathrm{eff}}_j$ 决定，而不是由 objective 名称自动决定。

## 4.4 pure SGD

主更新为

$$
W_{t+1}
=
W_t
-
\eta_tG_t,
\qquad
G_t
=
\sum_{e\in\mathcal B_t}
\omega_{e,t}\nabla_W\ell_e(W_t),
$$

不含 momentum、weight decay 或逐坐标预条件。

---

# 5. NTP 相干强化定理

## 5.1 共享 margin 条件

假设存在 $\|Q_\star\|_F=1$、$\operatorname{Row}(Q_\star)\subseteq U_\star$ 和 $\gamma>0$，使所有被采样事件满足

$$
\langle Q_\star,X_e\rangle_F\ge\gamma.
$$

该条件表示所有 prefix-position contrasts 在同一任务 separator 上相干，而不是要求完整梯度平行。

定义结构化更新

$$
S_T
:=
-\sum_{t<T}\eta_tG_tP_\star.
$$

定义其在共享 separator 上的相干投影

$$
\mathsf C_T
:=
\langle S_T,Q_\star\rangle_F.
$$

精确受控情形中 $X_eP_\star=X_e$。

## 定理 T2：NTP exposure 的相干增长下界

在受控树实现、contrast-NTP、共享 margin 和 pure SGD 条件下，并假设每个注册事件调用 $1$ 至 $s$ 个头部 mappings，则

$$
\operatorname{rank}(S_T)\le q_\star\le q_0=kr
$$

且

$$
\boxed{
\mathsf C_T
\ge
\gamma\mathsf A_T,
\qquad
\|S_T\|_F
\ge
\gamma\mathsf A_T
\ge
\frac{\gamma}{s}\mathsf E_T^{\mathrm{map}}.
}
$$

### 证明

由负梯度形式，

$$
S_T
=
\sum_{t<T}\eta_t
\sum_{e\in\mathcal B_t}
\omega_{e,t}\pi_e(W_t)X_e.
$$

每个 $X_e$ 的行空间位于 $U_\star$，所以 $S_T$ 的秩不超过 $q_\star$。再与 $Q_\star$ 取 Frobenius 内积：

$$
\mathsf C_T
=
\sum_{t,e}
\eta_t\omega_{e,t}\pi_e(W_t)
\langle X_e,Q_\star\rangle_F
\ge
\gamma\mathsf A_T.
$$

由 Cauchy--Schwarz 和 $\|Q_\star\|_F=1$ 得 $\|S_T\|_F\ge\gamma\mathsf A_T$；再由每个事件至多调用 $s$ 个头部 mappings，$\mathsf E_T^{\mathrm{map}}\le s\mathsf A_T$。证毕。

## 推论 T2A：full-position 的有限时间强化条件

设 full-position 与 root-only 使用相同 $Q_\star,\gamma$。若对某 $c>1$，

$$
\mathsf A_T^{\mathrm{full}}
\ge
c\,
\mathsf A_T^{\mathrm{root}},
$$

则 full-position 的相干增长下界至少大 $c$ 倍。

当每个事件的 active mapping 数固定为 $s_0$ 时，上式等价于比较 $\mathsf E_T^{\mathrm{map}}/s_0$。该条件必须由数据生成器的 mapping coverage、loss normalization 和实际 margin pressure 验证。若 root-only 已提供无限相干 exposure，它也可能达到同一渐近低有效秩；本文不声称 full-position 名称本身构成充分优势。

## 定理 T3：循环 NTP 暴露下的 pure-SGD task-spike 发散

考虑有限的固定 contrast-NTP 事件集。第 $t$ 步只采样事件 $e_t$，并执行

$$
W_{t+1}
=
W_t+\eta_t\pi_{e_t}(W_t)X_{e_t}.
$$

假设：

1. T2 的共享正 margin 成立；
2. $0<\eta_t<\infty$，且每个事件获得无限累计步长质量：
   $$
   \sum_{t:e_t=e}\eta_t=\infty
   \qquad
   \text{for every }e;
   $$
3. 不使用显式正则、momentum 或预条件。

则在精确受控实现中

$$
\boxed{
\operatorname{rank}(S_T)\le q_\star\le q_0,
\qquad
\mathsf A_T\to\infty,
\qquad
\mathsf E_T^{\mathrm{map}}\to\infty,
\qquad
\|S_T\|_F\to\infty.
}
$$

### 证明

由共享 margin 与 $M_X:=\max_e\|X_e\|_F<\infty$，

$$
\gamma\mathsf A_T
\le
\mathsf C_T
\le
M_X\mathsf A_T.
$$

反设 $\mathsf A_T$ 有界。则参数路径总长度满足

$$
\sum_t\|W_{t+1}-W_t\|_F
\le
M_X\sum_t\eta_t\pi_{e_t}(W_t)
=
M_X\mathsf A_\infty
<\infty,
$$

故 $W_t$ 收敛到某个有限矩阵 $W_\infty$。事件集有限，且 logistic pressure

$$
\pi_e(W)=\frac{1}{1+\exp\!\bigl(\langle W,X_e\rangle_F\bigr)}
$$

在每个有限 $W$ 上连续且严格为正，因此存在 $c>0$，使充分大 $t$ 上所有事件均满足 $\pi_e(W_t)\ge c$。于是任取事件 $e$，

$$
\mathsf A_\infty
\ge
\sum_{t\ge t_0:e_t=e}\eta_t \pi_e(W_t)
\ge
c\sum_{t\ge t_0:e_t=e}\eta_t
=\infty,
$$

矛盾。故 $\mathsf A_T\to\infty$。T2 随即给出 $\mathsf C_T\to\infty$ 与 $\|S_T\|_F\to\infty$，而 $\mathsf E_T^{\mathrm{map}}\ge\mathsf A_T$；秩上界仍由 T1 与 pure-SGD 行空间保持得到。证毕。

固定正步长下，若每个事件具有正采样概率，则 iid sampling 或 random reshuffling 以概率一满足累计步长条件。该定理不声称训练 loss 趋零、参数方向收敛、multiclass softmax 或共同训练特征的对应结论。

---

# 6. 近似包络与空间外更新

令

$$
\Delta W_T
=
S_T+R_T,
$$

$$
R_T
:=
-\sum_{t<T}\eta_tG_t(I-P_\star).
$$

相对于训练 filtration $\mathcal F_t$，写

$$
G_t(I-P_\star)=b_t+\xi_t,
\qquad
\mathbb E[\xi_t\mid\mathcal F_t]=0.
$$

定义

$$
B_T=\sum_{t<T}\eta_t\|b_t\|_F,
\qquad
V_T=
\sum_{t<T}\eta_t^2
\mathbb E[\|\xi_t\|_F^2\mid\mathcal F_t].
$$

## 定理 T4：小空间外偏差下的 task-spike dominance

令 $\Gamma_T:=\gamma\mathsf A_T$。若存在确定性尺度 $\underline{\Gamma}_T\uparrow\infty$，使

$$
\Pr(\Gamma_T\ge \underline{\Gamma}_T)\to1,
$$

$$
\frac{B_T}{\underline{\Gamma}_T}\xrightarrow{\mathbb P}0,
\qquad
\frac{\mathbb E[V_T]}{\underline{\Gamma}_T^2}\to0,
$$

则

$$
\frac{\|R_T\|_F}{\|S_T\|_F}
\xrightarrow{\mathbb P}0.
$$

### 证明

martingale 正交性给出

$$
\mathbb E
\left\|
\sum_{t<T}\eta_t\xi_t
\right\|_F^2
\le \mathbb E[V_T].
$$

由 Markov 不等式，随机项除以 $\underline{\Gamma}_T$ 后依概率趋零；bias 假设给出 $B_T/\underline{\Gamma}_T\xrightarrow{\mathbb P}0$。T2 又给出 $\|S_T\|_F\ge\Gamma_T\ge \underline{\Gamma}_T$ 以趋一概率成立，因此 $\|R_T\|_F/\|S_T\|_F\xrightarrow{\mathbb P}0$。证毕。

任务相关 tail mapping 若产生持续非零 $b_t$，可能使 $B_T=\Theta(\Gamma_T)$；它不能被命名为 nuisance 后自动消失。

---

# 7. Gaussian bulk、task outlier 与完整权重有效秩

## 7.1 Gaussian 基座

初始化为

$$
(W_0)_{ab}
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma_0^2/m),
$$

且独立于任务树与 mapping coordinates。

**本文直接证明并使用：**

1. $\operatorname{rank}(W_0)=\min(d,m)$ 几乎必然成立；
2. 对任意固定正交矩阵 $U,V$，$UW_0V\overset d=W_0$，故初始化无任务方向偏好。

作为高维直觉，当 $d/m\to\beta\in(0,\infty)$ 时，非零经验谱形成边缘位于

$$
\lambda_\pm=\sigma_0^2(1\pm\sqrt\beta)^2
$$

的连续 bulk；该极限定律不进入 T5--T6 的证明。

Gaussian bulk 不是严格相等的奇异值序列。

## 7.2 完整权重分解

令

$$
W_T
=
W_0+\Delta W_T
=
S_T+N_T,
\qquad
N_T:=W_0+R_T,
$$

且 $\operatorname{rank}(S_T)\le q_\star\le q_0$。

## 定理 T5：task outlier 与完整权重谱尾

对任意 $T$：

$$
\sigma_{q_0+1}(W_T)
\le
\|N_T\|_{\mathrm{op}},
$$

$$
\sigma_{q_0}(W_T)
\ge
\sigma_{q_0}(S_T)-\|N_T\|_{\mathrm{op}}.
$$

因此若

$$
\sigma_{q_0}(S_T)>2\|N_T\|_{\mathrm{op}},
$$

则

$$
\sigma_{q_0}(W_T)>\sigma_{q_0+1}(W_T),
$$

即存在严格的第 $q_0$ 个谱隙。

若

$$
\|S_T\|_F>\|N_T\|_F,
$$

则

$$
\boxed{
\tau_{q_0}(W_T)
\le
\frac{\|N_T\|_F^2}
{\bigl(\|S_T\|_F-\|N_T\|_F\bigr)^2}.
}
$$

### 证明

奇异值扰动不等式给出前两式。对谱尾，$S_T$ 是一个 rank-$q_0$ 候选近似；最佳 rank-$q_0$ 逼近恒等式给出

$$
\sum_{j>q_0}\sigma_j(W_T)^2
\le
\|W_T-S_T\|_F^2
=
\|N_T\|_F^2.
$$

反三角不等式给出

$$
\|W_T\|_F
\ge
\|S_T\|_F-\|N_T\|_F.
$$

相除即得。证毕。

## 推论 T5A：完整权重低有效秩

若

$$
\rho_F(T)
:=
\frac{\|W_0+R_T\|_F}{\|S_T\|_F}
\to0,
$$

则

$$
\boxed{
\tau_{q_0}(W_T)=\tau_{kr}(W_T)\to0.
}
$$

等价地，对任意固定 $\varepsilon>0$，充分大的 $T$ 满足

$$
r_{\mathrm{en}}(W_T;\varepsilon)\le q_0.
$$

## 推论 T5B：entropy effective rank

令

$$
p_j(T)=\frac{\sigma_j(W_T)^2}{\|W_T\|_F^2},
\qquad
r_{\mathrm{ent}}(W_T)
=
\exp\!\left[-\sum_{j=1}^n p_j(T)\log p_j(T)\right].
$$

当 $n=\min(d,m)$ 固定时，若 $q_0<n$、$\tau_{q_0}(W_T)\to0$，则

$$
\limsup_{T\to\infty}r_{\mathrm{ent}}(W_T)\le q_0.
$$

若宽度 $n$ 与 $T$ 同时增长，还需 $\tau_{q_0}(W_T)\log n\to0$ 等联合控制，不能直接使用固定维数推论。

---

# 8. 主定理

## 定理 T6：概率树—NTP—SGD—完整权重有效秩

考虑固定特征的二类 full-position contrast-NTP writer。假设：

1. 概率活动树的共享头部含至多 $k$ 个 causal mappings；
2. 每个 mapping 至多传递 $r$ 个 task freedoms；
3. 注册 writer 只接收 $a_e=\sum_jE_jc_{e,j}$，其中固定 $E_j$ 实现 mapping coordinates；
4. 全部注册 prefix-position events 对同一 $Q_\star$ 具有正 margin；
5. finite event set 作 singleton 更新，且每个事件满足 $\sum_{t:e_t=e}\eta_t=\infty$；
6. 优化器为无 momentum、无 weight decay、无预条件的 pure SGD；
7. $W_0$ 为独立 Gaussian 初始化；
8. $d,m,k,r$ 在 $T\to\infty$ 时固定，且 $kr\le\min(d,m)$。

令注册预算 $q_0:=kr$，实际任务维数为 $q_\star:=\dim U_\star\le q_0$。则：

$$
\operatorname{rank}(\Delta W_T)\le q_\star\le q_0
$$

对所有 $T$ 成立，

$$
\|\Delta W_T\|_F\to\infty,
$$

且

$$
\boxed{
\tau_{q_0}(W_T)=\tau_{kr}(W_T)\to0.
}
$$

因此完整 $W_T$ 在有限时刻一般仍满秩，但其能量秩渐近进入概率树预算 $kr$。

### 证明

T1 给出 $q_\star=\dim U_\star\le q_0=kr$。NTP 外积梯度和 pure SGD 给出 $\operatorname{Row}(\Delta W_T)\subseteq U_\star$。T2--T3 给出 mapping exposure 的相干累积与 $\|\Delta W_T\|_F\to\infty$。精确受控情形 $S_T=\Delta W_T$、$R_T=0$，故

$$
\rho_F(T)=\frac{\|W_0\|_F}{\|\Delta W_T\|_F}\to0.
$$

应用 T5A 即得。证毕。

## 条件推广 T6A

若 learned model 存在固定 $P_\star$、结构化部分满足 $\|S_T\|_F\to\infty$，并且

$$
\frac{\|R_T\|_F}{\|S_T\|_F}
\xrightarrow{\mathbb P}0,
$$

则同样有

$$
\tau_{kr}(W_T)\to0
$$

依概率成立。

该条件不能由“teacher 有树结构”或“NTP 被使用”单独推出。

---

# 9. learned MLP 与一层 Transformer

## 9.1 residual MLP

对

$$
m_{e,t}
=
\phi(W_{1,t}\operatorname{LN}(h_{e,t})),
\qquad
h^+_{e,t}=h_{e,t}+W_{2,t}m_{e,t},
$$

$W_2$ 的梯度仍为

$$
\nabla_{W_2}\ell_{e,t}
=
\delta^2_{e,t}m_{e,t}^\top.
$$

但 T6 只有在以下条件同时成立时才能推广：

$$
m_{e,t}=P_\star m_{e,t}+\epsilon^m_{e,t},
\qquad
\operatorname{rank}(P_\star)\le kr,
$$

$$
\frac{
\sum_t\eta_t\|G_{2,t}(I-P_\star)\|_F
}{\|S_{2,T}\|_F}
\to0,
$$

$$
\|S_{2,T}\|_F\to\infty.
$$

共同训练的 $W_1$、LayerNorm 和输出头可能旋转坐标或承担 margin 增长，所以这些条件是**条件证明**，不是架构定理。

## 9.2 一层 Transformer

对 attention output writer $W_O$ 和 MLP output writer $W_2$，梯度右因子分别为 attention aggregate 和 MLP activation。若各自满足稳定 $kr$ 维包络、相干增长与小累计泄漏，则 T4--T6 可逐块应用。

本文不推广到 $W_Q,W_K,W_V,W_1$、完整参数元组或多层 Transformer。

---

# 10. full-position NTP 的必要性边界

full-position NTP 在证明中承担两个具体对象：

1. 它定义所有 prefix-position events 的共享 writer 更新；
2. 它决定每个 mapping 的 raw/effective exposure 和相干 margin mass。

但是，低有效秩的逻辑必要条件是“无限且相干的低维 exposure”，不是 objective 名称。若 root-only loss 已经以共享正 margin 反复暴露同一组 mappings，它也可满足 T6。反之，若只有非 ROOT positions 提供某些 mappings 的可识别监督，root-only 会缺失相应 exposure。

因此 Protocol 必须比较 full-position 与 root-only 的 exposure、相干度和谱增长，不能只比较最终 rank。

---

# 11. two-phase、功能充分性与当前边界

two-phase 只保留一个直觉：任务方向可能先稳定，对应 singular gain 随后继续增长；它不进入 T1--T6 的证明依赖。

低参数谱也不自动表示任务功能充分。实验必须冻结其余参数，对 rank-$kr$ update 和结构投影 update 做 held-out replay，并与随机等秩方向比较。

现有受控 AdamW 实验支持参数位移与 finite task effect 可以低维，但没有验证本文新增的 Gaussian + pure-SGD + separable-NTP 增长链。

---

# 12. 可证伪预测

1. **受控实现：**shared-tree oracle writer 的任务激活并集维数不超过 $kr$。
2. **结构必要性：**disjoint-tree 在保持单事件 $s,r$ 不变时，其跨样本 union budget 增大。
3. **NTP exposure：**full-position 相对 root-only 的 task-spike 增长差异应由 $\mathsf{Exp}^{\mathrm{eff}}_j$ 与 gradient coherence 解释。
4. **中心化更新：**精确受控 pure SGD 满足 $\tau_{kr}(\Delta W_T)\approx0$。
5. **outlier 门：**$\rho_{\mathrm{op}}<1/2$ 时应出现严格第 $kr$ 个谱隙。
6. **完整有效秩门：**只有 $\rho_F$ 足够小，$\tau_{kr}(W_T)$ 才应进入低尾区。
7. **learned bridge：**capable MLP 若 activation intervention subspace 不跨 contexts 稳定，则概率树实现假设失败。
8. **功能门：**任务 rank-$kr$ replay 应优于随机等秩 replay。

---

# 13. 最终结论

## 已证明

- 受控概率树中的 $k$ 个共享 mappings、每个至多 $r$ 个自由度，严格产生维数至多 $kr$ 的 writer-input task space；
- full-position contrast-NTP 的 pressure-weighted exposure 在共享 margin 下给出 task-spike 增长下界；
- pure SGD 保持该 writer-input 行空间，且循环累计步长条件使 exposure 与 task spike 发散；
- 固定维数的精确模型中，发散 task spike 自动压过固定 Gaussian $W_0$，故完整 $W_T$ 满足 $\tau_{q_0}(W_T)=\tau_{kr}(W_T)\to0$；近似模型还需 $\|R_T\|_F/\|S_T\|_F\xrightarrow{\mathbb P}0$。

## 条件证明

- 小 bias/martingale leakage 下的近似包络；
- multiclass softmax NTP；
- learned residual MLP 的 $W_2$；
- 一层 Transformer 的 $W_O/W_2$。

## 未证明

- 概率树自动决定 learned network 的内部坐标；
- full-position 一定比任何 root-only objective 更快或更低秩；
- 完整 Transformer 参数元组、hidden state 或自然语言模型普遍低有效秩。

**唯一下一决策：**执行配套 Protocol 的 shared/disjoint-tree × full-position/root-only 因果比较，先验证 learned residual MLP 是否实现概率树的 $kr$ 维 activation envelope，并检查 exposure、相干增长、$\tau_{kr}(\Delta W_T)$、$\rho_F$ 与 $\tau_{kr}(W_T)$ 是否按定理链同时闭合。

---

# 文件入口

全文使用的推导均在本文或 [02_证明附录.md](02_证明附录.md) 内给出；假设与反例见 [03_物理先验与假设审计.md](03_物理先验与假设审计.md)。实验 Protocol 不包含在本 theory-only 同步包中。
