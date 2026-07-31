---
anchor_id: 15_00_01_spectral_learning_dynamics
parent_anchor: 15_00_covariance_head_gate_alignment
status: controlled_pass_real_insufficient_load_guard
canonical_language: en
canonical_file: 15_00_01_spectral_learning_dynamics_anchor.md
updated: 2026-07-30
---

# A15_00_01 线性 Router 的条件性频谱学习动力学


## 1. Problem Definition

父 anchor 已证明训练 endpoint 存在等能 head alignment，但未定位 10k 前的形成
原因。本 subanchor 只问：

> 在匹配专家优势信号的谱分布，并分开优化器与表征运动后，covariance 各向
> 异性是否因果性地缩短高方差 Gate 模式的学习时间；该签名是否出现在真实
> MoE 的前 2B tokens？

“谱加速”指某频带达到自身注册目标拟合比例所需时间更短，不是 raw logit 更大、
功能效用更高，或任意各向异性任务最终都对齐 head。

受控主指标为

$$
R_{M:H}(\rho)=\frac{T_M(\rho)}{T_H(\rho)},
\qquad
R_{T:H}(\rho)=\frac{T_T(\rho)}{T_H(\rho)}.
$$

$T_B(\rho)$ 是频带 $B$ 达到已知目标拟合比例 $\rho$ 所需的更新数或 tokens；
比值无量纲。它判断学习速度，不能证明专家功能价值。

## 2. Physical Priors

1. **条件性 covariance 乘子：**平衡且未饱和的 Gate 更新由专家优势与输入的
   交叉协方差决定；优势系数匹配时，大特征值缩短模式时间常数。
2. **各向同性对称：**平谱不指定 head；系统对齐预选方向必须来自任务、
   优化器或有限样本破缺。
3. **专家反馈不自动为正：**可训练专家可以放大、补偿或逆转初始排序。

## 3. Falsifiable Hypotheses

**H1：**专家优势谱均衡时，各向异性产生 $R_{M:H}>1$、$R_{T:H}>1$ 及随谱隙
增强的剂量关系；平谱或 whitening 消除排序。真实训练早期形成正的等能 head
contrast，并能把 raw gradient、实际 optimizer update 与表征贡献分开。

**最强 rival R1：**专家优势目标本来就偏 head；旋转目标或把目标移到 tail
即可改变学习方向，与 covariance 无因果关系。

**R2：**自适应优化器或表征基底漂移制造 endpoint 对齐；固定表示下有效，
真实 $W_t\times U_t$ 分解却不能归因到 Gate 更新。

**Pass：**固定专家受控测试支持 H1，whitening 消除速度排序，tail-only 仍能
学会，并且真实训练出现注册的早期签名。

**Fail：**有效且匹配信号后，各向异性不改变学习时间；或表面效应在 whitening
后仍存在，并由目标方向解释。

**Insufficient：**能力、专家优势、数值、保存密度或真实训练稳定性护栏失败；
或受控因果通过但真实签名无法确定。

## 4. Mathematical Model

对实际 Router 输入 $x$，令 $\Sigma=U\Lambda U^\top$、$\bar W=C_EW$，中心化
专家优势为 $a(x)=-C_E\ell(x)$。平衡 softmax Gate 满足

$$
\dot{\bar W}(0)=\frac1E\mathbb E[a(x)x^\top].
$$

若 $a(x)=Ax+\varepsilon$ 且
$\mathbb E[\varepsilon x^\top]=0$，则

$$
\dot{\bar W}(0)u_i=\frac{\lambda_i}{E}Au_i.
$$

注册局部二次模型为

$$
\dot w_i=-(\kappa\lambda_i+\beta)w_i
+\kappa\lambda_i a_i,
$$

模式学习时间为

$$
T_i(\rho)=\frac{-\log(1-\rho)}{\kappa\lambda_i+\beta}.
$$

完整证明和反例见
[self-contained 理论文档](../../../../../daily_research_reports/0731/router_spectral_learning_dynamics_theory_package/01_理论论文_线性MoE_Router的频谱学习动力学.md)。
该模型不能自行决定联合训练中的时间变化专家优势 $A_t$。

## 5. Computational Realization

**E03-S：**已知 covariance 基底、8-output 线性 Gate、注册专家优势目标；用
flat、anisotropic、whitened 和 tail-only 分开 covariance、任务方向和预条件。
固定优势通过后才进入 trainable-expert 阶段。

**E03-R：**小型 top-1 DCLM MoE 从初始化训练到最多 2B tokens；记录 $W_t$、
raw Gate gradient、实际 optimizer update、固定 probe 的 $U_t$、完整
$W_s\times U_t$ crossing、有符号频带交叉项、margin、flip、load 和
$C_EW_t$ 奇异谱。训练不使用 load-balance auxiliary loss；只使用可随 checkpoint
恢复、不反传的 expert-score bias 作为负载护栏，并把它与 $W_t$ 分开报告。

## 6. Minimal Falsification Tests

1. 固定优势下，以相同目标、初始化分布、优化器、样本和更新预算比较 flat 与
   至少一个 anisotropic 谱；
2. whiten anisotropic 输入后要求速度排序消失；把全部优势放入 tail 后要求
   tail 可学习；
3. 固定优势测试有效后，才比较 frozen 与 trainable experts 的额外反馈；
4. 真实训练以匹配方向 null 定义首次持久 head alignment，并分解 Gate 权重与
   表征基底运动。

## 7. Current Evidence

[E01](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E01_actual_router_input_band_response/summary_cn.md)
与
[E02](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E02_early_head_alignment_onset/summary_cn.md)
已经支持实际 Router 输入上的强等能 head alignment、非零 middle/tail 访问和
10k 后的相对展宽；没有观察形成时刻，也没有分开专家优势、优化器与表征原因。

E03-S 的正式记录包括
[Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/protocol_cn.md)、
[Summary](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/summary.md)
和
[Detailed evidence ledger](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_S_controlled_spectral_learning_dynamics/detailed.md)。
full run 对注册的受控 S0/S1 条款给出科学 **PASS**；该 PASS 只属于受控条款。

经审计的目标定义为

$$
A_{gate}=\tau A_{raw},\qquad \tau=0.25,
$$

其中 $A_{raw}$ 是方向匹配的专家分数系数，$A_{gate}$ 是 Gate logit 可达到的
目标。因此，$F_B$ 和 $T_B$ 比较的是 $W_t$ 与 $A_{gate}$；whitened 条件先把
学得权重映回原频谱坐标再计算。

八个 seeds 中，moderate 4:2:1 各向异性的中位数
$(D_{M:H},D_{T:H})=(0.69268,1.38588)$，strong 16:4:1 为
$(1.38477,2.77145)$，均超过匹配 flat-rotation null 的 q95
$(0.003277,0.003019)$。strong whitening 把中位数恢复到
$(0.000053,0.000637)$，位于 flat 95% 区间内；tail-only 目标在每个 seed 上的
held-out KL 降低均至少为 0.9999939。因此，在固定基底 Gaussian、纯 SGD 的
受控构造中，covariance 各向异性因果性地改变 Gate 模式的有限时间学习速度；
tail 较慢不能由“tail 目标无法表达或学不会”解释。

该结果不裁定真实 DCLM Router 为何形成 head alignment。E03-R 的正式记录包括
[Protocol](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/protocol_cn.md)、
[Summary](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/summary.md)
和
[Detailed evidence ledger](../../../../experiments/A15/15_00_covariance_head_gate_alignment/A15_00_E03_R_real_early_spectral_learning_dynamics/detailed.md)。
三个冻结源码的 full run 得到科学
**INSUFFICIENT（`insufficient_load_guard`）**：每个 seed 都出现某一专家占一个
20-step 层内负载的 80% 以上，首次失败窗口分别结束于 step 72、75、79；到
step 100，rolling maximum 已接近 0.99。这个历史有效性护栏无法恢复，因此
三个 job 均被停止。

step 50 及以前的已选有效点都不是双 contrast orientation-null candidate。
seed 29 和 43 在 step 120 的两个 contrast 均超过 orientation-null q95，但此时
负载已接近单专家集中，不能进入 basis bootstrap 或 $T_{form}$。actual-input
replay、基底、raw/applied identity、capacity、源码冻结和分析闭环护栏均通过。
因此，E03-R 没有回答真实形成问题，也不是对受控 E03-S 机制的负结果。S1
通过仍使 S2 具备执行资格，但 S2 尚未运行；具备资格不是正专家反馈证据，也
不会自动触发 S2。

## 8. Claim Boundary And Next Decision

**已支持：**在匹配可达 Gate 目标、固定 Gaussian 表征、trace-normalized 谱和
纯 SGD 下，较大的 covariance 特征值会缩短相应线性 Gate 模式的有限学习时间。
flat-spectrum、rotation-null、whitening、dose-response 与 tail-capability 对照
共同支持这一受控因果结论。

**在受控系统内已削弱：**目标方向、有限样本方向不对称、raw-logit 能量放大和
tail 不可表达，都不能解释注册的速度排序。

**尚未解决：**在负载稳定的真实 DCLM 轨迹中是否形成同一签名，它来自 raw
Gate gradient、AdamW 实际 update 还是表征基底运动，以及可训练 experts 是否
进一步放大该效应。注册的 E03-R score-bias 条件没有成为这一问题的稳定载体。
S2 是独立的、eligible 但未执行的实验，不是自动续跑项。

本 subanchor 不能声称 covariance 导致了现有 DCLM endpoint，不能声称所有
训练后的 Router 必然对齐 covariance head，不能把 E03-R 负载崩溃后的 contrast
称为 head formation，不能声称 middle/tail 缺乏功能价值，也不能声称专家反馈
为正或频谱路由改善 validation loss/FLOP。

**唯一下一决策：**是否批准一份新的 E03-R Protocol；它必须先独立验证负载
稳定机制、冻结归因边界，并在任何新的 2B-token full run 之前通过一个小型
pre-full 稳定性门槛。
