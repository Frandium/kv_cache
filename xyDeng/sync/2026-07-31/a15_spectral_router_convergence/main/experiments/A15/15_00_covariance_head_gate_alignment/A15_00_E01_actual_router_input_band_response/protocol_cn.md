---
experiment_id: A15_00_E01_actual_router_input_band_response
status: approved_for_implementation_smoke_and_full_run
execution_status: completed_strict_h1_fail_typed_result
result_summary: summary_cn.md
created: 2026-07-30
updated: 2026-07-30
primary_anchor: A15_00_covariance_head_gate_alignment
canonical_protocol: protocol.md
execution_scope: implementation_smoke_and_full_run
---

# Protocol 审核稿：A15_00_E01 实际 Router 输入上的多分辨率频带访问与训练分配

## 0. Approval Snapshot

**批准状态：** APPROVED。研究者于 2026-07-30 批准本 Protocol，并明确授权
实现、smoke 和完整冻结审计；本批准不包含新训练或 band-only dispatch。

**本实验确切回答：**

1. 线性 Gate 对实际 Router 表征的 head、middle、tail 和十二个细频带分别有
   多少等能访问强度；
2. 这些频带在真实 token 上分别产生多少专家相对 logit 响应，以及 native
   top-1 路由是否依赖它们；
3. 30k--40k 与 40k--80k 两个已保存训练区间的净 Gate 权重变化分别偏向哪些
   频带，endpoint 趋势来自 Gate 权重还是 Router 表征基底。

**本实验不回答：** 用 middle/tail 做分发会改善还是损害 held-out loss、共同
训练兼容性、专家形成或训练效率。“训练中效用较弱”在 E01 中只能表述为较弱
的等能增益、当前路由影响或净 Gate 权重分配，不能表述为较弱的功能价值。

**主 anchor：** [A15_00 实际 Router 输入上的频带访问与训练分配](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_00_covariance_head_gate_alignment/15_00_covariance_head_gate_alignment_anchor_cn.md)。

**唯一决策问题：** 相比 head，middle/tail 的等能访问与当前路由使用是否较
弱；两个训练区间是否都把更多 Gate 增益分配给 head，而不是由输入能量或
表征漂移造成表面趋势？

**物理 prior：** 输入方差会机械放大 head 的实际响应；优化还可能额外把更多
Gate 权重增益分配给高方差方向。

**核心模型项：**

$$
G_{\ell,A}(W,U)=\frac1{d_A}\|C_EW_\ell U_{\ell,A}\|_F^2.
$$

**主指标：**

$$
\mathbf B_\ell^{coarse}
=\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right).
$$

fine 12×64 的完整 $G_{\ell,b}$ profile 是强制交付项，不允许只给 coarse 或
head/rest 汇总。

**主 falsifier：** $V_H^\perp$ 高，但 $\mathbf B^{coarse}$ 不偏 head；或
30k--40k、40k--80k 的净 Gate 位移不持续偏 head，且 endpoint 变化主要来自
$U_t$ 漂移。

**实验性质：** frozen root-cause and metric audit。不是功能干预、Router
构造或联合训练。

**最小设置：**

- 两个现有谱系：LB 与 decommon；
- 12 layers、hidden width 768、8 experts、top-1；
- checkpoint 30k、40k、80k；
- 32 个 calibration 训练序列、64 个 evaluation 文档，每个 256 token；
- coarse 三组和 fine 十二带两种分辨率。

**checkpoint 事实：** 2026-07-30 已只读确认两个谱系的 30k、40k、80k
mp_rank_00_model_states.pt 均存在。若 S0 发现 30k 文件损坏或与 40k 坐标不
兼容，预注册 fallback 顺序为共同可用的 20k、再到 10k；fallback 必须发生在
任何主指标读取前并写入 amendment，不能事后按结果选择。

**必须运行的条件：** 六个 checkpoint endpoint；两个区间；完整
$W_{30/40/80}\times U_{30/40/80}$ crossing；coarse/fine 全指标；方向 null；
错误层基底；calibration 分片稳定性；center/DC；native no-op。

**无 practical 硬门槛：** 删除 $\log(1.25)$、$\log(1.10)$ 和 8/12 层门槛。
Haar q95 与 bootstrap 区间只负责判断结果能否区别随机方向或零；所有连续效应
大小、逐层结果和谱带数据必须报告。

**Pass — persistent head allocation：** 两个谱系分别满足：

1. 40k 与 80k 的 model-level coarse $H:M$、$H:T$ 对比均高于对应方向 null，
   且 paired bootstrap 下界高于 0；
2. 两个区间的 $\mathbf B^{update}$ 均偏 head，并超过各自 update null；
3. 两个区间的固定基底 $\Delta_W\mathbf B$ 下界均高于 0；
4. hard validity guards 全部通过。

**Fail：** guards 通过且结果精确支持 energy-only、middle/tail 等能增益不
弱于 head，或两个区间都没有 head-directed Gate 权重变化。

**Typed result 而非强行二元化：** 只有前段为正记 early-only；只有后段为正
记 late-only；两段相反记 non-monotonic/reversal；主要由 $U$ 解释记
representation-drift-only；两个谱系相反时分别报告，不能合并成普遍规律。

**Insufficient：** actual-input/no-op、checkpoint 坐标、基底稳定性、方向 null
或不确定性护栏失败；或区间不能区分正、零和负方向。

**允许 claim：** 两个被审计 Gate 在三个 checkpoint 上对各频带的等能访问、
真实响应、native 路由依赖，以及两个净训练区间的权重分配方向。

**不能 claim：** middle/tail 功能效用、band-only dispatch 的 loss 效果、
每一步梯度方向、从初始化开始的因果机制、所有模型普遍如此、训练效率提升。

**审批决定：** APPROVED FOR IMPLEMENTATION, SMOKE, AND FULL FROZEN AUDIT。

## 1. 术语与指标定义

| 术语 | 直白含义 | 具体对象 / 计算 | 单位 / 公式 | 判断作用 | 不能证明 |
| --- | --- | --- | --- | --- | --- |
| Router reference $g_\ell$ | Gate 变换前的上游表征 | MoE block 中送往 Gate 分支的张量 | activation | 验证 LB/decommon 输入变换 | Gate 实际 covariance |
| Gate-effective input $r_\ell$ | 线性 Gate 真正使用的表征 | 直接 hook mlp.gate pre-input；LB 为 $g$，decommon 为 $g-c$ | activation | covariance、native replay 与部署对象 | 表征语义 |
| expert input $h_\ell$ | sparse experts 收到的张量 | MoE block 专家分支输入 | activation | known-bad 对照 | Router 几何 |
| coarse head $H$ | 最大 covariance 方向 | ranks 1--64 | 64 directions | 与 middle/tail 比较 | 语义 common |
| coarse middle $M$ | 中方差方向 | ranks 65--320 | 256 directions | 中频组 | 功能效用 |
| coarse tail $T$ | 低方差方向 | ranks 321--768 | 448 directions | 尾部组 | 功能效用 |
| fine band $F_j$ | 连续 64 个 eigen-ranks | $F_j=[64(j-1)+1,64j]$ | 64 directions | 发现组内峰值/非单调性 | 语义频率 |
| $G_A$ | 每方向完全等能后的 Gate 增益 | $\|C_EWU_A\|_F^2/d_A$ | logit²/activation² | 访问强度主量 | token 使用或效用 |
| row-space share $\Psi_A$ | Gate 专家对比权重有多少落在该组 | $\|C_EWU_A\|_F^2/\|C_EW\|_F^2$ | fraction | 总权重分配 | 输入能量和路由效果 |
| $V_A^\perp$ | 真实 token 上该组总共推动多少相对 logits | $\mathbb E\|C_EWP_Ax\|^2$ | logit²/token | 实际总响应 | learned preference |
| $v_A^\perp$ | 每方向实际响应 | $V_A^\perp/d_A$ | logit²/token/direction | 防止宽 tail 机械占优 | 完全去掉 $\lambda_i$ |
| $S_A^\perp$ | 每单位组能量的响应 | $V_A^\perp/\mathbb E\|P_Ax\|^2$ | logit²/activation² | 部分去除能量 | 宽组内仍按 $\lambda_i$ 加权 |
| $\mathbf B^{coarse}$ | head 相对 middle/tail 的每方向纯增益 | $(\log G_H/G_M,\log G_H/G_T)$ | two dimensionless log ratios | endpoint 主比较 | 功能价值 |
| route flip $F_A$ | 去除该组后专家身份是否改变 | 第 8 节公式 | token fraction | 当前路由使用 | loss 改善 |
| margin change $D_A$ | 去除该组后 native margin 降多少 | 第 8 节公式 | logit | 当前决策依赖 | 分发效用 |
| $\mathbf B^{update,a\to b}$ | 区间净 Gate 位移偏向哪个 coarse 组 | 第 7 节公式 | log-ratio vector | 训练分配方向 | 位移大小/每步梯度 |
| $\Delta_W\mathbf B$ | 固定基底后，换 Gate 权重是否增强 endpoint 对比 | 第 7 节公式 | log-ratio change | Gate-weight effect | 40k 前路径 |
| $\Delta_U\mathbf B$ | 固定 Gate 后，换表征基底造成的对比变化 | 第 7 节公式 | log-ratio change | representation-drift rival | 漂移原因 |
| 功能效用 | middle/tail 分发对 loss 或兼容性的作用 | 未在 E01 计算 | loss / compatibility | 后续实验对象 | 不能由几何代理 |

## 2. Anchor Alignment

**决策问题：** middle/tail 的等能访问和当前 native 使用是否弱于 head，且两个
训练区间是否持续把更多 Gate 权重分配给 head？

**物理 prior：** 大 covariance 提供机械响应优势，也可能提供优化信号优势。

**核心模型项：** $G_A$ 与 $\mathbf B^{coarse}$。

**Falsifier：** $V_H^\perp$ 高但 $G_H$ 不高于 $G_M/G_T$；或两段净 Gate
位移不偏 head，endpoint 趋势由 $U$ 解释。

**Claim boundary：** 只裁定 access、current native use 和 saved-interval
training allocation，不裁定 functional dispatch utility。

## 3. Tested Hypothesis

### H1 — persistent head-directed allocation

1. 40k 与 80k 的 coarse $G_H$ 分别高于 $G_M$ 和 $G_T$；
2. fine profile 不被 coarse 平均隐藏，完整 12 带可见；
3. 30k--40k 和 40k--80k 的净 Gate 位移均在 coarse 与 fine 结果上更偏 head；
4. 固定基底的 Gate 主效应为正，而不是只有基底主效应；
5. middle/tail 可以被访问和用于 native 路由，但相对 head 较弱。

### H0-energy — energy-only

$V_H^\perp$ 高，但 $G_H$ 与 middle/tail 无稳定差异。

### H0-drift — representation-only

endpoint 对比变化主要由 $\Delta_U\mathbf B$ 解释，
$\Delta_W\mathbf B$ 近零。

### H0-stage — stage-specific / non-monotonic

两个训练区间的 Gate 方向不一致，不存在单一持续趋势。

## 4. Rival Explanations

1. **输入能量：** $V\rightarrow S\rightarrow G$ 分解；主结论由 $G$ 给出。
2. **coarse 维度不等：** $G$、$v$ 按方向数归一；同时报告总 $V$ 和
   row-space share $\Psi$。
3. **专家共有 logit 平移：** 使用 $C_E$；raw $W$ 结果仅作 debug。
4. **center/DC：** 在 Gate 实际输入坐标中，将
   $C_EW\mu^{(r)}$ 单列，不塞入任何 covariance band；decommon 的
   $c$ 已经包含在 $r=g-c$ 的定义中，不得再次相减。
5. **Gate 总范数/奇异值增长：** 方向 null 保持 $C_EW$ 或
   $C_E\Delta W$ 的非零奇异值。
6. **表征基底漂移：** 完整 $3\times3$ Gate × basis crossing。
7. **basis 不稳定：** calibration half-split、随机同维 overlap null 和独立
   half verdict。
8. **fine 多重比较：** simultaneous bootstrap/Haar envelope；不能事后挑
   12 带中的峰值。
9. **区间长度不同：** 30k--40k 为 10k steps，40k--80k 为 40k steps；
   不比较 raw magnitude 作为速度。
10. **模型特异性：** LB 与 decommon 分开裁定；不以两个模型代表普遍规律。

## 5. Data、模型、算法与目标

### 5.1 模型与冻结 checkpoint

| Lineage | Router mode | Checkpoint root | Frozen steps |
| --- | --- | --- | --- |
| decommon | running center；Gate 接收 $g-c$ | /mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768--linear_running_center_8gpu_gbs768-center_running-gate_off-acp_off-lb_0-linear/checkpoints | 30000, 40000, 80000 |
| LB | no center；load-balancing training | /mnt/bucket/MoE_Router/outputs/qwen_moe_runs/output_moe/qwen3-moe-H768-linear_nocenter_lb001_8gpu-center_off-gate_off-acp_off-lb_0.01-linear/checkpoints | 30000, 40000, 80000 |

每个 checkpoint 使用 checkpoint-STEP/mp_rank_00_model_states.pt。六个文件已
确认存在；执行前必须记录 byte size、sha256、模型配置、Gate shape、layer
count、expert ordering、center state、tokenizer 和代码版本。

**fallback：** 只在 S0、任何主指标读取前触发。若任一 30k 在完整性或坐标上
失败，选择两个谱系共同可用的最大早期 step：先 20k，再 10k。必须记录
amendment；两个谱系不得使用不同早期 step。

### 5.2 数据

- 训练顶层 shard：global-shard_01_of_10；
- held-out 顶层 shard：global-shard_02_of_10；
- calibration：从训练二进制 shard 固定抽取 32 sequences × 256 tokens；该
  uint32 token 流不保留原始 document boundary，因此不得将 sequence 写成
  source document；
- evaluation：固定 64 documents × 256 tokens；
- calibration 只拟合 $\mu,\Sigma,U$ 和方向 null；
- evaluation 计算 held-out $V/v/S$、route flip、margin 和图表；
- 六个 checkpoint 使用完全相同的 token ids、attention mask 和文档顺序；
- evaluation uncertainty 单位是 document，不把 token 当独立样本；basis
  half-split 与 calibration bootstrap 的单位是固定 training sequence。

### 5.3 Router 表征捕获与 native no-op

每层、每 checkpoint 同时捕获：

1. 上游 Router reference $g_\ell$；
2. mlp.gate pre-input $r_\ell$；
3. mlp.gate native output logits；
4. expert input $h_\ell$，只作 known-bad 对照。

LB 必须满足 $r_\ell=g_\ell$；decommon 必须满足
$r_\ell=g_\ell-c_\ell$。离线重算 logits 的相对 Frobenius error 必须
$\le10^{-5}$，top-1 agreement 必须为 1.0；否则停止解释。

### 5.4 Covariance 与双分辨率

对每个 model × checkpoint × layer，在 calibration 的
$r_\ell-\mu_\ell^{(r)}$ 上计算 covariance eigenbasis。任何使用 $g_\ell$ 或
$h_\ell$ 建立主基底的结果都违反 Protocol。

fine：

$$
F_j=[64(j-1)+1,64j],\qquad j=1,\ldots,12.
$$

coarse：

$$
H=F_1,\qquad M=F_2\cup F_3\cup F_4\cup F_5,\qquad
T=F_6\cup\cdots\cup F_{12}.
$$

### 5.5 统计与聚合

- primary aggregation：每个谱系内 12 层的 median，逐层结果全部保留；
- 12 层不是独立随机 seeds，不对层做伪显著性检验；
- checkpoint/basis bootstrap 使用相同重采样 document indices 做配对；
- calibration-basis bootstrap：AI 提议 200 次，需研究者审核；
- evaluation bootstrap：AI 提议 2000 次，需研究者审核；
- orientation null：AI 提议 256 个 thin Haar-Stiefel samples，需研究者审核；
- fine 12 带使用 simultaneous max-deviation envelope；
- 若任何层无法给出稳定 basis，该层标记 ineligible；只允许 layer-local 结论。
  只要 12 层不完整，就不得写“整个模型所有层”，model median 必须同时注明
  eligible set。

## 6. Conditions、Controls 与 Checkpoints

| Item | Anchor clause | Rival / model term | Why needed | Evidence role | Pass | Fail | Insufficient | Figure/table |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Six-checkpoint provenance | trajectory validity | coordinate compatibility | 净位移必须可比 | hard guard | hashes/configs compatible | incompatible | unresolved | checkpoint manifest |
| Actual Gate input/no-op | correct object | $r$ vs upstream $g$ / expert $h$ | 防止重复旧对象错误 | hard guard | error/top1 合格 | mismatch | unresolved | noop audit |
| 30k/40k/80k endpoints | access/current use | $G,V,F,D$ | 当前与轨迹起点 | primary | 全指标存在 | precise rival | interval wide | endpoint tables |
| Coarse H/M/T | registered comparison | $\mathbf B^{coarse}$ | 直接回答三组相对强弱 | primary | contrasts resolved | non-head pattern | uncertain | coarse table |
| Fine 12×64 | hidden within-group pattern | $G_b,V_b,F_b,D_b$ | 防止 coarse 平均掩盖 | mandatory support | all 12 reported | contradictory pattern | unstable basis | fine heatmap |
| 30k--40k update | early interval | $\mathbf B^{update},\Delta_W,\Delta_U$ | 早段趋势 | primary | typed verdict | typed rival | uncertain | trajectory table |
| 40k--80k update | late interval | same | 后段趋势 | primary | typed verdict | typed rival | uncertain | trajectory table |
| Full $3\times3$ crossing | W versus U | representation drift | 隔离共演化 | primary | all nine cells | U-only pattern | incompatible | crossing figure |
| Endpoint orientation null | W orientation | norm/singular values | 排除总尺度 | primary guard | observed > q95 | inside null | invalid null | null table |
| Update orientation null | $\Delta W$ orientation | tiny/random displacement | 排除随机更新方向 | primary guard | observed > q95 | inside null | tiny/invalid | null table |
| Wrong-layer basis | layer specificity | arbitrary basis | 检查本层坐标特殊性 | secondary | same-layer stronger | wrong equal/stronger | mismatch | control table |
| Half-split basis | eigenspace stability | finite calibration | 保护 band 含义 | hard guard | profile/verdict repeats | stable contradiction | unstable | stability table |
| Center/DC | non-band response | mean/center rival | 不把 DC 当 head | hard guard | separately reconstructed | leakage | unresolved | DC table |
| Expert-input replay | known-bad object | $h_\ell$ | 证明对象护栏有效 | negative control | differs as expected | surprising identity | unavailable | debug table |

## 7. Primary Metric

令

$$
C_E=I_E-\frac1E\mathbf1\mathbf1^\top,\qquad
\bar W=C_EW.
$$

对任意 coarse 或 fine 集合 $A$，

$$
G_{\ell,A}(W,U)=\frac1{d_A}\|\bar W_\ell U_{\ell,A}\|_F^2.
$$

coarse 主比较为

$$
\mathbf B_\ell^{coarse}(W,U)
=\left(B_{\ell,H:M},B_{\ell,H:T}\right)
=\left(
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,M}+\epsilon},
\log\frac{G_{\ell,H}+\epsilon}{G_{\ell,T}+\epsilon}
\right),
$$

其中 $\epsilon=10^{-12}$ 只作数值保护。$\exp(B_{H:M})$ 和
$\exp(B_{H:T})$ 分别是 head 相对 middle/tail 的每方向平方 logit 增益比。
不画 25% 或 10% 判定线，只画零线和方向 null envelope。

row-space 总分配另报

$$
\Psi_{\ell,A}
=\frac{\|\bar W_\ell U_{\ell,A}\|_F^2}
{\|\bar W_\ell\|_F^2}.
$$

随机各向同性方向下，coarse 期望维度份额为
$64/768$、$256/768$、$448/768$；$\Psi$ 不用于替代每方向 $G$。

### 7.1 Orientation null

对 $\bar W=L\Sigma R^\top$ 的非零 rank $r$，从 768 维 Haar-Stiefel
分布采样 $R_j^{null}\in\mathbb R^{768\times r}$，构造

$$
\bar W_j^{null}=L\Sigma R_j^{null\top}.
$$

它保持全部非零奇异值，只随机化相对 covariance basis 的输入方向。对
$C_E\Delta W$ 独立重复。每个 null replicate 使用与观察量相同的 layer
aggregation；coarse 使用 model-level q95，fine 使用 12 带 simultaneous
max envelope。

### 7.2 两段训练分配

令 $\mathcal T=\{30,40,80\}$，计算全部

$$
\mathbf B_{\ell;s,t}
=\mathbf B_\ell^{coarse}(W_{\ell,s},U_{\ell,t}),
\qquad s,t\in\mathcal T.
$$

对 $a\to b\in\{30\to40,40\to80\}$：

$$
\Delta W_\ell^{a\to b}=W_{\ell,b}-W_{\ell,a},
$$

$$
\mathbf B_\ell^{update,a\to b}
=\frac12\left[
\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,a})
+\mathbf B_\ell^{coarse}(\Delta W_\ell^{a\to b},U_{\ell,b})
\right],
$$

$$
\Delta_W\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;b,a}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;a,b})
\right],
$$

$$
\Delta_U\mathbf B_\ell^{a\to b}
=\frac12\left[
(\mathbf B_{\ell;a,b}-\mathbf B_{\ell;a,a})
+(\mathbf B_{\ell;b,b}-\mathbf B_{\ell;b,a})
\right].
$$

同时报告 $\|C_E\Delta W_\ell^{a\to b}\|_F$ 和 fine
$G_{\ell,F_j}(\Delta W)$。若位移接近数值精度，方向指标不可解释。

### 7.3 Evidence rule 与误判代价

**Support 一个 coarse 对比：** observed model-level statistic 超过对应 Haar
q95，且 paired document-bootstrap 95% interval 下界高于 0。

**不支持但非 Fail：** interval 包含 0；记 insufficient，不把小效应当零。

**Fail 一个 head-advantage 对比：** guards 通过且 interval 上界低于或等于 0，
或 observed statistic 稳定落在方向 null 内且区间足够窄。

false positive 会把机械能量或表征漂移误当训练偏好；false negative 会漏掉弱但
真实的 middle/tail 或 head 访问。因后续功能实验成本高，本实验优先控制 false
positive，同时完整报告连续效应避免把弱效应抹成不存在。

## 8. Secondary Metrics And Guards

### 8.1 实际响应、每方向响应与部分能量归一

$$
V^\perp_{\ell,A}
=\mathbb E\|C_EW_\ell P_{\ell,A}x_\ell\|^2,
\qquad
v^\perp_{\ell,A}=\frac{V^\perp_{\ell,A}}{d_A},
$$

$$
S^\perp_{\ell,A}
=\frac{V^\perp_{\ell,A}}
{\mathbb E\|P_{\ell,A}x_\ell\|^2}.
$$

coarse 同时给出 total $V$ 和 per-direction $v$；fine 各带等维。$S$ 在 wide
middle/tail 内仍按 $\lambda_i$ 加权，不作为纯 Gate 偏好结论。

### 8.2 当前 native 路由使用

令

$$
z_\ell^{(-A)}=z_\ell-C_EW_\ell P_{\ell,A}x_\ell.
$$

对每个 token 固定 native winner $e^*=\arg\max_e z_{\ell,e}$，定义

$$
m_{\rm native}(q)=q_{e^*}-\max_{e\ne e^*}q_e.
$$

即使去带后 winner 翻转，也继续用原 native winner 计算带符号 margin；这样
$D_A$ 测的是该频带对原 native 决策的支持，而不是干预后新 winner 的置信度。

$$
F_{\ell,A}
=\Pr[\arg\max z_\ell\ne\arg\max z_\ell^{(-A)}],
$$

$$
D_{\ell,A}
=\mathbb E[m_{\rm native}(z_\ell)-m_{\rm native}(z_\ell^{(-A)})].
$$

对 coarse 三组和 fine 十二带全部计算。由于 coarse 维度不同，$F/D$ 表示总
native 决策依赖，不解释为每方向效率。fine 等维结果用于定位。

### 8.3 Center/DC 与重构

单独报告 $C_EW\mu^{(r)}$。检查

$$
x_\ell\approx\sum_{j=1}^{12}P_{\ell,F_j}x_\ell
$$

和 offline logits reconstruction。若 evaluation 上跨带 covariance 使
$\mathbb E\|C_EWx\|^2$ 与 $\sum_jV^\perp_{F_j}$ 不可加，则报告 cross term，
不得把 $V$ 当严格份额分解。

### 8.4 Basis stability

用固定 16/16 calibration sequence halves 独立拟合 $U$。报告 coarse/fine
projector overlap、dimension-matched random-overlap null，以及两 half 的
完整 $G$ profile 和 coarse 对比。若 overlap 无法超过随机且 verdict 不复现，
对应层/带为 insufficient；不使用固定 0.75 人为阈值。

### 8.5 核心图合同

**Figure 1 — endpoint 全频带访问与实际使用**

- 问题：每个 checkpoint/layer 上，head、middle、tail 和十二带分别被访问和
  使用多少？
- 面板：fine $G_b$、$V_b^\perp$、$v_b^\perp$、$S_b^\perp$、$F_b$、$D_b$；
  coarse 三组摘要另列。
- 横轴：fine band 1--12；纵轴：layer 1--12；facet：model × checkpoint。
- 数据：calibration basis + held-out evaluation metrics。
- 允许结论：访问、实际响应和 native 决策依赖的频带分布。
- 不能证明：功能效用或训练因果。

**Figure 2 — coarse endpoint trajectory**

- 问题：30k、40k、80k 的 head-vs-middle/tail 等能对比如何变化？
- 指标：$B_{H:M}$、$B_{H:T}$，逐层点、model median、bootstrap interval、
  Haar envelope。
- 允许结论：endpoint 方向及是否超过随机 orientation。
- 不能证明：变化来自 $W$ 还是 $U$。

**Figure 3 — 两段 Gate × basis 分解**

- 问题：两段趋势来自 Gate 权重还是表征基底？
- 面板：完整 $3\times3$ crossing；每段 $\mathbf B^{update}$、
  $\Delta_W\mathbf B$、$\Delta_U\mathbf B$；fine $G(\Delta W)$。
- 允许结论：saved-interval 净权重分配与 representation drift。
- 不能证明：每步梯度、功能效用或速度差异。

## 9. Known Good / Known Bad / Known Confusing Cases

**Known good：**

- 直接 hook 的 Gate input 经同一个 Gate 可 replay native logits/top-1；
- 12 个 fine projectors 重构 centered Router representation；
- coarse projectors严格等于注册 fine bands 的并集；
- null 与原 $C_EW$ 或 $C_E\Delta W$ 奇异值在数值误差内一致；
- 人工 head/middle/tail-aligned $W$ 能恢复正确 coarse 对比；
- 人工 stage reversal 能得到相反的两段 trajectory label。

**Known bad：**

- 使用 $h_\ell$ 或任意专家输入替代 Router 表征；
- 只给 coarse、只给 head/rest，或只挑一个 fine peak；
- 用 $V$ 或 $S$ 单独声称 learned preference；
- 用 30k/80k endpoint 差直接声称 Gate 权重训练趋势；
- 比较 10k-step 与 40k-step 区间 raw magnitude 并称训练速度；
- 运行 middle/tail-only dispatch 后把 frozen expert mismatch 当频带效用。

**Known confusing：**

- $V_H$ 高、$G_H$ 不高：energy-only；
- $G_M/G_T$ 非零且 route ablation 有效，但低于 head：middle/tail 可见且使用，
  但相对较弱，不是“看不到”；
- $G$ 偏 head、$F/D$ 不偏 head：每方向权重偏头，但真实 token/决策使用不同；
- $\mathbf B^{update}$ 偏 head、$\Delta_W\mathbf B$ 近零：位移方向偏头但规模
  不足以改变 endpoint；
- 一段正一段零：stage-specific 或 saturation；
- 一段正一段负：non-monotonic/reversal；
- 两个模型相反：model-conditioned result，不是普遍规律。

## 10. Stage-Level Profiling Plan

| Stage | Local question | Input evidence | Pass / fail / unclear rule | Debug artifact | Handoff |
| --- | --- | --- | --- | --- | --- |
| S0 provenance | 六个 checkpoint 可比吗 | files, hashes, configs | compatible / amend fallback / stop | checkpoint_manifest.json | S1 |
| S1 object/no-op | 捕获的是 Router 真正使用的表征吗 | $g,r,h$, native logits | exact replay / stop | noop_audit.json | S2 |
| S2 basis | 双分辨率基底稳定且可重构吗 | covariance, halves, projectors | reproduce / layer insufficient | basis_stability.csv | S3 |
| S3 endpoints | 每个 band 被访问多少 | coarse/fine $G,\Psi$ + null | quantified/typed | endpoint_gain.csv | S4 |
| S4 current use | 每个 band 当前推动和改变路由多少 | $V/v/S/F/D$ | quantified/typed | band_use.csv | S5 |
| S5 trajectory | 两段净 Gate 位移偏向哪里 | $G(\Delta W),\mathbf B^{update}$ | interval labels | update_direction.csv | S6 |
| S6 decomposition | endpoint 变化来自 W 还是 U | $3\times3,\Delta_W,\Delta_U$ | W/U/mixed/unclear | crossing.csv | S7 |
| S7 aggregate | 两模型支持何种结论 | all layers + intervals | pass/fail/typed/insufficient | verdict.json | result record |

任一 hard guard 失败时停止主结论解释；允许保存 debug artifact。

### 10.1 批准后的 24 小时最小执行边界

本表是计划上限，不是已经测得的 runtime。E01 不进行 optimizer step，也不训练
四层或六层新模型；最多使用 8×5090 并行重放六个冻结 endpoint。若实际环境
少于八卡，保持条件不变，只降低并行度，不删减 checkpoint、layer 或频带。

| 批准后时窗 | 工作包 | 完成门槛 |
| --- | --- | --- |
| 0--3 h | S0 provenance、hook/no-op smoke、固定 token manifest | 六个 endpoint 坐标可比且 native replay 通过；否则停止或在主指标前触发 fallback |
| 3--9 h | 六个 endpoint 的 $g/r/h$/logit 捕获、coarse/fine basis 与 half-split | 全部 layer 有重构与稳定性状态 |
| 9--15 h | endpoint $G/\Psi/V/v/S/F/D$、错误层与 center/DC 对照 | 三个 checkpoint 的全带表齐全 |
| 15--20 h | 两段 $\Delta W$、完整 $3\times3$ crossing、orientation null | training-allocation 与 $W/U$ 分解表齐全 |
| 20--23 h | paired bootstrap、simultaneous envelope、三张注册图 | 所有区间和图合同完成 |
| 23--24 h | 数值审计、typed verdict、summary/detailed evidence skeleton | 可以交给研究者作结果判断；不能自动升级到功能 claim |

若 24 小时内 hard guard 未通过，交付 insufficient/debug 记录，不通过删条件或
事后缩窄到“好看”的 layer/band 来制造结论。

## 11. Algorithm Specification

**输入：**

- 两个谱系的 30k、40k、80k checkpoints；
- 固定 calibration/evaluation token ids；
- 每层 Gate weights、centers、$g,r,h$ 和 native logits。

**冻结参数：**

- hidden width 768；
- fine band width 64，共 12 带；
- coarse H=1--64，M=65--320，T=321--768；
- $\epsilon=10^{-12}$；
- orientation null seed 20260730；
- bootstrap seed 20260731；
- null/bootstrap 重复次数在研究者批准本 Protocol 时冻结。

**步骤：**

1. 校验六个 checkpoint 文件、hash、config、坐标和 expert ordering。
2. 若 30k provenance 失败，在任何指标读取前按 20k→10k fallback 规则处理并
   记录 amendment。
3. 对相同 token batch 捕获 $g,r,h$ 和 native logits；完成 no-op。
4. 每个 model × checkpoint × layer 在 calibration 的实际 Gate input $r$
   上拟合 $\mu^{(r)},\Sigma,U,\lambda$。
5. 构造 fine 12 projectors 与 coarse 三个并集 projector；检查重构。
6. 完成 half-split basis stability 与错误层对照。
7. 从 $W,U$ 计算 coarse/fine $G,\Psi$ 和 orientation null。
8. 在 evaluation 计算 coarse/fine $V,v,S,F,D$ 和 center/DC。
9. 形成完整 $W_{30/40/80}\times U_{30/40/80}$ crossing。
10. 对 30k--40k 和 40k--80k 计算
    $G(\Delta W)$、$\mathbf B^{update}$、
    $\Delta_W\mathbf B$、$\Delta_U\mathbf B$ 和 $\|C_E\Delta W\|_F$。
11. 使用配对 document bootstrap；fine 使用 simultaneous envelope。
12. 依 Section 12 生成 typed verdict；不运行任何 band-only dispatch。

**强制输出：**

- checkpoint_manifest.json
- noop_audit.json
- basis_stability.csv
- band_metrics_fine.csv
- band_metrics_coarse.csv
- endpoint_contrasts.csv
- route_ablation.csv
- gate_basis_crossing_3x3.csv
- update_direction_fine.csv
- orientation_null.csv
- verdict.json
- 三张注册中心图

**失败原因分类：** object/implementation failure、checkpoint incompatibility、
basis instability、energy-only prior failure、representation-drift rival、
stage-specific trajectory、metric imprecision。

## 12. Success / Failure / Insufficient Evidence

### 12.1 三个独立回答轴

每个 model × layer 必须分别输出：

1. **Access profile：** coarse/fine $G,\Psi$；
2. **Current-use profile：** coarse/fine $V,v,S,F,D$；
3. **Training-allocation profile：** 两段
   $G(\Delta W),\mathbf B^{update},\Delta_W\mathbf B,\Delta_U\mathbf B$。

不得用其中一个轴替代另一个。

### 12.2 Persistent head allocation — Full Pass

两个谱系分别满足 Approval Snapshot 的四项 Pass 条件。允许结论：

> 在两个被审计谱系中，40k/80k Gate 的每方向等能增益均偏向 coarse head；
> 30k--40k 和 40k--80k 的净 Gate 位移也都偏向 head，并在固定基底下增强
> head-vs-middle 与 head-vs-tail endpoint 对比。

即使 Full Pass，也必须单独报告 middle/tail 的非零访问和 route use；不能写成
“Router 看不到 middle/tail”。

### 12.3 Energy-only

$V_H^\perp$ 高，但 $\mathbf B^{coarse}$ 在 orientation null 内或不高于
middle/tail。允许结论：head 的实际响应优势可由输入能量解释，没有识别出纯
Gate head preference。

### 12.4 Middle/tail accessed but weaker

本 Protocol 不给“看得到/看不到”设置人为绝对阈值；access 是由 $G_M,G_T$
给出的连续强度，current use 是由 $V/F/D$ 给出的另一组连续强度。若这些量在
calibration half-split 中可复现，同时 coarse head 对比为正，允许结论：
middle/tail 在测得强度下可被 Gate 访问或参与 native 决策，但相对 head 较弱。
若只复现 $G$ 而没有复现 $F/D$，只能写“可访问”，不能写“当前路由在使用”。
任何情形都不能结论 middle/tail 功能效用较弱。

### 12.5 Middle/tail not weaker

至少一个 coarse 对比的区间上界 $\le0$，或 fine 同维 band 显示稳定的
middle/tail 增益不低于 head。允许结论：当前数据不支持 middle/tail 的
等能访问较弱；报告具体层、checkpoint 和 band。

### 12.6 Trajectory types

- 两段均 head-directed：persistent；
- 30k--40k 正、40k--80k 不可区分零：early-only / saturation；
- 前段不可区分零、后段正：late-only；
- 两段方向相反：non-monotonic / reversal；
- $\Delta_U$ 解释变化而 $\Delta_W$ 近零：representation-drift-only；
- 两模型不同：model-conditioned，不给 shared verdict。

### 12.7 Insufficient

- checkpoint hash/config/expert ordering 不兼容；
- actual-input/no-op 失败；
- coarse/fine projector 不重构；
- basis half-split 与随机 overlap 无法区分且 verdict 不复现；
- orientation null 不保持奇异值；
- $C_E\Delta W$ 接近数值精度；
- bootstrap 或 null 区间无法区分竞争解释；
- 缺少任一强制 fine/coarse 输出。

## 13. What This Cannot Claim

无论结果如何，E01 都不能证明：

1. 线性 Gate 在表达能力上无法读取 middle/tail；
2. middle/tail 信息没有功能价值；
3. middle/tail-only 或 band-only dispatch 会改善或损害 held-out loss；
4. covariance 大因果地产生了 Gate 梯度；
5. checkpoint 净位移代表每一步 optimizer update；
6. 专家因频带而形成了功能专业化；
7. 频谱分发改善训练路径、验证损失或 loss/FLOP；
8. 两个谱系代表所有模型、数据或 Router 配置。

回答“用 middle/tail 分发会发生什么”需要独立后续 Protocol，至少控制 native
linear score、专家负载、容量、token 数和 batch，并以 held-out loss 或独立
token 组的一步交叉更新兼容性为功能目标；E01 结果不能代替该实验。

## 14. Review Notes And Protocol Changes

### 2026-07-30 researcher revision

- 研究对象固定为 Router 真正使用的表征。
- 30k 存在时使用 30k；否则按 20k、10k 选择共同可用的最大早期 checkpoint。
- 已只读确认两个谱系的 30k、40k、80k 主状态文件存在，因此本版冻结三者。
- coarse 分辨率：head 1--64，middle 65--320，tail 321--768。
- fine 分辨率：12 个连续 64 维 band。
- 删除 25%、10% 和 8/12 practical 硬门槛；完整连续效应和逐带数据强制交付。
- 把“看得到”“当前使用”“训练分配”“功能效用”分开；E01 不回答后半部分
  functional dispatch effect。

### 2026-07-30 execution amendment before primary metrics

- 研究者批准 implementation、smoke 和 full frozen audit；不批准新训练。
- 训练日志确认 calibration source 为
  `/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10`。
- 该训练源只保留 uint32 token stream，不保留原始 document boundary。因此
  calibration 的统计单位由“document”澄清为固定、互不重叠的 256-token
  sequence；数量、训练/held-out 分离、频带、checkpoint 和主指标均不改变。
- evaluation 继续使用 held-out `global-shard_02_of_10` 的 64 个独立文档，
  paired bootstrap 继续以 evaluation document 为单位。

### Researcher section-by-section approval fields

- Q1 scope: access + current use + training allocation only: APPROVED
- Actual Router representation and no-op contract: APPROVED
- Coarse H/M/T boundaries: APPROVED
- Fine 12×64 boundaries: APPROVED
- Checkpoints 30k/40k/80k and fallback rule: APPROVED
- Primary metric $\mathbf B^{coarse}$ and mandatory full profiles: APPROVED
- No practical hard margin; retain Haar/bootstrap evidence rules: APPROVED
- Two-interval $3\times3$ W/U decomposition: APPROVED
- No-new-training and 24-hour execution envelope: APPROVED
- Pass/fail/typed/insufficient language: APPROVED
- Functional-utility exclusion: APPROVED
- Bootstrap/null counts (200 basis, 2000 evaluation, 256 null): APPROVED
- Approve English canonical protocol.md creation: APPROVED
- Approve implementation / smoke / full frozen audit: APPROVED
