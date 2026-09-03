---
anchor_id: 15_08_target_conditioned_layer_innovation
status: e04_eligible_h2_pass_h1prime_fail
canonical_language: en
companion_language: zh
updated: 2026-08-11
---

# A15_08 目标条件下的层新增信息

研究者判断记录：研究者中文原始判断。英文正式版：[Anchor](15_08_target_conditioned_layer_innovation_anchor.md)。

## 1. Problem Definition

A15 最终希望判断逐层不同的信息能否支持逐层 Router。当前更靠前的阻塞问题是：一次具名层写入改变 token 表征后，其中哪一部分是写入前状态还不能线性访问的目标相关信息？

唯一决策问题是：

> 在冻结 Qwen3-8B 的受控组合任务答案 token 上，第 25 层归一化 attention 更新中不能由写入前状态线性预测的部分，是否在旧状态和目标独立同预算对照之外提供留出终端 code 预测增益；若增益存在，TRAIN-only 条件创新矩阵的前两个方向能否在超过同秩随机与标签置乱对照的同时保留大部分增益？

“层更新”是在同一个第 25 层 post-attention normalization 下得到的两个状态之差。“新增信息”是在注册线性读出器族内，对冻结八分类目标产生留出预测增益。它不等于 Shannon 信息创造、参数中的事实知识或模型原生使用。

H2 Pass 只建立一个有边界的新增可访问性对象；H1' Pass 才会准入其
注册的二维压缩供后续功能审核，H1' Fail 则意味着最小充分秩仍未知。
本轮不训练 Router，不检验专家效用。

## 2. Physical Priors

1. 严格匹配 hook 的 residual-stream 状态共享同一环境坐标，所以精确差分是合法计算更新；token、hook 或 normalization 不匹配会使此前提失败。
2. 冻结网络不会创造外部信息，但 attention 可以让上下文区分在当前 token 上新变得可访问；它由旧状态条件下的留出目标风险暴露。
3. 若目标相关新增可访问性低维，则不使用确认标签学到的小子空间应保留完整更新增益，并超过同秩空对照。

## 3. Falsifiable Hypotheses

**主 H2——条件新增可访问性。**令 $X$ 为写入前状态、$R_U$ 为实际更新中
不能由 $X$ 线性预测的部分。在 $X$ 的冻结线性读出之外加入 $R_U$，会降低
untouched CONFIRMATION 上的终端 code 交叉熵，超过目标独立的平衡错配
对照，并与同维 $Z$-only 对 $X$-only 的比较同号。

**门控 H1'——注册的二维充分性。**只有 H2 Pass 后，TRAIN-only 条件创新
矩阵前两个方向才被检验：它们必须保留至少 $80\%$ 的完整 $R_U$ 增益，
并在匹配搜索预算后超过同秩随机方向和 TRAIN 标签置乱方向。八分类线性
目标的判别秩天然不超过七，因此它检验的是本对象的二维检索是否充分，
不是“整个表征是否低秩”。

**最强竞争解释。**更新可以非零且单独可读，却只是在缩放或重复编码 $X$ 已经提供的区分。它允许明显差分协方差和更新探针，但预测在 $X$ 之外没有稳定增益，也不超过匹配空方向。

H2 与门控 H1' 分别给出 Pass、Fail 或 Insufficient；H1' 不能挽救 H2。

## 4. Mathematical Model

令 $h$ 为第 25 层 attention 前 residual，$a$ 为 attention 写入，$N$ 为前后共同使用的 post-attention RMSNorm：

$$
X=N(h),\qquad Z=N(h+a),\qquad U=Z-X.
$$

$Z=X+U$ 说明 $U$ 是精确更新，但不说明它是新知识。无标签公共基 $V_{pool}$ 由 TRAIN 中 episode-centered $X$ 与 $Z$ 的 pooled covariance 求得；它让旧状态、新状态和更新在同一坐标中定位，而不是比较不同层各自的 rank。

从 $X$ 分别预测更新与中心化 one-hot 目标 $Y_c$，使用 TRAIN episode 交叉拟合残差：

$$
R_U=U-\widehat{\mathbb E}_{lin}[U\mid X],\qquad
R_Y=Y_c-\widehat{\mathbb E}_{lin}[Y_c\mid X].
$$

定义条件创新矩阵：

$$
K_{new}=C_{UY}S_Y^{+}C_{UY}^{\top},\qquad
C_{UY}=\frac1nR_U^{\top}R_Y,\qquad
S_Y=\frac1nR_Y^{\top}R_Y.
$$

$K_{new}$ 半正定且秩最多为七。相对 $\Sigma_{R_U}+\rho I$ 的广义特征方向优先选择“仍未被旧状态解释的目标相关协方差”相对“残余更新方差”更大的方向。它们只是候选；最终证据来自确认集增益。

主留出条件增益为：

$$
G_{true}=CE_{conf}(f_X(X))-CE_{conf}(f_X(X)+g_{full}(R_U)),
$$

单位为 nats/example。它只支持指定目标和线性读出器族中的新增可访问性。

## 5. Computational Realization

- **数据：**每个独立 episode 随机生成 $U\to V$ 与 $V\to C$ 两个双射、八个平衡终端 code，以及共享完全相同 context 和答案的一跳/两跳配对查询；TRAIN、DEVELOPMENT、CONFIRMATION 的 episode 与措辞家族互斥。
- **表征：**冻结 Qwen3-8B，在答案 token 和 block 25 存储 $X$、$Z$、$U$ 与限制选择 logits；两跳为主条件，一跳为次要复杂度对照。
- **公共基：**只用 TRAIN 的 episode-centered 数据拟合 $V_{pool}$，只负责定位 covariance 与 innovation mass。
- **差分审核：**验证 $Z=X+U$，并验证 $\Sigma_Z-\Sigma_X=\Sigma_U+\operatorname{Cov}(X,U)+\operatorname{Cov}(U,X)$。
- **条件矩阵：**按 episode 在 TRAIN 交叉拟合更新和目标预测器；DEVELOPMENT 选择正则化；冻结两个方向后只打开一次 CONFIRMATION。
- **读出与对照：**TRAIN/DEVELOPMENT 拟合旧状态 ridge、完整 $R_U$
  correction、同维 $Z$ readout、64 个精确平衡错配 correction，以及搜索
  预算匹配的二维空对照；selection ledger 冻结后，CONFIRMATION 才提供注册
  交叉熵差与 paired episode bootstrap。

确认标签不得构造公共基、残差模型、创新矩阵、秩、正则化、阈值或对照。

## 6. Minimal Falsification Tests

H2 的决定性比较是在同一确认样本上比较 old-only 与 old-plus-full-$R_U$。
每次 paired bootstrap 都从 true gain 中减去 64 个独立生成、目标严格独立的
平衡错配 bank 的 higher-method 95 分位。H2 Pass 要求 $G_{true}$、同维
$Z$-versus-$X$ gain 和容量对比 $T_{cap}$ 的 95% 下界都大于零；H2 Fail
要求所有 guard 有效且 $G_{true}$ 上界不大于零；否则为 Insufficient。

H1' 只在 H2 Pass 后判断。令 $G_2$ 为两个冻结条件创新坐标的增益，并
定义 $D_{80}=G_2-0.8G_{true}$。H1' Pass 要求 $D_{80}$、相对同秩随机
q95 和相对 TRAIN 标签置乱 q95 的三个 within-draw 对比 95% 下界都大于零；
任一上界不为正则 H1' Fail；其余为 Insufficient。

能力、平衡、hook identity、精确差分重建、合成 known-good/bad/confusing cases 与工件完整性只是有效性 guard，不能挽救失败的主比较。

## 7. Current Evidence

方差区间审核发现增长选择子空间没有超过同维随机方向，关闭了用方差增长
定义新增信息的路径。A15_08_E01 随后观察到正的更新读出增益，但其固定
cyclic 对照是可逆的目标重标记，且完整与二维比较对象不一致；其科学资格是
`PRIMARY_H2_INELIGIBLE_CONTROL_DESIGN`，不是 H2 verdict。

[A15_08_E02](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E02_fresh_balanced_mismatch_repair/summary.md)
使用全新 episode 修复了 E01 的代数缺陷。其数值数组映射为
`H2_PASS_H1_FAIL`：复杂确认条件上 $G_{true}=0.735082$
[0.715608, 0.756246]，同维状态增益为 0.721760
[0.702151, 0.742898]，$T_{cap}=0.734620$ [0.715064, 0.755466]，
二维保留完整点估计增益的 32.49%。但 E02 在 selection freeze 前使用
confirmation labels 物化了 confirmation mismatch maps，违反冻结的数据访问
规则。控制资格是 `INELIGIBLE_PROTOCOL_CONFIRMATION_LABEL_PREUSE`；
因此 E02 不能裁定 H2 和 H1'。

[A15_08_E03](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E03_fresh_confirmation_freeze_repair/summary.md)
产生了一份诊断数值包，但冷读合规审计发现：缺少 namespaced `episode_id`
和 `map_id`，两次提取缺少结构化 success receipt，simple 结果缺少逐 episode
与 bootstrap 数组，13 项 analysis-only manifest 也没有覆盖注册的完整实验
链。其控制资格因此是 `INELIGIBLE_GUARD`；E03 数字不能承担 H2/H1' verdict。

[A15_08_E04](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_E04_strict_conformance_repair/summary.md)
是全新的纯合规修复。它不改变科学对象、readout、两个冻结方向、控制族、
bootstrap 或阈值，只补齐身份、收据、次要数组和完整 manifest 合约；使用
seed 8103 的 128 个全新 episode，与 E01/E02/E03 在全部 ID、context 和 text
上零碰撞。59 个 artifact、39 个必需 family 的记录审计完整且 eligible。

在 E04 合格的复杂 confirmation 上，$G_{true}=0.767207$
$[0.751296,0.785146]$，$G_{state}=0.754508$
$[0.738226,0.773069]$，$T_{cap}=0.766839$
$[0.750643,0.784307]$，所以注册的局部 H2 Pass。两个冻结创新方向得到
$G_2=0.255052$ $[0.244938,0.264427]$，只保留完整点增益的 33.24%，且
$D_{80}=-0.358713$ $[-0.369760,-0.348675]$，所以 H1' Fail。它们相对
同秩随机和标签置乱的对比仍为正，说明方向不是空的，但不说明二维足够。

TRAIN-only 谱分析区分了两种“低秩”。复杂条件更新方差在公共基的谱头高度
集中（前 256 秩占 98.41%），但两个目标条件候选方向的平方质量在前 256 秩
只占 37.90%，质量中位秩为 374。七个非零广义特征值也较平，前两个只占
31.17%。因此“方差谱看起来低秩”不推出“目标检索对象二维充分”。由于这
两个候选方向没有通过留出充分性门，它们在公共基中的分布也不能被写成全部
新增可访问性的最终位置。

## 8. Claim Boundary And Next Decision

已经支持：匹配 residual 坐标允许精确计算差，但差分和谱方差本身不能定义
知识。在注册的目标、数据、层 transition 和线性读出器族内，E04 证明了旧
状态及平衡同预算竞争解释之外的新增可访问性；同时否定了“两个条件创新方向
能保留至少 80% 完整增益”的注册命题。这是 H2 的一个局部实例，也是局部
二维 H1' 的 Fail，不是对组会原始全局低秩 H1 的裁定。

尚未解决：同一个 $R_U$ 对象的最小充分任务读出秩；研究者现已明确暂停该
问题，不把 $r=3\ldots7$ 阶梯作为当前下一步。跨层/自然数据迁移、模型原生
使用和路由功能仍在后续。

不能声称：信息论创造、非线性新增、事实知识存储、MLP 原生使用、专家效用、Router 收益、自然语言普适性或逐层普遍规律。

**唯一下一决策：**从子
[A15_08_01 已批准 Protocol](../../../experiments/A15/15_08_target_conditioned_layer_innovation/A15_08_01_E01_layerwise_long_range_gain_and_representation_rank/protocol.md)
获得注册且合格的结果。子方向、公共基、两张图、描述性秩边界、数值合同、
实现与一次四卡运行均已批准；指定镜像/环境已经验证，但候选数据修订 A4 正等待 Block B 重新确认。当前不产生
科学结果，也不授权 Router。
