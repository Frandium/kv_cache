---
experiment_id: A15_02_01_E01_cross_update_compatibility_gate
status: completed_fail
created: 2026-07-30
updated: 2026-07-30
primary_anchor: 15_02_01_cross_update_compatibility_gate
canonical_protocol: protocol.md
approval_date: 2026-07-30
implementation_authorized: true
full_run_authorized: true
result: summary_cn.md
---

# Protocol 审核稿：middle / long-tail 的一步共同训练兼容性准入

## 0. Approval Snapshot

**执行结果（2026-07-30）：** 所有运行护栏通过，但 M/T/N 均未在两条谱系
同时通过预注册的 12-layer 80k Validation 准入。无候选停止规则生效，未运行
Final、40k、4-layer transfer 或 E02。见 [结果摘要](summary_cn.md)。

**审批状态：** 研究者已于 2026-07-30 批准 E01 实现与 full run；英文
canonical `protocol.md` 与本稿必须保持同一裁定合同。

**所属 subanchor：** [A15_02_01](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_02_middle_tail_functional_resolution/subanchors/15_02_01_cross_update_compatibility_gate_anchor_cn.md)。

**唯一决策问题：** 在控制 native Router 分数、margin、原生专家、负载、容量、
token 数、难度、范数、文档和 batch 后，actual Router input 的 middle、
long-tail 或 middle+long-tail 是否在未见文档上额外预测“两组 token 更新同一
专家时是否互助”，并超过同维随机与错误层基底？

**本实验角色：** 冻结静态分辨率诊断 + 局部功能准入。它不训练 Router，
不直接回答长期训练收益。

**主指标：** held-out compatibility 增量 $\Delta R_S^2$。它表示加入频带后，
对未见文档的一步兼容性多解释多少；它有准入权，但不能证明长期收益。

**最强证伪：** 频带确实产生不同邻域，但 $\Delta R_S^2$ 不高于零、同维
Haar 随机 q95 或错误层基底。此时“额外几何”成立，“额外功能分辨率”不成立。

**Pass：** 单个预注册候选 $S^*$ 在 12-layer final test 通过增量、随机与
错误层门，并在 4-layer branch checkpoint 上按同一锁定 document split 复现；
这是架构迁移检查而非第二份独立数据证据；此结果只解锁父
anchor 的 E02 训练审批。

**Fail：** 兼容性测量有效且区间足够窄，但三个候选均不能通过上述门。

**Insufficient：** 路由重放、局部步长、self-loss、数据独立、频谱稳定、
统计精度或 4-layer transfer 任一关键护栏失败。

## 1. 术语、对象与频带

| 对象 | 精确定义 | 为什么使用 | 能回答多少 | 不能回答 |
| --- | --- | --- | --- | --- |
| actual Router input $r_\ell$ | 直接 hook `mlp.gate` 收到的 pre-input；decommon 已包含部署时中心变换 | 保证分析的是 Gate 真正使用的表征 | 当前 Gate 输入频谱中的信息 | 专家输入或上游未变换表征 |
| Head $H$ | covariance eigen-ranks 1--64，64 维 | Q1 已知强信号，作为正向解释参考 | native score 是否已吸收 head | non-head 功能收益 |
| Middle $M$ | ranks 65--320，256 维 | 检查中方差非 head 信息 | middle 的独立增量 | tail 的贡献 |
| Long-tail $T$ | ranks 321--768，448 维 | 检查低方差方向 | tail 的独立增量 | 稀有词、稀有 token 或长尾数据收益 |
| Non-head $N$ | $M\cup T$，ranks 65--768，704 维 | 检查 middle+long-tail 联合 | 完整非 head 是否有增量 | middle 与 tail 谁贡献增量 |
| Native controls $X_{native}$ | 完整 Gate logits、margin、expert、load/capacity、token NLL、hidden/band/gradient norm、position、document-level aggregate 与 batch load | 排除 Router 已知信息与难度捷径 | 频带是否在线性分数之外增量预测 | 未观测混杂的完全消除 |
| 功能兼容性 | 一组 token 更新同一专家后，另一独立组的 loss 如何变化 | 用实际 loss 而非标签重合定义“适合放一起训练” | 局部共同训练关系 | 长期共同演化 |

这里的 “long-tail” 只指 covariance 的低方差方向，不指词频或数据频率。

## 2. 与 Anchor 的对应关系

本实验只裁定 [A15_02_01 subanchor](../../../../problem_anchors/15_linear_gate_spectral_training_bias/15_02_middle_tail_functional_resolution/subanchors/15_02_01_cross_update_compatibility_gate_anchor_cn.md)
的准入条款。三个层级必须分开：

1. **Q2-A 静态分辨率：** 是否产生 native scores 之外的新邻域；只说明“分得
   不一样”。
2. **Q2-B 局部功能：** 新信息是否预测一步交叉更新 loss；这是本实验的唯一
   决策层。
3. **Q2-C 联合训练：** 是否改善 matched-FLOP held-out NLL；属于父 anchor 的
   条件性 [E02 审核稿](../A15_02_E02_matched_spectral_dispatch_training/protocol_cn.md)。

Q2-A 不能替代 Q2-B，Q2-B 也不能替代 Q2-C。

## 3. H1 与最强 Rivals

**H1：** 至少一个 $S\in\{M,T,N\}$ 在 native controls 之外稳定预测一步共同
训练兼容性，超过同维随机方向和错误层基底，并能迁移到拟用于 8 卡训练的
4-layer branch checkpoint。

| Rival | 它预测什么 | 用什么比较区分 | 指标能回答多少 |
| --- | --- | --- | --- |
| R0：只有额外几何 | 新邻域很多，但不能预测真实交叉 loss | 静态 novelty 与 $\Delta R^2$ 分开报告 | 若 novelty 高而 $\Delta R^2$ 不过门，只否定功能解释 |
| R1：只是维度效应 | 任意同维子空间都同样有效 | 256 个同维 Haar bases | 排除“维度越大越容易拟合”，不排除其他结构先验 |
| R2：只是任意 non-head | $M/T$ 不优于 non-head 内随机方向 | non-head 内同维随机 bases | 检查 covariance rank 位置是否特殊；$N$ 因填满 non-head 不适用此对照 |
| R3：只是任意层几何 | 错误层同 rank 基底同样有效 | 预注册错误层映射 | 支持本层特异性，不证明因果方向 |
| R4：范数、难度或离群值 | 增量在加入 norm/NLL/gradient controls 后消失 | nuisance controls 与截尾稳健性 | 排除注册捷径，不保证无未测混杂 |
| R5：文档或 batch 泄漏 | 更换文档或 batch 配对后信号消失 | 文档级 split、pair permutation、batch-resampled ledger | 排除注册的数据泄漏 |
| R6：步长伪影 | 只有过大一步产生兼容性 | self-loss 与 $\eta/2$ 稳定性 | 验证局部 operationalization，不等于真实 optimizer dynamics |

## 4. 频谱估计合同

**在哪里估计：** 频谱只在冻结的 calibration token IDs 上估计，不在 Q2 的
fit、validation 或 final-test 文档上估计。12-layer 条件复用 Q1 已验证的
32×256 training sequences 与 actual-input bases；4-layer checkpoint 用相同
calibration token IDs 对自己的每层 actual Router input 重新估计基底。

对每个模型、checkpoint、layer 分别定义

$$
x_{i,\ell}=r_{i,\ell}-\mu_\ell,
\qquad
\Sigma_\ell=\mathbb E[x_{i,\ell}x_{i,\ell}^{\top}]
=U_\ell\Lambda_\ell U_\ell^{\top}.
$$

**为什么每层、每 checkpoint 单独估计：** Router 表征会随深度和训练状态改变；
共用一个基底会把错误层或旧 checkpoint 的漂移误当成频带效果。

**护栏：** actual-input replay、中心化重构、half-split projector overlap 和
eigen-rank 顺序必须通过 Q1 同级检查。该护栏只能证明频谱对象可复现，不能证明
其有功能。

## 5. 模型、层与数据分离

### 5.1 模型与层

| 条件 | Checkpoint | Q2-A 静态 | Q2-B 一步更新 | 为什么这样取 | 能回答多少 |
| --- | --- | --- | --- | --- | --- |
| 12-layer LB | 80k 主、40k 复现 | 全 12 层 | layers 1/6/12 | 覆盖浅/中/深且避免按结果挑层 | 两个保存状态内的层级稳定性 |
| 12-layer decommon | 80k 主、40k 复现 | 全 12 层 | layers 1/6/12 | 检查不同 Router 谱系是否同向 | 描述性跨谱系复现，不是 center/LB 因果消融 |
| 4-layer H768 branch | checkpoint-800，约 0.629B nominal tokens | 全 4 层 | 全 4 层 | 与后续 8×5090 treatment 完全同对象 | 是否允许向该四层训练配置迁移 |

### 5.2 数据

AI 提议冻结 512 个新的 DCLM held-out documents，每篇前 1024 个有效 token，
不得与 Q1 的 calibration 或 64-document evaluation 重合：

| Split | 文档数 | 用途 | 能回答多少 | 不能用于 |
| --- | ---: | --- | --- | --- |
| Operationalization | 64 | 步长、known-good/bad smoke、pair 构造可行性 | 指标是否能正常测 | 候选选择或最终结论 |
| Fit | 192 | 拟合 baseline 与 augmented ridge | 学习固定低容量预测器 | 调参后报告最终效果 |
| Validation | 128 | 选择单个 $S^*$ 与 ridge 正则 | 从 $M/T/N$ 中一次性选候选 | Pass 结论 |
| Final test | 128 | 单次锁定评估与 document bootstrap | 最终兼容性准入 | 再选择 band、层或超参 |

所有 split 按完整文档划分；同一文档不能跨 split。512 是待审核的 AI 提案；
若只读数据 manifest 无法满足，必须 amendment，不得静默减少。

## 6. A/B Token-Group 与一步更新流程

每个 target model × checkpoint × layer × expert 的 A/B ledger 只用 native
controls 构造；随后 $H/M/T/N$、random 和 wrong-layer 全部复用同一 ledger
与同一个兼容性 target。

### 6.1 Pair 合同

- 每组恰好 32 个 loss-bearing tokens，全部来自一个 1024-token document，
  且原生进入同一 target expert；
- A 与 B 必须来自两个不同 document、sequence 和 dataloader batch；
- 同一 model-layer-expert cell 内，每个 document 最多形成一个 group，同一 token
  不跨 pair 重复；若某 document 对该 expert 不足 32 个 token，则跳过该
  document；若最终不足注册 pair 数，该 cell 判 `insufficient`；
- pair 在完整 logits、margin、token NLL、hidden norm、position、native load
  和 capacity headroom 上匹配；
- 每个 model-checkpoint-layer-split cell 的目标上限为 256 pairs；专家间如何
  分配由下面的结果前可行性修订冻结。

**目的：** 让 A/B 的主要已知差异不来自专家身份、难度或 token 数。它能提高
局部比较的内部效度，并让 document 成为明确的重采样单位；它不能保证所有
语义因素都匹配。

### 6.1.1 结果前可行性修订（2026-07-30）

在尚未计算任何兼容性 target 或频带 feature 时，S1 的 route-only 检查发现：
decommon 的若干 layer-expert-split 中，原生路由到该专家的可用 token 总数不足
64。因而强制每个专家 32 pairs 在数学上不可定义，并会把原生专家停用误判成
频带没有功能。

最小修订保持上述所有 group 条件，只改变专家间样本分配：

- 每组仍为同一文档的 32 个 token；A/B 文档和逻辑 batch 独立，token 不复用；
- 先在每个 expert 内构造所有可行的匹配 pair；
- 每个 model-layer-split 最多 256 pairs，按原生 route mass 分配，受无复用
  pair 数约束；未用 quota 只在其余可行 experts 间重新分配；
- native controls 中保留 expert identity 与原生 load，并报告实际专家覆盖；
- 科学 split 每个 cell 至少需要 192 pairs；Operationalization 只选择测量步长，
  不承载结论，因此允许更少。

修订后的 estimand 是“native 路由实际 token 总体”的兼容性，不能单独回答几乎
停用的专家。条件允许时，equal-per-expert 结果只作敏感性分析。此修订不改变
文档、频带、功能 target、预测器、对照、选择或 Pass 规则，并在查看任何结果前
冻结。

### 6.2 双向一步更新

```mermaid
flowchart LR
  P["冻结模型并缓存 native routes"] --> S["从同一专家参数快照开始"]
  S --> GA["只用 A 的 masked loss 计算目标专家梯度"]
  GA --> UA["目标专家走一步；其他参数与 routes 不变"]
  UA --> EB["测 B 的 nat/token loss 变化"]
  EB --> R["恢复专家快照"]
  R --> GB["反向执行 B 更新、A 测量"]
  GB --> C["两方向平均，得到兼容性 C"]
```

令 $\theta_{\ell,e}$ 为 target expert 参数：

$$
\Delta_{A\rightarrow B}
=L_B(\theta_{\ell,e}-\eta\nabla_{\theta_{\ell,e}}L_A)
-L_B(\theta_{\ell,e}),
$$

$$
C_e(A,B)=-\frac12
\left(\Delta_{A\rightarrow B}+\Delta_{B\rightarrow A}\right).
$$

$C$ 的单位是 nat/token；$C>0$ 表示两组的局部更新平均互助，$C<0$ 表示冲突。
测它的目的，是给“适合放入同一专家训练”一个直接 loss 定义；它只能描述
冻结 checkpoint、固定 routes 和一步小更新，不能代表长期训练轨迹。

**更新规则：** 全模型冻结，只允许 target expert 更新；所有 MoE 层重放原生
routes，capacity 决策也重放。$\eta$ 只在 Operationalization split 选择；要求
A 更新降低 A 自身 loss、B 更新降低 B 自身 loss，并且 $\eta/2$ 下 $C$ 的符号
和 pair 排名稳定。Adam moments 不参与该局部步，使用显式 SGD probing step，
避免 checkpoint optimizer state 成为额外变量。

**Probe 精度护栏：** 更新前缓存原生 bfloat16 routes、winner 和 routing
weights；只在有限一步的局部 loss 测量中，把冻结 checkpoint 参数值无损提升为
float32 并关闭 autocast。结果前 smoke 发现，bfloat16 expert 输出的量化格点大于
计划测量的微小 loss 变化，会随机翻转 self-loss 符号。float32 probe 不改变参数
数值或路由，只去掉该测量格点。它回答 checkpoint 附近平滑的局部 loss 几何，
不能回答 bfloat16 部署精度本身的收益。

## 7. 三组频带分别做什么

对每个 $S\in\{M,T,N\}$ 都执行完全相同的两项分析：

1. **Q2-A：** 从 native controls 中线性残差化 band coordinates，再看
   residual-band 邻域与 native-logit 邻域的差异；目的：确认是否真的新增分辨率。
2. **Q2-B：** 用相同的 band pair features 预测真实 $C$，并计算 held-out
   $\Delta R^2$；目的：判断新增分辨率是否具有局部功能。

Q2-A 的“残差化”固定为：在 Fit documents 上用 ridge 拟合
$\hat q_{i,S}=f_S(X_{native,i})$，在 Validation 选择正则，然后只在 Final test
使用 $\tilde q_{i,S}=q_{i,S}-\hat q_{i,S}$。在同一 native expert 内，以
$\tilde q$ 的 cosine 取 $k=32$ 邻居，并与标准化 native-logit 邻居比较。
目的：把 Router 分数已线性解释的频带部分先拿掉；它不能排除 native scores
与频带之间更复杂的非线性冗余。

为限制 kNN 的计算量，Q2-A 在每个 Final-test document 中按
`SHA256(document_hash, token_position)` 固定选择 32 个可预测位置，共 4096
个 tokens；该选择不读取 band、expert 或结果。它只改变静态诊断的计算规模，
不改变 Q2-B 的完整 A/B ledger。

| Treatment | 维数 | 它专门回答 | 必要对照 | 结果不能被解释为 |
| --- | ---: | --- | --- | --- |
| $M$ | 256 | 中方差方向是否有增量 | full-space Haar-256、non-head Haar-256、wrong-layer M | tail 效果 |
| $T$ | 448 | 低方差方向是否有增量 | full-space Haar-448、non-head Haar-448、wrong-layer T | 稀有 token 效果 |
| $N=M+T$ | 704 | 全部 non-head 联合是否有增量 | full-space Haar-704、wrong-layer N | middle/tail 各自贡献；non-head 内随机对照不可定义 |
| $H$ reference | 64 | 已知 head 是否基本被 native logits 吸收 | full-space Haar-64 | Q2 候选；它不参与 $S^*$ 选择 |

若 $N$ 通过但 $M/T$ 均未通过，只允许说“联合 non-head feature block 有增量”；
不能把增量归因给其中任一子带。

## 8. 随机子空间与错误层对照

### 8.1 随机子空间

对 $k\in\{64,256,448,704\}$，以固定 seed 生成 $768\times k$ i.i.d.
Gaussian matrix，QR 后取正交列，得到 Haar-random $k$ 维子空间。每个
model × checkpoint × layer × $k$ 生成 256 个 orientations。

- **Full-space Haar：** 在完整 $\mathbb R^{768}$ 中抽样，回答“真频带是否优于
  任意同维方向”。
- **Non-head Haar：** 先在 $N$ 的 704 维坐标中抽样，再映回原空间；只对
  $M/T$ 定义，回答“特定 covariance rank 是否优于任意 non-head 方向”。
- **为什么 $N$ 没有 non-head Haar：** $N$ 已占满整个 704 维 non-head span；
  在其内部旋转不会改变 projector，因此不是有效对照。

随机基底只重新计算 band features；兼容性 $C$、A/B ledger 和 prediction splits
完全复用。这样 random gap 只比较方向，不重复采样或更新噪声。

### 8.2 错误层基底

- 12-layer：target layers 1/6/12 分别使用 source layers 6/12/1；
- 4-layer：target layers 1/2/3/4 分别使用 source layers 3/4/1/2；
- source 基底在 target layer 的 768 维坐标中直接投影，不重新拟合。

它检验本层频谱是否特殊；它不能排除跨层共享语义方向，因为共享方向可能让
错误层也有效。

## 9. 指标合同：目的、回答范围与边界

| 指标 | 精确计算 / 单位 | 测量目的 | 能回答多少 | 不能证明 |
| --- | --- | --- | --- | --- |
| Residual neighborhood novelty $N_S$ | final-test token 的 residual-band kNN 与 native-logit kNN 的 $1-|\cap|/k$，$k=32$，无量纲 | 检查频带是否增加静态分辨率 | 新划分是否存在、跨文档是否稳定 | 新划分是否有益 |
| $C_e(A,B)$ | 双向一步交叉 loss 的负平均，nat/token | 直接定义局部共同训练互助/冲突 | 冻结 expert 的一步功能关系 | 长期 loss 或路由收益 |
| Gradient cosine | $\langle g_A,g_B\rangle/(\|g_A\|\|g_B\|)$，$[-1,1]$ | 解释 $C$ 的一阶来源 | 两组梯度是否同向 | 有限步 loss 必然改善 |
| $\Delta R_S^2$ | $R^2_{test}(C\mid X_{native},\phi_S)-R^2_{test}(C\mid X_{native})$，无量纲 | **主指标：检查频带是否在线性 Router 信息之外增量预测功能** | 是否授予 E02 训练资格 | 训练后一定更好 |
| Random gap | $\Delta R_S^2-q_{0.95}(\Delta R^2_{R_k})$ | 排除同维高维几何 | 真频带是否超过随机方向 null | 频带因果 |
| Wrong-layer gap | $\Delta R_S^2-\Delta R_{wrong}^2$ | 排除任意层基底 | 本层 rank 结构是否更贴近 target | 层间信息不存在 |
| Nuisance ablation | 加/去 norm、NLL、gradient、outlier controls 后 $\Delta R^2$ 变化 | 检查捷径解释 | 信号是否依赖注册 nuisance | 所有混杂均排除 |
| Split / step stability | document bootstrap、40k replication、$\eta/2$ rank correlation | 检查可复现性 | 结果是否依赖某批文档/保存点/步长 | 普遍训练动力学 |

## 10. 预测模型与主统计量

对每组 token 先计算方向归一的 band mean coordinate：

$$
q_{i,S}=\frac{U_{\ell,S}^{\top}(r_{i,\ell}-\mu_\ell)}
{\|U_{\ell,S}^{\top}(r_{i,\ell}-\mu_\ell)\|_2+\epsilon},
\qquad
\bar q_{A,S}=\frac1{|A|}\sum_{i\in A}q_{i,S}.
$$

每个 band 只向模型增加两个 pair features：
$\phi_S=(\cos(\bar q_A,\bar q_B),\|\bar q_A-\bar q_B\|_2^2)$。band energy
单独进入 nuisance controls，因此维数大的 band 不会靠增加 predictor 参数获胜。

Baseline 与 augmented model 均为标准化 ridge regression；正则网格
$\{10^{-4},10^{-3},\ldots,10^4\}$ 只在 Validation 选择。使用低容量线性模型的
目的是测试稳定、可迁移的残差信号；若失败，只能否定这类预注册低容量增量，
不能证明不存在复杂非线性信息。

Document ID 和 batch ID 不作为可记忆的 one-hot predictor；它们只用于 split、
matching、cluster bootstrap 和 permutation null。进入 $X_{native}$ 的是可迁移
的 document aggregate（平均 NLL/norm）与 batch load/capacity 统计。

$$
\Delta R_S^2
=R^2_{final}(C\mid X_{native},\phi_S)
-R^2_{final}(C\mid X_{native}).
$$

置信区间以 document block bootstrap 2000 次估计；A/B pair 不被当作独立
document。随机 orientation null 使用 256 draws 的 empirical q95。

## 11. 候选选择与 Pass / Fail / Insufficient

### 11.1 只允许一次选择

1. 在 12-layer 80k **Validation** 上分别计算 $M/T/N$；
2. 候选必须在 LB 与 decommon 的预注册 layers 1/6/12 上具有同向的 model-level
   median $\Delta R^2$，并超过各自 random q95 与 wrong-layer；
3. 若多个候选合格，选择 paired document-bootstrap 下界最大的一个，记为
   $S^*$；tie 依次优先 $M$、$T$、$N$，避免因维数偏向更宽子空间；
4. 选择后锁定 band、feature、layers、step size 和 ridge；Final test 不得重选；
5. 若无候选，直接 `fail`，4-layer transfer 与 E02 均不启动。

### 11.2 Final Pass

`pass` 必须同时满足：

- 12-layer 80k final-test 的 $\Delta R_{S^*}^2$ document-bootstrap 95% 下界
  $>0$，并超过同维 full-space random q95、可定义时的 non-head random q95 和
  wrong-layer；LB 与 decommon 方向一致；
- 40k replication 同号且没有精确、相反的效果；它检验保存点稳定性，不以
  40k 重新选择候选；
- 4-layer checkpoint-800 对锁定 $S^*$ 的四层 pooled/median final-test
  $\Delta R^2$ 下界 $>0$，并超过对应 random 与 wrong-layer；
- route replay、self-loss、$\eta/2$、pair independence、basis stability 与
  nuisance guards 全部通过。

不设 10% 或 25% practical effect 硬门槛；报告连续 $\Delta R^2$、置信区间和
random gap。此 Pass 只表示“值得花 8 卡训练成本验证”。

### 11.3 Fail 与 Insufficient

- **Fail：** 测量与精度护栏通过，但无 $S^*$；或锁定候选在 final test 明确
  不高于零 / random / wrong-layer；或 4-layer transfer 明确不复现。
- **Insufficient：** final interval 过宽，或任一关键 operational guard 失败。
  Insufficient 不得被改写为“频带没有功能”。

## 12. 已知好、已知坏与易混淆检查

| 检查 | 目的 | 预期 | 能回答多少 |
| --- | --- | --- | --- |
| 同组 self-update | 验证更新符号与 loss mask | self-loss 降低 | 实现方向正确，不支持 H1 |
| 同一文档两半（仅 smoke） | 构造较容易的正相关 pair | $C$ 相对较高 | 指标有动态范围，不进入主分析 |
| 打乱 $C$ target | known-bad null | $\Delta R^2\approx0$ | predictor 不凭泄漏获胜 |
| Band feature 在 pair 内置换 | 方向 null | 增量消失 | feature-target 对齐必要 |
| 高 norm / 高 NLL 子集 | confusing case | controls 后结论不翻转 | 不是明显难度捷径 |
| Native no-op replay | 验证固定 routes | logits、winner、loss 重放一致 | operational validity |

## 13. 执行阶段与停止条件

1. **S0 provenance：** 冻结 checkpoint、token、basis、document split 与 seeds。
   目的：防止对象漂移；只能保证可复现。
2. **S1 measurement smoke：** 用 Operationalization split 验证 hook、replay、
   masked loss、restore、双向更新与步长。失败则停止。
3. **S2 Q2-A：** 全 12 层和四层模型运行静态 novelty。它不触发训练。
4. **S3 12-layer Q2-B fit/validation：** 生成一次 $C$ ledger，比较全部 bands 与
   controls，选择并锁定 $S^*$。无候选则停止。
5. **S4 12-layer final/40k replication：** 单次 final 裁定与保存点复现。
6. **S5 4-layer transfer：** 对锁定 $S^*$ 在全部四层复现；失败则 E02 保持阻塞。
7. **S6 evidence record：** 写 `summary.md` / `detailed.md`，更新 owning anchor；
   不自动提交训练。

任何阶段若 actual Router input、文档独立、参数 restore、loss replay 或频谱基底
失败，后续功能结论全部停止，不允许通过删 guard 继续。

## 14. 核心图表、结论边界与审核项

### 14.1 核心交付

1. **图 A：静态 novelty vs 功能 $\Delta R^2$。** 横轴是 novelty，纵轴是
   held-out 增量，颜色为 $H/M/T/N/random$。目的：让“不同划分”和“有用划分”
   一眼分开；不能显示长期训练收益。
2. **图 B：$M/T/N$ 相对 random/wrong-layer 的 final-test 增量。** 按模型、层、
   checkpoint 展示 document-bootstrap 区间。目的：裁定准入与稳定性。
3. **表 C：4-layer transfer gate。** 只报告锁定 $S^*$，避免再次挑选。

### 14.2 不能声称

即使 Pass，也不能声称：middle/tail routing 会降低长期 loss、会形成更好专家、
会改善训练效率、代表语义相似性、对所有层/尺度成立，或一步 SGD probe 就是
实际 AdamW 训练动力学。

### 14.3 请研究者重点审核

- [ ] **E01-C1：** 频谱用独立 calibration set；Q2 文档不重估基底；
- [ ] **E01-C2：** $M=65$--$320$、$T=321$--$768$、$N=65$--$768$；
- [ ] **E01-C3：** 12-layer Q2-B 固定 layers 1/6/12；4-layer 跑全部四层；
- [ ] **E01-C4：** 512 documents 的 64/192/128/128 split；
- [ ] **E01-C5：** 32 tokens/group、32 pairs/expert/split、pair 内文档和 batch 独立；
- [ ] **E01-C6：** 固定路由、只更新 target expert 的双向 SGD probe；
- [ ] **E01-C7：** 256 个 full-space Haar；$M/T$ 再加 non-head Haar；
- [ ] **E01-C8：** Validation 只选一次 $S^*$，Final test 不重选；
- [ ] **E01-C9：** 主门是 $\Delta R^2>0$ 的文档区间 + random q95 + wrong-layer，
  不设人为 practical threshold；
- [ ] **E01-C10：** 4-layer transfer 是 8 卡训练的硬前置；
- [x] **E01-C11：** 研究者已批准 E01 实现与运行；该批准不提前授权 E02
  在 E01 Fail/Insufficient 时运行。

**执行决定：** E01 已获批准。只有 E01 结果 Pass 后，父 anchor 的 E02
条件性执行授权才生效。
