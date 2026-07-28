# A14 从概率树与 NTP 到低秩参数更新子空间

这是 2026-07-28 同步到 `xyDeng` 分支的 theory-only 证明包。它只包含主文、
证明附录和物理先验/边界审计；所有证明依赖均在这三份文件内给出。

源目录：
`daily_research_reports/0728/NTP_hierarchical_low_rank_parameter_update_theory_package/`

三份源理论文件的组合 SHA-256：
`8b8fdc4e01e2985150b5a6bf6cd1cc5f7bf1cde3678be7cac065d89d492864b1`

## 推荐阅读顺序

1. [03_物理先验与假设审计.md](03_物理先验与假设审计.md)：先确认每条因果箭头的现实含义、数学条件与反例。
2. [01_理论论文_从概率树与NTP到低秩参数更新子空间.md](01_理论论文_从概率树与NTP到低秩参数更新子空间.md)：阅读 T1--T6 与主结论。
3. [02_证明附录.md](02_证明附录.md)：核验循环暴露、空间外项与完整权重谱尾的详细证明。

## 当前最强结论

在固定特征的受控 contrast-NTP writer 中，若：

- 概率树跨 contexts 复用至多 \(k\) 个 causal mappings；
- 每个 mapping 以固定坐标传递至多 \(r\) 个任务自由度；
- 所有注册 prefix-position events 对同一 separator 具有正 margin；
- 每个事件在 pure SGD 中获得无限累计步长质量；
- \(W_0\) 是独立 Gaussian 初始化，且矩阵维数随 \(T\) 固定；

则

$$
\operatorname{rank}(\Delta W_T)\le kr,
\qquad
\|\Delta W_T\|_F\to\infty,
\qquad
\tau_{kr}(W_T)\to0.
$$

因此，低秩结论首先属于参数更新子空间；完整 \(W_T\) 在有限时刻一般仍满秩，
但当发散 task spike 压过固定 Gaussian 基座后，其 rank-\(kr\) 外相对谱能量趋零。

## 证明链

$$
\text{共享概率树 mappings}
\Rightarrow
\dim U_\star\le kr
\Rightarrow
\text{NTP exposure 在共享 margin 上相干}
\Rightarrow
\text{pure-SGD task spike 发散}
\Rightarrow
\tau_{kr}(W_T)\to0.
$$

T3 采用自包含的有限路径长度反证：若 pressure-weighted exposure 有界，则参数
路径收敛到有限点；此时 logistic pressure 保持严格为正，与每个事件的无限累计
步长质量矛盾。

## 三种状态

### 已证明

- 受控固定 mapping injections 给出 \(\dim U_\star\le kr\)；
- NTP exposure 的相干增长下界；
- 循环 pure SGD 使 exposure 与低维 task spike 发散；
- 固定维数精确模型满足 \(\tau_{kr}(W_T)\to0\)。

### 条件证明

- 小 conditional bias / martingale leakage 下的 \(R_T=o_{\mathbb P}(S_T)\)；
- multiclass softmax；
- learned residual MLP 的 \(W_2\) 与一层 Transformer 的 \(W_O/W_2\)。

### 未证明

- teacher 的共享概率树自动产生 learned fixed coordinates；
- full-position NTP 无条件优于 root-only；
- 完整 Transformer 参数元组或自然语言模型普遍低有效秩。

## 明确排除

- 实验 Protocol、focus、内部审核清单和可选背景文献；
- 实验实现、运行记录、raw logs、checkpoints 或数据；
- graph、anchor、Research_System 根 `sync/` 的任何修改；
- “完整 \(W_T\) 有低代数秩”或“自然语言模型普遍低秩”的主张。
