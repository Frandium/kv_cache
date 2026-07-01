# 实验协议

## 输入

- 模型：`fdong/Qwen3-0.6B`，Transformers 标准 Qwen3Model。
- 数据：`/Users/bytedance/Desktop/dclm/part-*.txt` 中彼此独立的网页/新闻文档。
- 主实验：8 条 sequence，每条 1024 tokens。
- 对照实验：8 条 sequence，每条 512 tokens。
- 层：全部 28 层。

## 指标一：句内连续性与句间区分性

对 (X_l,A_l,H_l) 分别计算：

- adjacent cosine：同 sequence 相邻 token cosine；
- within-sequence random cosine：同 sequence 随机 token pair cosine；
- between-sequence cosine：不同 sequence token pair cosine；
- sequence gap：within-random 减 between；
- sequence centroid accuracy：使用每条 sequence 的偶数位置建立 centroid，预测奇数位置属于哪条 sequence。

Centered 指标在所有 sequence/token 的全局均值被移除后计算。

## 指标二：Residual 强方向由谁主导

对 (X_l,A_l,H_l) 的 centered token matrix 分别求 top-(k) PCA basis。报告：

\[
\operatorname{overlap}(U,V)=\frac{\|U^\top V\|_F^2}{k},
\]

以及 (X_l/A_l) top basis 对 (H_l) 最优 top basis 能量的恢复比例：

\[
R_X(k)=
\frac{\|\bar H_l U_{X,k}\|_F^2}
{\|\bar H_l U_{H,k}\|_F^2},
\qquad
R_A(k)=
\frac{\|\bar H_l U_{A,k}\|_F^2}
{\|\bar H_l U_{H,k}\|_F^2}.
\]

若 (R_X(k)>R_A(k)) 且 (H-X) overlap 高于 (H-A)，则支持 residual input 主导相加后强方向。

## 指标三：Attention 在 (H_l) 谱空间中的位置

将 centered (A_l) 投影到 (H_l) 的 common/middle bands：

\[
E_A(B)=\frac{\|\bar A_lU_{H,B}\|_F^2}{\|\bar A_l\|_F^2}.
\]

Tail energy 使用总能量减去 top-10% 投影能量。对每个 band 进一步计算 sequence gap 和 centroid accuracy，判断 context 信息位于哪里，而不只判断能量位于哪里。

## 判定

- 支持问题 1：(A_l) 的 centered adjacent/within similarity 和 sequence accuracy 在多数层显著高于 between baseline。
- 支持问题 2：多数层、多个 (k) 上 (R_X(k)>R_A(k))，且 (H-X) overlap 更高。
- 支持问题 3 的 middle/tail 版本：(A_l) 的 sequence gap/accuracy 在 middle/tail band 明显高于 common band。
- 证据不足：结论只在少数层出现，或 512/1024 两种长度方向相反。

## 输出

- `layer_metrics.csv`：每层全部标量指标；
- `summary.json`：配置和跨层汇总；
- `continuity_by_layer.png`；
- `sequence_accuracy_by_layer.png`；
- `top_subspace_dominance_by_layer.png`；
- `attention_energy_in_h_bands.png`；
- `attention_band_sequence_gap.png`。

## 补充：Band Attribution

仅报告 (A) 在 (H) band 中的能量比例不足以判断谁主导该 band。补充实验对每个 (H) band 同时计算：

\[
X_B=P_BX,\qquad A_B=P_BA,\qquad H_B=X_B+A_B.
\]

由于：

\[
\|H_B\|^2=\|X_B\|^2+\|A_B\|^2+2\langle X_B,A_B\rangle,
\]

结果同时报告：

- (X/A) 各自有多少能量落入该 band；
- band 内 (A) 的 norm share；
- cross term；
- 将 cross term 对称分配后的 Shapley energy share；
- (A_B) 与最终 (H_B) 的 adjacent-token continuity；
- A 自身谱 band 到 H 谱 band 的能量转移矩阵。

使用更细的 band：`0-1%, 1-2%, 2-5%, 5-10%, 10-20%, 20-50%, 50-100%`。
