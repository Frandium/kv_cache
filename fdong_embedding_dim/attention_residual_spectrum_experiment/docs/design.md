# Attention Context 在 Residual 谱空间中的位置

## 核心问题

对 Qwen3 每一层，记进入 attention residual 的表征为 (X_l)，不含 residual 的 attention 输出为 (A_l)，相加后的表征为：

\[
H_l=X_l+A_l.
\]

本实验回答三个问题：

1. (A_l) 是否在同一 sequence 内连续，同时能够区分不同 sequence？
2. (H_l) 的强谱方向是否仍主要由 (X_l) 决定，而不是由 (A_l) 决定？
3. (A_l) 的能量和 sequence-level 区分信息主要位于 (H_l) 谱空间的 top、middle 还是 tail band？

## 表征位置

标准 Qwen3 decoder layer 计算：

\[
X_l=\text{layer input},
\qquad
A_l=\operatorname{SelfAttn}(\operatorname{RMSNorm}(X_l)),
\qquad
H_l=X_l+A_l.
\]

Hook 分别挂在 decoder layer 输入和 `self_attn` 输出。这里的 (A_l) 已经过 `o_proj`，但尚未与 residual 相加。

## 谱空间

对跨 sequence、跨 token 收集的 (H_l) 做全局中心化：

\[
\bar H_l=H_l-\mathbb E[H_l].
\]

以 randomized PCA 得到 (H_l) 的 top 10% 方向，并定义：

\[
B_C=[0,1\%),\qquad B_M=[1\%,10\%),\qquad B_T=[10\%,100\%).
\]

同样计算 (X_l) 和 (A_l) 的 top subspace，用 principal-subspace overlap 与对 (H_l) 的重构能量判断谁主导 (H_l) 的强方向。

## 关键区分

本实验同时报告 raw 与 centered similarity。Raw cosine 可能被所有 token 共享的均值方向抬高；centered cosine 更接近 token/sequence 间的可分变化。

“Attention 位于 tail”需要同时满足两类证据：

1. (A_l) 在 (H_l) middle/tail band 中具有较高能量占比；
2. (A_l) 的 sequence-level 区分性主要由 middle/tail band 保留。

仅凭同 sequence cosine 高，不能推出 attention output 位于谱尾。

