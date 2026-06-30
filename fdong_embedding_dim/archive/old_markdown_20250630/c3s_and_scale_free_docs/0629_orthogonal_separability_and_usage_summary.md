# 0629 Orthogonal Separability and Common-Channel Usage Summary

## 0. 一句话结论

这组实验目前支持的主结论是：

\[
\text{参数矩阵的大奇异方向主要来自数据中的高频共享结构 / shared routing operator，}
\]

而不是单纯来自：

\[
\text{long-tail 表征里混入了 common component。}
\]

更细地说：

\[
\text{long-tail identity 可以在 residual subspace 中可分，}
\]

但模型的 prediction 仍可能依赖一个 shared top parameter channel，尤其是 attention routing 矩阵

\[
B_{qk}=W_q^\top W_k
\]

的 top singular channel。

因此，本轮实验把问题拆成了两件事：

\[
\text{可分性：long-tail 信息写在哪个子空间？}
\]

\[
\text{使用性：模型预测时实际依赖哪个参数通道？}
\]

这两件事不等价。

## 1. 原始问题

我们最初想验证的假设是：

\[
\text{long-tail 样本之所以使用 common/top 参数通道，}
\]

是因为：

\[
\text{long-tail 表征本身含有 common component。}
\]

更具体地，对于 tail hidden state：

\[
h_{\ell}=a_{\ell}c+r_{\ell}
\]

其中：

\[
c=\text{common/K direction}
\]

\[
a_{\ell}c=\text{tail 表征中的 common component}
\]

\[
r_{\ell}=\text{tail-specific residual component}
\]

我们想知道：

\[
\text{tail 使用 } B_{qk} \text{ top channel}
\]

到底是因为：

\[
a_{\ell}c
\]

还是因为数据任务本身存在 shared K-routing / common operator。

## 2. Toy 数据与模型

数据是 Fangdong-style K-token trigram task。

共有一个 shared token：

\[
K
\]

以及四个 group：

\[
A,B,C,D
\]

每个 group 有三个 token：

\[
G_0,G_1,G_2
\]

每个 group 都遵循同一个 transition pattern：

\[
G_0,G_1\rightarrow K
\]

\[
G_1,K\rightarrow G_2
\]

\[
K,G_2\rightarrow G_0
\]

\[
G_2,G_0\rightarrow G_1
\]

其中 \(A\) 是 common/high-frequency group，\(B,C,D\) 是 tail groups。

默认频率为：

\[
p_A=0.70,\quad p_B=p_C=p_D=0.10
\]

模型是 tied-embedding single-head attention toy：

\[
E\in\mathbb{R}^{13\times 32}
\]

\[
W_q,W_k,W_v\in\mathbb{R}^{32\times 32}
\]

\[
B_{qk}=W_q^\top W_k\in\mathbb{R}^{32\times 32}
\]

这里 \(B_{qk}\) 是 attention routing 的核心矩阵，因为 attention score 近似由：

\[
q^\top k
=
h_q^\top W_q^\top W_k h_k
=
h_q^\top B_{qk}h_k
\]

决定。

## 3. 实验变式

### 3.1 Natural Baseline

自然 baseline 是：

\[
\text{variant=baseline},\quad \text{tail\_common\_scale}=1
\]

这表示训练和 forward 都不额外修改 tail 的 common component。

### 3.2 Tail Common Scale

对 tail hidden state 做：

\[
h=a c+r
\]

然后改成：

\[
h_{\alpha}=\alpha a c+r
\]

其中：

\[
\alpha=\text{tail\_common\_scale}
\]

当：

\[
\alpha=1
\]

表示自然值。

当：

\[
\alpha=0
\]

表示删除 tail attention-input 中的 common projection。

注意：这个 scale 干预进入训练 forward，但它只作用在 attention 的 Q/K/V 输入 hidden state 上，不是完整 input/output 表征正交实验。

### 3.3 Orthogonal Update

每步 optimizer update 后，对 tail token embedding 做：

\[
E_{\mathrm{tail}}
\leftarrow
E_{\mathrm{tail}}
-
(E_{\mathrm{tail}}\cdot u_K)u_K
\]

其中：

\[
u_K=\frac{E_K}{\|E_K\|}
\]

这个方法让 tail embedding 尽量保持与 \(K/common\) direction 正交。

但它仍不是最严格的 input/output 正交，因为 forward 中输出头和 residual path 仍可能通过原始结构发生耦合。

### 3.4 Strict Orthogonal IO

严格版本在每次 forward 中构造：

\[
E_{\mathrm{io,tail}}
=
E_{\mathrm{tail}}
-
(E_{\mathrm{tail}}\cdot u_K)u_K
\]

然后 input、residual stream、output logits 全部使用 \(E_{\mathrm{io}}\)。

也就是说：

\[
\text{input embedding uses } E_{\mathrm{io}}
\]

\[
\text{residual uses } E_{\mathrm{io}}
\]

\[
\text{logits}=hE_{\mathrm{io}}^\top
\]

这个 setting 才回答：

\[
\text{如果 long-tail token 不允许在 K/common direction 上有投影，参数矩阵还会不会奇异？}
\]

## 4. 指标定义

### 4.1 Common Energy

\[
\mathrm{CommonEnergy}(h)
=
\frac{\|P_c h\|^2}{\|h\|^2}
\]

其中：

\[
P_c=u_Ku_K^\top
\]

它回答：

\[
\text{tail 表征里有多少能量落在 K/common direction 上？}
\]

### 4.2 Common Projection Separability

只看：

\[
a_i=\langle h_i,u_K\rangle
\]

能不能区分不同 tail target。

用 Fisher ratio：

\[
\mathrm{Fisher}
=
\frac{\sum_y n_y\|\mu_y-\mu\|^2}
{\sum_y\sum_{i:y_i=y}\|x_i-\mu_y\|^2}
\]

如果 \(x_i=a_i\)，就是 common projection Fisher。

它回答：

\[
\text{tail identity 是否能靠 common 投影值区分？}
\]

### 4.3 Residual Separability

先去掉 common projection：

\[
r_i=h_i-\langle h_i,u_K\rangle u_K
\]

再算 Fisher ratio 或 nearest-centroid accuracy。

它回答：

\[
\text{tail identity 是否能靠 residual/tail 子空间区分？}
\]

### 4.4 Bqk Top-Channel Spectral Metrics

对：

\[
B_{qk}=W_q^\top W_k
\]

做 SVD：

\[
B_{qk}=U\Sigma V^\top
\]

核心指标：

\[
\sigma_1
\]

\[
\mathrm{Top1Energy}
=
\frac{\sigma_1^2}{\sum_i\sigma_i^2}
\]

\[
\mathrm{EffRank}
=
\exp\left(-\sum_i p_i\log p_i\right),
\quad
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2}
\]

其中 top1 energy 越高、effective rank 越低，说明 \(B_{qk}\) 越尖、越接近单通道。

### 4.5 Bqk Top1 Ablation Damage

删掉 \(B_{qk}\) 的第一奇异分量：

\[
B_{qk}^{(-1)}
=
B_{qk}-\sigma_1u_1v_1^\top
\]

然后看 tail loss 增量：

\[
\Delta L_{\mathrm{tail}}
=
L_{\mathrm{tail}}(B_{qk}^{(-1)})
-
L_{\mathrm{tail}}(B_{qk})
\]

这个指标回答：

\[
\text{tail prediction 是否功能性依赖 } B_{qk} \text{ top channel？}
\]

它比单纯看奇异值更接近“使用程度”。

## 5. 第一组结果：Scale 干预

自然 baseline 中：

\[
\text{baseline},\quad \alpha=1
\]

有：

\[
\mathrm{tail\_common\_energy}=0.3406
\]

\[
\mathrm{common\_proj\_fisher}=1.16\times 10^{14}
\]

\[
\mathrm{residual\_fisher}=1.64\times 10^{14}
\]

\[
\mathrm{common\_centroid\_acc}=0.7778
\]

\[
\mathrm{residual\_centroid\_acc}=0.7778
\]

这说明：

\[
\text{tail 在 common projection 上可分，}
\]

但：

\[
\text{tail 在 residual subspace 上同样可分，甚至 Fisher 更高。}
\]

scale sweep 中，\(\alpha\) 增大确实让 common energy 增大：

\[
0.0687\rightarrow 0.2618\rightarrow 0.2841\rightarrow 0.3406\rightarrow 0.3896
\]

并且 common projection Fisher 也增大：

\[
1.08
\rightarrow
8.17\times 10^{12}
\rightarrow
3.12\times 10^{13}
\rightarrow
1.16\times 10^{14}
\rightarrow
2.54\times 10^{14}
\]

所以 scale 干预有效地改变了：

\[
\text{tail 在 common direction 上的可分性。}
\]

但是，tail 对 \(B_{qk}\) top1 的功能依赖并不随 \(\alpha\) 单调变化：

\[
\Delta L_{\mathrm{tail}}:
1.581,\ 0.00012,\ 0.252,\ 1.839,\ 0.0279
\]

因此不能说：

\[
\text{tail common component 越大，tail 越依赖 } B_{qk} \text{ top channel。}
\]

更准确是：

\[
\text{common projection 是一条可分通路，但模型是否使用 } B_{qk} \text{ top channel 还由任务结构和优化决定。}
\]

## 6. 第二组结果：Strict Orthogonal IO

比较自然 baseline：

\[
\text{baseline},\quad \alpha=1
\]

和：

\[
\text{strict\_orthogonal\_io},\quad \alpha=1
\]

自然 baseline：

\[
B_{qk}\ \mathrm{Top1Energy}=0.378
\]

\[
B_{qk}\ \mathrm{EffRank}=3.467
\]

\[
\Delta L_{\mathrm{tail}}=1.839
\]

strict orthogonal IO：

\[
B_{qk}\ \mathrm{Top1Energy}=0.922
\]

\[
B_{qk}\ \mathrm{EffRank}=1.436
\]

\[
\Delta L_{\mathrm{tail}}=5.752
\]

这说明：

\[
\text{即使 tail identity 不允许写在 K/common embedding direction 上，}
\]

\[
B_{qk} \text{ 仍然会形成极强的 top singular channel。}
\]

同时在 strict IO 中：

\[
\mathrm{common\_centroid\_acc}=0.5556
\]

\[
\mathrm{residual\_centroid\_acc}=0.7778
\]

说明：

\[
\text{tail identity 主要在 residual subspace 中可分。}
\]

但：

\[
\Delta L_{\mathrm{tail}}=5.752
\]

说明：

\[
\text{tail prediction 仍强烈依赖 } B_{qk} \text{ top routing channel。}
\]

所以可分性和使用性被分离了：

\[
\text{tail 信息写在 residual 子空间，}
\]

但：

\[
\text{prediction 仍经过 shared top parameter channel。}
\]

## 7. 第三组结果：Frequency Ablation

频率设置：

\[
\text{uniform}: (0.25,0.25,0.25,0.25)
\]

\[
\text{mild}: (0.40,0.20,0.20,0.20)
\]

\[
\text{current}: (0.70,0.10,0.10,0.10)
\]

\[
\text{extreme}: (0.90,0.0333,0.0333,0.0334)
\]

在 natural baseline 下：

\[
\mathrm{Top1Energy}:
0.266,\ 0.495,\ 0.378,\ 0.740
\]

\[
\mathrm{EffRank}:
4.052,\ 3.028,\ 3.467,\ 2.294
\]

\[
\Delta L_{\mathrm{tail}}:
0.0003,\ 1.188,\ 1.839,\ 3.125
\]

从 uniform 到 extreme 的总体趋势是：

\[
\text{频率越偏斜，} B_{qk} \text{ 越尖，tail 对 top channel 的功能依赖越强。}
\]

这个结果支持：

\[
\text{frequency imbalance + shared K-routing}
\Rightarrow
\text{top singular parameter channel}
\]

而不是：

\[
\text{tail/common 表征混合 alone}
\Rightarrow
\text{parameter singularity}
\]

## 8. 当前完整结论

### 8.1 参数奇异性的来源

当前实验支持：

\[
\text{参数奇异性主要来自高频 shared structure / K-routing。}
\]

也就是说，很多样本反复需要同一个 shared operation：

\[
G_0,G_1\rightarrow K
\]

\[
G_1,K\rightarrow G_2
\]

\[
K,G_2\rightarrow G_0
\]

这些 shared routing pattern 会反复更新 \(B_{qk}\) 的相似方向，最终形成：

\[
\text{large singular value / top parameter channel}
\]

### 8.2 Long-tail 可分性的位置

long-tail identity 不是一定要写在 common projection 上。

在 natural baseline 中：

\[
\text{common projection 和 residual subspace 都可分。}
\]

在 strict orthogonal IO 中：

\[
\text{residual subspace 更可分。}
\]

所以：

\[
\text{tail 是否在 common direction 上可分，取决于表征约束和训练方式。}
\]

### 8.3 模型实际使用的参数通道

即使 tail identity 在 residual 子空间可分，模型 prediction 仍可能依赖：

\[
B_{qk} \text{ top singular channel}
\]

因为这个 top channel 不是单纯用来编码 tail identity，而是用来执行 shared routing operator。

所以：

\[
\text{可分性}\ne\text{功能使用性}
\]

这是本轮实验最重要的 conceptual takeaway。

## 9. 与原始假设的关系

原始强假设是：

\[
\text{tail 使用 common/top channel 是因为 tail 表征里有 common component。}
\]

现在需要改成弱假设：

\[
\text{tail 表征中的 common component 提供了一条可分通路，}
\]

但：

\[
\text{top parameter channel 的形成主要由 shared data structure / frequency 驱动。}
\]

更准确地说：

\[
\text{common component 决定 tail 信息能否写进 common projection，}
\]

\[
\text{shared routing 决定参数矩阵是否形成高增益 top channel。}
\]

二者相关，但不是同一件事。

## 10. 与 Fangdong CRS 结论的关系

Fangdong 的 CRS 最终结论是：

\[
\text{谱更平} \not\Rightarrow \text{loss 更好}
\]

因为大奇异方向可能是 useful shared feature，而不是 bug。

我们的实验与这个结论一致：

\[
\text{即使 strict orthogonal IO 让 tail identity 转到 residual subspace，}
\]

\[
B_{qk} \text{ 仍然可能更尖，而且 tail prediction 更依赖它。}
\]

这说明：

\[
\text{top singular channel 可能是 shared routing feature。}
\]

因此，不能把目标简单设成：

\[
\text{消灭大奇异值}
\]

更合理的目标是：

\[
\text{区分哪些 top channel 是 useful shared operator，哪些会压制 tail residual learning。}
\]

## 11. 当前边界与不足

第一，common direction 在本 toy 中定义为：

\[
u_K=\frac{E_K}{\|E_K\|}
\]

这不是真实 LLM 中所有 common subspace。

第二，当前 toy 的 shared structure 非常强，因为所有 group 都共享 \(K\)。所以结论主要针对：

\[
\text{shared K-routing}
\]

第三，当前实验已经说明 frequency 和 shared structure 很重要，但还没有完全拆开：

\[
\text{frequency imbalance}
\]

和：

\[
\text{shared operator}
\]

下一步如果继续，应做：

\[
\text{universal K vs group-specific K vs no K}
\]

来区分 shared structure 本身的作用。

## 12. 最终总结

当前 0629 实验链条可以总结为：

\[
\text{frequency + shared K-routing}
\Rightarrow
\text{top singular parameter channel}
\]

\[
\text{representation constraint}
\Rightarrow
\text{tail identity 写在 common projection 或 residual subspace}
\]

\[
\text{prediction path}
\Rightarrow
\text{tail 是否功能性依赖 top channel}
\]

所以对“为什么主要使用 common/top 方向，而不是 long-tail/residual 方向”的回答是：

\[
\text{不是因为 residual 方向不可分，}
\]

而是因为：

\[
\text{shared high-frequency routing 形成了更高增益、更早稳定的参数通道。}
\]

long-tail 可以在 residual 子空间中可分，但只要它的 prediction 仍经过 shared routing operator，它就可能继续依赖 top singular parameter channel。

