# 均匀特征如何进入稳定专家分工

## Question

均匀出现的 features，能不能稳定、均匀、并且有功能价值地分到不同 experts 上？

这里的“稳定均匀”不是只看 expert load 是否平均，而是看三件事：

1. 同一个 feature 是否主要进入同一个 expert；
2. 不同 feature 是否没有被合并到同一个 expert；
3. 训练之后这个分区是否还能保持。

## Verdict

在均匀features分布合成数据实验上，

**1. 可以构造出 feature-level expert 初始化partition，这种partition在初始时能够将features均匀分开，并且训练后（1600steps）能够准确预测并且仍然保持feature-level specialized的partition.但它不会由 random gating 自然产生，需要依赖表征几何的分布。**

**2. 这种partition可以不依赖features的真实labels，由所有hidden states无监督聚类中心产生（但7/8个seeds的实验中成功，一个seed失败，失败原因是聚类时两类features中心过近发生混淆）**

具体结论是：

1. route-position hidden states 里确实有可聚类的 feature center；
2. 减去共同成分后，在干净路由位置上聚类，可以恢复 feature-to-expert 分区；
3. random gating 会被 common component 和随机几何带偏；
4. all-position clustering 会因为聚类对象选错而合并 features；
5. 真实数据集 DCLM 中 proxy feature 可发现、可线性路由，但普通训练会在 step 5/10 中再次发生gating混淆，无法保持初始的hidden states聚类结构。

所以当前主瓶颈不是“feature 是否存在”，而是：

```text
如何选对 hidden-state 聚类的对象；
如何让第 0 步已经形成的分区能够在训练早期（从而能够避免early lock-in现象发生）。
```

## Mechanism

路由位置 hidden state 写成：

$$
h_f = c + r_f + \epsilon_f
$$

其中：

- $c$ 是所有 feature 共享的共同成分；
- $r_f$ 是 feature $f$ 自己的剩余成分；
- $\epsilon_f$ 是噪声。

门控分数是：

$$
z_{f,e}=w_e^\top h_f=w_e^\top c+w_e^\top r_f+w_e^\top \epsilon_f
$$

这解释了为什么 feature 均匀不等于 expert 均匀：

- $w_e^\top c$ 会给某些 expert 一个对所有 feature 都存在的共同优势；
- random $w_e$ 不一定和 feature centers 对齐，所以多个 feature 可能被同一个 expert 接走；
- top-1 training 会放大早期优势，导致 lock-in 或 collapse。

因此正确路线不是指望 random gate 自己分开 feature，而是先找到 $r_f$ 的中心，再把 router 初始化到这个分区附近。

## Evidence

### 1. Feature center 是存在的，在表征集合中可见。

A06_08 显示：对 route-position residual hidden states 做 k-means，可以达到 `feature_NMI=1.0`、load $L=0$。

A06_09 显示：这个 pseudo-center 初始化在受控训练中可以保持到 1600 步。

A06_16 显示：去掉 learned absolute positional embedding 后，C0-C3 的 discovery 和 preservation 都通过。

### 2. Random gating 不会自然给出好分区。

A05_04_02 显示：toy dot-product setting 中，common logit 早于 collapse，并且 common-logit cancellation 可以显著提升 slot NMI。

A05_04_03 显示：真实 DCLM 中，step 0 的强 common-domination 版本被削弱，但 common component 仍然影响 load；到 step 10，common channel 会迅速放大。

所以 random gating 的失败不是因为 feature 不均匀，而是因为 router 看到的几何不是干净 feature geometry。

### 3. All-position clustering 失败是因为背景噪声过大，导致聚类中心偏离我们想要的feature中心。

A06_17 显示：route-only 和 slot 最后位置聚类都能达到 `feature_NMI=1.0`；但 all-position clustering 平均只有 `0.797`，经常把完整 feature 合并到同一个 expert。

这说明问题不是 route feature geometry 不存在，而是不能把所有 hidden states 都扔进同一个 k-means。

### 4. Real DCLM 的第一失败点是初始化几何会在早期训练时发生变换。

A06_10 发现 real DCLM hidden states 有稳定 proxy clusters。

A06_11 显示 proxy centers 可以转成 step-0 linear routing。

A06_12/A06_13 显示普通 DCLM 训练会把 raw-center routing 从 step-0 NMI `0.7549` 打到 step-5 `0.0410`、step-10 `0.0131`，而 loss 没有明显坏掉。

所以真实主线的下一步应当是 preservation / anti-feedback，而不是继续证明 proxy feature 存在。

### 5. 使用common-control的方法这个分区在受控环境中有功能价值，能够降低rare feature的loss。

A07_01 到 A07_03 显示：在 controlled D07 中，common-control 不只是让 load 更平均，还能降低 rare feature loss，并且 routed expert 有对应的 utility。

这说明 feature partition 值得保存，但目前只限 controlled synthetic setting。

## Boundary

现在可以说：

- feature-level partition 在受控路由位置上可达；
- random gating 不是可靠机制；
- all-position clustering 不是可靠 feature discovery object；
- real DCLM 当前坏在 early training preservation；
- controlled D07 支持这个分区有功能价值。

现在不能说：

- 真实语言语义专家已经形成；
- common subtraction 是最终方法；
- all-position clustering 可以直接用于真实数据；
- real DCLM training preservation 已经解决；
- A07 已经证明真实 checkpoint 有 expert utility。

## Ask

我想请导师裁定下一步：

**是否把下一张 anchor 定为 real-text early preservation / anti-feedback？**

具体问题是：

```text
能否让 step-0 已经存在的 proxy feature partition 穿过 step 5/10 的训练覆盖窗口，
同时不显著伤害 LM loss？
```

如果导师同意，下一步最小实验是：

1. 固定 A06_10/A06_11 的 proxy labels 和 raw-center initialization；
2. 对比普通训练、router freeze / delayed update、低 router learning rate、common/load anti-collapse、proxy-preservation auxiliary loss；
3. 主指标看 step-5/10 `proxy_route_NMI`，约束指标看 LM loss。

## One-Sentence Close

**现在不是继续证明 feature 存在，而是要测试：已经存在并可路由的 feature partition，能不能在真实文本训练早期被保存下来。**
