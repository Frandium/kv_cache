父问题：

uniform feature分布下，随机初始化的dot-product top-1 MoE为什么不能自然形成feature level的specialization？

↓

Anchor0603回答什么

如果输入本身是均匀高维分布，那么 random Gaussian dot-product gate 为什么仍有固定 load imbalance？

结论：
1. 在高维空间中，不同向量近似正交，因此由于gating centering方向分布不均的load imbalance减弱
2. 对于正式使用的dot-product，存在gating row-norm variation，这个因素会加剧load imbalance

作用：
为后续实验提供一个baseline和最理想情况，将最理想情况的机制拆解，从而分离出后续实验中出现 imbalance 情形可能的导致原因和机制。

↓

Anchor0604回答什么

当 uniform symbolic features 经过 embedding / transformer hidden-state formation 后，step-0 load imbalance 还是 row norm 主导吗？

结论：
1. 在这种情形下，hidden common component是synthetic hidden-state 情形下更显著的导致gating imbalance的主因。0603中的row-norm不是主因

作用：
1. 将实验推进到了real hidden state中，并且将row norm的imbalance作用程度收束，主机制转向了hidden-state common component. 
2. 同时保持推进，因为去掉common component，仍然存在imbalance，从而驱动我们去探究residual部分的导致imbalance来源

↓

Anchor0605回答什么

去掉 hidden common component 后，剩余 load 是 finite-sample noise，还是还有真实 residual geometry？

结论：
去掉 common 后，残余 hidden states 仍然不是各向同性的球状分布；residual covariance anisotropy / structured residual geometry 是剩余 imbalance 的主要候选解释，而不是有限样本噪声。

作用：
1. 对“centering 解决了初始化几何问题” 这个说法进行了严格化，防止过早得出错误结论

↓

Anchor0606回答什么

在 A06_05 的 residual geometry 仍存在的情况下，如果给 oracle feature labels / centroids，feature-level routing partition 是否可达？

结论：
可达。oracle feature centroid 可以做到 perfect held-out feature routing。

作用：
证明目标 partition 本身不是不可能；如果后面 label-free 方法失败，失败原因是方法没有找到 partition，而不是 partition 不存在。


↓

Anchor0607回答什么
不用 labels，只做 global common-centering / projection / whitening，能不能接近 A06_06 的 oracle partition？

结论：
不能。它们能显著改善 load，但几乎不改善 feature_NMI；whitening 甚至让 NMI 崩掉。

作用：
纠正了一个以往的错误观念：更加load balance不意味着feature的specialization的达成

↓

现在知道什么

1. uniform feature 不保证uniform routing，更不保证feature specialization
2. gate-only中，row-norm variance是另一个可以分离出的imbalance 机制
3. common是一个很大的load imbalance的source
4. common-centering之后，residual geometry仍然存在
5. label-free common/residual control还是找不到 pure feature partition

↓

还不知道什么

1. 是否存在label-free的discovery能够找到pseudo-feature的中心
2. 初期训练动力学：即使初始化找到了pseudo-feature centers，是否，初期训练是否能够保持这种partition和feature specialization
3. real dclm中，应该使用的是什么作为feature的定义
4. common control在真实训练中的作用：是否只能作为load的guard，但是主方法还需要进一步推理

↓

下一步最自然去哪

A. feature discovery before routing
在 residual hidden states 上做 clustering / dictionary learning / contrastive feature estimation，先找到 pseudo-feature centers，再初始化 router。

B. anti-lockin after partition
从 oracle 或 pseudo-oracle partition 出发，测试 top-1 training 什么时候破坏这个 partition，以及需要什么 anti-lockin 机制保护它。

如果进入 DCLM，必须先定义 feature/proxy specialization metric，不能只看 load。

----

# Setting reliability audit for 0605-0607

结论：0605-0607 的 setting 足够支持当前主线继续推进到“无标签 feature discovery”和“防止 early softmax lock-in”；但它们只支持 initialized synthetic hidden-state replay 这个边界内的判断，不能直接支持 training stability、真实 DCLM 表现、或所有 label-free 方法无效。

## 0605 是否可靠

可靠。

它解决的是：去掉 hidden-state common component 后，残余 load imbalance 是否只是有限样本噪声。

可靠点：

- 沿用 0604 的 hidden-state surface：4 个 pair 严格均匀，pair position 随机，background token 与 pair id 无关，router readout 固定在 slot span 的最后一个位置；因为本轮 `slot_token_len=1`，所以这个位置就是 `pair_start`，不是序列最后一个 token。final block 与其他 block 使用一致 router reference。
- 有 state / router / replay / whitening / sample-count / context-position audit，不只是看最终 load。
- 关键对照区分了 centered replay、matched isotropic residual、whitened residual 和 sample-count guard。
- 结论不是来自单一 seed 或单一 depth/readout，结果表覆盖多个 seed、depth、readout 条件。

边界：

- 0605 可以说“structured residual geometry / residual covariance anisotropy 是剩余 imbalance 的主要候选解释”，不应该写成“covariance alone 完整因果解释了所有 residual imbalance”。
- whitening 在这里是 diagnostic，不是已经证明可部署的方法。
- context-position 条件上的高 $L$ 值只能说明局部条件下仍有结构，不应反过来当作 aggregate imbalance 的唯一来源。

决策：

0605 足以排除“只是有限样本噪声”的路线，允许进入“残余几何是否包含可用 feature structure”的下一步。

## 0606 是否可靠

可靠，但它是 positive control，不是方法证明。

它解决的是：如果给定真实 pair feature label，当前 hidden-state 几何中是否存在可以被 router 读出的 feature-aligned partition。

可靠点：

- 有 calibration / eval split，避免只在同一批样本上构造和验证 centroid。
- feature identity 明确对应 4 个 pair，专家数也是 4，因此 oracle partition 是可检验的。
- 同时比较 oracle centroid、raw centroid、random/equal-norm 等 baseline。
- 判断指标包含 feature NMI、load、margin、route heatmap，不把 balanced load 误当 specialization。

边界：

- 0606 使用标签，因此只能证明“partition reachable”，不能证明 label-free discovery。
- raw centroid 也能成功，所以 common-centering 不是 oracle reachability 的必要条件；它更多说明 feature signal 本来就在 hidden state 中。
- 它不说明训练能自然到达该 partition，也不说明真实 DCLM feature 已经定义。

决策：

0606 足以保留“feature-level specialization is geometrically available”的主线，允许继续问“没有标签时能不能发现它”。

## 0607 是否可靠

可靠，作为 negative control 足够。

它解决的是：只靠 label-free 的 global common/residual controls，是否能接近 0606 的 oracle feature partition。

可靠点：

- 明确继承 0605 的 residual-geometry 问题和 0606 的 oracle upper bound。
- 有 dependency audit 和 label-leakage audit，确保 label-free 条件没有偷用 pair label。
- 同时看 load 和 feature NMI，直接证明“load improvement 不等于 specialization improvement”。
- label-free 条件下 feature NMI 没有接近 oracle，而 oracle 条件可达，因此失败不是因为 hidden state 没有 feature signal。

边界：

- 0607 只裁定 simple global controls：global centering、projection、whitening、random-row variants 等。
- 它不能否定 clustering、dictionary learning、contrastive objective、activation patching proxy、gradient proxy 等更强的 feature discovery 方法。
- heldout batch mean 是强 transductive diagnostic；因为它仍未恢复 feature NMI，所以 negative result 反而更强，但它本身不是部署方案。

决策：

0607 足以停止把“load balance repair”当成主线方法，下一步应转向 feature proxy / cluster discovery；同时把 common/residual control 降级为 load guard 或 diagnostic，而不是 specialization mechanism。

## 对当前 exp_line 的收紧

需要收紧三句话：

1. 0605 的主张应是“structured residual geometry / covariance anisotropy 仍然制造 router imbalance”，不是“covariance 单独完整解释一切”。
2. 0606 的主张应是“oracle feature partition 可达”，不是“common-centered residual 是唯一可达方式”。
3. 0607 的主张应是“simple label-free global controls 失败”，不是“所有 label-free 方法失败”。

## Proceed gate

可以推进，但推进对象应该是：

- feature discovery：activation cluster、token-function proxy、gradient proxy、contrastive grouping。
- anti-lockin：如果初始 routing 已有错误 basin，训练早期如何防止 softmax/top-k 把错误 partition 固化。
- DCLM 前置条件：先定义 feature/proxy metric；不能只用 load balance 作为 specialization 证据。

不应立即推进的是：

- 只优化 load balance 的 router method。
- 没有 feature proxy 的真实 DCLM 大实验。
- 把 0607 解释成 label-free 路线整体失败。
