# 结果索引：A06 特征中心、间隔与保持机制

## 主线

```text
A06_08: 能不能无标签找到特征中心
-> A06_09: 找到中心后训练能不能保持
-> A06_17_02-05: 为什么保持，以及间隔边界在哪里
-> 下一步: 正间隔到底来自特征残余方向还是高增益公共谱带
```

## 关键结果

| 实验 | 问题 | 结果 | 对机制的更新 |
|---|---|---|---|
| A06_08 | 路由位置无标签聚类能否找到特征中心？ | k-means / spherical k-means 在 full grid 达到 `feature_NMI=1.0`。 | 特征中心在合成路由位置上可发现。 |
| A06_09 | A06_08 的 pseudo center 初始化训练后能否保持？ | pseudo init 和 oracle init 都保持 `feature_NMI=1.0`。 | 好初始化可以进入可保持 basin，但当时没有解释为什么。 |
| A06_16 | 更接近真实输入混合后是否仍可发现和保持？ | no-position C0-C3 step-0 与 final NMI 均为 `1.000`。 | learned absolute position was the old confound; controlled route-position bridge passes. |
| A06_17 | all-position hidden states 能否直接聚类？ | route-only / slot offset 3 为 `1.000`；all-position mean 约 `0.797`。 | 失败主要是样本池错误，不是 feature geometry 不存在。 |
| A06_17_02 | 保持是否来自 router 主动追踪 center？ | center init NMI `1.000`，但 movement alignment 约 `-0.4`。 | active tracking 被削弱；正 margin 解释更强。 |
| A06_17_03 | 训练漂移是否吃掉初始边界？ | worst boundary consumption `<0.70`，dynamic switch rate `0.000`。 | observed preservation stays inside margin buffer. |
| A06_17_04 | margin 是不是因果边界？ | static crossing ratio 约 `1.04-1.07`; forced crossing breaks matched region. | margin is a usable geometric boundary in this controlled bridge. |
| A06_17_05 | exact center 是否必须？ | `rho<=0.70` preserves; pure random has preserve fraction `0.219`. | exact center not required; positive-margin basin is the object. |
| A06_18 | PCA/AE/SAE 能否替代 route-relevant selection？ | SAE reconstruction can be good but feature NMI is weak. | generic reconstruction objective is not route objective. |
| A06_19 | minimal-prior M1/M2 center search 是否稳定？ | M1/M2 do not reliably beat raw all-position. | tested weak-prior operationalizations are not enough. |
| A06_20 | route-logit common subtraction 能否救 feature recovery？ | route-logit subtraction changes load but does not improve feature NMI. | load-changing common removal is not feature recovery. |

## 当前一句话判断

受控设置中，特征中心初始化可以形成并保持分工；保持更像正间隔安全区，
不是路由向量主动追踪中心。下一步要判断这个正间隔是否由真正的特征残余
支撑。

## 不能推出

- 真实语言语义专家已经存在；
- 高归一化互信息等于专家有功能价值；
- 任意无标签聚类都能找到可用中心；
- 真实 DCLM 训练保持已经解决。

