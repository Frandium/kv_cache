# 下一步待办：从总间隔到特征残余间隔

## 背景

当前实验说明：正间隔可以解释受控合成设置中的保持。但“总间隔为正”还
不够强，因为它可能来自两种不同来源：

1. **特征残余方向**：真正区分 feature 的方向。
2. **高增益公共谱带**：许多样本共享、容易被训练放大的公共方向。

导师文档提醒：高频公共预测结构会形成高增益公共谱带，后续更具体的特征
学习可能被这个谱带吸引。因此，下一步要问的不是“有没有正间隔”，而是：

```text
正间隔到底来自哪里？
```

## 待办 1：正间隔准入门槛

候选初始化进入训练前，必须先通过以下检查：

1. 不使用特征标签生成候选中心。
2. 不直接使用“已知路由位置”作为方法输入。
3. 训练前 feature NMI 高于 all-position baseline。
4. 每个特征中心对匹配专家有正间隔。
5. 负载不能由少数专家吞并多个特征来伪装成功。

通过后再进入 preservation training。失败则回到 route-relevant state
selection 或 center construction。

## 待办 2：谱带-间隔分解审计

设 $C$ 是高增益公共谱带，$P_C$ 是投影算子。把 feature center 分解为：

$$
\mu_f=P_C\mu_f+(I-P_C)\mu_f.
$$

把 route margin 分解为：

$$
m_f=m_f^C+m_f^{\perp C}.
$$

其中：

- $m_f^C$ 是公共谱带贡献；
- $m_f^{\perp C}$ 是公共谱带外的特征残余贡献。

## 最小审计

1. **A06_08 discovery audit**  
   比较 route-position centers 和 all-position centers 的 $m_f^C$ /
   $m_f^{\perp C}$。判断成功中心是否有更强 residual margin。

2. **A06_09 / A06_17_02 preservation audit**  
   沿训练轨迹跟踪 $m_f^C(t)$ 和 $m_f^{\perp C}(t)$。判断保持的是
   residual margin 还是 common-band margin。

3. **A06_17_04 / A06_17_05 stress audit**  
   在 shrink / crossing / random-perturbation 条件下，看先掉的是
   common-band margin 还是 residual margin。

## 判定

**支持特征残余机制：**

```text
m_f^{perp C} 能区分成功和失败初始化；
m_f^{perp C} 比总 margin 更能解释 preservation；
common-band margin 不能单独解释 feature NMI。
```

**削弱特征残余机制：**

```text
总 margin 为正但主要来自 common band；
residual margin 不能预测 feature NMI 或 preservation；
成功/失败主要由 common-band margin 决定。
```

## 更新后的问题

如果支持：08 线和 09 线可以合并成
“发现特征残余中心并形成可保持残余间隔”。

如果削弱：当前结果只能说“路由稳定”，不能说“分工由真实特征结构支撑”。

