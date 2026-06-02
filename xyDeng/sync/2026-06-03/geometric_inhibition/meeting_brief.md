# Meeting Brief

## Meeting Summary

结论：

> 在 uniform multi-B synthetic 中，slot-stable initialization 能把 slot 信息注入 router 初始几何；在此基础上，geometric inhibition 能把 final route-slot NMI 稳定到 1.000，而且不是单纯提高 gate confidence。

最小理由：

1. Q1: C1 vs C0 的 step-0 route-slot NMI 是 0.242 vs 0.009；C4 vs C3 是 0.242 vs 0.008，说明 prototype 本身有 slot 信息。
2. Q2/Q3: C2 vs C1 final NMI 从 0.446 到 1.000；C5 vs C4 从 0.566 到 1.000。confidence 也上升，但 NMI 同时大幅上升，因此削弱“只是 gate sharper”的 rival。
3. Q5: 所有条件 target accuracy 都是 1.000，所以没有看到 routing 更整齐但任务变差的 tradeoff。

当前解释：

Observation: slot-stable init 提高 step-0 alignment；geometric inhibition 提高 final route-slot NMI 和 seed stability；selected gate confidence 上升但不是唯一变化。

Interpretation: prototype geometry 是可用的，但 init alone 不够稳定；显式 token margin / router-center separation 能抑制 route mixing。

## Meeting Preparation

**Question:**

在 top-1 MoE 中，如果我们先用 slot prototype 初始化 router，再加入 geometric inhibition，是否能稳定形成 slot-aligned routing？

**Why this matters?**

之前 ordinary top-1 router 的失败可能来自 early assignment lock-in。这个实验检查一个更清晰的机制：不是让模型自己发现 specialization，而是先给一个几何 prior，再问 inhibition 是否能防止 routing drift。

**Hypothesis:**

slot centroid prototype 含有可用的 slot direction；geometric inhibition 可以在训练中维持 assigned slot expert 和其他 experts 的 margin，并减少 route mixing。

**Evidence:**

最重要 metric 是 final route-slot NMI，因为本次组会目标是 routing stabilization，不是完整 utility specialization。

1. Q1 init 成功：C1/C4 的 step-0 NMI 都约 0.242，random baseline C0/C3 约 0.009。
2. Q2 geo 有额外贡献：C2/C5 final NMI 都是 1.000，且 final NMI std 为 0。
3. Q3 不是 confidence-only：C2 vs C1 confidence +0.028 且 NMI +0.554；C5 vs C4 confidence +0.029 且 NMI +0.434。
4. Q4 cosine：init-only 下 cosine 比 dot final NMI 更高，0.566 vs 0.446；加 geo 后两者都 1.000。
5. Q5 accuracy：所有条件 target accuracy 都是 1.000。

**Boundaries:**

这个结论只覆盖 uniform multi-B synthetic、external slot assignment $a(s,i)=s$、无 router bias、top-1 selected-gate MoE。

**Cannot claim yet:**

不能说真实语言数据中能自动获得 slot label；不能说 cosine 自然产生 specialization；不能说 Zipfian 高频/低频已经解决；不能说 expert utility 完整解决。

**Next Decision:**

组会中可以把它作为一个 clean positive mechanism：geometric inhibition stabilizes slot-assigned routing and weakens the confidence-only rival。

**Next Iteration:**

下一步测试 Zipfian multi-B 或 less-oracle prototype。如果在频率不平衡下 C2/C5 仍稳定，claim 可以推进到 frequency robustness；如果失败，则说明当前几何机制依赖 uniform exposure 或 oracle assignment。
