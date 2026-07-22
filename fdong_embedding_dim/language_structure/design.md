# 基于 Token 联合分布的递归语言 Pattern 模型

## 目标

我们希望完全从真实语料的 token 序列出发，检验语言是否具有以下统计结构：

- pattern 的频率跨越多个数量级；
- 高频 pattern 被大量更复杂的 pattern 复用；
- 复杂 pattern 可以递归地由少量低层 pattern 组合；
- 真实语料中的深层组合具有稳定复现和统计压缩价值。

整个分析只使用 token ID、位置和文档边界，不使用 LLM 参数、hidden state、embedding 或语义标注。

## 与谱空间猜想的关系

本研究来自 [natural_language_spectral_space.md](../meeting_docs/natural_language_spectral_space.md) 中的物理先验：自然语言具有层级化组合结构，高层 feature 由少量低层 feature 组合而成，低层 feature 可以被多个高层 feature 复用。

这里把该先验转化为一个独立于 LLM 的统计问题：仅根据 token 联合分布，检验真实语料中能否发现稳定、可复用、可继续组合的递归 pattern。谱空间不参与本实验的定义、发现或评价。

## 核心猜想

> 真实语言可以由一组反复复用的递归 token pattern 有效描述。Token 是叶子；两个已有 pattern 在有限窗口内的有序共现形成父 pattern；少量稳定父 pattern 继续参与更高层组合。

这条猜想支持以下可证伪预测：

1. 真实语料中存在大量跨文档稳定复现的高阶 pattern；
2. pattern support 呈长尾和多尺度分布；
3. 少量低层 pattern 具有很高的 parent reuse；
4. 递归 pattern 对真实语料的压缩和 held-out 复现显著强于保持低阶统计的随机语料。

## Pattern 的递归定义

设 token 词表为 $\mathcal V$。

### 原子 pattern

每个 token 都是一个深度为 0 的 pattern：

$$
\mathcal P_0=\{[v]:v\in\mathcal V\}.
$$

Token pattern $[v]$ 在文档位置 $i$ 出现时，其 occurrence interval 为：

$$
I([v])=[i,i].
$$

### 复合 pattern

给定已有 pattern $p,q$，定义：

$$
C_b(p,q),
$$

其中 $b=[d_{\min},d_{\max}]$ 是两个子 pattern 的 gap bucket。

一次有效 occurrence 需要满足：

1. $p$ 出现在 $q$ 之前；
2. 两个子 occurrence 使用互不重叠的 token 位置；
3. 两者间隔位于 $b$；
4. 父 occurrence interval 是两个子区间的最小包围区间。

父区间为：

$$
I(C_b(p,q))=
[\min(s_p,s_q),\max(e_p,e_q)].
$$

顺序属于 pattern identity：

$$
C_b(p,q)\neq C_b(q,p).
$$

递归 pattern universe 为：

$$
\mathcal P
=
\mathcal P_0
\cup
\{C_b(p,q):p,q\in\mathcal P\}.
$$

每个 pattern 都对应一棵有序二叉树：

```text
                  parent
                 /      \
             left        right
             /  \         /  \
          token token   token token
```

### 候选 pattern 与数据支持的 pattern

递归定义产生所有候选 pattern。真实语料分析只保留满足统计条件的 pattern：

$$
\mathcal A_l
=
\{p\in\mathcal P_l:
\operatorname{Keep}(p)=1\}.
$$

其中 $\mathcal A_l$ 是第 $l$ 层保留的 active pattern 集合。

## 每个 Pattern 的统计量

### Support

$$
n(p)=\text{pattern }p\text{ 的去重 occurrence 数}.
$$

同一文档中具有相同父 interval 的多种 child matching 只计一次。

### Document coverage

$$
c(p)=
\frac{\#\{d:p\text{ 在文档 }d\text{ 中出现}\}}
{\#\text{documents}}.
$$

Document coverage 用于减少网页重复句子和单文档 burst 对频率的影响。

### 子 pattern 关联强度

在相同 gap bucket 内计算 normalized PMI：

$$
\operatorname{NPMI}(p,q)
=
\frac{
\log\frac{P_b(p,q)}{P_b(p)P_b(q)}
}{
-\log P_b(p,q)
}.
$$

NPMI 高说明两个子 pattern 的共现强于其边际频率给出的基线。

### Held-out stability

语料按文档划分 train 和 validation。定义每百万 token occurrence rate：

$$
r_{\text{train}}(p)=\frac{n_{\text{train}}(p)}{T_{\text{train}}/10^6},
\qquad
r_{\text{valid}}(p)=\frac{n_{\text{valid}}(p)}{T_{\text{valid}}/10^6}.
$$

稳定性使用：

$$
s(p)=
\left|
\log\frac{r_{\text{valid}}(p)+\epsilon}
{r_{\text{train}}(p)+\epsilon}
\right|.
$$

第一版取 $\epsilon=10^{-12}$，仅用于避免数值除零；保留条件已经要求 validation support 大于零。

### Parent reuse

$$
R(p)=
\#\{q:p\text{ 是 }q\text{ 的直接 child}\}.
$$

它直接衡量一个低层 pattern 被多少种高层 pattern 复用。

### Pattern 规模

每个 pattern 同时记录：

- `depth`：递归树深度；
- `leaf_count`：叶子 token 数量；
- `span_p50/span_p90`：实际 occurrence span 的分位数；
- `left_id/right_id`：两个直接 child；
- `gap_bucket`：child 间距类型。

## 自底向上的 Pattern Mining

### 第一层：发现 `C-4-2`

记：

$$
C\text{-}4\text{-}2(a,b)
$$

为 token pattern $a$ 与 $b$ 在同一文档内有序共现、最小包围 span 不超过 4 的 pair pattern。程序流式扫描语料，只统计实际出现的 pair，不枚举词表的笛卡尔积。

从所有观测到的 `C-4-2` 中保留：

```text
support 足够高
+ NPMI 足够高
+ 在 held-out 文档中稳定复现
```

Support 保证 pattern 常见；NPMI 排除主要由两个高频 token 的边际频率造成的偶然共现；held-out stability 排除局部重复和单一文档 burst。保留集合记为 $\mathcal A_1$。

### 第二层：检验 `C-8-(C-4-2, C-4-2)`

主实验统计：

$$
C\text{-}8(p,q),
\qquad p,q\in\mathcal A_1,
\qquad p\neq q.
$$

一次有效 occurrence 同时满足：

1. $p$ 与 $q$ 是两个不同的 `C-4-2` pattern identity；
2. 两次 occurrence 使用的底层 token 位置互不重叠；
3. $p$ 在 $q$ 之前；
4. 父 pattern 的最小包围 span 不超过 8；
5. occurrence 不跨文档边界。

这一步直接检验：真实语料中稳定存在的局部 pair，能否进一步形成稳定的异质 pair-of-pairs。若这些二阶组合能跨文档复现、继续被不同父 pattern 复用，并显著强于 null corpus，就支持层次化合成 feature 的物理先验。

仅观察到 `C-8-(p,q)` 不构成证据，因为两个高频子 pattern 随机情况下也会共现。主统计量是二阶组合的 support、NPMI、held-out stability 和 parent reuse，并与保留 `C-4-2` 边际频率、破坏其配对关系的 null corpus 比较。

### 自组合与扩展组合

$p=q$ 的 `C-8-(p,p)` 单独统计为 self-composition。它可以反映重复、排比等真实结构，但不计入主实验的“不同低层 feature 形成高层 feature”指标。

完成主实验后，再加入 `token+pair`、`pair+token` 和不同深度 pattern 的组合，用于检验三叶、五叶等非平衡树。它们属于递归模型的扩展实验。

完整递归流程为：

```text
Level 0:
    token

Level 1, window ≤ 4:
    统计所有实际出现的 C-4-2
    → support + NPMI + held-out 筛选得到 A1

Level 2, window ≤ 8:
    主实验：C-8-(p, q), p,q ∈ A1, p ≠ q
    单独报告：C-8-(p, p)

扩展实验：
    token + retained pair
    retained pair + token
    不同深度 pattern + pattern

更高层, window ≤ 16, 32, ...:
    任意较低层 active pattern + 任意较低层 active pattern
    → 继续筛选
```

### Window schedule

第一版使用：

```text
level 1: maximum parent span = 4
level 2: maximum parent span = 8
level 3: maximum parent span = 16
level 4: maximum parent span = 32
```

每层内部再按真实 gap 分桶：

```text
0
1
2–3
4–7
8–15
16–31
```

Window 太小会漏掉长距离复用；window 太大会产生大量偶然共现。倍增 schedule 可以逐层观察新增结构。

## 需要追踪哪些 Pattern

程序从不构造 $|\mathcal A|^2$ 的完整 pattern 笛卡尔积，只处理语料中实际观察到的共现事件。

每层使用以下保留条件：

```text
train support ≥ min_train_support
validation support ≥ min_valid_support
NPMI ≥ min_npmi
held-out log-rate difference ≤ max_log_rate_diff
每个 support bucket 内 score 位于 beam 前列
```

第一版 10M-token pilot 的默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `train_fraction` | 0.8 | 按文档划分 80% train、20% validation |
| `max_depth` | 4 | 最多递归四层 |
| `window_schedule` | 4, 8, 16, 32 | 各层最大 parent span |
| `min_train_support` | 100 | train 中至少出现 100 次 |
| `min_valid_support` | 20 | validation 中至少出现 20 次 |
| `min_npmi` | 0.10 | 排除主要由边际频率解释的组合 |
| `max_log_rate_diff` | `log(2)` | train/valid 每百万 token rate 相差不超过 2 倍 |
| `beam_per_support_bucket` | 10,000 | 每个频率量级最多保留 1 万个 pattern |

Beam score 直接使用 NPMI；NPMI 相同时优先保留 document coverage 更高的 pattern。Support bucket 已控制频率量级，held-out 条件已控制复现稳定性。

Support bucket 使用 2 的幂次：

```text
[100, 199]
[200, 399]
[400, 799]
...
```

分桶 beam 保留不同频率量级的 pattern，避免所有名额被最高频组合占据。

这些值是 pilot 的实现参数。正式结论需要报告阈值扫描，并与随机语料使用相同参数。

## 可扩展计数方法

### 小规模原型

在 1M–10M tokens 上：

- 用 Python dictionary 精确统计实际观察到的 pair；
- pattern key 使用整数 tuple：`(left_id, right_id, gap_bucket)`；
- 每个文档独立处理，禁止 pattern 跨文档边界；
- 每层重新扫描语料，在文档内动态重建 active pattern occurrences。

### 大规模语料

在 100M tokens 以上使用两遍计数：

```text
第一遍：Count-Min Sketch 或 Space-Saving 找 heavy candidates
第二遍：只对 candidate keys 做 exact recount
```

最大 parent span 为 4 时，每个 token 最多与其后 3 个 token 形成有序 pair event，事件数与 token 数线性增长。内存只保存 sketch 和 retained candidate keys。

### 文档内 occurrence 生成

对每篇文档：

1. 创建 level-0 token occurrence；
2. 根据 pattern DAG 自底向上重建所有 active pattern occurrence；
3. 按 occurrence start 排序；
4. 使用双指针枚举窗口内实际出现的 pattern pair；
5. 检查顺序、gap、child token 不重叠；
6. 以 `(parent_key, start, end)` 去重；
7. 更新 candidate counter；
8. 文档结束后释放 occurrence。

这样不需要把全语料 occurrence posting list 常驻内存。

## 重复组合树的处理

相同叶子 pattern 可能有多种树：

$$
C(C(a,b),c),
\qquad
C(a,C(b,c)).
$$

第一版保留不同树结构，因为它们代表不同的组合假设。

完成统计后，对 occurrence 高度重合的 pattern 做等价检测：

$$
J(p,q)=
\frac{|O(p)\cap O(q)|}{|O(p)\cup O(q)|}.
$$

当 $J(p,q)\geq0.95$ 时，保留描述成本更低、复用度更高的树。大规模数据可以用 occurrence interval 的 MinHash 近似 Jaccard。

## 代码结构

```text
fdong_embedding_dim/language_structure/
├── design.md
├── pattern_types.py       # Pattern、Occurrence、gap bucket
├── corpus.py              # DCLM 文档读取与 tokenization
├── count_level.py         # 单层 candidate 生成与计数
├── mine_patterns.py       # 多层递归调度
├── score_patterns.py      # support、NPMI、stability、beam
├── evaluate_structure.py  # reuse、depth、coverage、null 对照
└── plot_structure.py      # 结果图
```

输入语料优先使用：

```text
/Users/bytedance/Desktop/dclm/part-*.txt
```

按完整文档切分 train/validation，避免同一文档片段进入两侧。

Tokenization 至少比较两种确定性方案：

- 项目当前使用的 Qwen tokenizer；
- byte 或固定词级 tokenization。

BPE merge 本身使用频率信息。跨 tokenization 复现实验可以判断结构是否依赖某个 tokenizer 的预先合并规则。

## 输出文件

### `patterns.parquet`

每行一个 retained pattern：

```text
pattern_id
left_id
right_id
depth
leaf_count
gap_bucket
train_support
valid_support
document_coverage
npmi
heldout_log_rate_diff
parent_reuse
span_p50
span_p90
score
```

### `level_summary.json`

每层记录：

- 输入 active pattern 数；
- 实际观察 pair event 数；
- candidate 数；
- 各拒绝原因数量；
- retained pattern 数；
- support、NPMI、depth、leaf count 和 span 分布。

### `examples.jsonl`

每个高频、高复用和深层 pattern 保存若干原文 occurrence，便于检查统计对象是否合理。

## 验证指标

主要图表：

1. 各 depth 的 support rank-frequency curve；
2. coverage 处于 10%、1%、0.1% 等区间的 pattern 数量；
3. parent reuse $R(p)$ 的分布；
4. `depth × support` 二维分布；
5. `leaf_count × span` 二维分布；
6. 每增加一层带来的语料压缩收益；
7. train 与 validation pattern rate 对比；
8. 真实语料与 null corpus 的 retained-pattern 数、深度和复用度对比。

Null corpus 至少包括：

- 保留 unigram frequency 的文档内 token shuffle；
- 保留 unigram/bigram transition 的一阶 Markov surrogate；
- 保留 window-4 局部结构、破坏跨 block 组合的 block shuffle；
- 保留每个 active `C-4-2` 的 occurrence 数和 span 分布、在相同位置区间内置换 pattern identity 的 occurrence-level permutation。

最后一种对照直接检验二阶结构：它保留一级 pattern 的边际频率和局部密度，破坏具体的 `C-8-(p,q)` 配对。

## 支持与失败条件

### 支持递归语言结构猜想

- 多个 depth 都产生大量 held-out 稳定 pattern；
- pattern support 在每层跨越多个数量级；
- 少量低层 pattern 具有很高 parent reuse；
- 深层 pattern 在 validation 中稳定复现；
- 真实语料的深度、复用和压缩收益显著超过 null corpus。

### 需要修改当前建模

- depth 1 以后几乎没有 pattern 通过 held-out 筛选；
- 深层 pattern 主要是同一短语的重复 parse tree；
- retained pattern 数、深度和复用度与 shuffled/Markov corpus 接近；
- 结果只在 BPE tokenizer 下出现；
- 统计结果对 support、NPMI 或 beam 阈值高度敏感。

## 最小实现顺序

第一轮只实现：

```text
1M tokens smoke test
→ 统计所有实际出现的 C-4-2
→ support + NPMI + held-out 筛选得到 A1
→ 统计 C-8-(p, q), p,q ∈ A1, p ≠ q
→ 单独统计 self-composition C-8-(p, p)
→ 输出 patterns.parquet、level_summary.json、examples.jsonl
→ 同参数运行 unigram shuffle 和 C-4-2 occurrence permutation
```

这一轮回答三个最小问题：

1. 实际候选数量和内存是否可控；
2. 异质 `C-8-(C-4-2, C-4-2)` 能否跨文档稳定复现；
3. 真实语料是否明显强于保留一级 pattern 边际频率的二阶 null 对照。

通过后再加入 `token+pair`、`pair+token`，并扩展到 10M–100M tokens、depth 4、Count-Min Sketch 和更多 null corpora。
