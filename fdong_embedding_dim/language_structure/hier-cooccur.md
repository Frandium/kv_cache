# Hierarchical Co-occurrence in Natural Language

## 一句话结论

> 仅使用真实语料中的 token、位置和文档边界，我们发现：稳定的低层 token pair 会以显著超出随机配对的方式形成稳定的上层组合，同一个低层 pattern 会被多个不同上层 pattern 复用。自然语言数据因此呈现出可直接统计的层次化组合结构。

本实验不使用语言模型参数、hidden state、embedding、梯度或语义标注。Qwen3 tokenizer 只负责把文本确定性地转换为 token ID。

## 要回答的问题

自然语言显然包含 token 共现。我们进一步检验：这些共现能否形成可递归复用的节点。

如果语言具有层次结构，应当观察到：

1. 一部分 token pair 在不同文档中稳定复现，形成低层节点；
2. 一部分低层节点继续稳定组合，形成上层节点；
3. 上层组合强于低层节点边际频率给出的随机基线；
4. 同一个低层节点进入多个不同上层节点，形成复用结构。

## Pattern 的递归定义

设 token 词表为 $\mathcal V$。

### 原子 Pattern

每个 token 是深度为 0 的 pattern：

$$
\mathcal P_0=\{[v]:v\in\mathcal V\}.
$$

Token $v$ 出现在位置 $i$ 时，occurrence interval 为：

$$
I([v])=[i,i].
$$

### 复合 Pattern

给定两个已经定义的 pattern $p,q$，若它们在同一文档内有序出现、底层 token 位置不重叠，定义：

$$
C_W(p,q),
$$

其中 $W$ 是父 pattern 允许的最大 span。父 occurrence interval 是两个 child interval 的最小包围区间：

$$
I(C_W(p,q))=
[\min(s_p,s_q),\max(e_p,e_q)].
$$

一次 occurrence 满足：

$$
|I(C_W(p,q))|\leq W.
$$

顺序和 child 间的 gap bucket 属于 pattern identity，因此：

$$
C_W(p,q)\neq C_W(q,p).
$$

完整 pattern 空间递归定义为：

$$
\mathcal P
=
\mathcal P_0
\cup
\{C_W(p,q):p,q\in\mathcal P\}.
$$

### 本实验使用的两层 Pattern

`C-4-2` 表示两个 token 在 span 不超过 4 的区间内有序共现：

$$
C\text{-}4\text{-}2(a,b)=C_4([a],[b]).
$$

`C-8-4` 表示两个已经保留的 `C-4-2` 在 span 不超过 8 的区间内继续组合：

$$
C\text{-}8\text{-}4(p,q)=C_8(p,q),
\qquad p,q\in\mathcal A_1.
$$

其中 $\mathcal A_1$ 是通过频率、关联强度和 held-out 稳定性筛选的 `C-4-2` 集合。

一个真实例子是：

```text
C-8-4: "to figure out what"
├── C-4-2: "to" + "figure"
└── C-4-2: "out" + "what"
```

低层 pair 在这里成为具有独立 identity、频率和复用度的节点；父 pattern 记录两个节点的组合关系。这正是层次结构在 token 联合分布中的操作化定义。

## 为什么采用这个定义

### 它完全由数据决定

每个 pattern 的 identity 来自 child identity、顺序和间距；每个 occurrence 来自 token 位置。整个过程可以用计数完成，不需要模型判断一句话表达了什么语义。

### 它同时表达组合与复用

父节点只保存直接 child。一个 child 可以进入多个不同 parent：

```text
低层 pattern p
├── C(p, q1)
├── C(p, q2)
└── C(q3, p)
```

因此挖掘结果形成有向无环图。节点的入边表示“由哪些直接下层节点组成”，出边表示“被多少上层节点复用”。

### 它允许语义、句法和格式使用同一个统计语言

`in an effort to`、`On the other hand`、`19th century`、日期和编号列表都可以由相同递归规则描述。对于解释 LLM 学习到的数据结构，它们都是训练分布中稳定、可共享的 pattern。

## 为什么递归定义能够覆盖任意有限共现

命题：给定任意有限、有序、互不重叠的 token occurrence 集合，只要最大 window 足以覆盖它们的最小包围区间，就存在一棵由上述递归规则生成的二叉组合树，其叶子恰好是这些 token occurrences。

证明采用叶子数量归纳。

当只有一个 token 时，它属于 $\mathcal P_0$。

假设任意少于 $m$ 个 token 的有序共现都能递归表示。对 $m$ 个 token，在任意相邻边界将其分成左右两个非空有序集合。根据归纳假设，左侧可以表示为 pattern $p$，右侧可以表示为 pattern $q$。取能够覆盖两者最小包围区间的 $W$，则：

$$
C_W(p,q)
$$

表示全部 $m$ 个 token。归纳完成。

因此：

- 任意有限 n-token 共现都能表示为递归二叉组合；
- 任意直接包含多个 child 的节点都能通过二叉括号化表示；
- 不同括号化对应不同的层次组合假设；
- 逐层增加 window 可以覆盖越来越长程的共现。

这个覆盖性说明递归定义不会因为 pattern 长度而遗漏某类有序共现。当前实验只实际计算到 `C-8-4`，验证的是第一层递归组合。

## 统计实验

### 数据

```text
corpus:                    Desktop/dclm
documents:                 10,000
Qwen3 tokens:              6,950,856
maximum tokens/document:   1,024
train/validation split:    80% / 20%，按文档稳定哈希划分
```

每个 DCLM 文档是一行 JSON 字符串。实验不允许 occurrence 跨越文档边界。

### C-4-2 计数

对 token 序列 $x_0,\ldots,x_{T-1}$，枚举：

$$
(x_i,x_{i+d},d),
\qquad d\in\{1,2,3\}.
$$

保留条件为：

```text
train support >= 100
validation support >= 20
NPMI >= 0.10
train/validation 每 token occurrence rate 相差不超过 2 倍
```

其中：

$$
\operatorname{NPMI}(p,q)
=
\frac{
\log\frac{P(p,q)}{P(p)P(q)}
}{
-\log P(p,q)
}.
$$

NPMI 衡量两个 child 的共现超过其边际频率基线的程度。

### C-8-4 计数

重新扫描 tokenized documents，重建所有 active `C-4-2` occurrence。对两个 occurrence $p,q$，仅在以下条件同时成立时生成父 pattern：

```text
p 位于 q 之前
p 与 q 的底层 token 位置不重叠
父 interval span <= 8
不跨文档边界
```

主结果统计 $p\neq q$ 的异质组合。$p=q$ 的 self-composition 单独报告。

`C-8-4` 保留条件为：

```text
train support >= 20
validation support >= 5
NPMI >= 0.10
train/validation 每 token occurrence rate 相差不超过 2 倍
```

### 一级边际保持的 Null

仅仅观察到两个高频 child 共现无法说明层次组合。Null 对每篇文档执行：

1. 保留每个 `C-4-2` occurrence 的 start、end 和 span；
2. 在相同 span 内置换 pattern identity；
3. 保留每个一级 pattern 在该文档中的 occurrence 数和 span 分布；
4. 重新统计 `C-8-4`。

真实语料和 null 都产生 4,898,336 个二层 occurrence events。两者的一级节点频率和局部 occurrence 密度相同，差别来自具体 child identity 的配对关系。

## 结果

### C-4-2

```text
train occurrence events:        16,564,305
validation occurrence events:    4,228,263
unique train candidates:          6,586,545
train support >= 100:                10,308
validation support >= 20:             9,989
NPMI >= 0.10:                         4,828
held-out stable:                       4,818
```

4,818 个最终一级 pattern 中：

```text
包含字母:       4,005  (83.1%)
包含数字:         646  (13.4%)
只有标点或空白:   167  (3.5%)
```

### C-8-4 与 Null

| 指标 | 真实语料 | Null |
|---|---:|---:|
| unique train candidates | 2,005,763 | 2,868,704 |
| train support 达标 | 13,633 | 3,719 |
| NPMI 达标 | 8,201 | 325 |
| held-out 稳定异质 pattern | **7,535** | **242** |
| held-out 稳定 self-composition | 78 | 66 |
| 异质 pattern median NPMI | 0.263 | 0.145 |
| 异质 pattern median train support | 42 | 33 |

Null 产生更多 unique identity，说明随机置换把同样数量的事件分散到更多组合中。真实语料把事件集中到反复出现的特定组合中。相同筛选条件下，真实语料的稳定异质上层节点数量是 null 的：

$$
\frac{7535}{242}=31.1.
$$

### 下层节点复用

在 4,818 个 `C-4-2` 中：

```text
被至少 1 个异质 C-8-4 parent 使用: 1,422
被至少 2 个异质 C-8-4 parents 使用:  934
被至少 5 个异质 C-8-4 parents 使用:  593
90th percentile parent reuse:             6
maximum parent reuse:                   409
```

这些 pattern 构成“少量 child 形成 parent、child 被多个 parent 复用”的两层 DAG。

## 低频但具有明确语义的 C-8-4

下表选择 train support 仅为 20–30 的异质 pattern。它们全部在 validation 中复现，并分布在多篇文档中。

| 上层 Pattern | Train / Valid | 文档数 | NPMI | 原文中的形式 |
|---|---:|---:|---:|---|
| `to figure + out what` | 26 / 7 | 28 | 0.844 | `try to figure out what to sign` |
| `in an + effort to` | 25 / 6 | 31 | 0.815 | `in an effort to defuse the immediate issue` |
| `in some + cases,` | 20 / 5 | 25 | 0.791 | `or may, in some cases, cause ...` |
| `to keep + in mind` | 25 / 5 | 29 | 0.749 | `what we have to keep in mind` |
| `On the + other hand` | 29 / 12 | 37 | 0.724 | `On the other hand, Mitsubishi ...` |
| `to find + out what` | 24 / 12 | 34 | 0.683 | `to find out what this was all about` |
| `something to + do with` | 24 / 11 | 34 | 0.680 | `may have something to do with ...` |

这些例子由统一的 token 计数程序自动发现。算法没有词组词典，也没有语义判别器。

### 数字与结构 Pattern

数字、日期和格式也是训练分布中的稳定结构：

| 上层 Pattern | Train / Valid | 文档数 | 原文中的形式 |
|---|---:|---:|---|
| `19th + century,` | 24 / 6 | 28 | `In the 19th century, ...` |
| `$...00 + million` | 23 / 8 | 27 | `$500 million`、`$100 million` |
| `July + date digits` | 29 / 8 | 35 | `Thursday, July 23, 2009` |
| `in early + 19xx` | 23 / 7 | 28 | `in the early 1900s`、`in the early 1980s` |

7,535 个真实异质 `C-8-4` 中，5,576 个至少包含一个数字，1,880 个包含字母且不含数字，79 个只包含标点或空白。高频头部包含大量编号、日期和列表结构；低频区域包含大量短语、句法和话语关系。

## 证据链

本实验在操作化定义下给出以下证据：

```text
真实 token 序列
→ 4,818 个跨文档稳定的低层 C-4-2 节点
→ 7,535 个跨文档稳定的异质 C-8-4 上层节点
→ 相同一级边际下，Null 只有 242 个
→ 934 个低层节点进入至少两个不同上层节点
→ 自动发现的低频节点具有清楚、可读的语言意义
```

因此，DCLM 自然语言数据包含显著的两层递归组合结构：反复出现的低层关系形成反复出现的上层关系，低层关系同时被多个上层关系调用。这一结论完全来自 token 联合分布。

## 结论边界

当前实验证明了 `C-4-2 → C-8-4` 这一层递归组合。以下问题留给下一阶段：

- `C-8-4` 能否继续形成稳定的 `C-16`、`C-32`；
- `token + C-4-2`、`C-4-2 + token` 等非平衡树能发现多少三叶、五叶结构；
- `although ... but`、`because ... so` 等长程关系在更大 window 中如何出现；
- 小 fan-in 是否由数据选择，而非二叉定义直接规定；
- 结果对 tokenizer、网页去重和不同语料是否稳定；
- 不同频率和类型的 pattern 如何进入 LLM 的不同谱方向。

## 文件与复现

- [完整 C-4-2 结果](./results/pilot_10k/c4_patterns.csv)
- [完整真实 C-8-4 结果](./results/pilot_10k/c8_real_patterns.csv)
- [C-8-4 Null 结果](./results/pilot_10k/c8_null_patterns.csv)
- [实验汇总](./results/pilot_10k/summary.json)
- [内容与分位数分析](./results/pilot_10k/analysis_summary.json)
- [实验代码](./mine_hierarchy.py)
- [详细实验参数](./experiment_design.md)

复现命令：

```bash
python3 fdong_embedding_dim/language_structure/mine_hierarchy.py \
  --data-dir /Users/bytedance/Desktop/dclm \
  --tokenizer-dir fdong/Qwen3-0.6B \
  --output-dir fdong_embedding_dim/language_structure/results/pilot_10k \
  --max-documents 10000 \
  --max-documents-per-file 128 \
  --tokenizer-batch-size 32 \
  --event-flush-documents 128
```

运行持续报告 documents、tokens、docs/s、tokens/s、事件数和 ETA。10,000-document pilot 总耗时 5 分 19 秒。
