# KV Cache Indexing and Knowledge Retrieval Notes

Date: 2026-05-09

## 1. Motivation

Long-context inference is increasingly constrained by KV cache capacity,
bandwidth, and per-token retrieval cost. In standard Transformer attention, each
new query vector scans all historical keys:

```text
score_i = q · k_i
attention = softmax(score)
output = sum_i attention_i * v_i
```

This is functionally similar to searching a huge corpus by scanning every
document for every query. Modern search engines, databases, vector databases,
and knowledge graphs avoid this pattern. They invest in indexing, hierarchical
storage, candidate recall, reranking, caching, and query understanding.

The core question is:

```text
What can large-scale knowledge retrieval systems teach us about K-cache lookup?
```

The working hypothesis is:

```text
K-cache retrieval is a high-frequency, low-latency, dynamically updated vector
indexing problem.
```

Attention should not be viewed only as dense matrix multiplication. It can also
be viewed as a retrieval process:

```text
current query q -> locate relevant historical K/V entries -> read V values
```

Therefore, the KV cache can be redesigned as an indexed memory system rather
than a flat sequence of token-level activations.

## 2. How Large-Scale Search Systems Retrieve Knowledge

A modern web search system does not scan the entire web at query time. It uses a
multi-stage pipeline.

### 2.1 Offline Construction

Before serving queries, a search system usually performs:

```text
crawl documents
-> clean and deduplicate
-> segment into pages/passages/blocks
-> extract keywords, entities, metadata, links, and quality signals
-> build inverted indexes
-> build vector indexes
-> build graph/entity indexes
-> build caches for hot queries and hot documents
```

The important idea is that retrieval cost is moved away from query time as much
as possible. Query time should use compact indexes, not raw full-corpus scans.

### 2.2 Query-Time Retrieval

At query time, the system typically does:

```text
query understanding
-> candidate recall
-> coarse ranking
-> expensive reranking
-> result fusion
```

The query is first parsed and interpreted. The system may identify keywords,
entities, relations, intent, time sensitivity, location, or task type.

Then multiple retrieval backends produce candidate documents:

```text
inverted index recall
vector nearest-neighbor recall
knowledge graph traversal
freshness/news index recall
personalization or cache recall
```

Only a small candidate set is sent into more expensive ranking models. This is a
central pattern:

```text
cheap broad recall -> medium-cost filtering -> expensive exact ranking
```

## 3. How Knowledge Graphs Are Built and Queried

A knowledge graph stores knowledge as entities, relations, and attributes.

```text
entity: Obama
entity: USA
relation: born_in
relation: president_of
attribute: date, location, type, confidence
```

The graph is often represented as triples:

```text
(subject, relation, object)
```

For example:

```text
(Einstein, born_in, Ulm)
(Qwen3, is_a, language_model)
(function_A, calls, function_B)
```

### 3.1 Construction Pipeline

A typical construction pipeline is:

```text
raw text / databases / logs / code
-> entity extraction
-> relation extraction
-> entity disambiguation
-> synonym and alias merging
-> confidence scoring
-> graph storage
-> index construction
```

Entity extraction identifies objects such as people, organizations, variables,
functions, files, products, dates, or concepts.

Relation extraction identifies edges such as:

```text
is_a
part_of
located_in
authored_by
calls
depends_on
mentions
causes
contradicts
supports
```

Entity disambiguation is necessary because one surface form can refer to many
objects, and one object can have many names.

### 3.2 Query Entry Points

The entry point can be:

```text
keyword
entity
natural language question
embedding vector
structured query
user context
```

For a natural language query such as:

```text
Where was Einstein born?
```

The system may convert it into:

```text
entity = Einstein
relation = born_in
lookup: (Einstein, born_in, ?)
```

For a complex query, the system may combine graph traversal with text retrieval:

```text
query -> identify entity/relation
-> retrieve graph neighborhood
-> retrieve supporting documents
-> rerank and synthesize answer
```

The key lesson is that graph retrieval often starts from a coarse semantic
object, not from individual words.

## 4. Mapping Search Concepts to KV Cache

The analogy between search systems and KV cache is direct:

| Search System | Transformer KV Cache |
| --- | --- |
| User query | Current attention query `q` |
| Document / passage | Historical token or token block |
| Document embedding | Key vector `k_i` |
| Document payload | Value vector `v_i` |
| Inverted/vector index | K-cache index |
| Candidate recall | Sparse token/block selection |
| Reranking | Exact attention over candidates |
| Hot cache | Recent/high-utility KV in HBM |
| Cold storage | Compressed/offloaded KV |
| Knowledge graph node | Semantic memory node |

Dense attention corresponds to full-corpus scanning:

```text
q attends to all k_i
```

Indexed attention should instead look like:

```text
q -> retrieve candidate blocks/tokens
-> exact q · k only on candidates
-> softmax over selected memory
-> read selected V
```

This changes the objective from:

```text
make every historical token equally addressable at every step
```

to:

```text
make relevant historical memory quickly locatable
```

## 5. What Can Be Borrowed for K-Cache Lookup

### 5.1 Inverted Index Analogy: Key Codes to Token Positions

In search engines:

```text
keyword -> document list
```

In K-cache:

```text
key code / cluster id -> token positions
```

Each historical key can be assigned to one or more learned buckets:

```text
k_i -> bucket_id
```

At query time:

```text
q -> likely bucket ids
-> gather token positions from those buckets
-> exact attention on gathered tokens
```

This resembles IVF, product quantization, locality-sensitive hashing, or learned
vector routing. It avoids scanning all tokens, but it introduces dynamic index
maintenance and GPU gather challenges.

### 5.2 Block-Level Indexing

Token-level indexing may be too expensive and too irregular for GPUs. A more
practical first step is block-level indexing:

```text
block_0 = tokens 0..15
block_1 = tokens 16..31
block_2 = tokens 32..47
```

Each block stores compact summaries:

```text
mean_key
max-like_key
learned_summary_key
multi_prototype_keys
recency / frequency / utility metadata
```

Query flow:

```text
q -> score block summaries
-> select top-k blocks
-> run exact attention over original K/V inside selected blocks
```

This is close to the current `qwen3_kcache_avg_topk` experiment, where averaged
K-cache blocks are used to select top blocks during decode. The likely weakness
is that a single important token can be hidden by averaging. A stronger design is
to store multiple summaries per block.

### 5.3 Multi-Prototype Block Summaries

A single average vector is often not enough. A block may contain several
unrelated semantic objects:

```text
token block = [function name, argument list, comment, unrelated variable]
```

A better block index can store:

```text
prototype_1: entity-like tokens
prototype_2: syntax/control tokens
prototype_3: rare or high-norm tokens
prototype_4: learned residual summary
```

Then query scoring becomes:

```text
block_score = max_j(q · prototype_j)
```

or:

```text
block_score = learned_router(q, prototypes)
```

This reduces the chance that a needle token is washed out by average pooling.

### 5.4 Hierarchical Retrieval

Search systems use layers of indexes. K-cache can do the same:

```text
Level 0: recent exact KV
Level 1: block summaries
Level 2: segment/topic/entity summaries
Level 3: global semantic memory
```

The retrieval path can be:

```text
q -> coarse semantic level
-> candidate segments
-> candidate blocks
-> exact token-level K/V
```

This matches the broader hierarchical memory idea:

```text
tokens -> blocks -> spans -> entities/topics -> global state
```

The deepest layers may not need token-resolution retrieval first. They can
retrieve semantic objects and only drill down to tokens when necessary.

### 5.5 Knowledge Graph Inspired KV Memory

During prefill, the model can build lightweight semantic nodes:

```text
entity node
topic node
function node
code symbol node
paragraph node
argument/claim node
event node
```

Each node stores:

```text
node embedding
node type
source token span pointers
related node pointers
confidence / utility score
```

Query flow:

```text
q -> retrieve relevant semantic nodes
-> expand node pointers to token/block spans
-> exact attention on original K/V
```

This is a graph-assisted attention mechanism. It does not require a symbolic,
human-readable graph at first. It can be a latent graph where nodes and edges are
learned but still provide retrieval structure.

### 5.6 Query Understanding for Attention Heads

Search systems classify query intent. Attention can also classify query type.

For each layer/head query, a small router can estimate:

```text
local continuation query
entity/reference query
long-range evidence query
syntax query
code symbol query
global topic query
```

Different query types should use different retrieval policies:

```text
local continuation -> recent window mostly enough
entity/reference -> entity nodes + mentions
code symbol -> definition/call-site blocks
global topic -> segment/topic summaries
rare token lookup -> high-salience token index
```

This suggests that K-cache indexing should be layer-aware and head-aware. Some
heads may remain dense/local, while others become explicit retrieval heads.

## 6. A Concrete Candidate Architecture

A practical architecture can use a three-tier memory layout.

### 6.1 Tier 1: Recent Exact KV

Keep the most recent tokens fully available:

```text
recent window = last 512 / 1024 / 2048 tokens
```

These tokens are always attended exactly. This protects local fluency, syntax,
and immediate continuation.

### 6.2 Tier 2: Indexed Block KV

Older tokens are divided into blocks:

```text
block size = 16 / 32 / 64 tokens
```

Each block stores:

```text
original K/V, possibly compressed or offloaded
several summary keys
block metadata
```

At decode time:

```text
q -> score summaries
-> select top-k blocks
-> gather original K/V for selected blocks
-> exact attention
```

### 6.3 Tier 3: Semantic Memory Nodes

Long contexts also maintain a smaller semantic memory:

```text
node = learned summary over a span/entity/topic/function
node -> token/block pointers
```

The semantic memory acts like a latent knowledge graph or table of contents.

Query flow:

```text
q -> semantic node retrieval
-> expand to blocks/tokens
-> exact attention
```

The final candidate set is a union:

```text
candidates =
  recent tokens
  ∪ top block-index tokens
  ∪ semantic-node-expanded tokens
  ∪ mandatory anchor tokens
```

## 7. Training Implications

Dense-trained models may not naturally produce index-friendly keys. Search
systems are designed around retrieval from the beginning. Similarly, models may
need training pressure to make KV memory indexable.

Possible training objectives:

```text
sparse attention training
router imitation from dense attention
oracle top-block distillation
contrastive query-block alignment
entity/span retrieval auxiliary loss
long-context evidence localization reward
fallback penalty for excessive dense retrieval
```

One useful recipe:

```text
1. Train/evaluate dense baseline.
2. Compute oracle sparse block selection from dense attention.
3. Train a router to imitate oracle block selection.
4. Add exact attention only over selected candidates.
5. Gradually reduce candidate budget.
6. Add long-context tasks that require far evidence.
```

This connects directly to the existing chunk-router and K-cache top-k
experiments.

## 8. System Implications

Indexing should be designed with GPU execution in mind. Token-level random
access may destroy efficiency, even if it reduces theoretical attention FLOPs.

Practical constraints:

```text
block-level gather is easier than token-level gather
fixed-size candidate budgets are easier to batch
recent window should remain contiguous
summary scoring should be cheap and dense
fallback full attention should remain available for uncertain cases
```

A useful serving layout:

```text
HBM:
  recent exact KV
  hot block KV
  all block summaries
  semantic node summaries

CPU DRAM:
  warm original KV blocks
  compressed older KV

Disk / remote:
  very cold replayable or heavily compressed memory
```

The query first consults summaries in HBM. Only selected older blocks need to be
loaded or expanded.

## 9. Major Risks

### 9.1 Average Summaries Can Hide Needles

If one important token is surrounded by unrelated tokens, average K may not
represent it. Multi-prototype summaries or learned routers are likely necessary.

### 9.2 K Vectors Are Not Stable Document Embeddings

Search embeddings are trained to be stable retrieval vectors. Transformer K
vectors are internal activations and may differ across layers, heads, positions,
and contexts. Indexing may need to be layer/head-specific.

### 9.3 Softmax Distribution Is Not Top-1 Retrieval

Attention often uses a distribution over many positions. Hard retrieval can
distort the distribution. Candidate recall must be high enough, and exact
attention should still happen after candidate selection.

### 9.4 GPU Kernels Matter

Sparse retrieval that relies on irregular gathers may be slower than dense
attention for moderate sequence lengths. The algorithm must be paired with
block-structured kernels.

### 9.5 Training and Inference Mismatch

If training uses dense attention but inference uses indexed sparse attention,
the model may not learn to store information in index-friendly forms. Sparse or
router-aware training should be considered.

## 10. Near-Term Experiment Ideas

### Experiment A: Multi-Summary Block Top-K

Extend average block selection:

```text
mean K summary
learned weighted K summary
top-n high-norm token summaries
2-4 learned prototypes per block
```

Compare against single-average top-k.

### Experiment B: Oracle Recall Analysis

Measure how often block summaries recover the dense-attention top tokens:

```text
recall@blocks
attention mass retained
PPL delta
needle-token survival rate
layer/head sensitivity
```

### Experiment C: Router Distillation

Use dense attention to produce oracle block labels:

```text
label block positive if it contains high attention mass
```

Train a lightweight router:

```text
router(q, block_summary) -> block score
```

Then run exact attention only on selected blocks.

### Experiment D: Semantic Node Pointers

For long documents or code, build span-level nodes:

```text
paragraph summaries
function summaries
entity mention clusters
```

Use query-to-node retrieval to propose blocks before token-level attention.

### Experiment E: Dynamic Candidate Budget

Use confidence to choose how much memory to retrieve:

```text
if top block scores are sharp:
    keep fewer blocks
else:
    keep more blocks or fallback dense
```

This mirrors search systems that spend more compute on ambiguous queries.

## 11. Summary

Modern knowledge retrieval systems show that large memory should not be scanned
flatly at query time. They use indexing, hierarchy, semantic objects, hot/cold
storage, multi-stage ranking, and fallback paths.

For KV cache, the analogous goal is:

```text
turn flat K-cache scanning into indexed memory retrieval
```

The most practical first direction is:

```text
block-level multi-prototype K index
-> learned query/block router
-> exact attention over selected original K/V
-> recent-window and fallback protection
```

The more ambitious direction is:

```text
latent knowledge-graph-like semantic memory
-> query retrieves semantic nodes
-> nodes expand to token/block evidence
-> exact attention recovers details
```

This frames KV cache not as passive stored activations, but as an actively
indexed memory system for long-context reasoning.
