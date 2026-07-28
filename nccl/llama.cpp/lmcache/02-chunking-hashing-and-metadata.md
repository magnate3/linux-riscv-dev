---
tags:
  - AI Infra
  - LMCache
  - KV Cache
  - 哈希
description: 解释 LMCache 如何把 token 序列切成可复用对象，以及哈希键、元数据和分布式对象标识如何共同决定命中语义。
---

# 第 2 章：Chunk 切分、哈希键与元数据管理

配套入口：

- [README.md](README.md)
- [00-index.md](00-index.md)
- [01-overall-architecture-and-core-abstractions.md](01-overall-architecture-and-core-abstractions.md)

这一章是 **LMCache 最核心的一章之一**。

因为一个缓存系统真正难的地方，从来不是“把东西存起来”，而是：**到底什么算同一个东西。**

对于 LMCache 来说，问题可以进一步翻译成：

- 一串 token 应该怎么切分成可复用单位？
- 这个单位的 key 应该怎么生成？
- 为什么相同文本有时能复用，有时绝对不能复用？
- 在分布式场景里，本地 key 和远端对象 key 又是什么关系？

如果这一层没设计好，后面的 offload、P2P、PD 全部都会失真。

## 技术背景

### 1. 为什么不能把整段 prompt 当成一个大 key

最直觉的做法是：把整段 prompt 哈希一下，命中就整段复用，miss 就整段重算。

这个做法理论上最简单，工程上却几乎不可用。

原因很直接：线上请求的复用结构通常不是“整段完全相同”，而是“前半段大量相同，后面稍微不同”。

典型例子：

- 多轮对话：前面几万 token 相同，最后一轮用户问题不一样。
- RAG：系统提示词和多数上下文块相同，只替换少量文档片段。
- 模板化请求：公共 prompt 一样，只变一些参数位。

如果整段做一个 key，那么只要尾部多一个 token，整段就 miss。这样缓存收益几乎归零。

所以 LMCache 必须支持 **部分命中**，而不是只支持整串命中。

### 2. 为什么又不能把粒度切得无限小

有人会马上想到另一个极端：那就每个 token 都单独算 key。

这也不现实。因为粒度越细：

- key 数量越多；
- 元数据数量越大；
- lookup 次数越多；
- 小对象管理成本越高；
- CPU / 网络 / 序列化开销更容易盖过真正收益。

所以 LMCache 需要的是一个折中点：

- 足够细，支持高命中率；
- 但又足够粗，让元数据和查找成本可控。

这就是 **chunk** 存在的根本原因。

### 3. 为什么 key 里必须带上“结构上下文”

KV Cache 不是普通业务缓存，它不是“这个文本对应一个 JSON 结果”这么简单。

同一段文本在下面这些条件改变时，产生的 KV 就不能安全复用：

- 模型变了；
- world size / rank 切法变了；
- KV dtype 变了；
- MLA 与非 MLA 切换了；
- 某些 request-specific tag 变了，比如 LoRA、租户隔离标签、多模态输入标识。

所以 KV 的 key 一定不是单纯的文本 hash，而是 **内容哈希 + 结构语义 + 请求维度** 的组合。

### 4. 为什么分布式场景还要再引入一层对象键

在单机里，`CacheEngineKey` 已经足够表达“这块 KV 是谁”。

但在分布式查找和远端共享里，还要解决另一个问题：

**这段 KV 在集群里究竟是哪一个 rank 的哪一份对象。**

于是 LMCache 又引入了分布式对象视角的 `ObjectKey`。它强调的是可路由、可跨 worker 标识，而不仅仅是本地缓存命中。

这两个键并存不是重复设计，而是分别服务于：

- 本地缓存编排；
- 分布式对象发现与传输。

## 技术核心（结合代码）

### 1. `TokenDatabase` 的本质：把 token 流翻译成“可复用对象序列”

主入口还是 `lmcache/v1/token_database.py`。

这里最关键的抽象是：

```python
process_tokens(
    tokens=None,
    hashes=None,
    offsets=None,
    mask=None,
    make_key=True,
    request_configs=None,
)
```

它的输出不是单个 key，而是一串：

```python
(start_idx, end_idx, key_or_hash)
```

这个接口非常值钱，因为它等价于说：

- 给我一串 token；
- 我告诉你每个可复用片段覆盖哪一段 token；
- 同时给出它的 hash 或完整 key。

LMCache 后面几乎所有核心路径都依赖这个抽象：

- `store()` 用它决定要保存哪些 chunk；
- `lookup()` 用它决定查哪些 chunk；
- `retrieve()` 用它决定从哪里开始能连续命中。

### 2. `ChunkedTokenDatabase`：默认主路径是固定 chunk 切分

默认实现是 `ChunkedTokenDatabase`。

它的工作逻辑可以压成三步：

1. 把 token 序列按 `chunk_size` 切块。
2. 对每块做前缀递推 hash。
3. 把 hash 包装成 `CacheEngineKey`。

源码主链路很清楚：

- `_chunk_tokens(...)` 负责固定长度切块；
- `_prefix_hash(...)` 负责递推前缀哈希；
- `process_tokens(...)` 负责产出 `(start, end, key)`。

等价伪代码如下：

```python
def process_tokens(tokens, mask=None):
    num_falses = count_prefix_false(mask)
    prefix_hash = NONE_HASH

    for chunk_id, token_chunk in enumerate(chunk(tokens, chunk_size)):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, total_len)

        prefix_hash = hash((prefix_hash, token_chunk, extra_keys))

        if start < num_falses:
            continue

        yield start, end, make_cache_engine_key(prefix_hash, request_configs)
```

这里有几个关键点必须理解。

#### 第一，chunk 是固定粒度，不是动态语义片段

默认路径下，LMCache 不是按句子、段落、语义边界切，而是按固定 `chunk_size` 切。这是非常典型的系统工程选择。

好处是：

- chunk 边界稳定；
- lookup 逻辑简单；
- offset 可预测；
- 与底层 page/block 管理更容易对齐。

坏处是：

- chunk 边界可能和真实语义边界无关；
- 如果文本差异刚好打在 chunk 中间，可能扩大 miss 范围。

LMCache 的选择说明它优先优化的是 **稳定性和工程可控性**，而不是理论上的最优 substring 复用。

#### 第二，哈希不是对每块独立算，而是前缀递推算

`_prefix_hash(...)` 的逻辑是：

```python
prefix_hash = NONE_HASH
for token_chunk in token_chunks:
    prefix_hash = hash((prefix_hash, token_chunk, extra_keys))
    yield prefix_hash
```

这意味着 chunk 2 的 hash 不只是 chunk 2 自身，还包含前面 chunk 1 的前缀信息。

这个设计的意义是：

- 它天然编码了“连续前缀”的语义；
- 可以保证 `[A][B]` 的第二块和 `[X][B]` 的第二块不会错误共享；
- 连续命中判定天然更安全。

这是 LMCache 做“前缀连续命中”而不是“任意散点命中”的关键基础。

#### 第三，`mask` 不是普通布尔过滤，而是前缀裁剪语义

源码里明确要求：`mask` 必须形如 `FFFFFTTTTT`，也就是 False 只能出现在前缀。

这个约束不是拍脑袋写的，而是因为 LMCache 的 lookup / retrieve 语义是：

- 前面这段已经由本地或其他系统处理过；
- 现在只需要从某个 chunk 边界往后继续匹配/加载。

如果允许中间出现断裂布尔位，连续 chunk 命中语义就会被破坏，内部很多 fast path 都会复杂化。

所以这里本质上是在强约束：**LMCache 当前优化的是连续 prefix reuse，不是通用稀疏子串复用。**

### 3. `save_unfull_chunk` 暴露了一个很典型的工程 trade-off

在 `_chunk_tokens(...)` 里，末尾不足一个完整 chunk 的部分是否保存，是可配置的：

```python
end = len(tokens) if save_unfull_chunk else (len(tokens) - len(tokens) % chunk_size)
```

这看起来像个小配置，实际非常体现系统取舍。

保存不完整 chunk 的好处：

- 更激进地利用可复用片段；
- 某些短请求或最后一段 prompt 也能吃到缓存收益。

代价：

- 对象粒度更碎；
- 尾块复用稳定性更差；
- 对 chunk 边界对齐友好的场景不一定划算。

面试里如果被问到“为什么很多系统偏爱完整块复用”，你就可以从这里展开：**块边界稳定会显著降低元数据复杂度、分配抖动和跨层交互成本。**

### 4. `SegmentTokenDatabase`：LMCache 已经为更高层的语义切分留了入口

除了固定 chunk，LMCache 还提供了 `SegmentTokenDatabase`。

它的核心思路是用特殊分隔符把 token 流拆成段：

- 通过 tokenizer 把 `blend_special_str` 编码成分隔 token；
- `_fast_split_by_subtensor(...)` 在 token 序列中滑窗查找分隔符；
- 每个 segment 单独 hash。

这条路径主要服务于 blending / 特殊段落复用场景。它说明 LMCache 并没有把“复用单位”写死为固定 chunk，而是在架构上给更高层的复用策略留了扩展点。

但你也要注意它的注释：未来可能要支持更快的 substring match。换句话说，当前实现仍然更偏工程可落地版本，而不是最强语义匹配引擎。

### 5. `CacheEngineKey`：本地缓存世界里的主键长什么样

`lmcache/utils.py` 里定义了 `CacheEngineKey`。

字段是：

```python
class CacheEngineKey:
    model_name
    world_size
    worker_id
    chunk_hash
    dtype
    request_configs
```

其中真正决定身份的核心部分是：

- `model_name`
- `world_size`
- `worker_id`
- `chunk_hash`
- `dtype`
- `tags`（由 request-specific config 派生）

`__hash__()` 和 `__eq__()` 都明确把这些字段纳入比较。这说明 LMCache 的设计原则非常保守：

**只要结构语义有变化，就宁愿 miss，也不要误命中。**

这是一个很典型的基础设施风格：先保证正确性，再追命中率。

### 6. `request_configs` 为什么会进 key 空间

很多人第一次看 `request_configs` 会觉得像“附加参数”。实际上它直接参与缓存隔离。

`CacheEngineKey.__post_init__()` 会把 `request_configs` 里以 `lmcache.tag.` 开头的键提取成 `tags`，并放进 key 的 hash 和相等性比较里。

这意味着同样一段 token、同样一个模型，只要 tag 不同，也会生成不同 key。

这是非常合理的。因为很多线上场景里，缓存共享不能只看文本，还要看上下文维度，比如：

- LoRA / adapter 身份；
- 租户隔离；
- 某类业务标签；
- 某些 request-level 复用策略。

如果这些因素不进入 key 空间，最坏情况就是把本不应该共享的 KV 错误复用掉。

### 7. 多模态占位和 request-specific 语义也会影响哈希稳定性

在 vLLM 集成层，`lmcache/integration/vllm/utils.py` 和 `vllm_v1_adapter.py` 还做了一件很重要的事：

- 提取 `sampling_params.extra_args` 里以 `lmcache.` 开头的配置；
- 对多模态 placeholder 用稳定哈希值覆盖 token id。

相关主线是：

- `extract_request_configs(...)`
- `extract_mm_features(...)`
- `apply_mm_hashes_to_token_ids(...)`

这说明 LMCache 很清楚一个事实：**“表面上相同的 token 序列” 不一定真表示同一个请求语义。**

多模态 placeholder 如果不转成稳定标识，或者 request-level tag 不进入 key，最终就会出现命中不一致甚至错误共享。

所以第 2 章你一定要建立一个意识：

**LMCache 的 key 从来不是单纯文本 hash，而是“可复用语义对象”的身份表达。**

### 8. `LMCacheMetadata`：保证 key 对应的 KV 结构是可解释的

单独有 key 还不够，因为 key 只告诉你“这是哪一块”，但没告诉你“这块 KV 长什么样”。

`lmcache/v1/metadata.py` 里的 `LMCacheMetadata` 提供了这层结构真相：

- `model_name`
- `world_size`
- `local_world_size`
- `worker_id`
- `local_worker_id`
- `kv_dtype`
- `kv_shape`
- `use_mla`
- `chunk_size`
- `kv_layer_groups_manager`

尤其关键的是：

- `get_shapes(num_tokens)`
- `get_dtypes()`

这两个方法直接把“某个 chunk 长度下，该分配怎样形状、怎样 dtype 的 KV 容器”暴露给了后面的分配和搬运层。

换句话说：

- `TokenDatabase` 决定 **对象边界与身份**；
- `LMCacheMetadata` 决定 **对象结构与物理形态**。

两者缺一不可。

### 9. `LayerCacheEngineKey`：为什么 layer 维度也会单独编码

`CacheEngineKey.split_layers(num_layers)` 会生成一组 `LayerCacheEngineKey`。

这说明 LMCache 不是强制把整块 KV 作为一个不可拆对象处理。对于 layerwise store / layerwise retrieve 场景，它会把：

- 同一个 chunk hash
- 同一组模型/rank/tag 语义

再细分到具体 `layer_id`。

这样做的好处是：

- 支持逐层加载、逐层保存；
- 更好地和 layerwise pipeline 对齐；
- 为后续更细颗粒度的回迁和 overlap 留出空间。

代价当然也存在：对象数进一步膨胀，元数据管理更复杂。所以这类模式更像是为了强性能路径而付出的复杂度预算。

### 10. `ObjectKey`：分布式对象世界里的主键

到了 `lmcache/v1/distributed/api.py`，LMCache 又定义了 `ObjectKey`：

```python
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
```

与本地 `CacheEngineKey` 相比，它更像一个“分布式存储对象 ID”。

这里最值得你看的是 `ComputeKVRank(...)`。它把：

- `world_size`
- `global_rank`
- `local_world_size`
- `local_rank`

编码成一个整数位图式标识。

这背后的含义是：在分布式系统里，**光知道 chunk hash 不够，还必须知道这块 KV 对应哪种并行切片语义。**

因为不同 TP/PP/rank 切法下，同一段文本的 KV 切片并不等价。

所以 `ObjectKey` 解决的其实不是“查 hash”，而是：

**在集群里，唯一标识一份可传输、可落地的 KV 对象。**

### 11. 为什么本地 `CacheEngineKey` 和远端 `ObjectKey` 要并存

这是面试里非常容易问出深度的一点。

答案是：它们服务的是两个不同语义层次。

`CacheEngineKey` 关心的是本地缓存编排：

- 这段 token 对应什么 chunk；
- 本地 backend 里有没有；
- 这个对象的 dtype / tags / layer 是什么。

`ObjectKey` 关心的是分布式对象发现：

- 这个 chunk hash 对应哪个 rank 切片；
- 远端系统该去哪里找它；
- 在分布式存储管理器里它的身份是什么。

如果强行只保留一种 key，往往要么本地逻辑变脏，要么远端语义表达不足。

### 12. 这套设计的本质：把“复用语义”和“存储语义”解耦

第 2 章最重要的系统直觉就是这句话。

LMCache 没有把“token 文本长什么样”“本地缓存怎么命中”“远端对象怎么标识”混成一个大杂烩，而是拆成了三件事：

1. `TokenDatabase`：把 token 流切成可复用对象。
2. `CacheEngineKey`：给本地缓存系统一个稳定身份。
3. `ObjectKey`：给分布式对象系统一个可路由身份。

这套拆法让后续的：

- local/offload backend
- P2P lookup
- PD 传输
- layerwise load

都可以在不推翻“命中语义”的前提下继续演进。

## 面试可能问到的问题

### 问题 1：为什么 chunk size 不是越大越好，也不是越小越好？

**满分回答思路：**

chunk size 本质上是在平衡 **命中灵活性** 和 **管理成本**。

chunk 太大时：

- 一点点尾部差异就会导致整大块 miss；
- 局部复用能力差；
- 多轮对话和 RAG 这种“高重叠、局部变化”场景收益下降。

chunk 太小时：

- key 数量、lookup 次数、元数据量都暴涨；
- 小对象分配和管理开销变重；
- 网络传输和序列化更容易被放大；
- 连续命中判断会更碎，系统抖动更明显。

所以合适的 chunk size 一般要结合：

- 典型 prompt 重复模式；
- page/block 粒度；
- CPU 和网络开销；
- backend 延迟模型。

如果再加一句工程判断，可以说：**LMCache 默认固定 chunk，本质上是在给命中率换一个可控的系统复杂度上界。**

### 问题 2：为什么 LMCache 用前缀递推 hash，而不是对每个 chunk 独立哈希？

**满分回答思路：**

因为 LMCache 想表达的不是“某块文本出现过”，而是“某块文本出现在这个前缀上下文之后”。

如果每个 chunk 独立哈希，那么 `[A][B]` 里的 `B` 和 `[X][B]` 里的 `B` 会得到同一个 chunk key，可能导致错误复用。对于 KV Cache 来说这是不可接受的，因为 attention 的结果依赖完整历史上下文。

前缀递推 hash 把前缀信息压进当前 chunk 的 hash，可以保证：

- 连续 chunk 命中语义正确；
- 前缀变化会传递到后续 chunk；
- lookup 的“最长连续前缀命中”有可靠基础。

代价是不能天然支持任意子串级别的自由复用，但这正是 LMCache 当前设计有意识做的取舍：优先支持 **prefix reuse**，而不是做一个复杂的 substring cache。

### 问题 3：为什么相同 prompt 的 KV 有时仍然不能复用？

**满分回答思路：**

因为 KV 的身份不仅由 prompt 决定，还由生成它的结构上下文决定。

至少要看这些因素：

- 模型名是否相同；
- KV dtype 是否相同；
- TP/PP/world size/rank 语义是否一致；
- 是否是 MLA；
- request-specific tag 是否一致；
- 多模态 placeholder 或 LoRA 等附加语义是否一致。

LMCache 把这些因素显式编码进 `CacheEngineKey`、`LMCacheMetadata`、`ObjectKey` 里，就是为了避免错复用。对基础设施来说，**一次错误命中比一百次 miss 更危险**，因为错误命中会直接破坏模型行为，而 miss 最多只会退化成重算。

---

这一章读完，你应该已经抓住 LMCache 的复用本质：

1. **不是缓存整段 prompt，而是缓存按 chunk/segment 切开的 KV 对象。**
2. **不是只做文本 hash，而是把模型、rank、dtype、tags 一起编码进身份。**
3. **不是只有一种 key，而是本地缓存 key 和分布式对象 key 各司其职。**

你发送“继续”，下一章我会写 **第 3 章：存储层、分层卸载与对象生命周期**，把 KV 从 GPU 对象变成多层存储系统这件事彻底讲透。