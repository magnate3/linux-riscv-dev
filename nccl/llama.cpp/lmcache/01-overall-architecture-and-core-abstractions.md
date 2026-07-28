---
tags:
  - AI Infra
  - LMCache
  - KV Cache
  - 源码导读
description: 从系统层视角拆解 LMCache 的整体架构、核心抽象和与 vLLM 的职责边界。
---

# 第 1 章：LMCache 整体架构与核心抽象

配套入口：

- [README.md](README.md)
- [00-index.md](00-index.md)

先给一句最重要的判断：**LMCache 不是“一个 KV Cache 功能”，而是把 KV Cache 从推理引擎内部，抽成一层独立的数据系统。**

如果只把它理解成“把 KV 从 GPU 挪到 CPU”，你会低估它；如果把它理解成“分布式对象缓存 + 引擎适配层 + 控制面”的组合，你才算真正看懂这个项目。

## 技术背景

没有 LMCache 之前，绝大多数 LLM Serving 系统的 KV Cache 都有三个硬限制。

### 1. KV 只在当前实例里有意义

vLLM 这类系统本身已经把 **单实例内** 的 KV 管理做得很强，比如 paged KV、块分配、连续 batching。但这些能力主要解决的是：

- 当前进程里的显存怎么切；
- 当前 worker 上的请求怎么复用本地已经算过的 KV；
- 当前调度周期里 block 怎么分配与回收。

它没有天然解决一个更贵的问题：**同样一段上下文，如果下次请求落到另一个 worker、另一个进程、甚至另一个节点，之前算过的 prefill 结果怎么继续复用。**

这正是企业场景里最痛的地方。

- **RAG**：知识块反复出现，但并不总是严格相同前缀。
- **多轮对话**：长历史对话每轮都要重复进入 attention。
- **共享系统 prompt / 模板 prompt**：不同用户请求大段公共前缀高度重复。
- **PD 分离部署**：prefill 在 A，decode 在 B，KV 需要跨机器流转。

如果没有一个独立 KV 层，结果就是：

- **TTFT 被重复 prefill 拉高**；
- **GPU 算力浪费在重复 attention 上**；
- **请求一旦漂移到别的实例，命中率立刻塌**；
- **系统优化停留在单实例局部最优，而不是全局最优。**

### 2. KV 一旦离开 GPU，就不再是“模型内部细节”

很多人第一次看 KV Cache 会以为这只是一个张量保存问题。但当你要把它拿到 GPU 外部复用时，问题立刻升级成系统设计题：

- 这段 KV 的 **复用单位** 怎么定义？
- 它的 **键** 怎么生成，怎么保证跨进程一致？
- 存在哪一层最划算，**GPU / CPU / disk / remote** 怎么选？
- 回迁时怎么避免同步阻塞？
- 多 worker 并发命中同一段对象时，如何避免远端热点被打爆？

所以 LMCache 解决的不是单一瓶颈，而是把 “KV 复用” 这件事系统化。

### 3. 真实线上环境要求它必须可降级、可插拔、可跨引擎

LMCache 不能写成一个和 vLLM 死绑的 patch，原因很直接：

- 推理引擎会演进，内部 KV 布局也会变；
- 不同引擎有不同 page/block 结构和 forward 生命周期；
- KV 复用层如果不能单独演化，维护成本会非常高。

所以它必须做三件事：

1. **把通用缓存逻辑抽出来**。
2. **把引擎相关布局藏到适配层里**。
3. **把健康检查、服务启动、lookup/offload 这些外围组件也统一纳管**。

这就是为什么 LMCache 最终长成今天这套结构，而不是一堆散落在 connector 里的工具函数。

## 技术核心（结合代码）

这一章只做一件事：把 LMCache 的主抽象和主链路一次性讲顺。

### 1. 总图先看清：LMCache 在 Serving 栈里的位置

最上层是 serving engine，比如 vLLM。它负责请求调度、paged KV block 管理、模型执行。

LMCache 插在它旁边，承担的是外部 KV 层的职责：

- 把 GPU 上的 KV 导出成可管理对象；
- 根据 token 序列生成 cache key；
- 把对象写进某个 backend；
- 在请求到来时先做 lookup，再决定是否回迁 KV；
- 提供 P2P / remote / PD 这些跨实例能力。

可以把它粗暴理解成下面这条链：

```text
vLLM/SGLang
  -> connector / service factory
  -> LMCacheManager
  -> LMCacheEngine
      -> TokenDatabase
      -> StorageManager
      -> GPUConnectorInterface
      -> EventManager
```

这几个抽象不是为了“代码好看”而拆的，而是每一层都在切一个不同的变化维度。

### 2. `LMCacheEngine` 是数据面的中心编排器

主入口在 `lmcache/v1/cache_engine.py` 的 `LMCacheEngine`。

它的构造函数已经把项目的设计意图写得很清楚：

- `config`：策略和行为开关。
- `metadata`：当前模型、rank、KV shape、dtype 等上下文。
- `token_database`：把 token 序列翻译成可查找的 key。
- `gpu_connector`：负责 GPU tensor 和外部 memory object 之间的互转。
- `broadcast_fn` / `broadcast_object_fn`：服务于多 rank 协调。

也就是说，`LMCacheEngine` 自己并不想知道 vLLM 的页表细节，也不想知道 remote backend 的网络实现细节。它只做“编排”：

1. 根据 token 序列切 chunk 并生成 key。
2. 为每个 chunk 分配 `MemoryObj`。
3. 调 `gpu_connector` 把 KV 从 GPU 搬到 `MemoryObj`。
4. 调 `storage_manager` 写入后端。
5. 反向加载时，先 lookup，再 get，再调 `gpu_connector` 写回 GPU。

它最核心的三个公开动作就是：

- `store(...)`
- `retrieve(...)`
- `lookup(...)`

你可以把它理解成 LMCache 的“数据面 API”。

### 3. `store -> lookup -> retrieve` 是最重要的三段主链路

#### `store(...)` 在干什么

`store(...)` 不是简单 `put(key, value)`，它是一个完整流水线。

核心步骤可以概括成：

```python
for start, end, key in token_database.process_tokens(tokens, mask):
    memory_obj = storage_manager.allocate(shape, dtype)
    memory_objs.append(memory_obj)
    keys.append(key)

gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
storage_manager.batched_put(keys, memory_objs, location=store_location)
```

这里最值得你注意的不是代码表面，而是它的系统含义：

- **先 key 化，再分配对象，再搬运，再落后端。**
- KV 在 LMCache 里不是直接拿一个大 tensor 到处传，而是先变成统一的 `MemoryObj`。
- 存储后端只需要理解 `MemoryObj`，不需要理解 vLLM 的内部 paged layout。

这就是典型的“引擎布局”和“缓存对象格式”分离。

#### `lookup(...)` 在干什么

`lookup(...)` 的目标不是立刻把 KV 全部取回来，而是先回答一个调度问题：

**当前这串 token，前面连续有多少 token 的 KV 已经存在？**

实现上它会：

1. 调 `token_database.process_tokens(...)` 把 token 序列切成 chunk key。
2. 调 `storage_manager.batched_contains(...)` 批量检查这些 key 在哪些 backend 存在。
3. 返回“连续命中的最大前缀长度”。

这个设计非常重要，因为调度侧真正关心的是：

- 这次 request 最多可以少算多少 token；
- 要不要给这次请求分配加载路径；
- 需要为哪些 block 预留空间。

也就是说，**lookup 是调度决策，不是数据搬运。** 这一步拆开以后，后面做异步 load、scheduler bypass、P2P 查找才有基础。

#### `retrieve(...)` 在干什么

`retrieve(...)` 才是真正的数据回迁路径。

它做的事情是：

1. 再次按 token 生成 chunk key。
2. 从 `StorageManager` 拿回对应 `MemoryObj`。
3. 调 `gpu_connector.batched_to_gpu(...)` 把这些对象写回引擎的 GPU KV buffer。
4. 返回一个 `ret_mask`，告诉上层哪些 token 成功命中并被装载。

这就解释了 LMCache 为什么天然要拆成 `lookup` 和 `retrieve` 两步：

- **lookup** 是轻量的“判断能省多少算力”；
- **retrieve** 是重型的“真把 KV 拉回来”。

如果两步混在一起，调度器很难做预算，异步流水线也很难展开。

### 4. `TokenDatabase` 解决的是“什么叫同一段 KV”

这个抽象在 `lmcache/v1/token_database.py`。

这是 LMCache 最关键、也最容易被低估的一层。因为所有缓存系统的本质都不是“存”，而是“怎么定义命中”。

LMCache 这里把问题抽成：

- 输入是一串 token；
- 输出是一组 chunk key；
- 每个 key 对应一个可复用的 KV 对象。

接口是 `process_tokens(...)`。

`ChunkedTokenDatabase` 是默认主路径，背后逻辑大致是：

```python
prefix_hash = NONE_HASH
for chunk in chunk(tokens, chunk_size):
    chunk_hash = hash((prefix_hash, chunk, extra_keys))
    key = CacheEngineKey(model_name, world_size, worker_id, chunk_hash, kv_dtype)
    yield start, end, key
    prefix_hash = chunk_hash
```

这套设计有三个工程意义：

1. **按 chunk 复用**，命中粒度比整段 prompt 更灵活。
2. **前缀相关哈希**，连续 chunk 的逻辑关系被保留下来。
3. **把模型与 rank 信息带进 key**，避免错模型、错布局复用。

所以 LMCache 并不是在缓存“prompt 文本”，而是在缓存“与模型布局强绑定的 chunk 化 KV 对象”。

### 5. `LMCacheMetadata` 解决的是“这个 KV 到底属于谁”

`lmcache/v1/metadata.py` 里的 `LMCacheMetadata` 很像配置对象，但它实际上承担了非常重要的契约作用。

它描述的是当前 KV 的结构上下文：

- `model_name`
- `world_size`
- `worker_id`
- `local_world_size`
- `kv_dtype`
- `kv_shape`
- `chunk_size`
- `use_mla`
- `engine_id`

这东西为什么重要？因为 KV Cache 不是纯数据，它隐含依赖：

- 当前模型层数；
- 每层 KV head 数和 head size；
- 是否是 MLA；
- 当前 TP/分布式拓扑。

`LMCacheMetadata.get_shapes()` 和 `get_dtypes()` 进一步把这些结构信息喂给内存分配和数据搬运层。也就是说，**metadata 是数据面各层共享的“结构真相来源”。**

没有这层，`StorageManager` 和 `GPUConnector` 根本无法在不理解完整引擎细节的前提下正确分配和搬运 KV。

### 6. `GPUConnectorInterface` 解决的是“怎么碰 GPU，但不把引擎耦死”

第 1 章先抓边界，不深入实现细节。

你只要先记住：LMCache 绝不让 `StorageManager` 直接理解 vLLM 的 paged KV buffer，也不让 `LMCacheEngine` 直接手写各种引擎 tensor layout 逻辑。

所以它把 GPU 相关操作独立抽象到 `lmcache/v1/gpu_connector/`。

`LMCacheEngine` 只会调用类似：

- `batched_from_gpu(...)`
- `batched_to_gpu(...)`

在 vLLM 场景里，真正的适配发生在：

- `lmcache/integration/vllm/lmcache_connector_v1.py`
- `lmcache/integration/vllm/vllm_v1_adapter.py`

外层 `LMCacheConnectorV1Dynamic` 对接的是 vLLM 的 `KVConnectorBase_V1` 生命周期，比如：

- `register_kv_caches(...)`
- `start_load_kv(...)`
- `wait_for_layer_load(...)`
- `save_kv_layer(...)`
- `request_finished(...)`

这里的设计重点是：

- **vLLM 只看到一个 connector**；
- **LMCache 内部真正复杂的逻辑则由 `LMCacheConnectorV1Impl` + `LMCacheManager` + `LMCacheEngine` 接住。**

这就是一个典型的“北向适配层 + 内核层”设计。

### 7. `StorageManager` 解决的是“KV 到底放哪里”

第 1 章先把职责讲清楚，下一章再深入后端细节。

`StorageManager` 位于 `lmcache/v1/storage_backend/storage_manager.py`，它的职责是统一调度各种存储 backend。

在 `LMCacheEngine` 看来，它只需要这些能力：

- 分配 `MemoryObj`
- `batched_put(...)`
- `batched_contains(...)`
- `batched_get(...)`
- 回收与删除

也就是说，`LMCacheEngine` 并不关心对象最终在：

- local CPU
- disk
- remote
- p2p
- pd

它只关心这个对象能不能被放进去、能不能被找出来、能不能被拿回来。

这种分层是 LMCache 能演化出多 backend 的根本原因。

### 8. `LMCacheManager` 解决的是“组件生命周期别散掉”

如果只有 `LMCacheEngine` 一个类，项目是跑不起来的。因为真实系统还有很多外围组件：

- lookup client / server
- offload server
- runtime plugin launcher
- internal API server
- health monitor

这些东西如果直接由 connector 到处 new，会立刻变成一锅粥。

所以 `lmcache/v1/manager.py` 定义了 `LMCacheManager`，专门管生命周期：

1. 从 `service_factory` 拿 metadata。
2. 创建或获取 engine。
3. 按角色选择性创建 lookup client/server、offload server、API server 等。
4. 在 `post_init()` 中完成引擎后初始化和健康检查挂接。
5. 在关闭阶段统一 stop 和 destroy。

这层的价值不是“多封装一层”，而是把 **引擎无关的组件生命周期管理** 从具体集成里抽了出来。

### 9. `BaseServiceFactory` + `VllmServiceFactory` 解决的是“同一个内核，适配不同引擎”

这套分层是 LMCache 很成熟的一点。

`lmcache/integration/base_service_factory.py` 先定义抽象工厂：

- 怎么创建 metadata
- 怎么创建 engine
- 怎么创建 lookup client/server
- 怎么创建 offload server
- 怎么创建 health monitor

然后 vLLM 在 `lmcache/integration/vllm/vllm_service_factory.py` 里把自己的规则补上。

例如它会按角色决定组件装配：

- `scheduler`：主要创建 lookup client，必要时不创建 engine。
- `worker`：创建 engine、lookup server、offload server。
- `DP rank 0`：额外创建 API server / plugin launcher。

这一层很关键，因为它说明 LMCache 不是简单地“给 vLLM 加几个 hook”，而是已经把 **组件装配策略** 也做成了可替换接口。

### 10. `LMCacheEngineBuilder` 解决的是实例复用与统一创建

在 `cache_engine.py` 末尾，`LMCacheEngineBuilder` 负责：

- 根据 `instance_id` 获取或创建 engine；
- 创建 token database；
- 记录 config / metadata / stats logger；
- 销毁时统一关闭 engine 和观测组件。

这意味着 engine 不是随便在每个角落 `LMCacheEngine(...)` new 出来的，而是通过 builder 统一管理实例身份。

这有两个好处：

1. 避免同一个 serving engine 进程里重复初始化一套 LMCache 内核。
2. 给 shutdown、stats、测试以及多角色路径留出统一收口点。

### 11. 把整条调用链串起来看一次

如果你从 vLLM 视角回看 LMCache，整条主线可以压缩成下面这张“面试口述版时序图”：

```text
请求进入 vLLM
  -> scheduler 侧通过 lookup client 判断 LMCache 命中多少 token
  -> worker 侧通过 connector 启动 load 或 save
  -> LMCacheConnectorV1Impl 调用 LMCacheManager
  -> LMCacheManager 持有 LMCacheEngine 和外围服务
  -> LMCacheEngine
       - 用 TokenDatabase 生成 chunk key
       - 用 StorageManager 查找/存取 MemoryObj
       - 用 GPUConnector 在 GPU KV 和 MemoryObj 之间搬运
  -> 命中则少算 prefill，未命中则回退到原生计算路径
```

理解到这里，你就已经抓住了 LMCache 的根：

- **不是替代 vLLM**，而是补一层“外部 KV 复用系统”；
- **不是只做存储**，而是做“查找 + 存取 + 搬运 + 生命周期 + 服务装配”；
- **不是只服务单机**，而是为跨实例、跨节点、PD 场景提前铺好抽象。**

## 面试可能问到的问题

### 问题 1：为什么 LMCache 不直接做成 vLLM 内部的一个模块，而要抽成相对独立的一层？

**满分回答思路：**

核心不是“代码组织偏好”，而是问题边界变了。vLLM 原生更擅长解决单实例内部的 paged KV 管理、block 分配和执行调度；LMCache要解决的是 KV 离开当前 GPU 之后的复用问题，包括对象切分、key 管理、跨进程一致性、分层存储、远端共享、PD 流转。这些问题已经超出单引擎内部缓存模块的范畴，更像一个独立的数据层。

继续往下答时要点出三个工程收益：

- 第一，**跨引擎复用抽象**。把通用逻辑留在 `LMCacheEngine`、`StorageManager`、`TokenDatabase`，把引擎差异压到 connector 和 service factory。
- 第二，**组件可以独立演进**。比如 remote backend、lookup server、operator，并不需要跟随 vLLM 内核一起改。
- 第三，**更容易做降级和隔离**。LMCache 异常时可以回退到 recompute，而不是把 serving engine 主链路一起拖死。

如果面试官追问“那为什么不全部写在 connector 里”，你就说 connector 只适合承接引擎生命周期和 GPU buffer 交互，不适合背负 token 哈希、backend 编排、健康检查、lookup/offload 服务这些职责，否则会迅速演化成巨型 God object。

### 问题 2：`LMCacheEngine` 为什么要拆成 `TokenDatabase`、`StorageManager`、`GPUConnectorInterface` 三层？

**满分回答思路：**

这是按“变化原因”拆层，而不是按功能名拆层。

- `TokenDatabase` 解决的是 **命中语义**：什么 token 序列算同一个可复用对象，chunk 怎么切，key 怎么生成。
- `StorageManager` 解决的是 **对象落点**：这些对象存到哪里、怎么查、怎么取、怎么回收。
- `GPUConnectorInterface` 解决的是 **设备布局耦合**：如何把外部对象和具体引擎的 GPU KV layout 互转。

这三个维度的变化频率和耦合对象完全不同：

- 换哈希和 chunk 策略，不该影响 remote backend。
- 新增 disk/remote/p2p backend，不该影响 vLLM paged tensor 逻辑。
- vLLM 升级 page/block 布局，不该逼着你改整个缓存 key 体系。

所以这是一种典型的稳定内核设计：把真正高频变化、外部依赖强的部分隔离掉，让 `LMCacheEngine` 只做编排。

### 问题 3：如果你要支持一个新的 serving engine，最小适配面在哪里？

**满分回答思路：**

最小适配面不是重写 `LMCacheEngine`，而是补齐两层：

1. **metadata 提取和 service factory**。
   需要把新引擎的模型配置、rank 信息、KV dtype/shape 转成 `LMCacheMetadata`，并决定 scheduler/worker 各自该起哪些组件。

2. **GPU connector / northbound connector**。
   需要把新引擎内部的 KV buffer 生命周期映射到 LMCache 的 `store / lookup / retrieve` 调用点，并实现 GPU tensor 与 `MemoryObj` 的搬运。

真正不该动的是：

- `TokenDatabase`
- `StorageManager`
- 大部分 `LMCacheEngine` 主逻辑

如果一个新引擎接入时需要大改这些通用层，通常说明抽象边界设计得不够干净，或者新引擎的 KV 语义与现有假设严重冲突。

---

这一章读完，你应该能回答三个最关键的问题：

1. **LMCache 在 Serving 栈里的准确位置是什么。**
2. **它为什么一定要拆成 Engine / TokenDB / Storage / GPUConnector / Manager 这些层。**
3. **它和 vLLM 的关系是“集成”而不是“绑定”。**

你发送“继续”，下一章我会写 **第 2 章：Chunk 切分、哈希键与元数据管理**，把 LMCache 最核心的复用语义彻底讲透。