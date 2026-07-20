# 深入浅出 llama.cpp（四）：手动操纵 KV Cache 实现增量对话

## 零、前言

在上一篇文章中，我们跑通了 `simple.cpp`。但细心的读者会发现，那个例子每轮都会重置。

如果要实现像 ChatGPT 那样“有记忆”的连续对话，核心就在于对 **KV Cache** 的管理。

为了深入理解，不如脱离高层封装，直接通过手动操作 `llama_batch` 结构体，来看看 `llama.cpp` 是如何通过 `n_past` 变量实现增量推理的。

## 一、核心概念：`llama_batch` 的结构

在 `llama.h` 中，`llama_batch` 是推理时的数据载体。它并不是一个简单的数组，而是一组指针：

```cpp
struct llama_batch {
    int32_t n_tokens;       // 本次处理的 token 数量
    llama_token * token;    // Token ID 数组
    float * embd;           // 嵌入向量（通常为 NULL）
    llama_pos * pos;        // 位置编码数组（决定了它在 KV Cache 的哪个位置）
    int32_t * n_seq_id;     // 每个 token 属于多少个序列
    llama_seq_id ** seq_id; // 序列 ID 数组
    int8_t * logits;        // 是否需要计算该 token 的 logits（概率分布）
};
```

当设置 `pos[i] = n_past` 时，实际上是在告诉模型：“把这个 Token 计算出的 Key 和 Value 存到 KV Cache 的第 `n_past` 个格子里”。

## 二、代码实现：一个简易的对话 REPL

我们的目标是实现一个 REPL（读取-求值-打印循环），模型能够记住之前聊过的所有内容。

### 1. 初始化 Batch
为了教学方便，我们初始化一个容量为 1 的 Batch，即每次只处理一个 Token。

```cpp
// 分配 1 个 Token 的空间：n_tokens_alloc=1, n_embd=0, n_seq_max=1
llama_batch batch = llama_batch_init(1, 0, 1);
int n_past = 0; // 核心：记录 KV Cache 中已经存了多少东西
```

### 2. 处理 Prompt：逐个“喂”入

把用户的输入分词，然后逐个送入模型。**关键在于 `n_past` 的递增**。

```cpp
for (size_t i = 0; i < tokens.size(); ++i) {
    batch.n_tokens = 1;
    batch.token[0]  = tokens[i];
    batch.pos[0]    = n_past;        // 告诉模型当前 Token 的位置
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;          // 默认序列 ID 为 0
    batch.logits[0] = (i == tokens.size() - 1); // 只计算最后一个 Token 的概率

    if (llama_decode(ctx, batch) != 0) { ...报错... }
    
    n_past++; // 每处理一个，位置指针向后移一位
}
```

### 3. 生成回答：将生成的 Token 喂回

生成过程其实是一个“自回归”过程：模型每产出一个 Token 就打印它，并立刻把它当做下一个输入喂回去，同时 `n_past` 继续增加。

```cpp
for (int i = 0; i < max_gen; ++i) {
    // 采样得到一个新的 Token
    llama_token tok = llama_sampler_sample(sampler, ctx, -1);
    
    // ... 打印 Token ...

    // 将生成的 Token 喂回模型
    batch.token[0] = tok;
    batch.pos[0]   = n_past; // 使用当前位置
    batch.logits[0] = true;

    llama_decode(ctx, batch);
    n_past++; // 这里的 n_past 保证了新生成的词接在之前内容的后面
}
```

> 注：如果你在运行此代码时发现回复断断续续，可以尝试将 `n_predict` 调大。同时注意 `n_ctx` 的限制，如果 `n_past` 超过了 `n_ctx`，模型会发生“溢出”，这时候就需要用到更高级的 KV Cache 循环管理技巧了。

## 三、深度解析：为什么这样能行？

### 1. 为什么不需要重新输入历史记录？

在第二轮对话开始时，`n_past` 并没有清零。

- **物理层面**：`llama_context` 内部维护了一块内存（KV Cache），里面存储了所有 `pos < n_past` 的 Token 的计算结果。
- **计算层面**：在 `llama_decode` 时，Attention 机制会自动将当前的 `batch.token` 与 KV Cache 中存储的旧向量进行计算。

### 2. BOS (Begin of Sentence) 的处理

代码中有一个细节：

```cpp
auto tokens = tokenize(vocab, prompt, n_past == 0);
```

只有在 `n_past == 0`（即第一次启动）时才添加 BOS 符号。如果每轮都加 BOS，会导致模型认为每句话都是新文章的开头，破坏上下文的连贯性。

### 3. 关于 CPU 推理的性能建议

在 `llama_context_params` 中设置 `n_threads`：

```cpp
cparams.n_threads = 8; // 根据你的 CPU 核心数设置
```

对于 CPU 二次开发来说，合理的线程数能显著降低 `llama_decode` 的延迟。

> 注：如果你用过 Ollama，就会惊讶于为什么在这份代码上 CPU 推理的速度奇慢。
> 这主要是因为这份代码是一个个 Token 喂入模型的，实际上 **批量化** 才是高效的办法。

---

## 四、结语

通过直接操作 `llama_batch` 和手动控制 `n_past`，我们实现了一个有状态的对话引擎。

**本章精华：**

1. **KV Cache 是自动追加的**：只要你不手动清理，模型就会一直记住 `pos` 对应的 Token。
2. **`n_past` 是逻辑时钟**：它同步了 Token 的位置信息和缓存的索引。
3. **单步 Decode vs 批量 Decode**：示例中为了清晰使用了单步（一次 1 个），在实际工程中，Prompt 部分应该一次性放入 Batch 处理（即 `batch.n_tokens = tokens.size()`），这样能充分利用 CPU 的并行计算能力。这个内容我将会在未来系统地补充。

虽然我们理解了底层原理，但你也发现了：手动管理 `batch` 的指针、处理 Token 到文字的转换、以及配置复杂的采样逻辑非常繁琐且容易出错。

在工业级开发中，我们没必要反复造轮子。下一篇我们将引入 `llama.cpp` 官方的 **`common` 工具库**，看看如何用“现代 C++”的方式极简地重构我们的对话机器人，并提升推理效率。