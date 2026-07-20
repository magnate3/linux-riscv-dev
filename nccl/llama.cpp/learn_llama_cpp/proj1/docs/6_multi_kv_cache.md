# 深入浅出 llama.cpp（六）：并行推理的艺术 —— 多序列管理与批量解码优化

## 零、前言

在上一章中，我们通过集成 `common` 库极大简化了代码逻辑，实现了一个可以连续对话的 REPL。但那个实现存在两个致命的性能缺陷：

1.  **逐字解码 (Serial Prefill)**：即使是输入的 Prompt，我们也是一个 Token 一个 Token 喂给模型的。这完全浪费了现代 CPU/GPU 的并行计算能力。
2.  **单点服务 (Single Stream)**：一个 `llama_context` 同时只能服务一个用户。如果想让 AI 同时处理两段不同的对话，难道要启动两个庞大的模型实例吗？

本章我们将深入 `llama.cpp` 的灵魂—— **多序列 (Multi-sequence) 管理**。我们将学习如何通过手动构建 `llama_batch` 实现 **批量预填充 (Batch Prefill)**，并让同一个模型实例同时处理多个独立的对话流。

## 一、核心原理：序列 ID (seq_id) 与并行

在 `llama.cpp` 的世界里，KV Cache 不是一个简单的线性数组，而是一个 **带有标签的存储池**。

每一个存入 KV Cache 的 Token 都绑定了两个关键属性：

- **`pos` (Position)**：它在该序列中的逻辑位置（决定了 Attention 观察多远）。
- **`seq_id` (Sequence ID)**：它属于哪一个独立的对话流。

在 `llama_decode` 进行计算时，`seq_id` 不同的 Token 在 KV Cache 中是 **相互不可见的（Masked）**。

这意味着：你可以在同一个 Batch 里混合输入来自 **不同用户** 的 Token，而模型会像拥有“分身术”一样，在内部逻辑上 **互不干扰地完成推理**。

## 二、从“逐字喂入”到“批量预填充”

在之前的代码中，我们循环调用 `common_batch_add`。现在我们要一次性把整个 Prompt 塞进 `llama_batch`。

### 1. 批量构建 Batch
```cpp
// 假设 tokens 是用户输入的 100 个分词结果
common_batch_clear(batch);

for (size_t i = 0; i < tokens.size(); ++i) {
    // 参数：batch, token_id, pos, seq_ids, 是否计算 logits
    // 我们将整个 prompt 作为一个 batch 传入
    // 只有最后一个 token 需要计算 logits (为了采样下一个词)
    bool is_last = (i == tokens.size() - 1);
    common_batch_add(batch, tokens[i], n_past + i, { 0 }, is_last);
}

// 核心优化：一次 decode 处理所有 token，这比循环调用快几十倍！
if (llama_decode(ctx, batch) != 0) { ... }
n_past += tokens.size();
```

## 三、实战：同时管理两个对话序列

我们要挑战一个更高难度的场景：**让模型同时续写两个完全不同的句子**。
- 序列 0: "The capital of France is"
- 序列 1: "Count from 1 to 5:"

### 1. 并行初始化
我们需要为每个序列维护自己的 `n_past`：
```cpp
int n_past_0 = 0;
int n_past_1 = 0;

auto tokens_0 = common_tokenize(vocab, "The capital of France is", true, true);
auto tokens_1 = common_tokenize(vocab, "Count from 1 to 5:", true, true);

common_batch_clear(batch);

// 将序列 0 放入 batch
for (size_t i = 0; i < tokens_0.size(); ++i) {
    common_batch_add(batch, tokens_0[i], n_past_0++, { 0 }, i == tokens_0.size() - 1);
}

// 将序列 1 放入同一个 batch
for (size_t i = 0; i < tokens_1.size(); ++i) {
    common_batch_add(batch, tokens_1[i], n_past_1++, { 1 }, i == tokens_1.size() - 1);
}

// 一次性处理两个序列的预填充！
llama_decode(ctx, batch);
```

### 2. 交替采样与生成
在生成阶段，每一轮我们可以从两个序列各自采样一个 Token，然后再把这两个 Token 组合成一个新的 Batch 送回去。

```cpp
// 使用负数索引进行初始采样，避开绝对索引计算
// -1 是最后一个有 logits 的 token (即 seq 1)
// -2 是倒数第二个有 logits 的 token (即 seq 0)
llama_token tok1 = llama_sampler_sample(sampler, ctx, -1);
llama_token tok0 = llama_sampler_sample(sampler, ctx, -2);

for (int step = 0; step < 15; ++step)
    {
        // 打印上一步采样的结果
        printf("\033[32m[Seq 0]\033[0m %s ", common_token_to_piece(ctx, tok0).c_str());
        printf("\033[33m[Seq 1]\033[0m %s\n", common_token_to_piece(ctx, tok1).c_str());

        // 准备下一轮 decode
        common_batch_clear(batch);
        common_batch_add(batch, tok0, n_past_0++, {0}, true); // 此时在 batch 中索引为 0
        common_batch_add(batch, tok1, n_past_0++, {1}, true); // 此时在 batch 中索引为 1

        // C. 执行下一次推理
        if (llama_decode(ctx, batch) != 0)
        {
            fprintf(stderr, "Decode failed at step %d\n", step);
            break;
        }

        // 循环中依然使用负数索引，逻辑最清晰
        tok1 = llama_sampler_sample(sampler, ctx, -1);
        tok0 = llama_sampler_sample(sampler, ctx, -2);

        if (llama_vocab_is_eog(vocab, tok0) && llama_vocab_is_eog(vocab, tok1))
            break;
    }
```

## 四、KV Cache 的清理：如何释放特定序列？

当一个对话结束时，我们需要释放它占用的内存。在新版 API 中，原来的 `llama_kv_cache_seq_rm` 已经移除，取而代之的是 `llama_memory` 系列函数：

```cpp
// 1. 获取内存管理句柄
llama_memory_t mem = llama_get_memory(ctx);

// 2. 删除序列 0 的所有缓存
// 参数：memory, seq_id, pos_start, pos_end (-1 表示全部)
llama_memory_seq_rm(mem, 0, -1, -1); 
```

这个操作非常轻量，它只是在内部逻辑上将这些缓存槽位标记为“可复用”，而不会影响序列 1 的数据。

## 五、性能对比：为什么要费这番功夫？

| 模式 | 处理 1000 Token Prompt 的速度 | 显存/内存利用率 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **逐字输入 (Chapter 4)** | 极慢 (受限于推理延迟) | 低 | 仅演示教学 |
| **批量预填充 (Batch Prefill)** | 极快 (受限于计算带宽) | 高 | 所有生产环境 |
| **多序列并行 (Concurrent)** | 中 (总吞吐量最高) | 极高 | AI 聊天服务器、高并发 Agent |

**核心结论**：`llama_decode` 的开销很大一部分在于启动内核计算。一次解码 100 个 Token 的速度远远快于解码 10 个 Token 10次。

## 六、结语

通过掌握 `llama_batch` 的批量构建和最新的 `llama_memory` 接口，我们已经实现了高性能、多并发的推理骨架。

**本章精华：**
1. **批量预填充 (Prefill)** 是提升吞吐量的关键，不要逐个输入 Token。
2. **`seq_id`** 是逻辑隔离不同请求的标签。
3. **`llama_memory`** 是新版 API 处理所有类型缓存统一句柄。

但在真实的长对话场景中，`n_ctx` 终究会被填满。下一章我们将学习 `llama.cpp` 中最高级的主题：**KV Cache 的位移与循环缓存 (Shift & Rolling Cache)**。

### 💡 开发者笔记

- **Logits 索引说明**：在使用 `llama_sampler_sample` 时，注意第三个参数 `idx`。它对应的是当前 `batch` 中 `logits[i] == true` 的那些 token 的相对索引。

-  **线程安全**：虽然一个 `llama_model` 可以共享，但一个 `llama_context` 在同一时间只能执行一个 `llama_decode`。并行是通过在 Batch 里塞入多个序列实现的，而不是通过 C++ 多线程同时调用同一个 Context 实现的。