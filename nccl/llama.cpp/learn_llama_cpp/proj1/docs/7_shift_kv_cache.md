# 深入浅出 llama.cpp（七）：突破上下文壁垒 —— KV Cache 的滚动与平移策略

## 零、前言

在之前的章节中，我们已经实现了多轮对话，甚至掌握了并行处理多个序列。但有一个现实的“物理极限”我们始终没有正面回应：**`n_ctx`（上下文窗口大小）**。

如果你的模型设置 `n_ctx = 2048`，当对话进行到第 2049 个 Token 时，程序会因为空间不足而报错或崩溃。为了让 AI 能进行“无限长”的对话（虽然它的记忆是有限的），我们需要像管理内存页一样管理 KV Cache。

本章我们将学习 `llama.cpp` 中最高级的技巧之一：**KV Cache Rolling (滚动更新)**。

## 一、核心痛点：当缓存满了，我们该怎么办？

当 KV Cache 达到上限时，通常有三种处理方案：

1. **直接清空**：AI 彻底失忆，用户体验极差。
2. **停住不写**：拒绝回答，程序报错。
3. **智能截断（Context Sliding）**：丢弃最早的一部分聊天记录，保留最新的上下文，同时 **永久保留** 系统提示词（System Prompt）。

方案 3 是目前所有主流 LLM 应用（如 ChatGPT, Claude）的标配。

## 二、底层原理：平移（Shift）而非重算

在以前，如果我们想删掉中间的 Token，可能需要重新计算整个序列。但在 `llama.cpp` 中，我们可以直接操作 KV Cache 的“逻辑索引”。

### 1. 逻辑删除 (`llama_memory_seq_rm`)

这个函数允许我们将 KV Cache 中指定范围（pos）的缓存标记为无效。
- **作用**：腾出物理空间。
- **注意**：它只是让这部分位置变为空白。

### 2. 逻辑平移 (`llama_memory_seq_add`)

这是一个极其强大的函数。它可以将一段位置区间的所有缓存，在逻辑上整体增加或减少一个偏移量。
- **场景**：如果我们删除了位置 `100~200` 的 Token，那么原先在 `201~500` 的 Token 就需要“向前平移” 100 个位置，变成 `101~400`。
- **优势**：**不需要重新推理模型**。它只是修改了 Attention 矩阵计算时的位置编码偏置。

## 三、示例程序：先上代码！

其实大部分的实现和之前的类似，但是在这份代码里我还做了一个小小的微调。

如果有亲自上手去跑过前面章节代码的读者，不难发现前面的章节里，模型像是在续写，而不是在和我们对话。

这实际上是因为缺少了训练时添加的 **起始和终止标识符** 导致的，在这份代码里，我顺手把这部分的问题解决掉了，现在它看起来就像是一个真正能对话的 AI 了！

```cpp
#include "llama.h"
#include "common.h"
#include <iostream>

void handle_kv_cache_overflow(llama_context *ctx, int &n_past, int n_keep)
{
    int n_ctx = llama_n_ctx(ctx);
    int n_discard = (n_past - n_keep) / 4;

    printf("\n\033[33m[KV Cache] 触发滚动：清理 %d 个旧 Token...\033[0m\n", n_discard);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);

    llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_past, -n_discard);

    n_past -= n_discard;

    printf("\033[32m[KV Cache] 滚动完成。当前 n_past: %d\033[0m\n", n_past);
}

int main()
{
    int n_ctx = 256; // 上下文的最大长度
    int n_past = 0; // 已经存入 KV Cache 的 Token 长度
    int n_keep = 0; // 保存系统提示词的长度

    // 模型初始化...

    // 上下文初始化...

    // 采样器初始化...

    // 构建系统提示词并加入 KV Cache...

    while (true)
    {
        std::string input;
        std::cout << "\nUser > ";
        std::getline(std::cin, input);

        std::string formatted_input = "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
        auto input_tokens = common_tokenize(vocab, formatted_input, false, true);

        // 在 Decode 之前检查空间
        if (n_past + (int)input_tokens.size() > n_ctx)
        {
            handle_kv_cache_overflow(ctx, n_past, n_keep);
        }

        // 构建 Batch 并推理
        common_batch_clear(batch);
        for (size_t i = 0; i < input_tokens.size(); ++i)
            common_batch_add(batch, input_tokens[i], n_past++, {0}, i == input_tokens.size() - 1);
        if (llama_decode(ctx, batch) != 0)
        {
            fprintf(stderr, "Batch decode failed\n");
            return 1;
        }

        std::cout << "AI: ";
        for (size_t i = 0; i < 2048; ++i)
        {
            auto id = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, id))
            {
                common_batch_clear(batch);
                common_batch_add(batch, id, n_past++, {0}, false);
                llama_decode(ctx, batch);
                break;
            }

            std::string piece = common_token_to_piece(ctx, id);
            std::cout << piece << std::flush;

            common_batch_clear(batch);

            // 在生成每个新词前，也要检查 KV Cache 溢出
            if (n_past + 1 > n_ctx)
            {
                handle_kv_cache_overflow(ctx, n_past, n_keep);
            }

            common_batch_add(batch, id, n_past++, {0}, true);
            if (llama_decode(ctx, batch) != 0)
            {
                fprintf(stderr, "Batch decode failed\n");
                return 1;
            }
        }

        std::cout << std::endl;
    }

    // 清理资源...

    return 0;
}
```

## 四、深度解析滚动函数实现

以下是对 `handle_kv_cache_overflow` 函数的深度解析：

```cpp
/**
 * @brief 现代 KV Cache 滚动函数
 * @param ctx 上下文指针
 * @param n_past 当前的位置计数器（引用传递）
 * @param n_keep 需要永久保留的 Token 数量（如系统指令）
 */
void handle_kv_cache_overflow(llama_context *ctx, int &n_past, int n_keep) {
    int n_ctx = llama_n_ctx(ctx);
    // 策略：腾出 1/4 的空间，避免频繁触发清理
    int n_discard = (n_past - n_keep) / 4; 

    // 1. 删除紧跟在系统提示词（n_keep）之后的旧对话
    // 区间：[n_keep, n_keep + n_discard)
    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);

    // 2. 将剩余的 Token 逻辑前移
    // 将 [n_keep + n_discard, n_past) 平移 -n_discard 位
    llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_past, -n_discard);

    // 3. 更新业务层的位置计数器
    n_past -= n_discard;
}
```

### 关键点：
- **`n_keep`**：这是你的“灵魂锁”。通常它等于系统指令（System Prompt）的长度。通过设置 `n_keep`，可以确保 AI 永远记得它自己的身份。
- **负偏移量**：`llama_memory_seq_add` 传入 `-n_discard`，实现了索引的“左移”，让 KV Cache 重新变得紧凑。

## 五、为什么要在采样和 Decode 时都检查溢出？

在示例代码中，有两次溢出检查：

1. **用户输入后**：用户可能输入了一大段长文本，直接撑爆缓存。
2. **生成每个 Token 前**：AI 在“吐字”过程中，每增加一个 Token 都在消耗空间，必须在 `llama_decode` 之前确保有空位。

这种双重检查是构建工业级鲁棒性程序的标准做法。

## 六、性能与体验的权衡

| 清理比例 | 触发频率 | 记忆保留度 | 性能开销 |
| :--- | :--- | :--- | :--- |
| **每次清理 1 个** | 极高 (每字触发) | 最高 | 高 (API 调用频繁) |
| **清理 1/4 (推荐)** | 中 | 适中 | 极低 |
| **清理 1/2** | 低 | 较低 | 极低 |

**建议**：在 C++ 开发中，**一次性清理 10%~25% 的缓存** 是性价比最高的策略。

## 七、结语

到此为止，我们已经攻克了 `llama.cpp` 二次开发中最硬核的基础设施：**动态 KV Cache 管理**。

我们不仅学会了如何让模型“开口说话”，还掌握了如何通过“逻辑平移”技术，让模型在有限的内存空间里拥有“无限”的记忆流。至此，你的推理引擎已经具备了在生产环境下稳定运行的“骨架”。

但一个优秀的 AI 引擎，仅仅有骨架是不够的。如果你仔细观察 `llama_decode` 后的结果，你会发现模型输出的是一串被称为 **Logits** 的浮点数——它们只是概率，而不是文字。

- 为什么有时候模型说话像复读机？
- 为什么同一个问题，模型每次给出的答案都不一样？
- 我们该如何限制模型，让它只能输出 JSON 格式，而不会乱说话？

这一切的秘密都藏在 **采样（Sampling）** 逻辑中。从下一章开始，我们将开启一个新的系列：**“采样的艺术”**。

我们将从最基础的贪婪采样讲起，一路深入到复杂的采样链（Sampler Chain）以及强大的 GBNF 语法约束。