# 深入浅出 llama.cpp（五）：拥抱 C++ 环境 —— 集成官方 common 工具库

## 零、前言

在之前的开发中，我们一直试图保持“纯净”，只调用 `llama.h` 提供的 C 接口。但在处理“批量填充 Batch”或“复杂的采样策略”时，代码开始变得臃肿且难以维护。

`llama.cpp` 官方其实为我们准备了一套非常强大的 C++ 工具库，就在源码的 `common/` 文件夹里。虽然它不是核心库的一部分，但几乎所有开发者都会把它带上。

今天，我们将通过集成 `common` 库，把上一章那个硬核的对话机器人重构成一个逻辑清晰、易于扩展的“现代版”程序。

## 一、为什么需要 common 库？

`llama.h` 提供的是“原子”级别的接口，而 `common` 库把这些原子组合成了“分子”：

- **`common.h`**：提供了 `common_batch_add` 等辅助函数。上一章我们手动赋值 `batch.token[i] = ...` 的操作，现在只需一行。
- **`sampling.h`**：提供了完善的采样链。支持 Top-K、Top-P、温度等参数的链式调用，不再需要手写复杂的采样逻辑。
- **字符处理**：`common_token_to_piece` 自动处理了复杂的缓冲区分配，让 Token 转文字变得像调用 `std::string` 一样简单。

## 二、如何在 CMake 中集成 common？

由于 `common` 库依赖于 `llama` 核心库，且包含多个源文件，我们需要在 `CMakeLists.txt` 中开启相关配置：

```cmake
# 1. 开启 common 编译选项
set(LLAMA_BUILD_COMMON ON)

# 2. 引入 common 所在的目录
include_directories(llama.cpp/common)

# 3. 链接时同时链接 llama 和 common
target_link_libraries(main_app PRIVATE llama common)
```

## 三、代码进化：更优雅的推理逻辑

引入 `common` 后，我们的对话机器人发生了质变。以下是基于 `common` 库优化后的完整 REPL 示例。

### 核心改进分析：

1.  **简化分词 (`common_tokenize`)**：不再需要调用两次 `llama_tokenize` 来获取长度，一个函数直接返回 `std::vector<llama_token>`。
2.  **简化 Batch 管理 (`common_batch_add`)**：通过辅助函数自动填充位置信息（pos）和序列 ID（seq_id），彻底告别繁琐的指针操作。
3.  **简化输出处理 (`common_token_to_piece`)**：自动处理 Token 碎片（特别是处理 UTF-8 多字节字符时），直接返回可打印的字符串。

### 完整示例代码：

```cpp
#include "llama.h"
#include "common.h"
#include "sampling.h"
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char **argv)
{
    std::string model_path = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // 1. 初始化后端（这部分保持原样）
    ggml_backend_load_all();

    // 2. 加载模型和上下文
    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 8;
    llama_context *ctx = llama_init_from_model(model, cparams);

    // 3. 简化采样器：使用 llama_sampler 配合 common 逻辑
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 4. 优化：初始化一个足够大的 Batch (例如 512)，用于批量处理 Prompt
    llama_batch batch = llama_batch_init(512, 0, 1);

    int n_past = 0;
    printf("=== KV Cache REPL (common.h optimized) ===\n");

    while (true)
    {
        std::string user_input;
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, user_input) || user_input == "/exit")
            break;

        std::string prompt = "User: " + user_input + "\nAssistant:";

        // --- 简化点 1：使用 common_tokenize ---
        // 参数：vocab, text, add_bos, parse_special
        auto tokens = common_tokenize(vocab, prompt, n_past == 0, true);

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            // --- 简化点 2：使用 common_batch_clear 和 common_batch_add ---
            // 将整个 Prompt 加入 Batch，只对最后一个 token 请求 logits
            // 替代了 batch.n_tokens = 1; batch.token[0] = ... 等一系列操作
            common_batch_clear(batch);
            common_batch_add(batch, tokens[i], n_past, {0}, i == tokens.size() - 1);

            if (llama_decode(ctx, batch) != 0)
            {
                fprintf(stderr, "Decode failed\n");
                return 1;
            }
            n_past++;
        }

        // --- 2. 生成回答 ---
        std::cout << "AI: ";
        for (int i = 0; i < 16; ++i)
        {
            // 采样
            llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok))
                break;

            // --- 简化点 3：使用 common_token_to_piece ---
            // 它自动处理 buffer 并返回 std::string
            std::string piece = common_token_to_piece(ctx, tok);
            std::cout << piece;
            std::cout.flush();

            // 和先前加入 Token 时一样的简化过程...
            common_batch_clear(batch);
            common_batch_add(batch, tok, n_past, {0}, true);

            if (llama_decode(ctx, batch) != 0)
            {
                fprintf(stderr, "Decode failed\n");
                return 1;
            }
            n_past++;
        }
        std::cout << std::endl;
    }

    // 清理
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
```

## 四、（可选）重新配置 VSCode 代码提示

在引入 `common/` 目录后，由于 `llama.cpp` 源码结构复杂（存在多个同名的 `common.h`），传统的递归包含路径会导致 IntelliSense 混乱。

推荐使用 **CMake Tools** 插件提供的配置。打开 `.vscode/c_cpp_properties.json`：

```json
{
    "configurations": [
        {
            "includePath": ["${default}"],
            "configurationProvider": "ms-vscode.cmake-tools",
            // ...
        }, 
        // ...
    ],
    // ...
}
```

通过这种方式，VSCode 会直接读取 `CMakeLists.txt` 中的 `include_directories` 信息，确保代码提示与实际编译环境完全一致。

## 五、结语

通过集成 `common` 库，我们不仅精简了代码，还利用 `common_batch_add` 为后续的复杂操作打下了基础。

现在，我们已经拥有了一个功能完备的本地对话 Demo。但在实际生产环境中，一个 AI 后端通常需要同时处理多个用户的请求。

在下一章中，我们将迎接最挑战性的部分：**多序列（Multi-sequence）管理**。我们将学习如何在同一个 KV Cache 空间内，让模型同时和多个人聊天而互不干扰。
