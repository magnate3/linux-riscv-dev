#include "common.h"
#include "llama.h"
#include <vector>
#include <string>
/*
llama_batch 与 seq_id:
在 C++ API 中，多流通过 seq_id 实现。在 common_batch_add 时，我们将 system_prompt 的 seq_id 设置为 {0, 1}，表示这两个序列（Slot）共享这段 KV Cache。
KV Cache 逻辑位置 (pos):
前缀复用的前提是 pos（位置偏移）一致。所有 Slot 都从 pos=0 开始共享前缀，之后各自的私有输入从 n_past_common 开始增长。
llama_decode:
这是真正的执行函数。它会检查 Batch 里的 Token，如果发现多个 seq_id 指向同一个 pos 的同一个 token，推理引擎会自动优化（取决于底层实现，通常在 llama.cpp 中通过序列管理来复用）
*/
int main(int argc, char ** argv) {

    // 2. 加载模型
    common_params params;
    llama_backend_init();
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    int  N_CTX = 2048;
    int n_parallel = 32;
    params.n_parallel = n_parallel;
    cparams.n_batch = 32; // 确保上下文足够容纳前缀 + 输出
    cparams.n_ctx = N_CTX; // 确保上下文足够容纳前缀 + 输出
    //cparams.n_parallel = 32;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context * ctx = llama_init_from_model(model, cparams);

    auto tokenize = [&](std::string text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
    // 3. 准备公共前缀 (System Prompt)
    std::string system_prompt = "你是一个AI助手。";
    auto tokens_system = tokenize(system_prompt, true);

    // 4. 定义两个不同的用户请求 (Slot 0 和 Slot 1)
    std::string user_1 = "北京天气怎么样？";
    std::string user_2 = "上海有什么好玩的？";
    auto tokens_1 = tokenize(user_1, false);
    auto tokens_2 = tokenize(user_2, false);

    // 5. 构建 Batch (核心：前缀复用逻辑)
    // llama_batch 用于一次性将多个 Slot 的 Token 发给 GPU
    // // 第三个参数改为 n_parallel，表示一个 Token 最多可以同时属于 4 个序列
    llama_batch batch = llama_batch_init(cparams.n_batch, 0, n_parallel);

    // 将公共前缀加入 Batch (只需计算一次，后续 Slot 共享)
    for (size_t i = 0; i < tokens_system.size(); ++i) {
        // common_batch_add 是封装好的工具函数，将 token 加入批处理
        // 注意：这里逻辑上让系统提示词占据 KV Cache 的起始位置 [0, tokens_system.size())
        common_batch_add(batch, tokens_system[i], i, {0, 1}, false); 
    }

    // 分别加入两个 Slot 独有的用户输入 (接在系统提示词之后)
    int n_past_common = tokens_system.size();
    for (size_t i = 0; i < tokens_1.size(); ++i) {
        common_batch_add(batch, tokens_1[i], n_past_common + i, {0}, false);
    }
    for (size_t i = 0; i < tokens_2.size(); ++i) {
        common_batch_add(batch, tokens_2[i], n_past_common + i, {1}, false);
    }

    // 6. 执行推理
    if (llama_decode(ctx, batch) == 0) {
        printf("成功利用前缀复用处理了两个并发流！\n");
    }

    // 7. 清理
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free (model);
    llama_backend_free();

    return 0;
}

