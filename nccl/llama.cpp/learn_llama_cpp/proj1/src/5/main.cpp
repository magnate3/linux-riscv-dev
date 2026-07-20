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