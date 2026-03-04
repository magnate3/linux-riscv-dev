#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

// 辅助函数：分词
static std::vector<llama_token> tokenize(const struct llama_vocab *vocab, const std::string &text, bool add_bos)
{
    // 获取所需长度
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_bos, true);
    if (n_tokens < 0)
    {
        std::vector<llama_token> result(-n_tokens);
        if (llama_tokenize(vocab, text.c_str(), text.size(), result.data(), result.size(), add_bos, true) < 0)
        {
            fprintf(stderr, "Tokenize failed\n");
        }
        return result;
    }
    std::vector<llama_token> result(n_tokens);
    llama_tokenize(vocab, text.c_str(), text.size(), result.data(), result.size(), add_bos, true);
    return result;
}

int main(int argc, char **argv)
{
    std::string model_path = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    // 重要：对于 CPU 推理，建议设置线程数
    cparams.n_threads = 8;

    llama_context *ctx = llama_init_from_model(model, cparams);

    // 创建采样器
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 初始化 Batch：分配 1 个 Token 的空间
    // llama_batch_init(n_tokens_alloc, n_embd, n_seq_max)
    llama_batch batch = llama_batch_init(1, 0, 1);

    int n_past = 0;
    printf("=== KV Cache REPL (Ctrl+C to exit) ===\n");

    while (true)
    {
        std::string user_input;
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, user_input) || user_input == "/exit")
            break;

        // 包装简单的指令格式
        std::string prompt = "User: " + user_input + "\nAssistant:";
        // 只有第一次对话才加 BOS
        auto tokens = tokenize(vocab, prompt, n_past == 0);

        // ---- 1. 喂入 Prompt ----
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = (i == tokens.size() - 1); // 只计算最后一个 token 的 logits

            if (llama_decode(ctx, batch) != 0)
            {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            n_past++;
        }

        // ---- 2. 生成回答 ----
        std::cout << "AI: ";
        for (int i = 0; i < 16; ++i)
        {
            // 采样
            llama_token tok = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, tok))
                break;

            // 打印
            char buf[128];
            int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
            if (n > 0)
            {
                std::cout.write(buf, n);
                std::cout.flush();
            }

            // 将生成的 Token 喂回模型
            batch.n_tokens = 1;
            batch.token[0] = tok;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            if (llama_decode(ctx, batch) != 0)
            {
                fprintf(stderr, "decode failed\n");
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