#include "llama.h"
#include "common.h"
#include <iostream>

/**
 * @brief 现代 KV Cache 滚动函数
 * @param ctx 上下文指针
 * @param n_past 当前的位置计数器（引用传递）
 * @param n_keep 需要永久保留的 Token 数量（如系统指令）
 */
void handle_kv_cache_overflow(llama_context *ctx, int &n_past, int n_keep)
{
    int n_ctx = llama_n_ctx(ctx);
    // 每次溢出时，我们腾出剩余空间（除了n_keep）的 1/4
    int n_discard = (n_past - n_keep) / 4;

    printf("\n\033[33m[KV Cache] 触发滚动：清理 %d 个旧 Token...\033[0m\n", n_discard);

    // 1. 移除紧跟在 n_keep 之后的 n_discard 个 Token
    // 逻辑：删除 pos 在 [n_keep, n_keep + n_discard) 之间的缓存
    llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);

    // 2. 将剩下的 [n_keep + n_discard, n_past) 向前平移 n_discard 个位置
    // 这样它们在下一次 Attention 计算时，逻辑上就紧跟在 n_keep 后面了
    llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_past, -n_discard);

    // 3. 更新业务层的位置计数器
    n_past -= n_discard;

    printf("\033[32m[KV Cache] 滚动完成。当前 n_past: %d\033[0m\n", n_past);
}

int main()
{
    int n_ctx = 256;
    int n_past = 0; // 已经存入 KV Cache 的 Token 长度
    int n_keep = 0; // 保存系统提示词的长度
    std::string model_path = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    auto *model = llama_model_load_from_file(model_path.c_str(), mparams);
    const auto *vocab = llama_model_get_vocab(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_threads = 8;
    auto *ctx = llama_init_from_model(model, cparams);

    auto sparams = llama_sampler_chain_default_params();
    auto *sampler = llama_sampler_chain_init(sparams);
    // 按需添加采样插件（注意顺序，通常先添加过滤类，最后添加概率类）
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 包装系统提示词
    std::string system_content = "You are a polite assistant.";
    // Qwen 格式: <|im_start|>system\n{content}<|im_end|>\n
    std::string formatted_sys = "<|im_start|>system\n" + system_content + "<|im_end|>\n";
    auto prompt_toks = common_tokenize(vocab, formatted_sys, true, true);

    auto batch = llama_batch_init(n_ctx, 0, 1);
    n_keep = prompt_toks.size();

    common_batch_clear(batch);
    for (size_t i = 0; i < prompt_toks.size(); ++i)
        common_batch_add(batch, prompt_toks[i], n_past++, {0}, i == prompt_toks.size() - 1);
    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "Batch decode failed\n");
        return 1;
    }

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

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}