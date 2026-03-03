#include "llama.h"
#include "common.h"
#include "sampling.h"
#include <vector>
#include <string>
#include <iostream>

int main(int argc, char **argv)
{
    // 1. 模型路径（请确保路径正确）
    std::string model_path = "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    // 2. 初始化后端和模型
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    // 3. 初始化上下文：注意 n_ctx 要足够容纳多个序列
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_threads = 8;
    // 这个参数决定了 KV Cache 内部如何划分逻辑槽位
    cparams.n_seq_max = 2;
    llama_context *ctx = llama_init_from_model(model, cparams);

    // 4. 初始化采样器
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 5. 准备两个不同的 Prompt 和序列 ID
    std::string prompt_0 = "What is the capital of France?";
    std::string prompt_1 = "Count from 1 to 5:";

    auto tokens_0 = common_tokenize(vocab, prompt_0, true, true);
    auto tokens_1 = common_tokenize(vocab, prompt_1, true, true);

    // 6. 核心：初始化一个足够大的 Batch 来进行批量处理
    // 能够一次性容纳两个 Prompt 的总和
    llama_batch batch = llama_batch_init(tokens_0.size() + tokens_1.size(), 0, 1);

    // 为每个序列维护一个独立的 KV Cache 位置指针
    int n0 = 0, n1 = 0;

    // --- 第一阶段：批量预填充 ---
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens_0.size(); ++i)
        common_batch_add(batch, tokens_0[i], n0++, {0}, i == tokens_0.size() - 1);
    for (size_t i = 0; i < tokens_1.size(); ++i)
        common_batch_add(batch, tokens_1[i], n1++, {1}, i == tokens_1.size() - 1);

    printf("\n[Step 1] Batch Prefilling for Seq 0 and Seq 1...\n");

    // 只需要调用一次 decode，模型就会并行处理两段 Prompt
    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "Batch decode failed\n");
        return 1;
    }

    // --- 第二阶段：交替生成 (Parallel Generation) ---
    printf("[Step 2] Generating results...\n\n");

    // 使用负数索引进行初始采样，避开绝对索引计算
    // -1 是最后一个有 logits 的 token (即 seq 1)
    // -2 是倒数第二个有 logits 的 token (即 seq 0)
    llama_token tok1 = llama_sampler_sample(sampler, ctx, -1);
    llama_token tok0 = llama_sampler_sample(sampler, ctx, -2);

    // 释放大 batch，换成生成用的小 batch (容量为 2)
    llama_batch_free(batch);
    batch = llama_batch_init(2, 0, 1);

    for (int step = 0; step < 15; ++step)
    {
        // 打印上一步采样的结果
        printf("\033[32m[Seq 0]\033[0m %s ", common_token_to_piece(ctx, tok0).c_str());
        printf("\033[33m[Seq 1]\033[0m %s\n", common_token_to_piece(ctx, tok1).c_str());

        // 准备下一轮 decode
        common_batch_clear(batch);
        common_batch_add(batch, tok0, n0++, {0}, true); // 此时在 batch 中索引为 0
        common_batch_add(batch, tok1, n1++, {1}, true); // 此时在 batch 中索引为 1

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

    // --- 第三阶段：清理特定序列 ---
    printf("[Step 3] Cleaning up Seq 0 from KV Cache...\n");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    // 释放资源
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}