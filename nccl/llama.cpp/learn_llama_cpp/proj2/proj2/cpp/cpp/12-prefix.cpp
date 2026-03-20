#include "llama.h"
#include <vector>
#include <cstdio>
#include <string>
static void
llama_batch_clear(struct llama_batch &batch)
{
    batch.n_tokens = 0;
}
void llama_batch_add(struct llama_batch& batch, llama_token id, llama_pos pos,
                     bool logits) {
  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.logits[batch.n_tokens] = logits;

  // Only provide a simple seq_id of {0}
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = 0;

  batch.n_tokens++;
}

static void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}
int main() {
    // 1. 初始化模型和上下文
    llama_model_params mparams = llama_model_default_params();
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // 确保上下文足够容纳前缀 + 输出
    const llama_vocab * vocab = llama_model_get_vocab(model);
    //llama_context * ctx = llama_new_context_with_model(model, cparams);
    llama_context * ctx = llama_init_from_model(model, cparams);
    // 2. 准备 Prompt：[前缀] + [问题]
    std::string prefix = "这是一个非常长的系统提示词，包含大量背景资料...";
    std::string query1 = " 请总结一下。";
    std::string query2 = " 请换个角度分析。";

    // 将文本转为 Token
    auto tokenize = [&](std::string text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };

    auto tokens_prefix = tokenize(prefix, true);
    auto tokens_query1 = tokenize(query1, false);
    auto tokens_query2 = tokenize(query2, false);

    // 3. 第一次推理：处理 [前缀 + 问题1]
    llama_batch batch = llama_batch_init(cparams.n_ctx, 0, 1);
    
    // 将前缀和问题1放入 batch (pos 从 0 开始)
    std::vector<llama_token> full_prompt1 = tokens_prefix;
    full_prompt1.insert(full_prompt1.end(), tokens_query1.begin(), tokens_query1.end());

    for (size_t i = 0; i < full_prompt1.size(); ++i) {
        llama_batch_add(batch, full_prompt1[i], i,  i == full_prompt1.size() - 1);
        llama_batch_add(batch, full_prompt1[i], i, {0}, i == full_prompt1.size() - 1);
    }
    // 第一次：处理 [Prefix + Query1]
    auto t_start1 = ggml_time_us();
    // 此时会计算所有 Token 的 KV 并存在内存中
    llama_decode(ctx, batch); 
    auto t_end1 = ggml_time_us();
    printf("第一次耗时 (冷启动): %8.2f ms\n", (t_end1 - t_start1) / 1000.0);
    printf("\n--- 第一次推理完成 (处理了前缀 + Query1) ---\n");

    // 4. 第二次推理：重用前缀缓存，只处理 [问题2]
    // 关键：清除 KV Cache 中属于 Query1 的部分，保留 Prefix 部分
    // 前缀的长度是 tokens_prefix.size()
    int n_past_prefix = tokens_prefix.size();
    
    // 移除从 n_past_prefix 开始的所有 KV 缓存（即删掉 Query1 的部分）
    //llama_kv_cache_seq_rm(ctx, 0, n_past_prefix, -1);
     llama_memory_seq_rm (llama_get_memory(ctx),0, n_past_prefix, -1);

    // 构造新 batch，只包含 Query2，但它的 pos 从 n_past_prefix 开始
    llama_batch_clear(batch);

    /* refer to  genc/cc/interop/backends/llamacpp.cc
      for (size_t i = 0; i < tokenized_prompt.size(); ++i) {
    llama_batch_add(batch, tokenized_prompt[i], i, false);
  }

  // Set the last token to output logits.
      batch.logits[batch.n_tokens - 1] = true;
    */
    for (size_t i = 0; i < tokens_query2.size(); ++i) {
        //llama_batch_add(batch, tokens_query2[i], n_past_prefix + i,  i == tokens_query2.size() - 1);
        llama_batch_add(batch, tokens_query2[i], n_past_prefix + i,{0},  i == tokens_query2.size() - 1);
    }

    // 第二次：处理 [Query2] (接在 Prefix 后面)
    auto t_start2 = ggml_time_us();
    // 这一次 decode 会非常快，因为它只计算 Query2 的几个 Token
    llama_decode(ctx, batch);
    auto t_end2 = ggml_time_us();
    printf("第二次耗时 (复用前缀): %8.2f ms\n", (t_end2 - t_start2) / 1000.0);
    printf("\n--- 第二次推理完成 (仅计算了 Query2，命中前缀缓存) ---\n");

    // 5. 释放资源
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
