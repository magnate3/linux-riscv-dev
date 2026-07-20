#include "llama.h"
#include <vector>
#include <cstdio>
#include <string>
#define N_SEQ_MAX 8
struct Stream {
    int id;                         // 流 ID
    int slot_id;                    // 对应的 llama seq_id
    int n_past = 0;                 // 当前流在 KV Cache 中的进度
    std::vector<llama_token> pending_tokens; // 待处理的 Token (Pre-fill 或新生成的词)
    bool is_prefill = true; // 标记是否是该流的第一轮（需要耦合前缀）
    bool active = false;            // 是否激活
    bool completed = false;         // 是否生成结束
};
void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens]    = id;
    batch.pos[batch.n_tokens]      = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;

    batch.n_tokens++;
}
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
    cparams.n_seq_max = N_SEQ_MAX; // 确保上下文足够容纳前缀 + 输出
    const llama_vocab * vocab = llama_model_get_vocab(model);
    //llama_context * ctx = llama_new_context_with_model(model, cparams);
    llama_context * ctx = llama_init_from_model(model, cparams);
    auto tokenize = [&](std::string text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
    // 2. 定义公共前缀 (System Prompt)
    std::vector<llama_token> tokens_prefix=tokenize("you are a expert at coding",true);
    int n_prefix = tokens_prefix.size();

    // 将公共前缀存入 seq_id 0
    llama_batch batch = llama_batch_init(cparams.n_ctx, 0, N_SEQ_MAX);
    for (int i = 0; i < n_prefix; ++i) {
        // logits = false (前缀处理阶段通常不需要输出)
        llama_batch_add(batch, tokens_prefix[i], i, {0}, false);
    }
    llama_decode(ctx, batch); // 计算并缓存公共前缀
    // --- 模拟高并发：两个不同的用户 Slot ---
    // 用户 A 的第一轮对话 (Slot/Seq_id 1)
    int user_a_slot = 1;
    std::vector<llama_token> tokens_user_a = tokenize("how to code hello world in c",false);
    
    llama_batch_clear(batch);
    for (int i = 0; i < (int)tokens_user_a.size(); ++i) {
        // 关键：指定该 Token 属于 seq_id 0 (共享前缀) 和 seq_id 1 (用户私有)
        // 位置从 n_prefix 开始
        llama_batch_add(batch, tokens_user_a[i], n_prefix + i, { user_a_slot}, i == (int)tokens_user_a.size()-1);
    }
    llama_memory_seq_cp(llama_get_memory(ctx), 0, user_a_slot, -1, -1);
    // 第一次：处理 [Prefix + Query1]
    auto t_start1 = ggml_time_us();
    // 此时会计算所有 Token 的 KV 并存在内存中
    llama_decode(ctx, batch); 
    auto t_end1 = ggml_time_us();
    printf("第一次耗时 (冷启动): %8.2f ms\n", (t_end1 - t_start1) / 1000.0);
    printf("\n--- 第一次推理完成 (处理了前缀 + Query1) ---\n");

        // 用户 B 的第一轮对话 (Slot/Seq_id 2) - 完全独立
    int user_b_slot = 2;
    std::vector<llama_token> tokens_user_b = tokenize("how to code hello world in c and in python",false);
    
    llama_batch_clear(batch);
    for (int i = 0; i < (int)tokens_user_b.size(); ++i) {
        // 同样共享 seq_id 0，但私有部分存入 seq_id 2
        llama_batch_add(batch, tokens_user_b[i], n_prefix + i, {user_b_slot}, i == (int)tokens_user_b.size()-1);
    }

    llama_memory_seq_cp(llama_get_memory(ctx), 0, user_b_slot, -1, -1);
    // 第一次：处理 [Prefix + Query1]
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
