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
static void
llama_batch_clear(struct llama_batch &batch)
{
    batch.n_tokens = 0;
}
// 检查特定 Slot 在某个位置是否有数据
bool is_pos_cached(llama_context * ctx, int slot_id, int pos) {
    // 获取当前总占用数
    //int cells_before = llama_get_kv_cache_used_cells(ctx);
    int cells_before = llama_memory_seq_pos_max(llama_get_memory(ctx), 0); 
    // 尝试删除该位置的一个 cell (p0=pos, p1=pos+1)
    // 注意：这会破坏该位置的数据，仅用于测试或清理逻辑
    llama_memory_seq_rm(llama_get_memory(ctx), slot_id, pos, pos + 1);
    
    int cells_after = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    
    // 如果占用数减少了，说明该位置之前确实有数据
    return cells_after < cells_before;
}

int main(int argc, char ** argv) {

    // 2. 加载模型
    //common_params params;
    llama_backend_init();
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    int n_parallel = 4;
    int  N_CTX = 2048*n_parallel;
    //params.n_parallel = n_parallel;
    cparams.n_batch = 64; // 确保上下文足够容纳前缀 + 输出
    cparams.n_ctx = N_CTX; // 确保上下文足够容纳前缀 + 输出
    cparams.n_seq_max       = n_parallel;
#if 0
     ctx_params.seed  = 1234;
    ctx_params.n_ctx   = n_kv_req;
    ctx_params.n_batch = std::max(n_len, n_parallel);
    ctx_params.n_seq_max       = n_parallel;
    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

#endif
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
    std::string system_prompt = "you are a AI assistant";
    auto tokens_system = tokenize(system_prompt, true);

    // 4. 定义两个不同的用户请求 (Slot 0 和 Slot 1)
    std::string user_1 = "what about the weather of shanghai";
    std::string user_2 = "where is the captital of china";
    auto tokens_1 = tokenize(user_1, false);
    auto tokens_2 = tokenize(user_2, false);
#if 0
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
#else
   // 重点：n_seq_max 设为 1，因为我们后续用 cp 逻辑分发
    llama_batch batch = llama_batch_init(cparams.n_batch, 0, n_parallel);
    for (size_t i = 0; i < tokens_system.size(); ++i) {
        common_batch_add(batch, tokens_system[i], i, {0}, false);
    }
    
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "前缀预热失败\n");
        return 1;
    }
    int n_past_prefix = tokens_system.size();
    printf("公共前缀已加载到 Slot 0，长度: %d\n", n_past_prefix);

    // 3. 动态分发给 Slot 1 和 Slot 2 (前缀复用)
    // 这一步是瞬时的，不消耗 GPU 计算资源
    //llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_past_prefix);
    //llama_memory_seq_cp(llama_get_memory(ctx), 0, 2, 0, n_past_prefix);

    // 4. 并发处理两个不同的用户请求
    //llama_batch_free(batch);
#if 1
    llama_batch_clear(batch); // 重置 batch 准备新数据
    //batch = llama_batch_init(cparams.n_batch, 0, n_parallel); // 每个 token 只归属一个 slot
        // 将两个 Slot 的 Token 同时放入 Batch 执行 (Continuous Batching)
    for (size_t i = 0; i < tokens_1.size(); ++i) {
        common_batch_add(batch, tokens_1[i], n_past_prefix + i, {1}, true);
        //common_batch_add(batch, tokens_1[i], n_past_prefix + i, {1}, (i == tokens_1.size()-1));
    }
    for (size_t i = 0; i < tokens_2.size(); ++i) {
        common_batch_add(batch, tokens_2[i], n_past_prefix + i, {2}, true);
        //common_batch_add(batch, tokens_2[i], n_past_prefix + i, {2}, (i == tokens_2.size()-1));
    }

   llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_past_prefix);
   llama_memory_seq_cp(llama_get_memory(ctx), 0, 2, 0, n_past_prefix);
    if (llama_decode(ctx, batch) == 0) {
        printf("并发推理成功！Slot 1 和 Slot 2 均复用了 Slot 0 的前缀。\n");
    }
    if(is_pos_cached(ctx, 1, n_past_prefix +1)){
        printf("slot1 pos %d cached \n",n_past_prefix +1);
    }
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < n_past_prefix; ++i) {
             if(is_pos_cached(ctx, j, i)){
                 printf("slot %d pos %d cached \n",j,i);
             }
        } 
    }
    int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0); 
    if(n_ctx_used >= llama_n_ctx(ctx)){
         printf("ctx kv cache is full \n");
    }
#else
     batch.n_tokens = 0; // 重置 batch 计数
    // 填充 Slot 1 的数据 (接在 p_len 之后)
    for (size_t i = 0; i < tokens_1.size(); ++i) {
        int idx = batch.n_tokens++;
        batch.token[idx]    = tokens_1[i];
        batch.pos[idx]      = n_past_prefix+ i;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 1; // 指定给 Slot 1
        batch.logits[idx]   = (i == tokens_1.size() - 1); // 仅最后一位算 Logits
    }

    // 填充 Slot 2 的数据 (接在 p_len 之后)
    for (size_t i = 0; i < tokens_2.size(); ++i) {
        int idx = batch.n_tokens++;
        batch.token[idx]    = tokens_2[i];
        batch.pos[idx]      =  n_past_prefix + i;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 2; // 指定给 Slot 2
        batch.logits[idx]   = (i == tokens_2.size() - 1);
    }

    // 执行连续批处理推理 (Continuous Batching)
    if (llama_decode(ctx, batch) == 0) {
        printf("成功！Slot 1 和 2 独立运行并复用了 Slot 0 的前缀。\n");
    }

#endif
       // 5. 动态清理：释放 Slot 1 的私有部分，保留前缀供下次 cp
    llama_memory_seq_rm(llama_get_memory(ctx), 1, n_past_prefix, -1);
    llama_memory_seq_rm(llama_get_memory(ctx), 2, n_past_prefix, -1);
#endif

    // 7. 清理
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free (model);
    llama_backend_free();

    return 0;
}

