#include "llama.h"
#include <vector>
#include <cstdio>
#include <string>
#define N_SEQ_MAX 8
// --- 定义单个并发流的状态 ---
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
// 简单的 Token 转文字辅助函数
std::string token_to_piece(const struct llama_vocab* vocab, llama_token token) {
    std::vector<char> result(32);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    } else {
        result.resize(n_tokens);
    }
    return std::string(result.data(), result.size());
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

void process_sec_turn(llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_prefix) {
}
void process_decode(llama_batch & batch,llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_ctx) {
#if 1
        // 遍历 batch 找到需要采样的 token
        for (int i = 0; i < batch.n_tokens; ++i) {
            // 只有标记了 logits 为 true 的位置才能调用 llama_get_logits_ith
            if (batch.logits[i]) {
                int slot_id = batch.seq_id[i][0]; // 获取该 logits 属于哪个流
                struct Stream & s = streams[slot_id -1];
                float * logits = llama_get_logits_ith(ctx, i);
                
                 // 贪婪采样
                 llama_token next_token = 0;
                 float max_p = -1e10;
                 for (int v = 0; v < llama_vocab_n_tokens(vocab); ++v) {
                     if (logits[v] > max_p) { max_p = logits[v]; next_token = v; }
                 }

                 // 转换并输出
                 std::string piece = token_to_piece(vocab, next_token);
                 printf("[Stream %d]: %s\n", s.id, piece.c_str());
                 fflush(stdout);

                 // 检查结束条件
                 if (next_token == llama_vocab_eos(vocab) || s.n_past >= n_ctx) {
                     s.active = false;
                     s.completed = true;
                     // 释放该槽位的 KV 缓存，让出显存
                     //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
                     llama_memory_seq_rm (llama_get_memory(ctx),s.slot_id, -1, -1);
                     printf("[Stream %d]: generation finsh and kv cache free\n", s.id);
                 } else {
                     // 将新生成的 Token 放入下一轮处理队列
                     s.pending_tokens.push_back(next_token);
                 }
            }
        }
#endif
}
void process_first_turn_2(llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_prefix,int n_ctx) {
    //int n_parallel = streams.size(); 
    llama_batch batch = llama_batch_init(4096, 0, N_SEQ_MAX);
    //llama_batch batch = llama_batch_init(2048, 0, N_SEQ_MAX);

    //llama_batch_clear(batch);
    
    llama_batch_clear(batch);
    for (auto & s : streams) {
        if (!s.active) continue;

        // --- 核心修复 1: 物理抹除目标槽位状态 ---
        // 这一步消除所有潜在的 "diverged" 历史
        //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, -1, -1);

        // --- 核心修复 2: 执行物理克隆 ---
        // 将 ID 0 的前缀状态同步到当前流的 ID 中
        // 注意：确保你的 type_k/v 是 F16，否则这里可能断言失败
        //llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, 0, n_prefix);
        llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, -1, -1);
        // 删除n_prefix+1
        llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, n_prefix+1, -1);
        // --- 核心修复 3: 构造 Batch ---
        for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
            bool is_last = (i == s.pending_tokens.size() - 1);
            // 【重要】：这里只传 {s.slot_id}，不要传 {0, s.slot_id}
            // 因为物理克隆已经完成了“继承”，逻辑耦合反而会引发冲突
           //  // 确保位置 (pos) 从 n_prefix 严格开始，不要包含重复的 BOS (Token 1)
            //llama_batch_add(batch, s.pending_tokens[i], n_prefix + i, {s.slot_id}, is_last);
            llama_batch_add(batch, s.pending_tokens[i], n_prefix + i, {s.slot_id}, is_last);
        }
#if 0
        if (batch.n_tokens > 0) {
            if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            continue;
            }
        }
#endif
        s.n_past = n_prefix + s.pending_tokens.size();
        s.pending_tokens.clear();
    }
#if 1 
    if (batch.n_tokens > 0) {
      // 执行当前流的 Pre-fill
        if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            //continue;
        }
    }
#endif
    process_decode(batch,ctx,vocab,streams,n_ctx);
    //llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_prefix);
       // 清理本次处理完的输入
    //for (auto & s : streams) { if (s.is_prefill) s.pending_tokens.clear(); }
    llama_batch_free(batch);
}
// 假设 n_prefix 是公共前缀(ID 0)的长度
//void process_first_turn(struct llama_batch & batch, llama_context * ctx, std::vector<Stream> & streams, int n_prefix) {
void process_first_turn(llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_prefix,int n_ctx) {
    //int n_parallel = streams.size(); 
    llama_batch batch = llama_batch_init(4096, 0, N_SEQ_MAX);
    //llama_batch batch = llama_batch_init(2048, 0, N_SEQ_MAX);

    //llama_batch_clear(batch);
    
    for (auto & s : streams) {
        if (!s.active) continue;

        // --- 核心修复 1: 物理抹除目标槽位状态 ---
        // 这一步消除所有潜在的 "diverged" 历史
        //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, -1, -1);

        // --- 核心修复 2: 执行物理克隆 ---
        // 将 ID 0 的前缀状态同步到当前流的 ID 中
        // 注意：确保你的 type_k/v 是 F16，否则这里可能断言失败
        //llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, 0, n_prefix);
        llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, -1, -1);
        // 删除n_prefix+1
        llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, n_prefix+1, -1);
        // --- 核心修复 3: 构造 Batch ---
        llama_batch_clear(batch);
        for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
            bool is_last = (i == s.pending_tokens.size() - 1);
            // 【重要】：这里只传 {s.slot_id}，不要传 {0, s.slot_id}
            // 因为物理克隆已经完成了“继承”，逻辑耦合反而会引发冲突
           //  // 确保位置 (pos) 从 n_prefix 严格开始，不要包含重复的 BOS (Token 1)
            //llama_batch_add(batch, s.pending_tokens[i], n_prefix + i, {s.slot_id}, is_last);
            llama_batch_add(batch, s.pending_tokens[i], n_prefix + i, {s.slot_id}, is_last);
        }
#if 1
        if (batch.n_tokens > 0) {
            if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            continue;
            }
        }
#endif
        s.n_past = n_prefix + s.pending_tokens.size();
        s.pending_tokens.clear();
        process_decode(batch,ctx,vocab,streams,n_ctx);
    }
#if 0 
    if (batch.n_tokens > 0) {
      // 执行当前流的 Pre-fill
        if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            //continue;
        }
    }
#endif
    //llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_prefix);
       // 清理本次处理完的输入
    //for (auto & s : streams) { if (s.is_prefill) s.pending_tokens.clear(); }
    llama_batch_free(batch);
}

void process_first_turn(llama_context * ctx,std::vector<llama_token> & prefix_tokens, std::vector<Stream> & streams, int n_prefix) {
    // 关键点 1：n_seq_max 必须等于你的最大 Slot 数量（例如 4 或 8）
    // 这样 batch 内部才有空间同时存储多个 seq_id
    int n_parallel = streams.size(); 
    llama_batch batch = llama_batch_init(2048, 0, n_parallel);

    // --- 第一步：预热公共前缀 (仅在 Slot 0 为空时执行) ---
    // 如果 Slot 0 已经有数据，可以跳过此步
    //if (llama_get_kv_cache_used_cells(ctx) == 0) {
        std::vector<int> all_slots;
        for (int i = 0; i < n_parallel; ++i) all_slots.push_back(i);

        for (int i = 0; i < n_prefix; ++i) {
            // 关键点 2：将前缀 Token 同时挂载到所有 Slot ID
            // 这样 Slot 1, 2, 3... 逻辑上立刻拥有了前缀，无需 seq_cp
            common_batch_add(batch, prefix_tokens[i], i, all_slots, false);
        }
        if (llama_decode(ctx, batch) != 0) {
            printf("prefill decode failed!\n");
        }
        llama_batch_clear(batch);
    //}

    // --- 第二步：并行处理各个流的增量输入 (Prefill) ---
    for (auto & s : streams) {
        if (!s.active || !s.is_prefill) continue;

        // 注意：这里不再调用 llama_kv_cache_seq_cp()，避开 is_full 报错

        for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
            bool is_last = (i == s.pending_tokens.size() - 1);
            // 关键点 3：只给当前流的 ID 添加增量 Token
            // 位置从 n_prefix 开始接续
            common_batch_add(batch, s.pending_tokens[i], n_prefix + i, {0, s.slot_id }, is_last);
        }

        s.n_past = n_prefix + s.pending_tokens.size();
        s.is_prefill = false;
    }

    // 统一执行 Batch 推理
    if (batch.n_tokens > 0) {
        if (llama_decode(ctx, batch) != 0) {
            printf("Batch prefill failed!\n");
        }
    }

    // 清ec
    for (auto & s : streams) { s.pending_tokens.clear(); }
    llama_batch_free(batch);
}

void process_first_turn(struct llama_batch & batch, llama_context * ctx, std::vector<Stream> & streams, int n_prefix) {

    //llama_batch_clear(batch);
    printf("%s %d \n",__func__,__LINE__); 
    for (auto & s : streams) {
        if (!s.active) continue;

        // --- 核心修复 1: 物理抹除目标槽位状态 ---
        // 这一步消除所有潜在的 "diverged" 历史
        //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, -1, -1);

        // --- 核心修复 2: 执行物理克隆 ---
        // 将 ID 0 的前缀状态同步到当前流的 ID 中
        // 注意：确保你的 type_k/v 是 F16，否则这里可能断言失败
        llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, -1, -1);

        // --- 核心修复 3: 构造 Batch ---
        llama_batch_clear(batch);
        for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
            bool is_last = (i == s.pending_tokens.size() - 1);
            // 【重要】：这里只传 {s.slot_id}，不要传 {0, s.slot_id}
            // 因为物理克隆已经完成了“继承”，逻辑耦合反而会引发冲突
           //  // 确保位置 (pos) 从 n_prefix 严格开始，不要包含重复的 BOS (Token 1)
            llama_batch_add(batch, s.pending_tokens[i], n_prefix + i, {s.slot_id}, is_last);
        }
    if (batch.n_tokens > 0) {
        // 执行当前流的 Pre-fill
        if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            continue;
        }
    }

        s.n_past = n_prefix + s.pending_tokens.size();
    }
#if 0
    if (batch.n_tokens > 0) {
        // 执行当前流的 Pre-fill
        if (llama_decode(ctx, batch) != 0) {
            printf("Stream decode failed!\n");
            //continue;
        }
    }
#endif 
       // 清理本次处理完的输入
    for (auto & s : streams) { if (s.is_prefill) s.pending_tokens.clear(); }
}
int main() {
    // 1. 初始化模型和上下文
    llama_model_params mparams = llama_model_default_params();
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    int  N_CTX = 8196;
    // 明确关闭 Flash Attention
    //cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; 
    cparams.n_ctx = N_CTX; // 确保上下文足够容纳前缀 + 输出
    //开启 KV 缓存量化 (Q8_0) 以节省高并发下的显存
    //cparams.type_k = GGML_TYPE_Q8_0; 
    //cparams.type_v = GGML_TYPE_Q8_0;
    // 2. 关键修复：允许每个 Token 同时关联多个 Sequence ID
    // 如果你有关联 {0, slot_id} 的操作，这里至少设为 2
    cparams.n_seq_max = N_SEQ_MAX; // 建议设为你的最大并发 Slot 数 + 1
    const llama_vocab * vocab = llama_model_get_vocab(model);
    //llama_context * ctx = llama_new_context_with_model(model, cparams);
    llama_context * ctx = llama_init_from_model(model, cparams);
    auto tokenize = [&](std::string text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
        // 3. 处理【公共前缀】(seq_id = 0)
    std::string prefix_text = "you are a perfect expert";
    std::vector<llama_token> prefix_tokens(prefix_text.size() + 2);
    int n_prefix = llama_tokenize(vocab, prefix_text.c_str(), prefix_text.size(), prefix_tokens.data(), prefix_tokens.size(), true, true);
    prefix_tokens.resize(n_prefix);

    llama_batch batch = llama_batch_init(cparams.n_ctx, 0, 1);
    //llama_batch batch = llama_batch_init(cparams.n_ctx, 0, N_SEQ_MAX);
    for (int i = 0; i < n_prefix; ++i) {
        llama_batch_add(batch, prefix_tokens[i], i, {0}, false);
    }
    if (llama_decode(ctx, batch) != 0) { printf("Prefix decode failed\n"); return 1; }
    printf("[System] common prefix : %d tokens\n", n_prefix);
    // 4. 初始化多个并发流 (模拟两个用户同时提问)
    std::vector<Stream> streams(2);
    std::string queries[] = {"talk a story about Los Angeles Lakers", "who is the best superstar of lakers?"};

    for (int i = 0; i < 2; ++i) {
        streams[i].id = i;
        streams[i].slot_id = i + 1; // seq_id 0 被前缀占用
        streams[i].active = true;
        streams[i].n_past = n_prefix;
#if 0

        // 发送一个“空”或者“虚假”的 Decode 激活
        // 可以在 seq_cp 之前，用一个无意义的 Token（比如 BOS）对 target_slot 做一次极简的 decode 来激活它。
        // 1. 先激活 target_slot (用一个 Token 占位)
        llama_batch_clear(batch);
        llama_batch_add(batch, llama_vocab_bos(vocab), 0, {streams[i].slot_id}, false);
        llama_decode(ctx, batch); 
        // 2. 清除刚才那个占位符的 KV (确保位置 0 干净)
        llama_memory_seq_rm(llama_get_memory(ctx), streams[i].slot_id, -1, -1);
        // 【核心】从 seq_id 0 物理复制前缀 KV 缓存到当前流的 slot
        //llama_memory_seq_rm(llama_get_memory(ctx),streams[i].slot_id, -1, -1);
        //llama_kv_cache_seq_cp(ctx, 0, streams[i].slot_id, 0, n_prefix);
        // 3. 现在执行 cp 绝对安全，因为 target_slot 已经在系统里“注册”过了
        llama_memory_seq_cp(llama_get_memory(ctx), 0, streams[i].slot_id, 0, n_prefix);

#else
        //llama_memory_seq_cp(llama_get_memory(ctx), 0, streams[i].slot_id, 0, n_prefix);
#endif
        // 将用户问题转为 Token
        std::vector<llama_token> q_tokens(queries[i].size() + 2);
        int n = llama_tokenize(vocab, queries[i].c_str(), queries[i].size(), q_tokens.data(), q_tokens.size(), false, true);
        q_tokens.resize(n);
        streams[i].pending_tokens = q_tokens;
    }
#if 0
    // 5. 【主循环】连续批处理 (Continuous Batching)
    while (true) {
        llama_batch_clear(batch);
        bool has_work = false;

        // 收集所有流待处理的 Token (不管是 Pre-fill 还是上一轮生成的)
        for (auto & s : streams) {
            if (!s.active) continue;
            has_work = true;

            // 确保 slot 1 没有任何残留状态
            //llama_kv_cache_seq_rm(ctx, 1, -1, -1)
            //llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, -1, -1);
            if(s.is_prefill)  {
                    llama_memory_seq_rm(llama_get_memory(ctx), s.slot_id, -1, -1);
                    llama_memory_seq_cp(llama_get_memory(ctx), 0, s.slot_id, -1, -1);
            }
            for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
                // 只有每个流输入序列的最后一个 Token 才需要计算 Logits 用于推理
                bool is_last = (i == s.pending_tokens.size() - 1);
                if(s.is_prefill)  {
                    printf(" add slot 0 and slot %d \n",s.slot_id);
                    //通过 {0, slot_id} 的初次绑定，llama.cpp 内部会自动创建一个“虚拟链接”。这比物理拷贝 seq_cp 快得多，且不会因为量化格式不支持而 Core Dump。
                    //llama_batch_add(batch, s.pending_tokens[i], s.n_past + i, {0,s.slot_id}, is_last);
                    llama_batch_add(batch, s.pending_tokens[i], s.n_past + i, {s.slot_id}, is_last);
                }
                else
                {
                    llama_batch_add(batch, s.pending_tokens[i], s.n_past + i, {s.slot_id}, is_last);
                } 
            }
            s.n_past += s.pending_tokens.size();
            s.pending_tokens.clear();
            s.is_prefill = false; // 第一轮结束，关闭耦合
        }

        if (!has_work || batch.n_tokens == 0) break;

         if (batch.n_tokens > 0) {
             // 执行单次批处理推理
             if (llama_decode(ctx, batch) != 0) {
                  printf("decode stream failed\n");
                 break;
             }
         }
         //llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, 0, n_prefix);
        // 对有输出需求的流进行采样
        int output_idx = 0;
        for (auto & s : streams) {
            if (!s.active) continue;

            // 获取 Logits：索引对应我们在 batch 中设置 logits=true 的顺序
            auto * logits = llama_get_logits_ith(ctx, output_idx);
            
            // 贪婪采样
            llama_token next_token = 0;
            float max_p = -1e10;
            for (int v = 0; v < llama_vocab_n_tokens(vocab); ++v) {
                if (logits[v] > max_p) { max_p = logits[v]; next_token = v; }
            }

            // 转换并输出
            std::string piece = token_to_piece(vocab, next_token);
            printf("[Stream %d]: %s\n", s.id, piece.c_str());
            fflush(stdout);

            // 检查结束条件
            if (next_token == llama_vocab_eos(vocab) || s.n_past >= N_CTX) {
                s.active = false;
                s.completed = true;
                // 释放该槽位的 KV 缓存，让出显存
                //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
                llama_memory_seq_rm (llama_get_memory(ctx),s.slot_id, -1, -1);
                printf("[Stream %d]: generation finsh and kv cache free\n", s.id);
            } else {
                // 将新生成的 Token 放入下一轮处理队列
                s.pending_tokens.push_back(next_token);
            }
            output_idx++;
        }
    }    
#else
     //process_first_turn(ctx,prefix_tokens,streams,n_prefix); 
     //process_first_turn(ctx,streams,n_prefix); 
     // *************************
     //process_first_turn(ctx,vocab,streams,n_prefix,cparams.n_ctx); 
     process_first_turn_2(ctx,vocab,streams,n_prefix,cparams.n_ctx); 
     //process_first_turn(batch,ctx,streams,n_prefix); 
     // 遍历 batch 找到需要采样的 token
#if 1
    // 5. 【主循环】连续批处理 (Continuous Batching)
    while (true) {
        llama_batch_clear(batch);
        bool has_work = false;

        // 收集所有流待处理的 Token (不管是 Pre-fill 还是上一轮生成的)
        for (auto & s : streams) {
            if (!s.active) continue;
            has_work = true;
            for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
                // 只有每个流输入序列的最后一个 Token 才需要计算 Logits 用于推理
                bool is_last = (i == s.pending_tokens.size() - 1);
                    llama_batch_add(batch, s.pending_tokens[i], s.n_past + i, {s.slot_id}, is_last);
            }
            s.n_past += s.pending_tokens.size();
            s.pending_tokens.clear();
        }

        if (!has_work || batch.n_tokens == 0) break;

         if (batch.n_tokens > 0) {
             // 执行单次批处理推理
             if (llama_decode(ctx, batch) != 0) {
                  printf("decode stream failed\n");
                 break;
             }
         }
         process_decode(batch,ctx,vocab,streams,cparams.n_ctx); 
      }
#endif
#endif
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
