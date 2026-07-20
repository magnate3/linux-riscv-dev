#include "llama.h"
#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
std::vector<llama_token>  llama_tokenize(const llama_context * ctx,std::string text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 3);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    tokens.resize(n);
    return tokens;
};
llama_token bridge_tokens(llama_context * ctx_tgt, llama_context * ctx_dft, llama_token t_dft) {
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_model * model_dft= llama_get_model(ctx_dft);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    // 将源模型的 ID 转为字符串片段 (Piece)
    char buf[128];
    int n = llama_token_to_piece(vocab_dft, t_dft, buf, sizeof(buf),0,false);
    if (n < 0) return 0;
    std::string s(buf, n);

    // 将字符串片段重新 Tokenize 成目标模型的 ID
    std::vector<llama_token> t_dst = llama_tokenize(ctx_tgt, s, false);
    return t_dst.empty() ? 0 : t_dst[0];
}

// 1. 严格对齐的贪婪采样
llama_token sample_greedy(struct llama_context * ctx, int idx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab =  llama_vocab_n_tokens(vocab);;
    float * logits = llama_get_logits_ith(ctx, idx);
    int max_id = 0;
    float max_p = -1e10;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > max_p) { max_p = logits[i]; max_id = i; }
    }
    return (llama_token)max_id;
}

// 实现逻辑：在给定的 logits 数组中找到数值最大的索引（即 Token ID）
llama_token llama_sample_token_greedy(const struct llama_vocab * vocab, float * logits) {
    // 1. 获取模型词表大小 (Vocab Size)
    const int n_vocab =  llama_vocab_n_tokens(vocab);;

    // 2. 初始化最大值寻找逻辑
    int   max_id    = 0;
    float max_logit = logits[0];

    // 3. 遍历所有 Logits（在 CPU 上，现代编译器会自动优化此循环）
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            max_id    = i;
        }
    }

    return (llama_token)max_id;
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
std::string custom_common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

std::string custom_common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return custom_common_detokenize(vocab, tokens, special);
}
int vocab_match(const struct llama_context * ctx_tgt,const struct llama_context * ctx_dft) {
    const char * test_word = "Hello";
    // 分别获取两个模型的 ID
    llama_token id_tgt = llama_tokenize(ctx_tgt, test_word, false)[0];
    llama_token id_dft = llama_tokenize(ctx_dft, test_word, false)[0];
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_model * model_dft= llama_get_model(ctx_dft);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    const int n_vocab_tgt =  llama_vocab_n_tokens(vocab_tgt);;
    const int n_vocab_dft =  llama_vocab_n_tokens(vocab_dft);;
    printf("n_vocab_tgt %d,n_vocab_dft %d \n",n_vocab_tgt ,n_vocab_dft);
    printf("Target ID for 'Hello': %d\n", id_tgt);
    printf("Draft ID for 'Hello':  %d\n", id_dft);
    
    if (id_tgt != id_dft) {
        printf("vocab mismatch \n");
        return -1;
    }
    return 0;
}
int main() {

#if 0
#else
    // 1. 初始化后端 (CPU 模式)
    llama_backend_init();

    // 2. 加载大模型 (Target) - CPU 优化配置
    auto t_params = llama_model_default_params();
    t_params.n_gpu_layers = 0; // 强制 0 层在 GPU，确保纯 CPU 运行
    std::string tgt_model_path = "/workspace/qwen/models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
    //std::string tgt_model_path = "/workspace/qwen/models/AMD-Llama-135m.Q8_0.gguf";
    //std::string tgt_model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    //llama_model * model_tgt = llama_model_load_from_file("llama-3-8b.gguf", t_params);
    llama_model * model_tgt = llama_model_load_from_file(tgt_model_path.c_str(), t_params);
    
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    // 3. 加载小模型 (Draft) - 同样强制 CPU
    auto d_params = llama_model_default_params();
    d_params.n_gpu_layers = 0;
    std::string dft_model_path = "/workspace/qwen/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    //std::string dft_model_path = "/workspace/qwen/models/AMD-Llama-135m.Q8_0.gguf";
    llama_model * model_dft = llama_model_load_from_file(dft_model_path.c_str(), d_params);
    //llama_model * model_dft = llama_model_load_from_file("llama-3-135m.gguf", d_params);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    // 4. 创建 Context (设置多线程)
    auto c_params = llama_context_default_params();
    c_params.n_ctx = 2048*4;
    // 关键：针对 CPU 逻辑核心数进行优化（通常设为物理核心数）
    c_params.n_threads = 8; 
    c_params.n_threads_batch = 8;

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, c_params);
    llama_context * ctx_dft = llama_init_from_model(model_dft, c_params);
    auto tokenize = [&](const llama_vocab * vocab,std::string& text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
#if 1
    if(vocab_match(ctx_tgt, ctx_dft)){
        //llama_free(ctx_tgt); llama_model_free(model_tgt);
        //llama_free(ctx_dft); llama_model_free(model_dft);
        //llama_backend_free();
        //return -1;
    }
#endif
    // 5. 协同推理循环
    //std::vector<llama_token> tokens = { llama_vocab_bos(vocab_tgt) };
       // 5. 准备 Prompt 和初始 Token
    std::string prompt = "The capital of France is";
    std::vector<llama_token> tokens = tokenize(vocab_tgt, prompt, false);
    //std::vector<llama_token> tokens = tokenize(vocab_tgt, prompt, true);
    //int n_predict = 50; // 总生成长度
    int n_draft = 4; // CPU 上建议设置 4-8 之间
    
#if 1
    int n_past = 0;
       // 大小模型必须同步完成 Prefill，建立相同的 KV Cache 基础
    llama_batch batch_pre = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch_pre.token[i] = tokens[i];
        batch_pre.pos[i]   = i;
        batch_pre.n_seq_id[i] = 1;
        // seq_id (二级指针)
        batch_pre.seq_id[i][0] = 0;
        ////batch.n_seq_id[i] 必须先设为 1
        //batch_pre.n_seq_id[i] = 1;
        //if (batch_pre.seq_id[i]) {
        //    batch_pre.seq_id[i][0] = 0; // 将该 token 分配给序列 0
        //}
        batch_pre.logits[i] = (i == tokens.size() - 1); // 只有最后一位需要 logits
    }
    batch_pre.n_tokens =  tokens.size();
    if (llama_decode(ctx_dft, batch_pre) != 0) {
        fprintf(stderr, "Draft decode failed at pos \n");
        goto fail1;
    }
    if (llama_decode(ctx_tgt, batch_pre) != 0) {
        fprintf(stderr, "tgt decode failed at pos \n");
        goto fail1;
    }
    n_past = tokens.size(); 
#endif
#endif
    // 3. 投机循环
    n_draft = 4;
    while (tokens.size() < 100) {
        llama_token last_confirmed = tokens.back();
        llama_token current_input = last_confirmed;
        std::vector<llama_token> draft_tokens;
        int n_accept = 0;
        llama_token next_token = 0;
        // --- A. 小模型顺序盲猜 ---
        for (int i = 0; i < n_draft; ++i) {
            llama_batch b_dft = llama_batch_init(1, 0, 1);
            b_dft.n_tokens = 1;
            b_dft.token[0] = current_input;
            b_dft.pos[0]   = n_past + i;
            b_dft.n_seq_id[0] = 1;
            b_dft.seq_id[0][0] = 0;
            b_dft.logits[0] = true;

            if (llama_decode(ctx_dft, b_dft) != 0) {
                fprintf(stderr, "Draft decode failed at pos %d\n", b_dft.pos[0]);
                llama_batch_free(b_dft);
                goto fail1;
            }
            llama_token sampled = sample_greedy(ctx_dft, 0);
#if 0
            sampled = bridge_tokens(ctx_tgt, ctx_dft, sampled);
#endif
            draft_tokens.push_back(sampled);
            current_input = sampled; // 自回归：下一次输入是本次输出
            llama_batch_free(b_dft);
        }

        // --- B. 大模型验证 (核心修正点) ---
        // 验证 Batch 必须包含小模型猜的 Token 的前驱，从而获得对比 Logits
        llama_batch b_tgt = llama_batch_init(n_draft, 0, 1);
        b_tgt.n_tokens = n_draft;
        for (int i = 0; i < n_draft; ++i) {
            // 输入上一个 Token，预测下一个以对齐小模型的 draft_tokens[i]
#if 0
            b_tgt.token[i]  = draft_tokens[i];
            b_tgt.pos[i]    = n_past + i +1;
#else
            b_tgt.token[i]  = (i == 0) ? last_confirmed : draft_tokens[i-1];
            b_tgt.pos[i]    = n_past + i ;
#endif
            b_tgt.n_seq_id[i] = 1;
            b_tgt.seq_id[i][0] = 0;
            b_tgt.logits[i] = true;
        }

        if (llama_decode(ctx_tgt, b_tgt) != 0) {
            fprintf(stderr, "Target verify failed!\n");
            llama_batch_free(b_tgt);
            goto fail1 ;
        }
        // --- C. 验证比对 ---

        for (int i = 0; i < n_draft; ++i) {
            llama_token t_tgt = sample_greedy(ctx_tgt, i);
            if (t_tgt == draft_tokens[i]) {
                n_accept++;
                //last_confirmed = t_tgt;
                //tokens.push_back(t_tgt);
                //printf("accept draft token  %d \n", t_tgt);
            } else {
                // 猜错了，这一位修正为大模型的正确 Token
                next_token = t_tgt;
                //printf("accept tgt token  %d \n", t_tgt);
                break;
            }
        }
        
#if 1
        // 如果全部猜对，大模型还需多给一个额外的 Token（投机采样的红利）
        if (n_accept == n_draft) {
             // 实际上大模型在最后一位也输出了预测，可以采样出来
             // 但为简化逻辑，这里让下一轮循环处理
             next_token = sample_greedy(ctx_tgt, n_draft - 1); 
        }
        for (int i = 0; i < n_accept; ++i) {
            tokens.push_back(draft_tokens[i]);
            //printf("%s", llama_token_to_piece(vocab_tgt, draft_tokens[i]).c_str());
        }
        // 修正并同步
        tokens.push_back(next_token);
        n_past += n_accept + 1;
        if (next_token == llama_vocab_eos(vocab_tgt)){
              goto fail1;
        }
        // Autoregressive Inference
#if 1      
        if (n_accept == n_draft) {
             llama_batch b_dft = llama_batch_init(1, 0, 1);
             b_dft.n_tokens = 1;
             b_dft.token[0] = next_token;
             b_dft.pos[0]   = n_past-1;
             b_dft.n_seq_id[0] = 1;
             b_dft.seq_id[0][0] = 0;
             b_dft.logits[0] = true;

            if (llama_decode(ctx_dft, b_dft) != 0) {
                fprintf(stderr, "Draft decode failed at pos %d\n", b_dft.pos[0]);
                llama_batch_free(b_dft);
                goto fail1;
            }
            llama_batch_free(b_dft);
             llama_batch b_tgt = llama_batch_init(1, 0, 1);
             b_tgt.n_tokens = 1;
             b_tgt.token[0] = next_token;
             b_tgt.pos[0]   = n_past-1;
             b_tgt.n_seq_id[0] = 1;
             b_tgt.seq_id[0][0] = 0;
             b_tgt.logits[0] = true;

            if (llama_decode(ctx_tgt, b_tgt) != 0) {
                fprintf(stderr, "Target decode failed at pos %d\n", b_tgt.pos[0]);
                llama_batch_free(b_tgt);
                goto fail1;
            }
            llama_batch_free(b_tgt);
        }
#endif
        // 【核心修复】清理 KV Cache 残留，确保 Y = X + 1
        llama_memory_seq_rm(llama_get_memory(ctx_dft),0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt),0, n_past, -1);
#else
           // D. 同步策略：如果全中，直接物理拷贝 KV Cache
    if (n_accept == n_draft) {
        // 逻辑确认：大模型此时已经拥有了包含最新 n_draft 个 token 的完整 KV Cache
        // 我们将其“镜像”给小模型，实现 prefix reuse
        //llama_kv_cache_seq_rm(ctx_dft, 0, 0, -1); 
        llama_memory_seq_rm(llama_get_memory(ctx_dft),0, 0, -1);
        //llama_kv_cache_seq_cp(ctx_tgt, 0, 0, 0, n_past + n_draft); 
        
        llama_memory_seq_cp(llama_get_memory(ctx_tgt),0, 0, 0, n_past + n_draft);
        n_past += n_draft; 
        // 注意：这种情况下我们还可以继续从大模型的最后一个 logits 采样一个额外的奖励 token
        llama_token bonus = sample_greedy(ctx_tgt, n_draft - 1);
        tokens.push_back(bonus);
        n_past += 1;
    } else {
        // 部分命中或没命中，走常规的 rm 截断逻辑
        n_past += n_accept + 1;
        // 【核心修复】清理 KV Cache 残留，确保 Y = X + 1
        llama_memory_seq_rm (llama_get_memory(ctx_dft),0, n_past, -1);
        llama_memory_seq_rm (llama_get_memory(ctx_tgt),0, n_past, -1);
    }
#endif
        if(n_accept == n_draft)
        printf("Accept ALL : %d/%d | Total: %zu\n", n_accept, n_draft, tokens.size());
        if(n_accept > 0 && n_accept < n_draft)
        printf("Accept part: %d/%d | Total: %zu\n", n_accept, n_draft, tokens.size());
        llama_batch_free(b_tgt);
    }
    std::cout << custom_common_detokenize(ctx_dft,tokens,true) << std::endl;
fail1:
    llama_free(ctx_tgt); llama_model_free(model_tgt);
    llama_free(ctx_dft); llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}

