#include "llama.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator> // Required for std::distance
llama_batch custom_llama_batch_get_one(llama_token * tokens, int32_t n_tokens, llama_pos pos_start, llama_seq_id seq_id) {
    // 1. 初始化一个空的 batch 结构
    // 这里的 1 代表分配 1 个 token 的空间，0 代表不分配额外的 embd 空间
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]    = tokens[i];      // 填入 Token ID
        batch.pos[i]      = pos_start + i;  // 设置位置 (至关重要，决定 KV Cache 存哪)
        batch.n_seq_id[i] = 1;              // 该 token 属于多少个序列
        batch.seq_id[i][0] = seq_id;         // 具体的序列 ID
        batch.logits[i]   = true;           // 是否需要模型输出该位置的 Logits
    }
    batch.n_tokens = n_tokens;
    return batch;
}
llama_token  sample_greedy(const struct llama_vocab * vocab,struct llama_context * ctx, int idx) {
    //const int n_vocab = llama_n_vocab(llama_get_model(ctx));
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
std::vector<llama_token>  llama_tokenize(const llama_context * ctx,std::string text, bool add_bos) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 3);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    tokens.resize(n);
    return tokens;
};
int vocab_match(const struct llama_context * ctx_tgt,const struct llama_context * ctx_dft) {
    const char * test_word = "Hello";
    // 分别获取两个模型的 ID
    llama_token id_tgt = llama_tokenize(ctx_tgt, test_word, false)[0];
    llama_token id_dft = llama_tokenize(ctx_dft, test_word, false)[0];
    printf("Target ID for 'Hello': %d\n", id_tgt);
    printf("Draft ID for 'Hello':  %d\n", id_dft);
    
    if (id_tgt != id_dft) {
        printf("vocab mismatch \n");
        return -1;
    }
    return 0;
}
int main() {
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
    std::string dft_model_path = "/workspace/qwen/models/AMD-Llama-135m.Q8_0.gguf";
    llama_model * model_dft = llama_model_load_from_file(dft_model_path.c_str(), d_params);
    //llama_model * model_dft = llama_model_load_from_file("llama-3-135m.gguf", d_params);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    // 4. 创建 Context (设置多线程)
    auto c_params = llama_context_default_params();
    c_params.n_ctx = 2048;
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
#if 0
    if(vocab_match(ctx_tgt, ctx_dft)){
        llama_free(ctx_tgt); llama_model_free(model_tgt);
        llama_free(ctx_dft); llama_model_free(model_dft);
        llama_backend_free();
        return -1;
    }
#endif
    // 5. 协同推理循环
    //std::vector<llama_token> tokens = { llama_vocab_bos(vocab_tgt) };
       // 5. 准备 Prompt 和初始 Token
    std::string prompt = "The capital of France is";
    std::vector<llama_token> tokens = tokenize(vocab_tgt, prompt, true);
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
        batch_pre.seq_id[i][0] = 0;
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
#if 0
    while (tokens.size() < 100) {
        int p_start = n_past;
        // 【步骤 A: 小模型
        std::vector<llama_token> draft_tokens;
        for (int i = 0; i < n_draft; ++i) {
            llama_batch b_dft = custom_llama_batch_get_one(&tokens.back(), 1, p_start + i - 1, 0);
                   // 构造单 token batch。注意：必须开启 logits 才能采样
            //llama_batch b_dft = llama_batch_init(1, 0, 1);
            //b_dft.token[0]  = tokens.back();
            //b_dft.pos[0]    = p_start + i - 1;
            //b_dft.n_seq_id[0] = 1;
            //b_dft.seq_id[0][0] = 0;
            //b_dft.logits[0] = true; // 开启 logits 以便小模型采样
            printf("dft decode n_tokens %d !\n",b_dft.n_tokens);

            if (b_dft.n_tokens > 0) {
                   if (llama_decode(ctx_dft, b_dft) != 0) {
                       printf("dft decode failed!\n");
                       break; 
                   }
                   else {
                       printf("dft decode succ!\n");
                   }
            }
            llama_token id = llama_sample_token_greedy(vocab_dft, llama_get_logits(ctx_dft));
            draft_tokens.push_back(id);
            tokens.push_back(id);
            llama_batch_free(b_dft);
        }

        // 【步骤 B: 大模型批量验证】
        // CPU 处理 Batch 同样比单 Token 串行快，因为它利用了向量化指令 (AVX2/AVX512)
        llama_batch b_tgt = llama_batch_init(n_draft, 0, 1);
        for (int i = 0; i < n_draft; ++i) {
            llama_batch_add(b_tgt, draft_tokens[i], p_start + i - 1, {0}, true);
        }

        if (b_tgt.n_tokens > 0) {
               if (llama_decode(ctx_dft, b_tgt) != 0) {
                   printf("tgt decode failed!\n");
                   break; 
               }
        }
        // 【步骤 C: 验证并剪枝 KV Cache】
        int n_accept = 0;
        for (int i = 0; i < n_draft; ++i) {
            llama_token tgt_id = llama_sample_token_greedy(vocab_tgt, llama_get_logits_ith(ctx_tgt, i));
            if (tgt_id == draft_tokens[i]) {
                n_accept++;
            } else {
                tokens[p_start + i] = tgt_id; // 修正错误的 token
                break;
            }
        }
        llama_batch_free(b_tgt);
        // 同步位置，清理多余的 KV Cache (前缀复用的核心操作)
        int n_keep = p_start + n_accept;
        tokens.resize(n_keep + 1);
        llama_memory_seq_rm (llama_get_memory(ctx_dft),0, n_keep, -1);
        llama_memory_seq_rm (llama_get_memory(ctx_tgt),0, n_keep, -1);
        std::cout << "Step speedup: " << n_accept << " tokens accepted." << std::endl;
    }
#else
while (tokens.size() < 20) {
        std::vector<llama_token> draft_tokens;
        llama_token last_confirmed = tokens.back();
        // --- A. 小模型预测 ---
        for (int i = 0; i < n_draft; ++i) {
            llama_batch b_dft = llama_batch_init(1, 0, 1);
            //b_dft.token[0]  = tokens.back();
            b_dft.token[0] = (i == 0) ? last_confirmed : tokens.back();
            b_dft.pos[0]    = n_past + i -1; // 修正：位置必须衔接上一个 token
            b_dft.n_seq_id[0] = 1;
            b_dft.seq_id[0][0] = 0;
            b_dft.logits[0] = true;
            b_dft.n_tokens = 1;
            if (llama_decode(ctx_dft, b_dft) != 0) {
                fprintf(stderr, "Draft decode failed at pos %d\n", b_dft.pos[0]);
                return 1;
            }
            
            llama_token sampled = sample_greedy(vocab_dft,ctx_dft, 0);
            draft_tokens.push_back(sampled);
            tokens.push_back(sampled);
            llama_batch_free(b_dft);
        }

        // --- B. 大模型批量验证 ---
        llama_batch b_tgt = llama_batch_init(n_draft, 0, 1);
        for (int i = 0; i < n_draft; ++i) {
            //b_tgt.token[i]    = draft_tokens[i];
            b_tgt.token[i]  = (i == 0) ? last_confirmed : draft_tokens[i-1];
            b_tgt.pos[i]      = n_past+ i-1; // 验证位置从 n_past 开始
            b_tgt.n_seq_id[i] = 1;
            b_tgt.seq_id[i][0] = 0;
            b_tgt.logits[i]   = true;
        }
        b_tgt.n_tokens = n_draft;
        if (llama_decode(ctx_tgt, b_tgt) != 0) {
            fprintf(stderr, "Target verify failed!\n");
            return 1;
        }

        // --- C. 验证与回退 ---
        int n_accept = 0;
        llama_token next_token = 0;
        for (int i = 0; i < n_draft; ++i) {
            llama_token tgt_id = sample_greedy(vocab_tgt,ctx_tgt, i);
            printf("Draft ID: %d | Target ID: %d\n", draft_tokens[i], tgt_id);
            if (tgt_id == draft_tokens[i]) {
                n_accept++;
                last_confirmed = tgt_id;
                tokens.push_back(tgt_id);
                //printf("accept draft token  %d \n", tgt_id);
            } else {
                //printf("accept tgt token  %d \n", tgt_id);
                //tokens[n_past + i] = tgt_id; // 修正为大模型的正确 Token
                next_token = tgt_id;
                break;
            }
        }

        // 如果全部猜对，大模型还需多给一个额外的 Token（投机采样的红利）
        if (n_accept == n_draft) {
             // 实际上大模型在最后一位也输出了预测，可以采样出来
             // 但为简化逻辑，这里让下一轮循环处理
             next_token = sample_greedy(vocab_tgt,ctx_tgt, n_draft - 1); 
        }

        // 修正并同步
        tokens.push_back(next_token);
        // --- D. 【核心修复】同步清理 KV Cache ---
        n_past += n_accept+1; // 更新大模型确认过后的位置
        //tokens.resize(n_past + 1); // 保留确认的部分 + 一个修正 Token

        llama_memory_seq_rm (llama_get_memory(ctx_dft),0, n_past, -1);
        llama_memory_seq_rm (llama_get_memory(ctx_tgt),0, n_past, -1);
        //std::cout << "Step speedup: " << n_accept << " tokens accepted." << std::endl;
        // 清理掉 KV Cache 中所有 n_past 之后（即猜错或未验证）的残留
        printf(" | Accept: %d/%d | Next Pos: %d\n", n_accept, n_draft, n_past);
        llama_batch_free(b_tgt);
    }

#endif
    std::cout << custom_common_detokenize(ctx_tgt,tokens,true) << std::endl;
fail1:
    // 6. 销毁
    llama_free(ctx_tgt); llama_model_free(model_tgt);
    llama_free(ctx_dft); llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}

