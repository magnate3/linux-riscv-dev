
#include "common.h"
#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-impl.h" 
#include <iostream>
#include <vector>
#include <cstring>
#define TGT_STREAM 8
#define TGT_STREAM3 3
#define TGT_STREAM2 2
#define TGT_STREAM1 1
#define N_SEQ_MAX (TGT_STREAM +1)
struct token_position {
    size_t seq_id;
    size_t index;
    token_position() : seq_id(0), index(0) {}
    token_position(size_t s, size_t i) : seq_id(s), index(i) {}

    std::string to_string() const {
        return "{ seq_id: " + std::to_string(seq_id) + ", index: " + std::to_string(index) + " }";
    }
};
std::string string_from_batch(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, batch.token[i]);

        detokenized.erase(
                std::remove_if(
                    detokenized.begin(),
                    detokenized.end(),
                    [](const unsigned char c) { return !std::isprint(c); }),
                detokenized.end());

        buf << "\n"          << std::to_string(i)
            << ", token '"   << detokenized << "'"
            << ", pos "      << std::to_string(batch.pos[i])
            << ", n_seq_id " << std::to_string(batch.n_seq_id[i])
            << ", seq_id "   << std::to_string(batch.seq_id[i][0])
            << ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";


    return buf.str();
}

void print_batch(llama_batch batch) {
    fprintf(stderr, "batch.n_tokens: %d\n", batch.n_tokens);
    fprintf(stderr, "batch.tokens: [");
    for (int i = 0; i < batch.n_tokens; i++) {
        fprintf(stderr, "%d, ", batch.token[i]);
    }
    fprintf(stderr, "]\n");
}
std::string string_from_tokens(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        detokenized.erase(
            std::remove_if(
                detokenized.begin(),
                detokenized.end(),
                [](const unsigned char c) { return !std::isprint(c); }),
            detokenized.end());

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}
std::unordered_map<llama_token, std::vector<token_position>> find_common_tokens(
        const std::vector<std::vector<llama_token>>& input_tokens,
        llama_model* model) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (input_tokens.empty()) {
        return {};
    }

    std::unordered_map<llama_token, std::unordered_map<size_t, token_position>> token_positions;
    for (size_t seq_id = 0; seq_id < input_tokens.size(); ++seq_id) {
        const auto& current_vec = input_tokens[seq_id];
        for (size_t token_idx = 0; token_idx < current_vec.size(); ++token_idx) {
            llama_token token = current_vec[token_idx];
            if (token_positions[token].find(seq_id) == token_positions[token].end()) {
                token_positions[token][seq_id] = token_position(seq_id, token_idx);
            }
        }
    }

    std::unordered_map<llama_token, std::vector<token_position>> common_tokens;
    for (const auto& entry : token_positions) {
        if (llama_vocab_get_add_bos(vocab) && entry.first == 1) {
            continue;
        }
        if (entry.second.size() > 1) {
            std::vector<token_position> positions;
            positions.reserve(entry.second.size());
            for (const auto& seq_pos : entry.second) {
                positions.push_back(seq_pos.second);
            }
            common_tokens[entry.first] = std::move(positions);
        }
    }

    return common_tokens;
}
// Compute similarity between two token sequences (stub for Phase 3)
// longest common prefix
//size_t common_lcp(const llama_tokens & a, const llama_tokens & b);
//// longet common subsequence
//size_t common_lcs(const llama_tokens & a, const llama_tokens & b);
float compute_similarity(
    const std::vector<llama_token>& a,
    const std::vector<llama_token>& b
) {
    // Longest Common Prefix (LCP) approach
    size_t common_prefix = 0;
    size_t max_len = std::min(a.size(), b.size());
    for (size_t i = 0; i < max_len; i++) {
        if (a[i] == b[i]) {
            common_prefix++;
        } else {
            break;
        }
    }

    if (a.empty() && b.empty()) return 1.0f;
    if (a.empty() || b.empty()) return 0.0f;

    return static_cast<float>(common_prefix) / static_cast<float>(std::max(a.size(), b.size()));
}
void print_common_tokens(std::unordered_map<llama_token, std::vector<token_position>> common_tokens) {
    for (const auto& token_info : common_tokens) {
        printf("Token id [%d] in common at positions:\n", token_info.first);
        for (const auto& pos : token_info.second) {
            printf("  Sequence %zu, Index %zu\n", pos.seq_id, pos.index);
        }
    }
}
std::vector<llama_token> tokenize_prompt(llama_model* model, std::string prompt) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int add_bos_token = llama_vocab_get_add_bos(vocab);
    const bool add_bos  = add_bos_token != -1 ? bool(add_bos_token) :
        (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM); // SPM = SentencePiece Model

    int n_tokens = prompt.length() + add_bos;
    std::vector<llama_token> input_tokens(n_tokens);
    n_tokens = llama_tokenize(vocab,
                              prompt.data(),
                              prompt.length(),
                              input_tokens.data(),
                              input_tokens.size(),
                              true,
                              false);
    if (n_tokens < 0) {
        input_tokens.resize(-n_tokens);
        llama_tokenize(vocab,
                prompt.data(),
                prompt.length(),
                input_tokens.data(),
                input_tokens.size(), add_bos, false);
    } else {
        input_tokens.resize(n_tokens);
    }
    return input_tokens;
}

void print_kv_physical_info(struct llama_context * ctx, const int32_t target_s1, const int32_t target_s2,llama_pos p0, llama_pos p1) {
    // 1. 访问内部的 kv_self 结构
    // kv_self 包含了所有关于物理 Cell 的元数据和 Tensor 指针
    //auto & kv = ctx->kv_self;
    struct llama_memory_i * mem=  llama_get_memory(ctx)->get_kv();
    if(NULL == mem)
    {
	 std::cout << "memory is NULL " <<std::endl;
	 return ;
    }
    //if(mem->get_status() != LLAMA_MEMORY_STATUS_SUCCESS)
    //{
    //     std::cout << "memory is not in successful status " <<std::endl;
    //}
    llama_kv_cache * kv =  dynamic_cast<llama_kv_cache*>(mem);
    const uint32_t n_ctx  = ctx->n_ctx();
    std::vector<llama_kv_cells> & v_cells = kv->get_kv_cells();
    int out = false;
    //std::vector<llama_kv_cells> & v_cells = llama_get_memory(ctx)->get_kv_cells();
    for (int32_t i = 0; i < n_ctx; ++i) {
        if(target_s1 != i && target_s2 != i){
            continue;
        }
        //const auto & cells = v_cells[i];
        auto & cells = v_cells[i];
        //if(cells.get_used()<=0){
        if(cells.size()<=0){
            continue;
        }
        for (uint32_t i = 0; i < cells.size(); ++i) {
             out = false;
             if (!cells.pos_in(i, p0, p1)) {
                  continue;
             }
             else {
                 printf(" cells[%d] pos_in %d-%d \t",i,p0,p1);
                 out = true;
             }
#if 1
             //for(int s = p0; s < p1; ++s){ 
              //for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
              std::vector<llama_pos> & cells_pos =  cells.get_cells_pos();
              for (uint32_t s = 0; s < cells_pos.size(); ++s) {
                 if (cells.seq_has(s, target_s1) && cells.seq_has(s, target_s2)) {
                     // cells.pos_get(idx)
                     printf(" cells seq[%d] has stream %d,%d \t",s,target_s1,target_s2);
                     out = true;
                 }
                 else {
                     //printf("\n");
                 }
             }
#else
              //using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

              //// the bitset seq[i] tells us which sequences are currently occupying the i-th cell
              //std::vector<seq_set_t> seq;
              for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                 if (cells.seq_has(s, target_s1)) {
                     // cells.pos_get(idx)
                     printf(" cells seq[%d] has stream %d\t",s,target_s1);
                     out = true;
                 }
                 if (cells.seq_has(s, target_s2)) {
                     // cells.pos_get(idx)
                     printf(" cells seq[%d] has stream %d\t",s,target_s2);
                     out = true;
                 }
             }
#endif
             if(out){
                 printf("\n");
             }
        }
     }
}
void print_kv_physical_info(struct llama_context * ctx, const int32_t target_s,llama_pos p0, llama_pos p1) {
    // 1. 访问内部的 kv_self 结构
    // kv_self 包含了所有关于物理 Cell 的元数据和 Tensor 指针
    //auto & kv = ctx->kv_self;
    struct llama_memory_i * mem=  llama_get_memory(ctx)->get_kv();
    if(NULL == mem)
    {
	 std::cout << "memory is NULL " <<std::endl;
	 return ;
    }
    //if(mem->get_status() != LLAMA_MEMORY_STATUS_SUCCESS)
    //{
    //     std::cout << "memory is not in successful status " <<std::endl;
    //}
    llama_kv_cache * kv =  dynamic_cast<llama_kv_cache*>(mem);
    const uint32_t n_ctx  = ctx->n_ctx();
    std::vector<llama_kv_cells> & v_cells = kv->get_kv_cells();
    //std::vector<llama_kv_cells> & v_cells = llama_get_memory(ctx)->get_kv_cells();
    for (int32_t i = 0; i < n_ctx; ++i) {
        if(target_s != i){
            continue;
        }
        const auto & cells = v_cells[i];
        //if(cells.get_used()<=0){
        if(cells.size()<=0){
            continue;
        }
        for (uint32_t i = 0; i < cells.size(); ++i) {
             if (!cells.pos_in(i, p0, p1)) {
                 continue;
             }
             printf(" cells pos[%d] pos_in %d-%d \t",i,p0,p1);
             if (cells.seq_has(i, target_s)) {
                 printf(" cells seq[%d] has stream %d \n",i,target_s);
             }
        }
#if 1
        {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                        if (cells.is_empty(i)) {
                            ss += '.';
                        } else {
                            assert(cells.seq_count(i) >= 1);
                            //ss += "cells[" + std::to_string(i)+"]";
                            if (cells.seq_count(i) == 1) {
                                ss += std::to_string(cells.seq_get(i));
                            } else {
                                ss += 'M';
                            }
                        }
                        if (i%256 == 255) {
                            ss += " *";
                            ss += '\n';
                        }
            }    
            printf("\n  seq data :%s\n", ss.c_str());
       }
       {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                std::string cur;
                if (cells.is_empty(i)) {
                    cur = '.';
                } else {
                    //cur = "cells[" +  std::to_string(i) + "]" +  std::to_string(cells.pos_get(i));
                    cur = std::to_string(cells.pos_get(i));
                }
                const int n = cur.size();
                for (int j = 0; j < 5 - n; ++j) {
                    cur += ' ';
                }
                ss += cur;
                if (i%256 == 255) {
                    ss += " *";
                }
                if (i%64 == 63) {
                    ss += '\n';
                }
            }
            printf("\n pos data: %s\n", ss.c_str());
       }
       for (int s = 0; s <  LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_pos_min(s) < 0) {
                continue;
            }

            printf("min[%d] = %5d, max[%d] = %5d\n", s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
      }
#endif
    }
}
// 获取第 il 层、物理索引为 i 的 K 缓存字节偏移
size_t get_kv_th_phys_offset(struct llama_context * ctx, int il, int32_t i) {
#if 0
    auto & kv_self = ctx->kv_self;
    auto & hparams = ctx->model.hparams;

    // 1. 获取基础维度
    // n_embd_gqa 是 Head_Dim * KV_Heads 的结果
    uint32_t n_embd_gqa = hparams.n_embd_gqa(); 
    ggml_type type_k    = kv_self.k->type;

    // 2. 计算 RowSize (单个物理槽位的字节数)
    // 使用 ggml_row_size 兼容所有量化格式 (如 Q8_0, Q4_K)
    size_t row_size = ggml_row_size(type_k, n_embd_gqa);

    // 3. 计算 LayerSize (单层总容量)
    // 注意：n_ctx 是 v_cells.size()，即总物理槽位数
    size_t layer_size = row_size * kv_self.n_ctx;

    // 4. 最终偏移量
    size_t total_offset = (size_t)il * layer_size + (size_t)i * row_size;
    return total_offset;
#endif
    return 0;
}

// 获取实际物理指针
void * get_phys_addr_k(struct llama_context * ctx, int il, int32_t i) {
    size_t offset = get_kv_th_phys_offset(ctx, il, i);
    //return (char *)ctx->kv_self.k->data + offset;
    return NULL;
}

llama_batch create_batch(int size, std::vector<std::vector<llama_token>> input_tokens, llama_model* model) {
    int n_prompts = input_tokens.size();
    printf("Creating new llama_batch with %d sequences\n", n_prompts);

    auto common_tokens = find_common_tokens(input_tokens, model);
    if (common_tokens.empty()) {
        printf("No common tokens found. Beginning of Sequence (BOS) is not considered\n");
    } else {
        print_common_tokens(common_tokens);
    }
    printf("\n");

    // Create a single batch for all prompts.
    llama_batch batch = llama_batch_init(size, 0, n_prompts);

    for (size_t s = 0; s < input_tokens.size(); s++) {
        std::vector<llama_token> prompt_tokens = input_tokens[s];
        printf("Processing prompt %ld, nr tokens: %ld (batch_n_tokens: %d)\n", s, prompt_tokens.size(),  batch.n_tokens);
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            int token_id = prompt_tokens[i];
            int idx = batch.n_tokens;
            printf("  idx: %d, token_id: %d \n", idx, token_id);
            batch.token[idx] = token_id;
            batch.pos[idx] = i;

            /*
            auto it = common_tokens.find(token_id);
            if (it != common_tokens.end()) {
                std::vector<token_position> tps = it->second;
                batch.n_seq_id[idx] = tps.size();
                for (size_t j = 0; j < tps.size(); j++) {
                    batch.seq_id[idx][j] = tps[j].seq_id;
                }
            } else {
            */
                batch.n_seq_id[idx] = 1;
                batch.seq_id[idx][0] = s;  // the sequence id
            /*}
            printf("    n_seq_id: %u\n", batch.n_seq_id[idx]);
            for (int i = 0; i < batch.n_seq_id[idx]; i++) {
                printf("    seq_id[%d]: %u\n", i, batch.seq_id[idx][i]);
            }*/
            batch.logits[idx] = i == prompt_tokens.size() - 1;
            batch.n_tokens++;
            //printf("idx: %4d, token: %6d, seq_id: %ld, logits: %d\n", idx, token_id, s, batch.logits[idx]);
        }
        printf("\n");
    }
    return batch;
}

int main() {
    // 1. 初始化后端 (CPU 模式)
    llama_backend_init();

    // 2. 加载大模型 (Target) - CPU 优化配置
    auto t_params = llama_model_default_params();
    t_params.n_gpu_layers = 0; // 强制 0 层在 GPU，确保纯 CPU 运行
    // 3. 加载小模型 (Draft) - 同样强制 CPU
    auto d_params = llama_model_default_params();
    d_params.n_gpu_layers = 0;
    std::string dft_model_path = "/workspace/qwen/models/AMD-Llama-135m.Q8_0.gguf";
    llama_model * model_dft = llama_model_load_from_file(dft_model_path.c_str(), d_params);
    //llama_model * model_dft = llama_model_load_from_file("llama-3-135m.gguf", d_params);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    // 4. 创建 Context (设置多线程)
    auto c_params = llama_context_default_params();
    c_params.n_ctx = 1024;
    // 关键：针对 CPU 逻辑核心数进行优化（通常设为物理核心数）
    c_params.n_threads = 8; 
    c_params.n_threads_batch = 8;
    c_params.n_batch = 80;
    c_params.n_ubatch = 32;
    c_params.n_seq_max = N_SEQ_MAX;
    llama_context * ctx_dft = llama_init_from_model(model_dft, c_params);
    std::string prompt1 = "What is the capital of Sweden?";
    //std::string prompt2 = "How many r's are there in strawberry?";
    std::string prompt2 = "What is the capital of France? the the capital of France is the largetst city of France";
    std::vector<llama_token> input_tokens1 = tokenize_prompt(model_dft, prompt1);
    std::vector<llama_token> input_tokens2 = tokenize_prompt(model_dft, prompt2);
    //std::vector<llama_token> input_tokens3 = tokenize_prompt(model, prompt3);

    llama_batch batch = create_batch(512, {input_tokens1, input_tokens2}, model_dft);
    print_batch(batch);
    std::cout << string_from_batch(ctx_dft,batch) <<std::endl;

    //debug_ubatch(ctx,*model,batch);

    if (llama_decode(ctx_dft, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }
    print_kv_physical_info(ctx_dft,0,1,0,5);
    print_kv_physical_info(ctx_dft,0,0,5);
    print_kv_physical_info(ctx_dft,1,0,5);
fail1:
    llama_batch_free(batch);
    llama_free(ctx_dft); 
    llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}

