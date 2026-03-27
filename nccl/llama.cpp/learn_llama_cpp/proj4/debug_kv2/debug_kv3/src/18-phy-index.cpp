
#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-impl.h" 
#include <iostream>
#include <vector>
#include <cstring>

void print_kv_physical_info(struct llama_context * ctx, int32_t target_pos) {
    // 1. 访问内部的 kv_self 结构
    // kv_self 包含了所有关于物理 Cell 的元数据和 Tensor 指针
    //auto & kv = ctx->kv_self;
    struct llama_memory_i * mem=  llama_get_memory(ctx)->get_kv();
    if(NULL == mem)
    {
	 std::cout << "memory is NULL " <<std::endl;
	 return ;
    }
    llama_kv_cache * kv =  dynamic_cast<llama_kv_cache*>(mem);
    const uint32_t n_ctx  = ctx->n_ctx();
    std::vector<llama_kv_cells> & v_cells = kv->get_kv_cells();
    //std::vector<llama_kv_cells> & v_cells = llama_get_memory(ctx)->get_kv_cells();
    for (int32_t i = 0; i < n_ctx; ++i) {
        const auto & cells = v_cells[i];
        //if(cells.get_used()<=0){
        if(cells.size()<=0){
            continue;
        }

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
#if 0
        if (cell.pos < 0) {
            continue; 
        }
        /// 3. 物理索引 i 存储了有效数据
        // 打印：物理索引 -> 逻辑位置 (Pos) | 所属序列 (Seq ID)
        //printf("Index[%5d]: Pos(%4d) | SeqCount(%zu)\n", i, cell.pos, cell.seq_id.size());
        
        // 如果开启了 Prefix Caching，一个 Cell 可能属于多个 Seq
        for (auto sid : cell.seq_id) {
             // 处理共享逻辑...
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
    c_params.n_ctx = 16;
    // 关键：针对 CPU 逻辑核心数进行优化（通常设为物理核心数）
    c_params.n_threads = 8; 
    c_params.n_threads_batch = 8;

    llama_context * ctx_dft = llama_init_from_model(model_dft, c_params);
    auto tokenize = [&](const llama_vocab * vocab,std::string& text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
       // 5. 准备 Prompt 和初始 Token
    std::string prompt = "The capital of France is";
    std::vector<llama_token> tokens = tokenize(vocab_dft, prompt, true);
    
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
    n_past = tokens.size(); 
    print_kv_physical_info(ctx_dft,0);
fail1:
    llama_free(ctx_dft); 
    llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}

