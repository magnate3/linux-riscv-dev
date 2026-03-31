
#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-impl.h" 
#include <iostream>
#include <vector>
#include <cstring>
#define TGT_STREAM 8
#define TGT_STREAM4 4
#define TGT_STREAM3 3
#define TGT_STREAM2 2
#define TGT_STREAM1 1
#define N_SEQ_MAX (TGT_STREAM +1)
void print_kv_physical_info(struct llama_context * ctx) {
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
    for (int32_t i = 0; i < n_ctx; ++i) {
        auto & cells = v_cells[i];
        if( cells.size() <=0){
            continue;
        }
        for (uint32_t i = 0; i < cells.size(); ++i) {
             if(cells.seq_count(i)>=1){
                 printf(" cells[%d] seq count %d \n",i, cells.seq_count(i));
             }
        }
   }
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
                 printf(" cells pos[%d] pos_in %d-%d \t",i,p0,p1);
                 out = true;
             }
             if(cells.seq_count(i)>=1){
                 printf(" cells seq count %d \t",cells.seq_count(i));
                 out = true;
             }
#if 1
             //for(int s = p0; s < p1; ++s){ 
              //std::vector<llama_pos> & cells_pos =  cells.get_cells_pos();
              //for (uint32_t s = 0; s < cells_pos.size(); ++s) {
              for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
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
             if(cells.seq_count(i)>=1){
                 printf(" cells seq count %d \t",cells.seq_count(i));
             }
             printf(" cells pos[%d] pos_in %d-%d \t",i,p0,p1);
             if (cells.seq_has(i, target_s)) {
                 printf(" cells seq[%d] has stream %d \n",i,target_s);
             }
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
    //c_params.kv_unified= true;
    //c_params.kv_unified= false;

    c_params.n_batch = 80;
    c_params.n_ubatch = 32;
    c_params.n_seq_max = N_SEQ_MAX;
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
    int prefix_len = tokens.size();

    std::string sub = "where is Paris";
    std::vector<llama_token> tokens_sub = tokenize(vocab_dft, sub, true);
    int n_past = 0;
    llama_pos pos1=0,pos2=0;
       // 大小模型必须同步完成 Prefill，建立相同的 KV Cache 基础
    llama_batch batch_pre = llama_batch_init(tokens.size(), 0, N_SEQ_MAX);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch_pre.token[i] = tokens[i];
        batch_pre.pos[i]   = i;
        batch_pre.n_seq_id[i] = 1;
        batch_pre.seq_id[i][0] = TGT_STREAM;
        batch_pre.logits[i] = (i == tokens.size() - 1); // 只有最后一位需要 logits
    }
    batch_pre.n_tokens =  tokens.size();
    if (llama_decode(ctx_dft, batch_pre) != 0) {
        fprintf(stderr, "Draft decode failed at pos \n");
        llama_batch_free(batch_pre);
        llama_free(ctx_dft); 
        llama_model_free(model_dft);
        llama_backend_free();
        exit(0);
    }
#if 0
    n_past = tokens.size(); 
    printf("n_past %d \n",n_past);
    print_kv_physical_info(ctx_dft,TGT_STREAM,0,n_past -1);
    pos1 = 0;
    pos2 = 2;
    llama_memory_seq_rm (llama_get_memory(ctx_dft),TGT_STREAM, 0, 2);
    //print_kv_physical_info(ctx_dft,TGT_STREAM,0,n_past -1);
    print_kv_physical_info(ctx_dft,TGT_STREAM,pos2,n_past -1);
#else
    n_past = tokens.size(); 
    printf("n_past %d \n",n_past);
    //print_kv_physical_info(ctx_dft,TGT_STREAM,0,n_past -1);
    pos1 = 0;
    pos2 = 2;
    llama_memory_seq_cp(llama_get_memory(ctx_dft),TGT_STREAM,TGT_STREAM2,-1, -1);
    llama_memory_seq_cp(llama_get_memory(ctx_dft),TGT_STREAM,TGT_STREAM4,-1, -1);
    print_kv_physical_info(ctx_dft,TGT_STREAM4,TGT_STREAM2,pos1,n_past -1);
    llama_memory_seq_rm(llama_get_memory(ctx_dft),TGT_STREAM2,prefix_len, -1);
    llama_batch batch_sub = llama_batch_init(tokens_sub.size(), 0, N_SEQ_MAX);
#if 1
    for (size_t i = 0; i < tokens_sub.size(); ++i) {
        batch_sub.token[i] = tokens_sub[i];
        batch_sub.pos[i]   = i+prefix_len;
        batch_sub.n_seq_id[i] = 1;
        batch_sub.seq_id[i][0] = TGT_STREAM2;
        batch_sub.logits[i] = (i == tokens_sub.size() - 1); // 只有最后一位需要 logits
    }
    batch_sub.n_tokens =  tokens_sub.size();
    if (llama_decode(ctx_dft, batch_sub) != 0) {
        fprintf(stderr, "Draft decode failed at pos \n");
        llama_batch_free(batch_pre);
        llama_batch_free(batch_sub);
        llama_free(ctx_dft); 
        llama_model_free(model_dft);
        llama_backend_free();
        return -1;
    }
    print_kv_physical_info(ctx_dft,TGT_STREAM,0,n_past -1);
    //print_kv_physical_info(ctx_dft,TGT_STREAM,0,n_past -1);
    //print_kv_physical_info(ctx_dft,TGT_STREAM2,0,n_past -1);
    //llama_memory_seq_rm (llama_get_memory(ctx_dft),TGT_STREAM2, pos1, pos2);
    //print_kv_physical_info(ctx_dft,TGT_STREAM,pos1,n_past -1);
    //print_kv_physical_info(ctx_dft,TGT_STREAM2,pos1,n_past -1);
    //print_kv_physical_info(ctx_dft,TGT_STREAM2,pos2,n_past -1);
    //print_kv_physical_info(ctx_dft,TGT_STREAM,TGT_STREAM2,pos1,n_past -1);
    //llama_memory_seq_rm(llama_get_memory(ctx_dft),TGT_STREAM2,-1, -1);
    //llama_memory_seq_cp(llama_get_memory(ctx_dft),TGT_STREAM,TGT_STREAM2,-1, -1);
    print_kv_physical_info(ctx_dft,TGT_STREAM2,0,tokens.size()  + tokens_sub.size() -1);
    print_kv_physical_info(ctx_dft,TGT_STREAM,TGT_STREAM2,pos1,n_past -1);
#endif
#if 0
    llama_batch batch_sub2 = llama_batch_init(tokens.size() + tokens_sub.size(), 0, N_SEQ_MAX);
    batch_sub2.n_tokens = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch_sub2.token[i] = tokens[i];
        batch_sub2.pos[i]   = i;
        batch_sub2.n_seq_id[i] = 1;
        batch_sub2.seq_id[i][0] = TGT_STREAM1;
        batch_sub2.logits[i] = (i == tokens.size() - 1); // 只有最后一位需要 logits
    }
    batch_sub2.n_tokens =  tokens.size();
    int index = tokens.size();
    for (size_t i = 0; i < tokens_sub.size(); ++i) {
        batch_sub2.token[index] = tokens_sub[i];
        batch_sub2.pos[index]   = index;
        batch_sub2.n_seq_id[index] = 1;
        batch_sub2.seq_id[index][0] = TGT_STREAM3;
        batch_sub2.logits[index] = (i == tokens_sub.size() - 1); // 只有最后一位需要 logits
        ++ index;
    }
    batch_sub2.n_tokens +=  tokens_sub.size();
    if (llama_decode(ctx_dft, batch_sub2) != 0) {
        fprintf(stderr, "Draft decode batch_sub2 failed \n");
        goto fail1;
    }
    pos1 = 0;
    print_kv_physical_info(ctx_dft,TGT_STREAM1,TGT_STREAM3,pos1,tokens.size() + tokens_sub.size());
    //print_kv_physical_info(ctx_dft,TGT_STREAM1,pos1,tokens.size());
    llama_batch_free(batch_sub2);
#endif
#endif
fail1:
    llama_batch_free(batch_pre);
    llama_batch_free(batch_sub);
    llama_free(ctx_dft); 
    llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}

