#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-impl.h" 

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#define MAIN_STREAM 0
#define N_SEQ_MAX 64

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello my name is\" -n 32 -np 4\n", argv[0]);
    LOG("\n");
}
// 查看Top-10候选token
void print_top_candidates(common_sampler * sampler, int top_n = 10) {
    // 获取排序后的候选
    auto * candidates = common_sampler_get_candidates(sampler, true);
    
    printf("Top %d candidates:\n", top_n);
    for (int i = 0; i < std::min(top_n, (int)candidates->size); i++) {
        const auto & cand = candidates->data[i];
        printf("  %2d: token=%5d, prob=%.4f, logit=%.4f\n",
               i + 1, cand.id, cand.p, cand.logit);
    }
}
common_sampler * init_speculative_sampler(const llama_model * model, float temp) {
    struct common_params_sampling params;
    //params.temp = 0;         // 投机采样中，大小模型的温度建议保持一致
    params.temp = temp;         // 投机采样中，大小模型的温度建议保持一致
    params.top_k = 40;          // 常见的采样参数
    params.top_p = 0.95f;
    params.penalty_repeat = 1.1f;     
    params.penalty_last_n = 64; 
    return common_sampler_init(model, params);
}

void print_kv_physical_info(struct llama_context * ctx, const int32_t target_s,const int32_t max_target_s,llama_pos p0, llama_pos p1) {
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
             //if (cells.seq_has(i, target_s)) {
             //    printf(" cells seq[%d] has stream %d ",i,target_s);
             //}
             for(int j =target_s; j <= max_target_s; ++j ){
                 if (cells.seq_has(i, j)) {
                     if(j ==target_s){
                         printf(" cells seq[%d] has stream %d ",i,j);
                     }
                     else {
                         printf(" ,%d ",j);
                     }
                 }
             }
             printf("\n");
        }
#if 0
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
#endif
       for (int s = 0; s <  LLAMA_MAX_SEQ; ++s) {
            if (cells.seq_pos_min(s) < 0) {
                continue;
            }

            printf("min[%d] = %5d, max[%d] = %5d\n", s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
      }
    }
}
int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "Hello my name is";
    params.n_predict = 32;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    common_init();

    // number of parallel batches
    int n_parallel = params.n_parallel;

    // total length of the sequences including the prompt
    int n_predict = params.n_predict;
    
    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = common_tokenize(vocab, params.prompt, true);

    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size())*n_parallel;

    // initialize the context

    llama_context_params ctx_params = common_context_params_to_llama(params);

    ctx_params.n_ctx   = 4096*4;
    //ctx_params.n_ctx   = n_kv_req*4;
    ctx_params.n_batch = std::max(n_predict, n_parallel);
    ctx_params.n_seq_max = N_SEQ_MAX; 
    ctx_params.offload_kqv = true;
    ctx_params.n_batch = 512; 
    ctx_params.kv_unified =  true;
#if 0
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    std::vector<llama_sampler_seq_config> sampler_configs;

    for (int32_t i = 0; i < n_parallel; ++i) {
        llama_sampler * smpl = llama_sampler_chain_init(sparams);

        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.sampling.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.sampling.top_p, params.sampling.min_keep));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp (params.sampling.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist (params.sampling.seed));

        sampler_configs.push_back({ i, smpl });
    }

    if (params.sampling.backend_sampling) {
        ctx_params.samplers   = sampler_configs.data();
        ctx_params.n_samplers = sampler_configs.size();
    }
#else
    std::vector<common_sampler*> sampler_configs;
    for (int s = 0; s < n_parallel; ++s) {
        auto smpl = init_speculative_sampler(model, 0.8f);
        sampler_configs.push_back( smpl);
    }
    common_sampler * smpl =  sampler_configs.front(); 
#endif
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);

    LOG_INF("\n%s: n_predict = %d, n_ctx = %d, n_batch = %u, n_parallel = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, ctx_params.n_batch, n_parallel, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_ERR("%s: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n", __func__,  n_kv_req);
        LOG_ERR("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    LOG("\n");

    for (auto id : tokens_list) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    // create a llama_batch
    // we use this object to submit token data for decoding
    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, n_parallel);
    //llama_batch batch = llama_batch_init(std::max(tokens_list.size(), (size_t) n_parallel), 0, n_parallel);
#if 1
    std::vector<llama_seq_id> seq_ids(n_parallel, 0);
    for (int32_t i = 0; i < n_parallel; ++i) {
        seq_ids[i] = i;
    }

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        common_batch_add(batch, tokens_list[i], i, seq_ids, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        common_batch_clear(batch);
        common_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
    }
#else

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); ++i) {
        common_batch_add(batch, tokens_list[i], i, {MAIN_STREAM}, false);
    }
    GGML_ASSERT(batch.n_tokens == (int) tokens_list.size());
#endif
    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    //// assign the system KV cache to all parallel sequences
    //// this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    //for (int32_t i = 1; i < n_parallel; ++i) {
    //    llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
    //}

    if (n_parallel > 1) {
        LOG("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur    = batch.n_tokens;
    int n_decode = 0;
    int n_past_tgt = batch.n_tokens;
    int s_parent = MAIN_STREAM;
    int n_seq_cur  = 1;
    const auto t_main_start = ggml_time_us();
    int n_seq_dft =  n_parallel;
    int n_draft =  n_parallel;
    float p_draft_split = n_seq_dft <= 1 ? 0 : 0.2f;
    llama_token last_token_id = tokens_list.back();
    std::vector<llama_token > i_last_token(n_parallel,last_token_id);
    print_kv_physical_info(ctx,MAIN_STREAM,n_parallel,0,n_cur);
    while (n_cur <= n_predict) {
        n_seq_cur = 1;
        // prepare the next batch
        common_batch_clear(batch);
        //if (sampler_configs[0]) {
        //    common_sampler_free(sampler_configs[0]);
        //}
        //sampler_configs[0] = common_sampler_clone(smpl);
        // sample the next token for each parallel sequence / stream
#if 1
        std::vector<bool> skips(n_seq_dft, false);
        std::vector<bool> drafting(n_seq_dft, true);
        last_token_id = i_last_token[0];
        //common_batch_add(batch, last_token_id, n_past_tgt, { 0 }, true);
#endif
        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0 || skips[i]) {
                // the stream has already finished
                continue;
            }
            std::vector<int> sa(1, i);
            //last_token_id = i_last_token[i];
            //common_batch_add(batch, last_token_id, n_past_tgt, { i }, true);
            //const llama_token new_token_id = llama_sampler_sample(sampler_configs[i].sampler, ctx, i_batch[i]);
            
            //const llama_token new_token_id = common_sampler_sample(sampler_configs[i], ctx, i_batch[i], true);
            //common_sampler_sample(sampler_configs[i], ctx, 0, true);
            common_sampler_sample(sampler_configs[i], ctx, i_batch[i], true);
            const auto * cur_p = common_sampler_get_candidates(sampler_configs[i], true);
            // is it an end of generation? -> mark the stream as finished
            
             // attempt to split the branch if the probability is high enough
           for (int f = 1; f < 8; ++f) {
               if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                   if(i == n_seq_cur){
                       ++ n_seq_cur;
                   }
                   else {
                       printf("n_cur %d splitting seq %3d into %3d\n",n_cur, i, n_seq_cur);
                  
                       llama_memory_seq_rm(llama_get_memory(ctx),    n_seq_cur, -1, -1);
                       llama_memory_seq_cp(llama_get_memory(ctx), i, n_seq_cur, -1, -1);

                       // all previous tokens from this branch are now also part of the new branch
                       for (int t = 0; t < batch.n_tokens; ++t) {
                           for (int p = 0; p < batch.n_seq_id[t]; ++p) {
                               if (batch.seq_id[t][p] == i) {
                                   batch.seq_id[t][batch.n_seq_id[t]] = n_seq_cur;
                                   batch.n_seq_id[t]++;
                                   break;
                               }
                           }
                       }
                       //if (sampler_configs[n_seq_cur]) {
                       //   common_sampler_free(sampler_configs[n_seq_cur]);
                       //}
                       //sampler_configs[n_seq_cur] = common_sampler_clone(sampler_configs[i]);
                       sa.push_back(n_seq_cur);
                       skips[n_seq_cur] = true;
                       ++ n_seq_cur;
                    }
                }
                else {
                     break;
                }
           }

            //streams[i] += common_token_to_piece(ctx, new_token_id);

            //i_batch[i] = n_past_tgt -1;
            i_batch[i] = batch.n_tokens ;

#if 0
            // push this new token for next evaluation
            common_batch_add(batch, new_token_id, n_cur, { i }, true);
#else
           /*for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
               printf(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                       k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx, cur_p->data[k].id).c_str());
           }
           */
           for (int is = 0; is < (int) sa.size(); ++is) {
                const llama_token id = cur_p->data[is].id;
                const int s = sa[is];
                //common_batch_add(batch, new_token_id, n_cur, { i }, true);
                if (llama_vocab_is_eog(vocab, id) || n_cur == n_predict) {
                    i_batch[i] = -1;
                    LOG("\n");
                    if (n_parallel > 1) {
                        LOG_INF("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                    }

                    continue;
                }
                i_last_token[i] = id;
                streams[i] += common_token_to_piece(ctx, id);
                common_sampler_accept(sampler_configs[i], id, true);
                common_batch_add(batch, id, n_cur, { s }, true);
                //if (batch_tgt.n_tokens > n_draft) {
                //        drafts[s].drafting = false;
                //}
           }
#endif
            n_decode += 1;
            //common_sampler_accept(sampler_configs[i], new_token_id, true);
            //print_top_candidates(sampler_configs[i], 5);
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;
        n_past_tgt += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    if (n_parallel >= 1) {
        LOG("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG("sequence %d:\n\n%s%s\n\n", i, params.prompt.c_str(), streams[i].c_str());
        }
    }

    print_kv_physical_info(ctx,MAIN_STREAM,n_parallel,0,n_cur);
    const auto t_main_end = ggml_time_us();

    LOG_INF("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    LOG("\n");
    //llama_perf_sampler_print(sampler_configs[0].sampler);
    for (int s = 0; s < n_parallel; ++s) {
        common_sampler_free(sampler_configs[s]);
    }
    llama_perf_context_print(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);


    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
