#include "llama.h"
#include "common.h"
#if DEBUG_GRAPH
#include "llama-graph.h"
#include "llama-model.h"
#endif
#define  DEBUG_KV 1
#if DEBUG_KV
#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#endif
#include "llama-sampling.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <iostream>

std::string string_from_batch(const struct llama_context * ctx, const struct llama_batch & batch);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
std::string string_from_ubatch(const struct llama_context * ctx, const struct llama_ubatch & batch);
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

struct token_position {
    size_t seq_id;
    size_t index;
    token_position() : seq_id(0), index(0) {}
    token_position(size_t s, size_t i) : seq_id(s), index(i) {}

    std::string to_string() const {
        return "{ seq_id: " + std::to_string(seq_id) + ", index: " + std::to_string(index) + " }";
    }
};

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
#if DEBUG_GRAPH
void debug_llm_build_qwen3(const llama_model & model) {

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();
}
#endif
int debug_batch(struct llama_context * ctx, const struct llama_model& model) {
    // Process sequence
    std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.seq_id[i][0] = 0;
        batch.n_seq_id[i] = 1;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = tokens.size();
    
    // Decode (reservoir state automatically managed)
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
    }
    return 0;
}
void debug_set_input_k_idxs(bool v_trans, const struct llama_model& model ,llama_kv_cache & kv_cache, const llama_ubatch * ubatch, const llama_kv_cache::slot_info& sinfo) {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    //GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    //int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*kv_cache.get_size();

	std::cout<< "stream " << s << " idx, k data ";
        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            //data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
	    std::cout<< offs + sinfo.idxs[s][i]  << " ";
        }
	std::cout<< std::endl;
    }
}
void debug_set_input_v_idxs(bool v_trans, const struct llama_model& model ,llama_kv_cache & kv_cache, const llama_ubatch * ubatch, const llama_kv_cache::slot_info& sinfo) {
    const uint32_t n_tokens = ubatch->n_tokens;
    //GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    //GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    //int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_cache.get_size();
	    std::cout<< "stream " << s << " idx, v data ";

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                //data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
	        std::cout<< offs + sinfo.idxs[s][i]  << " ";

            }
	    std::cout<< std::endl;
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = kv_cache.get_size();

        const int64_t n_embd_v_gqa = model.hparams.n_embd_v_gqa_max();
        //const int64_t n_embd_v_gqa = kv_cache.hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
	        std::cout<< "stream " << s << " idx, kv data ";
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    //data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
	            std::cout<< offs + j*kv_size + sinfo.idxs[s][i]  << " ";
                }
	        std::cout<< std::endl;
            }
        }
    }
}
// refer to llama_context::encode
int debug_ubatch(const struct llama_context * ctx, const struct llama_model& model, const struct llama_batch & batch_inp) {
#if DEBUG_KV_IN_CLASS
    return 0;
#endif
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();
    std::unique_ptr<llama_batch_allocr> balloc= std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd());

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, ctx->get_cparams().kv_unified ? LLAMA_MAX_SEQ : ctx->get_cparams().n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);
    std::cout<<string_from_ubatch(ctx,ubatch)<<std::endl;
#if 1
    struct llama_memory_i * mem=  llama_get_memory(ctx)->get_kv();
    if(NULL == mem)
    {
	 std::cout << "memory is NULL " <<std::endl;
	 return -1;
    }
   llama_kv_cache * kv =  dynamic_cast<llama_kv_cache*>(mem);
   llama_kv_cache::slot_info sinfo = kv->find_slot(ubatch, false);
   kv->debug_set_input_v_idxs(NULL,&ubatch, sinfo);
   kv->debug_set_input_k_idxs(NULL,&ubatch, sinfo);
#endif
#if 0
   int kv_size =16384;
   int n_seq_max = 2;
   int n_pad = 2;
   int seq_id = 0;
   int seq_id2 = 1;
    auto kv_cache = std::make_unique<llama_kv_cache>(
        model, GGML_TYPE_F16, GGML_TYPE_F16,
        /*v_trans=*/true, /*offload=*/true, /*unified=*/false,
        kv_size, n_seq_max, n_pad,
        /*n_swa=*/0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr);
    const auto n_kv = kv_cache.get()->get_n_kv(sinfo);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
	    std::cout<< "stream " << s  << " idxs size : " << sinfo.idxs[s].size() <<  "  number of streams = " << sinfo.s1 - sinfo.s0 + 1 << std::endl;
    }
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            //auto & cells = v_cells[sinfo.strm[s]];
            //auto & head  = v_heads[sinfo.strm[s]];

            //cells.set(sinfo.idxs[s], it->v_cells[s]);
            //head = it->v_heads_old[s];

	  std::cout<< "stream " << s << " idx ";
	 for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
		 const auto idx = sinfo.idxs[s][ii];
		  //cells.pos_set(idx, ubatch.pos[i]);
		  std::cout<< idx  << ","; 
         }
	  std::cout<< std::endl;
   }
   debug_set_input_v_idxs(false,model,*(kv_cache.get()), &ubatch, sinfo);
   debug_set_input_k_idxs(false,model,*(kv_cache.get()), &ubatch, sinfo);
   //debug_set_input_v_idxs(true,model,*(kv_cache.get()), &ubatch, sinfo);
#endif
 #if 0
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
    for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }
#endif
    return 0;
}
void debug_kv_cache(const struct llama_model& model, const struct llama_batch & batch) {

	 //llama_memory_context_ptr mctx;
	 //const auto & ubatch = mctx->get_ubatch();
	llama_kv_cache_context *mctx;
	// Create KV cache
   int kv_size =16384;
   int n_seq_max = 2;
   int n_pad = 2;
   int seq_id = 0;
   int seq_id2 = 1;
   int delta = 3;
    auto kv_cache = std::make_unique<llama_kv_cache>(
        model, GGML_TYPE_F16, GGML_TYPE_F16,
        /*v_trans=*/true, /*offload=*/true, /*unified=*/false,
        kv_size, n_seq_max, n_pad,
        /*n_swa=*/0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr);
    
    // Manage sequences
    kv_cache->seq_add(seq_id, 0, 16,delta);  
    //kv_cache->seq_add(seq_id, 0, -1,0);  
    kv_cache->seq_add(seq_id2, 0, -1,0);  
    kv_cache->seq_rm(seq_id, 0, -1);  // remove all positions for a sequence
    kv_cache->seq_cp(0, 1, 0, -1);    // copy sequence 0 to sequence 1
    kv_cache->seq_keep(seq_id);        // keep only this sequence
    kv_cache->clear(true);             // clear all cached data
}
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

std::string string_from_ubatch(const struct llama_context * ctx, const struct llama_ubatch & batch) {
    std::stringstream buf;

    buf << "ubatch [ ";

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
            << ", seq_id "   << std::to_string(batch.seq_id[i][0]);
            //<< ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";


    return buf.str();
}
std::string token_as_string(llama_model* model, llama_token token) {
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int lsplit = 0;
    bool special = false;
    std::vector<char> piece(8, 0);
    int n_tokens = llama_token_to_piece(vocab, token, piece.data(), piece.size(), lsplit, special);
    if (n_tokens < 0) {
        piece.resize(-n_tokens);
        llama_token_to_piece(vocab, token, piece.data(), piece.size(), lsplit, special);
    } else {
        piece.resize(n_tokens);
    }
    return std::string(piece.data(), piece.size());
}




const char* RED = "\033[0;31m";
const char* GREEN = "\033[0;32m";
const char* BLUE = "\033[0;34m";
const char* ORANGE = "\033[0;33m";  // Actually yellow, but often appears as orange in many terminals
const char* RESET = "\033[0m";
#if 0
int main(int argc, char** argv) {
    fprintf(stdout, "llama.cpp batch exploration\n");
    llama_model_params model_params = llama_model_default_params();
    //std::string model_path = "models/llama-2-7b-chat.Q4_K_M.gguf";
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    //std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";
    //std::string model_path = "models/mamba-1.4b-f16.gguf";

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    // This prompt is 69 tokens
    //std::string prompt1 = R"(You are an AI assistant specializing in task completion. Your goal is to provide clear, concise, and accurate responses to user queries. Always maintain a helpful and professional tone. If a request is unclear, ask for clarification. Prioritize user safety and ethical considerations in your answers.)";
    std::string prompt1 = "What is the capital of Sweden?";
    //std::string prompt2 = "How many r's are there in strawberry?";
    std::string prompt2 = "What is the capital of France?";

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    ctx_params.n_seq_max = 2;
    ctx_params.n_batch = 80;
    ctx_params.n_ubatch = 32;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("%sprompt1: %s%s\n", BLUE, prompt1.c_str(), RESET);
    printf("%sprompt2: %s%s\n", ORANGE, prompt2.c_str(), RESET);

    // Tokenize the prompts.
    std::vector<llama_token> input_tokens1 = tokenize_prompt(model, prompt1);
    std::vector<llama_token> input_tokens2 = tokenize_prompt(model, prompt2);

    llama_batch batch = create_batch(512, {input_tokens1, input_tokens2}, model);
    print_batch(batch);
    std::cout << string_from_batch(ctx,batch) <<std::endl;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(3));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    llama_token sp_token_seq1 = llama_sampler_sample(sampler, ctx, input_tokens1.size()-1);
    std::string sp_str1 = token_as_string(model, sp_token_seq1);
    printf("%snew_token_seq1: %d : token_str1 [%s]%s\n", BLUE, sp_token_seq1, sp_str1.c_str(), RESET);

    llama_sampler_reset(sampler);

    llama_token sp_token_seq2 = llama_sampler_sample(sampler, ctx, input_tokens1.size() + input_tokens2.size()-1);
    std::string sp_str2 = token_as_string(model, sp_token_seq2);
    printf("%snew_token_seq2: %d : token_str2 [%s]%s\n", ORANGE, sp_token_seq2, sp_str2.c_str(), RESET);

    int decode_calls = 10;

    int pos1 = input_tokens1.size();
    int pos2 = input_tokens2.size();
    std::vector<std::string> seq_1_output;
    std::vector<std::string> seq_2_output;

    while (decode_calls--) {
        llama_batch update_batch = llama_batch_init(2, 0, 2);
        update_batch.token[0] = sp_token_seq1;
        update_batch.token[1] = sp_token_seq2;
        update_batch.pos[0] = pos1++;
        update_batch.pos[1] = pos2++;
        update_batch.n_tokens = 2;

        update_batch.n_seq_id[0] = 1;
        update_batch.seq_id[0][0] = 0;
        update_batch.logits[0] = true;

        update_batch.n_seq_id[1] = 1;
        update_batch.seq_id[1][0] = 1;
        update_batch.logits[1] = true;

        if (llama_decode(ctx, update_batch) != 0) {
            fprintf(stderr, "llama_decode() failed\n");
            return 1;
        }

        sp_token_seq1 = llama_sampler_sample(sampler, ctx, 0);
        std::string sp_str1 = token_as_string(model, sp_token_seq1);
        seq_1_output.push_back(sp_str1);
        printf("%snew_token_seq1: %d : token_str1 [%s]%s\n", BLUE, sp_token_seq1, sp_str1.c_str(), RESET);
        //print_colored_token("prompt1: ", BLUE);

        llama_sampler_reset(sampler);

        sp_token_seq2 = llama_sampler_sample(sampler, ctx, 1);
        std::string sp_str2 = token_as_string(model, sp_token_seq2);
        seq_2_output.push_back(sp_str2);
        printf("%snew_token_seq2: %d : token_str2 [%s]%s\n", ORANGE, sp_token_seq2, sp_str2.c_str(), RESET);

        llama_batch_free(update_batch);
    }
    printf("sequence 1 output:\n");
    for (size_t i = 0; i < seq_1_output.size(); i++) {
        printf("%s%s%s", BLUE, seq_1_output[i].c_str(), RESET);
    }
    printf("\n");
    printf("sequence 2 output:\n");
    for (size_t i = 0; i < seq_2_output.size(); i++) {
        printf("%s%s%s", ORANGE, seq_2_output[i].c_str(), RESET);
    }
    printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    llama_sampler_free(sampler);

    return 0;
}
#else
int main(int argc, char** argv) {
    fprintf(stdout, "llama.cpp batch exploration\n");
    llama_model_params model_params = llama_model_default_params();
    //std::string model_path = "models/llama-2-7b-chat.Q4_K_M.gguf";
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    //std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";
    //std::string model_path = "models/mamba-1.4b-f16.gguf";

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    // This prompt is 69 tokens
    //std::string prompt1 = R"(You are an AI assistant specializing in task completion. Your goal is to provide clear, concise, and accurate responses to user queries. Always maintain a helpful and professional tone. If a request is unclear, ask for clarification. Prioritize user safety and ethical considerations in your answers.)";
    std::string prompt1 = "What is the capital of Sweden?";
    //std::string prompt2 = "How many r's are there in strawberry?";
    std::string prompt2 = "What is the capital of France? the the capital of France is the largetst city of France";
    //std::string prompt2 = "What is the capital of France?";
    //std::string prompt3 = "What is the capital of china?";

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    ctx_params.n_seq_max = 2;
    ctx_params.n_batch = 80;
    ctx_params.n_ubatch = 32;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("%sprompt1: %s%s\n", BLUE, prompt1.c_str(), RESET);
    printf("%sprompt2: %s%s\n", ORANGE, prompt2.c_str(), RESET);

    // Tokenize the prompts.
    std::vector<llama_token> input_tokens1 = tokenize_prompt(model, prompt1);
    std::vector<llama_token> input_tokens2 = tokenize_prompt(model, prompt2);
    //std::vector<llama_token> input_tokens3 = tokenize_prompt(model, prompt3);

    llama_batch batch = create_batch(512, {input_tokens1, input_tokens2}, model);
    print_batch(batch);
    std::cout << string_from_batch(ctx,batch) <<std::endl;

    //debug_ubatch(ctx,*model,batch);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }
    //llama_batch batch2 = create_batch(512, {input_tokens2, input_tokens3}, model);
    //print_batch(batch2);
    //std::cout << string_from_batch(ctx,batch2) <<std::endl;

    //if (llama_decode(ctx, batch2) != 0) {
    //    fprintf(stderr, "llama_decode() failed\n");
    //    return 1;
    //}
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(3));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    llama_token sp_token_seq1 = llama_sampler_sample(sampler, ctx, input_tokens1.size()-1);
    std::string sp_str1 = token_as_string(model, sp_token_seq1);
    printf("%snew_token_seq1: %d : token_str1 [%s]%s\n", BLUE, sp_token_seq1, sp_str1.c_str(), RESET);

    llama_sampler_reset(sampler);

    llama_token sp_token_seq2 = llama_sampler_sample(sampler, ctx, input_tokens1.size() + input_tokens2.size()-1);
    std::string sp_str2 = token_as_string(model, sp_token_seq2);
    printf("%snew_token_seq2: %d : token_str2 [%s]%s\n", ORANGE, sp_token_seq2, sp_str2.c_str(), RESET);

    int decode_calls = 10;

    int pos1 = input_tokens1.size();
    int pos2 = input_tokens2.size();
    std::vector<std::string> seq_1_output;
    std::vector<std::string> seq_2_output;

#if 1
    while (decode_calls--) {
        llama_batch update_batch = llama_batch_init(2, 0, 2);
        update_batch.token[0] = sp_token_seq1;
        update_batch.token[1] = sp_token_seq2;
        update_batch.pos[0] = pos1++;
        update_batch.pos[1] = pos2++;
        update_batch.n_tokens = 2;

        update_batch.n_seq_id[0] = 1;
        update_batch.seq_id[0][0] = 0;
        update_batch.logits[0] = true;

        update_batch.n_seq_id[1] = 1;
        update_batch.seq_id[1][0] = 1;
        update_batch.logits[1] = true;

        if (llama_decode(ctx, update_batch) != 0) {
            fprintf(stderr, "llama_decode() failed\n");
            return 1;
        }

        sp_token_seq1 = llama_sampler_sample(sampler, ctx, 0);
        std::string sp_str1 = token_as_string(model, sp_token_seq1);
        seq_1_output.push_back(sp_str1);
        printf("%snew_token_seq1: %d : token_str1 [%s]%s\n", BLUE, sp_token_seq1, sp_str1.c_str(), RESET);
        //print_colored_token("prompt1: ", BLUE);

        llama_sampler_reset(sampler);

        sp_token_seq2 = llama_sampler_sample(sampler, ctx, 1);
        std::string sp_str2 = token_as_string(model, sp_token_seq2);
        seq_2_output.push_back(sp_str2);
        printf("%snew_token_seq2: %d : token_str2 [%s]%s\n", ORANGE, sp_token_seq2, sp_str2.c_str(), RESET);
        //debug_ubatch(ctx,*model,update_batch);

        llama_batch_free(update_batch);
    }
    printf("sequence 1 output:\n");
    for (size_t i = 0; i < seq_1_output.size(); i++) {
        printf("%s%s%s", BLUE, seq_1_output[i].c_str(), RESET);
    }
    printf("\n");
    printf("sequence 2 output:\n");
    for (size_t i = 0; i < seq_2_output.size(); i++) {
        printf("%s%s%s", ORANGE, seq_2_output[i].c_str(), RESET);
    }
    printf("\n");
 #endif
    //debug_kv_cache(*model,batch);
    //debug_ubatch(ctx,*model,batch);
    //debug_ubatch(ctx,*model,batch2);
    llama_batch_free(batch);
    //llama_batch_free(batch2);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    llama_sampler_free(sampler);

    return 0;
}
#endif
