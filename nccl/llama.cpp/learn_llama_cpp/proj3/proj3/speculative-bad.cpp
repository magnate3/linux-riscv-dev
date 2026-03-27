//#include "build-info.h"

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <memory>

#include <nlohmann/json.hpp>
#define LOG_TOKENS_TOSTR_PRETTY(ctx, tokens)                                 \
    [&tokens, &ctx]()                                                        \
    {                                                                        \
        std::stringstream buf;                                               \
        buf << "[ ";                                                         \
                                                                             \
        bool first = true;                                                   \
        for (const auto &token : tokens)                                     \
        {                                                                    \
            if (!first)                                                      \
                buf << ", ";                                                 \
            else                                                             \
                first = false;                                               \
                                                                             \
            auto detokenized = llama_token_to_piece(ctx, token);             \
                                                                             \
            detokenized.erase(                                               \
                std::remove_if(                                              \
                    detokenized.begin(),                                     \
                    detokenized.end(),                                       \
                    [](const unsigned char c) { return !std::isprint(c); }), \
                detokenized.end());                                          \
                                                                             \
            buf                                                              \
                << "'" << detokenized << "'"                                 \
                << ":" << std::to_string(token);                             \
        }                                                                    \
        buf << " ]";                                                         \
                                                                             \
        return buf.str();                                                    \
    }()                                                                      \
        .c_str()


struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens,
               llama_pos   pos_0,
            llama_seq_id   seq_id) {
    return {
        /*n_tokens    =*/ n_tokens,
        /*tokens      =*/ tokens,
        /*embd        =*/ nullptr,
        /*pos         =*/ nullptr,
        /*seq_id      =*/ nullptr,
        /*logits      =*/ nullptr,
        /*all_pos_0   =*/ pos_0,
        /*all_pos_1   =*/ 1,
        /*all_seq_id  =*/ seq_id,
    };
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    return common_token_to_piece(ctx, token);
}
static float frand() {
    return (float) rand() / RAND_MAX;
}

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;

    //struct llama_sampling_context * ctx_sampling;
    struct common_sampler * ctx_sampling = nullptr;
};

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    //const int n_seq_dft = params.n_parallel;
    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = 5;

    // TODO: make this configurable
    const float p_accept = -1.0f; // always draft n_draft tokens, no early stopping
    const float p_split  = 0.10f;


    // init llama.cpp
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
    // streaming prompts from json
    //std::ifstream dataset_file("/home/hedgehog/llama_cpp_osd/data/raw_data/spider_validation.json", std::ifstream::binary);
    //Json::Value data;
    std::fstream fJson("data/raw_data/spider_validation.json");
    std::stringstream buffer;
    buffer << fJson.rdbuf();
    auto data = nlohmann::json::parse(buffer.str());
	
    int data_counter = 0;
    for (auto question : data) {
	if (data_counter == 0) {
		data_counter += 1;
		continue;
	}
	//auto conv = question["conversation"];
	
	//std::string prompt;
	//for (auto sentence : conv) {
	//	auto content = sentence["content"];
	//	prompt = content.get<std::string>();
	//	break;
	//}
	
	//std::string prompt = std::to_string(prompt_in);

	//printf("\nprompt: %s\n", prompt);
	

    	// load the target model
    	//params.logits_all = true;
	

        auto tokenize = [&](const llama_vocab * vocab,std::string& text, bool add_bos) {
            std::vector<llama_token> tokens(text.size() + 3);
            int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
            tokens.resize(n);
            return tokens;
        };
        std::string prompt = "The capital of France is";
        // tokenize the prompt
        std::vector<llama_token> inp = tokenize(vocab_tgt, prompt, true);
        //inp = ::llama_tokenize(ctx_tgt, params.prompt, true);
        //inp = ::llama_tokenize(ctx_tgt, params.prompt, true);
	//std::cout << typeid(params.prompt).name() << '\n';
	//std::cout << typeid(prompt).name() << '\n';
	
	for (int i = 0; i < (int) inp.size(); i++) {
        	// Printing the element at
        	// index 'i' of vector
		std::cout << inp[i] << " ";
    	}
	std::cout << std::endl;

        const int max_context_size     = llama_n_ctx(ctx_tgt);
        const int max_tokens_list_size = max_context_size - 4;
	
	printf("\nprompt length: %d\n", (int) inp.size());

        if ((int) inp.size() > max_tokens_list_size) {
            fprintf(stderr, "%s: error: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
            return 1;
        }

        fprintf(stderr, "\n\n");

        for (auto id : inp) {
            fprintf(stderr, "%s", llama_token_to_piece(ctx_tgt, id).c_str());
        }

        fflush(stderr);

        const int n_input = inp.size();

        const auto t_enc_start = ggml_time_us();
	
        // eval the prompt with both models
        llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
        llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));
        llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input,     0,           0));

	//printf("\ndebugging ckpt.\n");

        const auto t_enc_end = ggml_time_us();

        // the 2 models should have the same vocab
        //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

        // how many tokens to draft each time
        int n_draft = 4;

        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;

        int n_past_tgt = inp.size();
        int n_past_dft = inp.size();

        // used to determine end of generation
        bool has_eos = false;

        // target model sampling context
        //struct llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);
        struct common_sampler *  ctx_sampling = common_sampler_init(model_tgt, params.sampling);

        // draft sequence data
        std::vector<seq_draft> drafts(n_seq_dft);
	
	//printf("\ndebugging ckpt.\n");

        params.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
        params.sparams.temp = std::max(0.01f, params.sparams.temp);

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].ctx_sampling = common_sampler_init(model_dft, params.sampling);
        }

        llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, 1);
        llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, n_seq_dft);

        const auto t_dec_start = ggml_time_us();

        // sample from the last token of the prompt
        drafts[0].i_batch_tgt.resize(1);
        drafts[0].i_batch_tgt[0] = 0;
	
        while (true) {
            // print current draft sequences
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) {
                    continue;
                }

                const auto & tokens = drafts[s].tokens;

                printf("draft %d: %s\n", s, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens).c_str());
            }

            int i_dft  = 0;
            int s_keep = 0;

            while (true) {
                LOG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);

                // sample from the target model
                llama_token id = llama_sampling_sample(ctx_sampling, ctx_tgt, NULL, drafts[s_keep].i_batch_tgt[i_dft]);
		
                llama_sampling_accept(ctx_sampling, ctx_tgt, id, true);
		
                //LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, ctx_sampling->prev).c_str());

                const std::string token_str = llama_token_to_piece(ctx_tgt, id);
		
                printf("%s", token_str.c_str());
		fflush(stdout);

                if (id == llama_token_eos(model_tgt)) {
                    
		    has_eos = true;
                }
	
                ++n_predict;

                // check if the target token matches any of the drafts
                {
                    bool matches = false;

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                            LOG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, id, token_str.c_str());

                            s_keep = s;
                            matches = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }

                    if (matches) {
                        ++n_accept;
                        ++n_past_tgt;
                        ++n_past_dft;
                        ++i_dft;

                        continue;
                    }
                }

                LOG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

                // TODO: simplify
                {
                    LOG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                    llama_kv_cache_seq_keep(ctx_dft, s_keep);
                    llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                    llama_kv_cache_seq_keep(ctx_dft, 0);

                    llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                    llama_kv_cache_seq_keep(ctx_tgt, s_keep);
                    llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                    llama_kv_cache_seq_keep(ctx_tgt, 0);
                }

                for (int s = 0; s < n_seq_dft; ++s) {
                    drafts[s].active = false;
                    drafts[s].tokens.clear();
                    drafts[s].i_batch_tgt.clear();
                }
                // note: will be erased after the speculation phase
                drafts[0].tokens.push_back(id);
                drafts[0].i_batch_tgt.push_back(0);

                llama_batch_clear(batch_dft);
                llama_batch_add  (batch_dft, id, n_past_dft, { 0 }, true);

                llama_kv_cache_seq_rm(ctx_dft, 0, n_past_dft, -1);
                llama_decode         (ctx_dft, batch_dft);

                ++n_past_dft;
		
                break;
            }

            if (n_predict > params.n_predict || has_eos) {
		break;
            }

            llama_sampling_cp(ctx_sampling, drafts[0].ctx_sampling);

            int n_seq_cur  = 1;
            int n_past_cur = n_past_dft;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active   = false;
                drafts[s].drafting = false;
            }
            drafts[0].active      = true;
            drafts[0].drafting    = true;
            drafts[0].i_batch_dft = 0;

            llama_batch_clear(batch_tgt);
            llama_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

            // sample n_draft tokens from the draft model using tree-based sampling
            for (int i = 0; i < n_draft; ++i) {
                batch_dft.n_tokens = 0;

                for (int s = 0; s < n_seq_dft; ++s) {
                    drafts[s].skip = false;
                }

                for (int s = 0; s < n_seq_dft; ++s) {
                    if (!drafts[s].drafting || drafts[s].skip) {
                        continue;
                    }

                    llama_sampling_sample(drafts[s].ctx_sampling, ctx_dft, NULL, drafts[s].i_batch_dft);

                    const auto & cur_p = drafts[s].ctx_sampling->cur;

                    for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p.size()); ++k) {
                        LOG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                k, s, i, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
                    }

                    if (cur_p[0].p < p_accept) {
                        LOG("stopping drafting for seq %3d, probability too low: %.3f < %.3f\n", s, cur_p[0].p, p_accept);
                        drafts[s].drafting = false;
                        continue;
                    }

                    std::vector<int> sa(1, s);

                    // attempt to split the branch if the probability is high enough
                    for (int f = 1; f < 8; ++f) {
                        if (n_seq_cur < n_seq_dft && cur_p[f].p > p_split) {
                            LOG("splitting seq %3d into %3d\n", s, n_seq_cur);

                            llama_kv_cache_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                            llama_kv_cache_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                            // all previous tokens from this branch are now also part of the new branch
                            for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                                for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                    if (batch_tgt.seq_id[t][p] == s) {
                                        batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                        batch_tgt.n_seq_id[t]++;
                                        
					break;
                                    }
                                }
                            }

                            // copy the draft state
                            drafts[n_seq_cur].active   = true;
                            drafts[n_seq_cur].drafting = true;
                            drafts[n_seq_cur].skip     = true;

                            drafts[n_seq_cur].tokens      = drafts[s].tokens;
                            drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                            drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                            llama_sampling_cp(drafts[s].ctx_sampling, drafts[n_seq_cur].ctx_sampling);

                            sa.push_back(n_seq_cur);

                            n_seq_cur++;
                        } else {
                            break;
                        }
                    }

                    // add drafted token for each sequence
                    for (int is = 0; is < (int) sa.size(); ++is) {
                        const llama_token id = cur_p[is].id;

                        const int s = sa[is];

                        common_sampler_accept(drafts[s].ctx_sampling, ctx_dft, id, true);

                        drafts[s].tokens.push_back(id);

                        // add unique drafted tokens to the target batch
                        drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                        llama_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                        // add the token to the batch for batched decoding with the draft model
                        drafts[s].i_batch_dft = batch_dft.n_tokens;

                        llama_batch_add(batch_dft, id, n_past_cur, { s }, true);

                        if (batch_tgt.n_tokens > n_draft) {
                            drafts[s].drafting = false;
                        }
                    }
                }

                // no sequence is drafting anymore
                if (batch_dft.n_tokens == 0) {
		    break;
                }

                // evaluate the drafted tokens on the draft model
                llama_decode(ctx_dft, batch_dft);
                ++n_past_cur;
                ++n_drafted;

                if (batch_tgt.n_tokens > n_draft) {
                    break;
                }
            }

            // evaluate the target model on the drafted tokens
            {
                llama_kv_cache_seq_keep(ctx_tgt, 0);
                for (int s = 1; s < n_seq_dft; ++s) {
                    llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
                }

                //LOG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt));
                llama_decode(ctx_tgt, batch_tgt);
                ++n_past_tgt;
            }

            // the first token is always proposed by the traget model before the speculation loop so we erase it here
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) {
                    continue;
                }

                drafts[s].tokens.erase(drafts[s].tokens.begin());
            }
        }

        auto t_dec_end = ggml_time_us();

        LOG_TEE("\n\n");

        LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

        LOG_TEE("\n");
        LOG_TEE("n_draft   = %d\n", n_draft);
        LOG_TEE("n_predict = %d\n", n_predict);
        LOG_TEE("n_drafted = %d\n", n_drafted);
        LOG_TEE("n_accept  = %d\n", n_accept);
        LOG_TEE("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

        LOG_TEE("\ndraft:\n");
        llama_print_timings(ctx_dft);

        LOG_TEE("\ntarget:\n");
        llama_print_timings(ctx_tgt);

        common_sampler_free(ctx_sampling);
        for (int s = 0; s < n_seq_dft; ++s) {
            common_sampler_free(drafts[s].ctx_sampling);
        }

        llama_batch_free(batch_dft);

        llama_free(ctx_tgt);
        llama_free_model(model_tgt);

        llama_free(ctx_dft);
        llama_free_model(model_dft);

        llama_backend_free();

        fprintf(stderr, "\n\n");

	data_counter += 1;
    }

    return 0;
}
