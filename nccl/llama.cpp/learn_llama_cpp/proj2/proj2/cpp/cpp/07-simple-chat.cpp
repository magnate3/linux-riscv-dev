#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono> // NEW: For Timing
using Clock = std::chrono::high_resolution_clock;

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers]\n", argv[0]);
    printf("\n");
}

#if 0
void handle_kv_cache_overflow(llama_context *ctx, int &n_past, int n_keep)
{
     int n_ctx = llama_n_ctx(ctx);
     int n_discard = (n_past - n_keep) / 4;
     if (n_past <= n_keep) {
	 return;
     }

     printf("\n\033[33m[KV Cache] roll and clear %d   Tokens...\033[0m\n", n_discard);
     llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);
     llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_past, -n_discard);
     n_past -= n_discard;
     printf("\033[32m[KV Cache] roll finnish , current n_past: %d\033[0m\n", n_past);
}
#else
void handle_kv_cache_overflow(llama_context *ctx, int &n_past, int n_keep)
{
     int n_ctx = llama_n_ctx(ctx);
     int n_discard = (n_past - n_keep) / 4;
     if (n_past <= n_keep) {
	 return;
     }

     //printf("\n\033[33m[KV Cache] roll and clear %d   Tokens...\033[0m\n", n_discard);
     llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep, n_keep + n_discard);
     llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep + n_discard, n_past, -n_discard);
     n_past -= n_discard;
     //printf("\033[32m[KV Cache] roll finnish , current n_past: %d\033[0m\n", n_past);
}
#endif

void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}
void batch_add_seq(llama_batch &batch, llama_token token, int pos, int32_t seq_id, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int ngl = 99;
    int n_ctx = 2048;

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) {
                    n_ctx = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    ngl = std::stoi(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    //int n_predict = prompt.size();
    int n_predict = 256;
    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    ctx_params.no_perf = false;


    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
#if 0
    // helper function to evaluate a prompt and generate a response
    auto generate = [&](const std::string & prompt) {
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
	//const int n_keep = prompt.size(); 
        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }
  
  size_t n_parallel = 4;
  llama_batch batch = llama_batch_init(std::max(prompt_tokens.size(), n_parallel), 0, n_parallel);
  
  {
      std::vector<llama_seq_id> seq_ids(n_parallel, 0);
      for (int32_t i = 0; i < n_parallel; ++i)
      {
          seq_ids[i] = i;
      }
  
      for (size_t i = 0; i < prompt_tokens.size(); ++i)
      {
          batch.token[batch.n_tokens] = prompt_tokens[i];
          batch.pos[batch.n_tokens] = i;
          batch.n_seq_id[batch.n_tokens] = seq_ids.size();
          for (size_t i = 0; i < seq_ids.size(); ++i)
          {
              batch.seq_id[batch.n_tokens][i] = seq_ids[i];
          }
          batch.logits[batch.n_tokens] = false;
  
          batch.n_tokens++;
  
          //common_batch_add(batch, tokens_list[i], i, seq_ids, false);
      }
  }
  
  batch.logits[batch.n_tokens - 1] = true;
  
  std::vector<std::string> streams(n_parallel);
  std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);
  
  int n_cur = batch.n_tokens;
  bool isFinished = false;
  
  while (isFinished == false)
  {        
      if (llama_decode(ctx, batch))
      {
          GGML_ABORT("failed to decode\n");
      }
      batch.n_tokens = 0;
      for (int32_t i = 0; i < n_parallel; ++i)
      {
          if (i_batch[i] < 0)
          {
              continue;
          }
  
          llama_token new_token_id = llama_sampler_sample(smpl, ctx, i_batch[i]);
  
          if (llama_vocab_is_eog(vocab, new_token_id))
          {
              isFinished = true;
              break;
          }
                 char buf[256];
                 int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                 if (n < 0) {
                     GGML_ABORT("failed to convert token to piece\n");
                 }
                 std::string piece(buf, n);
                 printf("%s", piece.c_str());
                 fflush(stdout);
                 streams[i]  += piece;
  
          i_batch[i] = batch.n_tokens;
  
          batch.token[batch.n_tokens] = new_token_id;
          batch.pos[batch.n_tokens] = n_cur;
          batch.n_seq_id[batch.n_tokens] = 1;
          batch.seq_id[batch.n_tokens][0] = i;
          batch.logits[batch.n_tokens] = true;
  
          batch.n_tokens++;
      }
  }
  
  std::string response = "";
  for (int32_t i = 0; i < n_parallel; ++i)
  {
      response += streams[i];
      response += "\n";
  }
      return response;
    };
#else
    //std::string LLMEngine::query(const std::string& prompt, int max_tokens) {
   auto generate = [&](const std::string & prompt) {
 
   int max_tokens = 128;
   if (!model) return std::string("Error: No model loaded.");

    if (ctx) { llama_free(ctx); ctx = nullptr; }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx = llama_init_from_model(model, ctx_params);

    const int n_prompt_max = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, false);
    std::vector<llama_token> prompt_tokens(n_prompt_max);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), true, false);

    if (n_prompt < 0) return std::string("Error: Tokenization failed.");

    llama_batch batch = llama_batch_init(2048, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch_add_seq(batch, prompt_tokens[i], i, 0, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return std::string("Error: Prompt decoding failed.");
    }

    auto t_start_gen = Clock::now();
    std::string result = "";
    int n_cur = batch.n_tokens;

    for (int i = 0; i < max_tokens; i++) {
        auto n_vocab_size = llama_vocab_n_tokens(vocab);
        auto * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        llama_token new_token_id = 0;
        float max_prob = -1e9;

        for (int j = 0; j < n_vocab_size; j++) {
            if (logits[j] > max_prob) {
                max_prob = logits[j];
                new_token_id = j;
            }
        }

        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);

        if (n >= 0) {
            std::string piece(buf, n);
            std::cout << piece << std::flush;
            result += piece;
        }

        //stats.tokensGenerated++;

        batch.n_tokens = 0;
        batch_add_seq(batch, new_token_id, n_cur, 0, true);
        n_cur++;

        if (llama_decode(ctx, batch) != 0) break;
    }

    auto t_end_gen = Clock::now();
    //stats.generationTimeMs = std::chrono::duration<double, std::milli>(t_end_gen - t_start_gen).count();

    std::cout << std::endl;
    llama_batch_free(batch);
    return result;
};
#endif
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int prev_len = 0;
    while (true) {
        // get user input
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

        // add the user input to the message list and format it
        messages.push_back({"user", strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }

        // remove previous messages to obtain the prompt to generate the response
        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

#if 0
	llama_memory_breakdown_print(ctx);
#endif
        // generate a response
        printf("\033[33m");
        std::string response = generate(prompt);
        printf("\n\033[0m");

        // add the response to the messages
        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }
    }

    // free resources
    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
