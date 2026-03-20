#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono> // NEW: For Timing

#include "LLMEngine.h"
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
    LLMEngine engine;
    if(!engine.loadModel(model_path))
    {
	printf("model load fail \n");

    }
    std::vector<llama_chat_message> messages;
    llama_model * model = engine.getModel();
    llama_context* ctx = engine.getCtx();
    int prev_len = 0;
    if(NULL == ctx || NULL == model){
            engine.unloadModel();
	    printf("model init fail \n");
	    return -1;
     }
    std::vector<char> formatted(llama_n_ctx(ctx));
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
        std::string response = engine.query(prompt,128);
        printf("\n\033[0m");

        // add the response to the messages
        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }
    }
//out:
    engine.unloadModel();
    return 0;
}
