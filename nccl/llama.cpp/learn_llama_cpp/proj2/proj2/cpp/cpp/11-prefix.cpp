#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

#include <vector>
#define MODEL_PATH "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
#define N_CTX 4096
#define N_THREADS 4

int main(int argc, char **argv) {
    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(MODEL_PATH, model_params);
    if (!model) {
        fprintf(stderr, "�load model fail\n");
        return 1;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx      = N_CTX;
    ctx_params.n_threads  = N_THREADS;
    //ctx_params.paged_kv   = true; 
    //ctx_params.n_kv_split = 2;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ctx create fail�\n");
        return 2;
    }

    // 5 轮对话，每一轮都在前一轮基础上追加
    const char *rounds[5] = {
        "Hello, who are you?",
        "Hello, who are you? I am a user.",
        "Hello, who are you? I am a user. What can you do?",
        "Hello, who are you? I am a user. What can you do? Can you write code?",
        "Hello, who are you? I am a user. What can you do? Can you write code? Please write a hello world in C."
    };

    //llama_token tokens[2048];

    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    //int n_tokens = llama_tokenize(model, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, false);
    int n_tokens;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (int r = 0; r < 5; r++) {
        printf("==================== Round %d ====================\n", r + 1);

	n_tokens = llama_tokenize(vocab, rounds[r], strlen(rounds[r]), tokens.data(), tokens.size(), true, false);



        if (n_tokens <= 0) {
            fprintf(stderr, "tokenize 失败\n");
            break;
        }
	size_t matched = 0;
#if 0
        size_t matched = llama_kv_cache_find_prefix(
            llama_get_kv_cache(ctx),
            tokens.data(),
            n_tokens
        );
#endif
        size_t new_decode = n_tokens - matched;

        printf("Total tokens: %4d\n", n_tokens);
        printf("Matched prefix: %4zu reused)\n", matched);
        printf("New to decode:  %4zu \n", new_decode);

        //if (llama_decode(ctx, tokens, n_tokens) != 0) {
	if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens)) != 0) {
            fprintf(stderr, "decode fail");
            break;
        }

        printf("\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
