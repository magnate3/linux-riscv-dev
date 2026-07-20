#include "LLMEngine.h"
#include <iostream>
#include <cstring>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;

// Silence llama.cpp logs
static void log_silencer(ggml_log_level level, const char * text, void * user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

LLMEngine::LLMEngine() {
    // Register silencer before backend init
    llama_log_set(log_silencer, nullptr);
    llama_backend_init();
}

LLMEngine::~LLMEngine() {
    unloadModel();
    llama_backend_free();
}

bool LLMEngine::loadModel(const std::string& modelPath) {
    auto start = Clock::now(); 

    if (isLoaded()) {
        std::cout << "[Engine] Unloading previous model..." << std::endl;
        unloadModel();
    }

    std::cout << "[Engine] Loading model: " << modelPath << "..." << std::endl;

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(modelPath.c_str(), model_params);

    if (model == nullptr) return false;

    vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; 
    ctx_params.n_batch = 2048; 

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        unloadModel();
        return false;
    }

    auto end = Clock::now(); 
    stats.loadTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "[Metrics] Model Load Time: " << stats.loadTimeMs << " ms" << std::endl;
    return true;
}

void LLMEngine::unloadModel() {
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    vocab = nullptr;
}

bool LLMEngine::isLoaded() const { return model != nullptr; }

void LLMEngine::batch_add_seq(llama_batch &batch, llama_token token, int pos, int32_t seq_id, bool logits) {
    batch.token[batch.n_tokens] = token;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = seq_id;
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

std::string LLMEngine::query(const std::string& prompt, int max_tokens) {
    if (!model) return "Error: No model loaded.";

    if (ctx) { llama_free(ctx); ctx = nullptr; }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; 
    ctx_params.n_batch = 2048;
    ctx = llama_init_from_model(model, ctx_params);

    const int n_prompt_max = -llama_tokenize(vocab, prompt.c_str(), prompt.length(), NULL, 0, true, false);
    std::vector<llama_token> prompt_tokens(n_prompt_max);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.length(), prompt_tokens.data(), prompt_tokens.size(), true, false);
    
    if (n_prompt < 0) return "Error: Tokenization failed.";

    llama_batch batch = llama_batch_init(2048, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch_add_seq(batch, prompt_tokens[i], i, 0, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return "Error: Prompt decoding failed.";
    }

    auto t_start_gen = Clock::now();
    std::string result = "";
    int n_cur = batch.n_tokens;
    int n_decode = 0;
    stats.tokensGenerated = 0;
    const auto t_main_start = ggml_time_us();
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

        stats.tokensGenerated++;

        batch.n_tokens = 0;
        batch_add_seq(batch, new_token_id, n_cur, 0, true);
        n_cur++;

        if (llama_decode(ctx, batch) != 0) break;
    }

    auto t_end_gen = Clock::now();
    stats.generationTimeMs = std::chrono::duration<double, std::milli>(t_end_gen - t_start_gen).count();

    std::cout << std::endl; 
    const auto t_main_end = ggml_time_us();
    n_decode = stats.tokensGenerated;

   printf("%s: decoded %d tokens in %.2f s, speed: %.2f t/s", __func__, n_decode,
		(t_main_end - t_main_start) / 1000000.0f,
		n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    llama_batch_free(batch);
    
    return result;
}
