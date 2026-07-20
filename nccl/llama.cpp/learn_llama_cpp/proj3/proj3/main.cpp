#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "llama.h"
//void print_batch(struct llama_model* model , struct llama_ubatch * ubatch)
void print_batch(const struct llama_vocab * vocab, struct llama_batch * ubatch)
{
     //llama_token_get_text(const struct llama_model * model, llama_token token)
     std::cout << "____ubatch token0____" << llama_vocab_get_text(vocab,ubatch->token[0]) << std::endl;
     //std::cout << llama_token_get_text(vocab,ubatch->token[0]) << std::endl;
}


// TODO:
// Fix GGML bindings

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-path.gguf>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    // 1. Set llama model and context parameters
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;          // Using 99 to utilize GPU acceleration, tested on Apple Silicon and
                                        // library correctly utilizes Metal API to take advantage of the chip.

    struct llama_context_params ctx_params = llama_context_default_params();

    ctx_params.embeddings = false;
    ctx_params.n_ctx     = 8192;        // context size
    ctx_params.n_threads = 4;         // CPU threads for generation (adjust as desired)

    // 2. Load the model
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    // 3. Create a new context
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        fprintf(stderr, "Error: Failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    // Set up conversation
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    const char* tmpl = llama_model_chat_template(model, /* name */ nullptr);
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    
    // Add a system message to guide the model's behavior
    const char* system_message = "You are a helpful, friendly AI assistant. Respond to users in a conversational way. "
                               "Be concise, helpful, and engage with the user's actual query.";
    messages.push_back({"system", system_message});
    
    // initialize the sampler, sets temperature, distribution for model sampling
    llama_sampler* smpl_chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl_chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string user_input;
    bool first_prompt = true;
    
    std::cout << "Conversation started. Type 'exit' to end.\n";
    std::cout << "User: ";
    
    while (std::getline(std::cin, user_input)) {
        if (user_input == "exit") {
            break;
        }
        
        // Add the user input to the message history
        messages.push_back({"user", strdup(user_input.c_str())});
        
        // Format the conversation using the chat template
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, 
                                               formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, 
                                              formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "Failed to apply the chat template\n");
            continue;
        }

        // Get the formatted conversation
        std::string prompt(formatted.begin(), formatted.begin() + new_len);

        // Prepare for token generation
        const bool is_first = first_prompt;
        first_prompt = false;

        // Tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            fprintf(stderr, "Failed to tokenize the prompt\n");
            continue;
        }

        // Prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        // Check if the context has enough space
        int n_ctx = llama_n_ctx(ctx);
#if 0
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
#else
	int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
#endif
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "Context size exceeded, clearing conversation history\n");
            
            // Reset conversation by clearing KV cache and message history
            //llama_kv_cache_clear(ctx);
	    llama_memory_clear(llama_get_memory(ctx), true);
            
            // Keep the system message and the current user message
            llama_chat_message system_msg = messages[0];
            llama_chat_message last_msg = messages.back();
            
            // Clear and rebuild messages
            for (size_t i = 1; i < messages.size(); i++) {
                free((void*)messages[i].content);
            }
            messages.clear();
            messages.push_back(system_msg);
            messages.push_back(last_msg);
            
            // Reformat and try again
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, 
                                              formatted.data(), formatted.size());
            prompt = std::string(formatted.begin(), formatted.begin() + new_len);
            
            // Re-tokenize with fresh context
            const int n_retry_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
            prompt_tokens.resize(n_retry_tokens);
            llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
            batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        }

        // Process the batch
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Failed to decode\n");
            continue;
        }

        std::string response;
        std::cout << "AI: ";

        auto start_time = std::chrono::high_resolution_clock::now();
        int token_count = 0;

        // Generate the response tokens
        while (true) {
            // Sample the next token
            llama_token new_token_id = llama_sampler_sample(smpl_chain, ctx, -1);

            // Check if token generated is end of generation token
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            token_count++;

            // Convert the token to a string
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "Failed to convert token to piece\n");
                break;
            }
            
            std::string piece(buf, n);
            std::cout << piece << std::flush;
            response += piece;

            // Prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
            
            // Process this new token
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "Failed to decode\n");
                break;
            }
            //print_batch(vocab,&batch); 
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double tokens_per_second = token_count / elapsed.count();


        std::cout << "\n\n[Stats: Generated " << token_count << " tokens in " 
        << std::fixed << elapsed.count() << " seconds | "
        << std::fixed << tokens_per_second << " tokens/sec]\n\n";

        // Add the model's response to the message history
        messages.push_back({"assistant", strdup(response.c_str())});
        std::cout << "\n\nUser: ";
    }

    // Clean up messages
    for (size_t i = 1; i < messages.size(); i++) {
        free((void*)messages[i].content);
    }
    
    llama_sampler_free(smpl_chain); // also frees the individual samplers added to chain
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
