#include "llama.h"
#include <iostream>
#include <vector>
#include <string>


// Here’s a corrected version of your chat loop that actually generates responses:
int main() {
    // Load model
    struct llama_model_params model_params = llama_model_default_params();
    struct llama_context_params ctx_params = llama_context_default_params();

    struct llama_model *model = llama_load_model_from_file("./models/gguf-model.bin", model_params);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    struct llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context\n";
        return 1;
    }

    std::vector<llama_token> conversation_history; // Store conversation tokens

    while (true) {
        std::string user_input;
        std::cout << "You: ";
        std::getline(std::cin, user_input);

        if (user_input == "exit") break; // Exit condition

        // Tokenize user input
        std::vector<llama_token> tokens(512);
        int n_tokens = llama_tokenize(ctx, user_input.c_str(), tokens.data(), tokens.size(), true);
        if (n_tokens < 0) {
            std::cerr << "Tokenization failed!\n";
            continue;
        }

        // Append tokens to conversation history
        conversation_history.insert(conversation_history.end(), tokens.begin(), tokens.begin() + n_tokens);

        // Process all tokens in conversation history
        if (llama_decode(ctx, conversation_history.data(), conversation_history.size()) != 0) {
            std::cerr << "Decoding failed!\n";
            break;
        }

        // Generate response
        std::cout << "AI: ";
        for (int i = 0; i < 50; i++) {  // Generate 50 tokens (adjust as needed)
            llama_token next_token = llama_sample_top_p(ctx, nullptr, 0.9);
            if (next_token == llama_token_eos(ctx)) break; // Stop at end-of-sequence token
            
            std::cout << llama_token_to_str(ctx, next_token);
            conversation_history.push_back(next_token); // Append response to history
        }
        std::cout << "\n";
    }

    // Cleanup
    llama_free(ctx);
    return 0;
}

// Example: Generating One Token at a Time
// Tokenize user input
std::vector<llama_token> tokens(512);
int n_tokens = llama_tokenize(ctx, user_input.c_str(), tokens.data(), tokens.size(), true);

// Run the model to process input tokens
if (llama_decode(ctx, tokens.data(), n_tokens) != 0) {
    std::cerr << "Decoding failed!\n";
}

// Sample and generate response
for (int i = 0; i < 50; i++) { // Generate 50 tokens (adjust as needed)
    llama_token next_token = llama_sample_top_p(ctx, nullptr, 0.9); // Sample from logits
    if (next_token == llama_token_eos(ctx)) break; // Stop at end-of-sequence token

    std::cout << llama_token_to_str(ctx, next_token); // Print generated word
}