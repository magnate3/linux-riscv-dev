/**
 * KV Cache Single-Thread Debug Program
 * 
 * Directly calls llama.cpp C API for easy step-by-step debugging of KV cache behavior
 * 
 * Build (from llama.cpp root directory):
 *   cmake -B build -DCMAKE_BUILD_TYPE=Debug
 *   cmake --build build --config Debug -j $(nproc)
 *   cmake --build build --target debug_kv_single_thread
 * 
 * Run:
 *   ./build/bin/debug_kv_single_thread /path/to/model.gguf
 * 
 * GDB Debugging:
 *   gdb --args ./build/bin/debug_kv_single_thread /path/to/model.gguf
 *   (gdb) break llama_kv_cache::find_slot
 *   (gdb) break llama_kv_cache::apply_ubatch
 *   (gdb) run
 */

#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// ============================================================
// Helper Functions
// ============================================================

static void print_separator(const char* title) {
    printf("\n");
    printf("============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

static void print_kv_cache_info(llama_context* ctx) {
    // Get KV cache usage info
    llama_memory_t mem = llama_get_memory(ctx);
    uint32_t n_kv_total = llama_n_ctx(ctx);
    
    // Simplified output - detailed cell usage should be inspected in GDB
    printf("  KV Cache: n_ctx=%d, memory=%p\n", n_kv_total, (void*)mem);
}

// ============================================================
// Main Test Program
// ============================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    
    print_separator("1. Initialize Backend");
    
    // Initialize ggml backend
    ggml_backend_load_all();
    
    print_separator("2. Load Model");
    
    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for easier debugging
    
    printf("  Loading model: %s\n", model_path);
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        printf("  Error: Failed to load model!\n");
        return 1;
    }
    printf("  Model loaded successfully!\n");
    
    print_separator("3. Create Context (KV Cache allocated here)");
    
    // Context parameters - critical settings!
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 256;         // Total 256 cells
    ctx_params.n_batch = 64;        // Max 64 tokens per batch
    ctx_params.n_ubatch = 32;       // 32 tokens per ubatch
    ctx_params.n_seq_max = 1;       // Set to 1 to simplify debugging (unified mode)
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // Disable Flash Attention
    ctx_params.offload_kqv = false; // Disable KQV offload to GPU
    ctx_params.op_offload = false;  // Disable op offload to GPU
    
    printf("  n_ctx = %d (number of KV cache cells)\n", ctx_params.n_ctx);
    printf("  n_batch = %d (logical batch size)\n", ctx_params.n_batch);
    printf("  n_ubatch = %d (physical ubatch size)\n", ctx_params.n_ubatch);
    printf("  n_seq_max = %d (max sequences)\n", ctx_params.n_seq_max);
    
    printf("\n  >>> Set breakpoint here: llama_kv_cache::llama_kv_cache <<<\n");
    printf("  >>> Inspect: kv_size, n_stream, unified params <<<\n\n");
    
    // Create context - KV cache is allocated here!
    // ★★★ Breakpoint: llama_kv_cache constructor ★★★
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        printf("  Error: Failed to create context!\n");
        llama_model_free(model);
        return 1;
    }
    printf("  Context created successfully!\n");
    print_kv_cache_info(ctx);
    
    print_separator("4. Tokenize Test Prompts");
    
    // Two test prompts to simulate concurrent requests
    const char* prompt1 = "Hello, my name is Alice and I am a";
    const char* prompt2 = "The capital of France is Paris, which";
    
    // Tokenize
    const int max_tokens = 64;
    std::vector<llama_token> tokens1(max_tokens);
    std::vector<llama_token> tokens2(max_tokens);
    
    int n_tokens1 = llama_tokenize(
        llama_model_get_vocab(model),
        prompt1, strlen(prompt1),
        tokens1.data(), max_tokens,
        true,   // add_special (BOS)
        false   // parse_special
    );
    tokens1.resize(n_tokens1);
    
    int n_tokens2 = llama_tokenize(
        llama_model_get_vocab(model),
        prompt2, strlen(prompt2),
        tokens2.data(), max_tokens,
        true,
        false
    );
    tokens2.resize(n_tokens2);
    
    printf("  Prompt 1: \"%s\"\n", prompt1);
    printf("    -> %d tokens\n", n_tokens1);
    
    printf("  Prompt 2: \"%s\"\n", prompt2);
    printf("    -> %d tokens\n", n_tokens2);
    
    print_separator("5. Scenario A: Single Sequence Prefill");
    
    printf("  Processing Prompt 1 (seq_id = 0)\n");
    printf("\n  >>> Set breakpoint here: llama_kv_cache::find_slot <<<\n");
    printf("  >>> Inspect: ubatch.n_tokens, ubatch.seq_id <<<\n\n");
    
    // Create batch
    llama_batch batch1 = llama_batch_init(ctx_params.n_batch, 0, 1);
    
    // Fill batch - seq_id = 0
    for (int i = 0; i < n_tokens1; i++) {
        common_batch_add(batch1, tokens1[i], i, { 0 }, false);
    }
    batch1.logits[batch1.n_tokens - 1] = true;  // Last token needs logits
    
    printf("  Batch 1: n_tokens=%d, seq_id=0\n", batch1.n_tokens);
    print_kv_cache_info(ctx);
    
    // ★★★ Breakpoints: find_slot(), apply_ubatch() ★★★
    printf("\n  Calling llama_decode() ...\n");
    int ret = llama_decode(ctx, batch1);
    if (ret != 0) {
        printf("  Error: llama_decode failed! ret=%d\n", ret);
    } else {
        printf("  Decode successful!\n");
    }
    
    print_kv_cache_info(ctx);
    llama_batch_free(batch1);
    
    print_separator("6. Scenario B: Continue Generation (Decode Phase)");
    
    printf("  Continue generation for seq_id=0 with next token\n");
    printf("  Observe: How KV cache appends new cell\n\n");
    
    // Simulate generation phase: add one token at a time
    llama_token next_token = 1000;  // Assume sampled token
    
    llama_batch batch2 = llama_batch_init(ctx_params.n_batch, 0, 1);
    common_batch_add(batch2, next_token, n_tokens1, { 0 }, true);  // pos = n_tokens1
    
    printf("  Batch 2: n_tokens=%d, seq_id=0, pos=%d\n", batch2.n_tokens, n_tokens1);
    print_kv_cache_info(ctx);
    
    printf("\n  >>> Observe how find_slot() finds next free cell <<<\n\n");
    
    printf("\n  Calling llama_decode() ...\n");
    ret = llama_decode(ctx, batch2);
    if (ret != 0) {
        printf("  Error: llama_decode failed! ret=%d\n", ret);
    } else {
        printf("  Decode successful!\n");
    }
    
    print_kv_cache_info(ctx);
    llama_batch_free(batch2);
    
    print_separator("7. Scenario C: Generate More Tokens");
    
    printf("  Generate 3 tokens sequentially, observe continuous cell allocation\n\n");
    
    for (int gen = 0; gen < 3; gen++) {
        llama_batch batch_gen = llama_batch_init(ctx_params.n_batch, 0, 1);
        llama_token tok = 1001 + gen;
        int pos = n_tokens1 + 1 + gen;
        
        common_batch_add(batch_gen, tok, pos, { 0 }, true);
        printf("  Generate token %d: pos=%d\n", gen + 1, pos);
        
        ret = llama_decode(ctx, batch_gen);
        if (ret != 0) {
            printf("  Error: llama_decode failed!\n");
        }
        llama_batch_free(batch_gen);
    }
    
    print_kv_cache_info(ctx);
    
    print_separator("8. Scenario D: Clear KV (Simulate Request Completion)");
    
    printf("  Clear KV cache for seq_id=0\n\n");
    
    printf("  >>> Observe: how llama_memory_seq_rm() frees cells <<<\n\n");
    
    llama_memory_t mem = llama_get_memory(ctx);
    bool removed = llama_memory_seq_rm(mem, 0, -1, -1);
    printf("  llama_memory_seq_rm(seq_id=0): %s\n", removed ? "success" : "failed");
    
    print_kv_cache_info(ctx);
    
    print_separator("9. Scenario E: New Request Reuses Freed Space");
    
    printf("  Send new request (seq_id=0), should reuse previously freed space\n\n");
    
    const char* prompt3 = "In machine learning, neural networks";
    std::vector<llama_token> tokens3(max_tokens);
    int n_tokens3 = llama_tokenize(
        llama_model_get_vocab(model),
        prompt3, strlen(prompt3),
        tokens3.data(), max_tokens,
        true, false
    );
    tokens3.resize(n_tokens3);
    
    llama_batch batch4 = llama_batch_init(ctx_params.n_batch, 0, 1);
    for (int i = 0; i < n_tokens3; i++) {
        common_batch_add(batch4, tokens3[i], i, { 0 }, false);  // seq_id = 0 (reuse)
    }
    batch4.logits[batch4.n_tokens - 1] = true;
    
    printf("  Batch 4: n_tokens=%d, seq_id=0\n", batch4.n_tokens);
    
    printf("\n  >>> Observe: how find_slot() finds previously freed space <<<\n\n");
    
    printf("\n  Calling llama_decode() ...\n");
    ret = llama_decode(ctx, batch4);
    if (ret != 0) {
        printf("  Error: llama_decode failed! ret=%d\n", ret);
    } else {
        printf("  Decode successful!\n");
    }
    
    print_kv_cache_info(ctx);
    llama_batch_free(batch4);
    
    print_separator("10. Cleanup");
    
    llama_free(ctx);
    llama_model_free(model);
    
    printf("  Cleanup completed!\n\n");
    
    print_separator("Debugging Tips");
    
    printf("  Key breakpoints:\n");
    printf("    1. llama_kv_cache::llama_kv_cache - constructor, observe initialization\n");
    printf("    2. llama_kv_cache::find_slot     - find free cells\n");
    printf("    3. llama_kv_cache::apply_ubatch  - update cell metadata\n");
    printf("    4. llama_kv_cache::cpy_k/cpy_v   - write K/V data\n");
    printf("    5. llama_kv_cache::seq_rm        - remove sequence\n");
    printf("\n");
    printf("  Key variables:\n");
    printf("    - kv_size: total number of cells\n");
    printf("    - n_stream: number of streams (unified=1)\n");
    printf("    - v_cells[s]: cell state for each stream\n");
    printf("    - v_heads[s]: search head for each stream\n");
    printf("    - ubatch.seq_id: sequence ID of tokens in batch\n");
    printf("    - sinfo.idxs: list of allocated cell indices\n");
    printf("\n");
    
    return 0;
}
