// End-to-end test for KV cache compaction
//
// Loads a small model, prefills context, compacts via attention matching,
// writes back compacted state, and generates tokens.
//
// Validates:
//   - State round-trip (save → parse → compact → write → load)
//   - Generation works after compaction (no crashes, produces tokens)
//   - Compacted state is smaller than original
//
// Usage:
//   test-kv-compact-e2e -m <model_path> [-c <ctx_size>] [--compact-ratio <ratio>]
//
// Requires a small model (e.g., TinyLlama-1.1B, Qwen2-0.5B, or similar).
// The test uses a fixed prompt and validates basic generation quality.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kv-compact-math.h"
#include "kv-compact-state.h"

static const char * TEST_PROMPT =
    "The quick brown fox jumps over the lazy dog. "
    "In a small village nestled between rolling hills, there lived an old clockmaker "
    "who spent his days repairing timepieces from all over the country. His workshop "
    "was filled with the gentle ticking of hundreds of clocks, each one telling its "
    "own story of time. The clockmaker believed that every clock had a soul, and he "
    "treated each one with the utmost care and respect.";

static std::string generate_n_tokens(llama_context * ctx, llama_model * model,
                                     const llama_vocab * vocab,
                                     common_params & params,
                                     llama_pos start_pos, int n) {
    std::string output;
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    for (int i = 0; i < n; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        output += common_token_to_piece(vocab, id);
        common_sampler_accept(smpl, id, true);

        common_batch_clear(batch);
        common_batch_add(batch, id, start_pos + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) break;
    }

    common_sampler_free(smpl);
    llama_batch_free(batch);
    return output;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered stdout for debug
    common_params params;
    float compact_ratio = 0.5f;  // keep 50% — conservative for testing

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        fprintf(stderr, "Usage: %s -m <model> [-c <ctx_size>]\n", argv[0]);
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();
    assert(ctx && "Failed to create context");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_ctx     = llama_n_ctx(ctx);
    const int n_layer   = llama_model_n_layer(model);
    const int n_head    = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    const int n_embd    = llama_model_n_embd(model);
    const int d_k       = n_embd / n_head;
    const int d_v       = d_k;
    const enum llama_rope_type rope_type = llama_model_rope_type(model);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                     rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    printf("E2E test: %d layers, %d KV heads, d_k=%d, ctx=%d, rope_type=%d, n_pos_per_embd=%u\n",
           n_layer, n_head_kv, d_k, n_ctx, (int)rope_type, n_pos_per_embd);

    // ---- Test 1: Prefill ----
    printf("\n[TEST 1] Prefill...\n");

    std::vector<llama_token> tokens = common_tokenize(vocab, TEST_PROMPT, true, false);
    const int n_tokens = (int) tokens.size();
    printf("  Input: %d tokens\n", n_tokens);
    assert(n_tokens >= 16 && "Prompt too short");

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }
    int rc = llama_decode(ctx, batch);
    assert(rc == 0 && "Prefill failed");
    printf("  PASS: prefill complete\n");

    // ---- Test 2: Save state ----
    printf("\n[TEST 2] Save state...\n");

    const size_t state_size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state_buf(state_size);
    size_t saved = llama_state_seq_get_data(ctx, state_buf.data(), state_buf.size(), 0);
    assert(saved > 0 && "Failed to save state");
    printf("  PASS: saved %zu bytes (%.2f MB)\n", saved, saved / (1024.0 * 1024.0));

    // ---- Test 3: Generate with full cache (reference) ----
    printf("\n[TEST 3] Generate with full cache...\n");

    const int n_gen = 32;
    std::string full_output = generate_n_tokens(ctx, model, vocab, params, n_tokens, n_gen);
    assert(!full_output.empty() && "No tokens generated with full cache");
    printf("  PASS: generated %zu chars: \"%.60s...\"\n", full_output.size(), full_output.c_str());

    // ---- Test 4: Parse state buffer ----
    printf("\n[TEST 4] Parse state buffer...\n");

    parsed_kv_state kv_state;
    bool parsed = kv_state.parse(state_buf.data(), saved, n_pos_per_embd);
    assert(parsed && "Failed to parse state buffer");
    assert(kv_state.n_stream > 0);
    assert(kv_state.streams[0].cell_count == (uint32_t)n_tokens);
    // Debug: dump first 64 bytes of state buffer as hex + u32 values
    printf("  State buffer first 64 bytes:\n  ");
    for (int i = 0; i < 64 && i < (int)saved; i++) {
        printf("%02x ", state_buf[i]);
        if ((i + 1) % 16 == 0) printf("\n  ");
    }
    printf("\n  As u32: ");
    for (int i = 0; i < 16 && i * 4 < (int)saved; i++) {
        uint32_t val;
        memcpy(&val, state_buf.data() + i * 4, 4);
        printf("[%d]=%u ", i, val);
    }
    printf("\n");

    // For hybrid models (SSM+MoE), only attention layers have KV cache
    // The state may contain fewer layers than llama_model_n_layer() reports
    const uint32_t n_kv_layer = kv_state.streams[0].n_layer;
    printf("  n_stream=%u, cell_count=%u, n_kv_layer=%u, n_layer=%d\n",
           kv_state.n_stream, kv_state.streams[0].cell_count, n_kv_layer, n_layer);
    printf("  State buffer: %zu bytes saved\n", saved);
    if (n_kv_layer == 0) {
        printf("  ERROR: n_kv_layer=0 — parser could not find any KV layers.\n");
        printf("  This may indicate a parsing issue with the state format.\n");
        llama_batch_free(batch);
        llama_backend_free();
        return 1;
    }

    // Validate K/V dimensions
    const auto & ld0 = kv_state.streams[0].layers[0];
    int parsed_embd_k = ld0.n_embd_k_gqa();
    int expected_embd_k = n_head_kv * d_k;
    printf("  K embd: parsed=%d, expected=%d (n_head_kv=%d, d_k=%d, n_head=%d, n_embd=%d)\n",
           parsed_embd_k, expected_embd_k, n_head_kv, d_k, n_head, n_embd);
    // n_embd_k_gqa may differ from n_head_kv * (n_embd/n_head) for some architectures
    // (e.g., models with different K/V head dimensions). Use parsed value as ground truth.
    int actual_d_k = parsed_embd_k / n_head_kv;
    int actual_d_v = ld0.n_embd_v_gqa_computed() / n_head_kv;
    printf("  Actual d_k=%d, d_v=%d (from state parser)\n", actual_d_k, actual_d_v);
    assert(parsed_embd_k > 0 && "K dimension is zero");
    assert(parsed_embd_k % n_head_kv == 0 && "K dimension not divisible by n_head_kv");
    printf("  PASS: dimensions valid\n");

    // ---- Test 5: Compact all layers ----
    printf("\n[TEST 5] Compact all layers...\n");

    const int t = std::max(1, (int)(n_tokens * compact_ratio));
    const int n_ref = std::max(8, n_tokens / 4);
    const int T = n_tokens;
    const int n_embd_k_gqa = parsed_embd_k;
    const int n_embd_v_gqa = ld0.n_embd_v_gqa_computed();
    const float inv_sqrt_dk = 1.0f / sqrtf((float) actual_d_k);
    const int ref_start = T - n_ref;
    const auto & sd = kv_state.streams[0];

    // Global importance scoring
    std::vector<float> global_importance(T, 0.0f);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];

        for (int h = 0; h < n_head_kv; h++) {
            for (int qi = 0; qi < n_ref; qi++) {
                const float * q_row = ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * actual_d_k;
                for (int ki = 0; ki < T; ki++) {
                    const float * k_row = ld.K.data() + ki * n_embd_k_gqa + h * actual_d_k;
                    float dot = 0.0f;
                    for (int d = 0; d < actual_d_k; d++) dot += q_row[d] * k_row[d];
                    // We just need max attention for importance, compute simplified
                    // (skip full softmax for speed in test)
                    float score = dot * inv_sqrt_dk;
                    if (score > global_importance[ki]) global_importance[ki] = score;
                }
            }
        }
    }

    // Select top-t
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_importance[a] > global_importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());

    printf("  Selected %d / %d positions\n", t, T);

    // Per-layer, per-head NNLS + C_v (simplified: use compact_head_highest_attn on extracted heads)
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            // For the E2E test, just copy original V at selected indices (skip NNLS/LSQ for speed)
            cv_all[l][h].resize(t * actual_d_v);
            for (int j = 0; j < t; j++) {
                const float * v_row = ld.V.data() + selected[j] * n_embd_v_gqa + h * actual_d_v;
                memcpy(cv_all[l][h].data() + j * actual_d_v, v_row, actual_d_v * sizeof(float));
            }
        }
    }

    printf("  PASS: compacted all %u layers × %d heads\n", sd.n_layer, n_head_kv);

    // ---- Test 6: Build compacted state ----
    printf("\n[TEST 6] Build compacted state...\n");

    auto compacted_buf = build_compacted_state(kv_state, selected, cv_all, n_head_kv, actual_d_k, actual_d_v, n_pos_per_embd);
    assert(!compacted_buf.empty() && "Failed to build compacted state");
    printf("  Compacted: %zu bytes (%.1f%% of original %zu bytes)\n",
           compacted_buf.size(), 100.0 * compacted_buf.size() / saved, saved);
    if (!kv_state.trailing_data.empty()) {
        printf("  Trailing data (recurrent section): %zu bytes preserved\n",
               kv_state.trailing_data.size());
    }
    assert(compacted_buf.size() <= saved && "Compacted state should not be larger");
    printf("  PASS\n");

    // ---- Test 7: Load compacted state and generate ----
    printf("\n[TEST 7] Load compacted state and generate...\n");

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(), compacted_buf.size(), 0);
    assert(loaded > 0 && "Failed to load compacted state");

    llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    printf("  Loaded. Max pos: %d\n", (int)pos_max);

    std::string compact_output = generate_n_tokens(ctx, model, vocab, params, pos_max + 1, n_gen);
    assert(!compact_output.empty() && "No tokens generated after compaction");
    printf("  PASS: generated %zu chars: \"%.60s...\"\n",
           compact_output.size(), compact_output.c_str());

    // ---- Test 8: Verify state round-trip (parse compacted) ----
    printf("\n[TEST 8] Verify compacted state re-parses...\n");

    parsed_kv_state kv_compact;
    bool reparsed = kv_compact.parse(compacted_buf.data(), compacted_buf.size(), n_pos_per_embd);
    assert(reparsed && "Failed to re-parse compacted state");
    assert(kv_compact.streams[0].cell_count == (uint32_t)t);
    assert(kv_compact.streams[0].n_layer == n_kv_layer);

    // Verify positions are preserved (original positions, not 0..t-1)
    for (int j = 0; j < t; j++) {
        int orig_pos = sd.cells[selected[j]].pos;
        int comp_pos = kv_compact.streams[0].cells[j].pos;
        assert(orig_pos == comp_pos && "Position mismatch after round-trip");
    }
    printf("  PASS: %u cells, positions preserved\n", kv_compact.streams[0].cell_count);

    // ---- Summary ----
    printf("\n=== All E2E tests passed! ===\n");
    printf("  Full cache output:      \"%.60s...\"\n", full_output.c_str());
    printf("  Compacted cache output: \"%.60s...\"\n", compact_output.c_str());
    printf("  Compression: %d → %d tokens (%.1fx)\n", n_tokens, t, (float)n_tokens / t);
    printf("  State size: %zu → %zu bytes (%.1f%%)\n",
           saved, compacted_buf.size(), 100.0 * compacted_buf.size() / saved);

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
