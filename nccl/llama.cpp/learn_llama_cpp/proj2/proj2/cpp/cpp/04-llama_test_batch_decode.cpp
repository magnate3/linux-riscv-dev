#include "llama.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
using namespace std;

// The code exports plain C function llama_test_batch_decode()
// that is intended to test
// llama_batch_get_one() vs llama_batch_init()
// new API against
// https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
// batch implementation and usage details in decode are here:
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-batch.cpp
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp
// The code is fail-fast and performance is not a goal.

struct ModelDeleter {
    void operator()(llama_model* model) const {
        if (model) llama_model_free(model);
    }
};
static bool LLAMA_BATCH_GET_ONE; // llama_batch_get_one() vs llama_batch_init()

static llama_seq_id seq = 0;

static const int32_t n_batch  = 64;
static const int32_t n_ubatch = 32;

static const int32_t predict = 8; // number of tokens to predict

static const char* prompt =
    "Jabberwocky\n"
    "\n"
    "'Twas brillig, and the slithy toves\n"
    "Did gyre and gimble in the wabe;\n"
    "All mimsy were the borogoves,\n"
    "And the mome raths outgrabe.\n"
    "\n"
    "\"Beware the Jabberwock, my son!\n"
    "The jaws that bite, the claws that catch!\n"
    "Beware the Jubjub bird, and shun\n"
    "The frumious Bandersnatch!\"\n"
    "\n"
    "He took his vorpal sword in hand:\n"
    "Long time the manxome foe he sought—\n"
    "So rested he by the Tumtum tree,\n"
    "And stood awhile in thought.\n"
    "\n"
    "And as in uffish thought he stood,\n"
    "The Jabberwock, with eyes of flame,\n"
    "Came whiffling through the tulgey wood,\n"
    "And burbled as it came!\n"
    "\n"
    "One, two! One, two! And through and through\n"
    "The vorpal blade went snicker-snack!\n"
    "He left it dead, and with its head\n"
    "He went galumphing back.\n"
    "\n"
    "\"And hast thou slain the Jabberwock?\n"
    "Come to my arms, my beamish boy!\n"
    "O frabjous day! Callooh! Callay!\"\n"
    "He chortled in his joy.\n"
    "\n"
    "'Twas brillig, and the slithy toves\n"
    "Did gyre and gimble in the wabe;\n"
    "All mimsy were the borogoves,\n"
    "And the mome raths outgrabe.";

struct str {
    char* data;
    int32_t bytes;
};

static void str_free(struct str * s) {
    if (s->data) { free(s->data); }
    memset(s, 0, sizeof(*s));
}

static bool str_grow(struct str * s, size_t bytes) {
    assert(bytes <= INT32_MAX);
    assert(bytes > s->bytes); // only growing supported
    char* data = s->data != NULL ? (char*)realloc(s->data, (size_t)bytes) :
                                   (char*)malloc((size_t)bytes);
    if (data) {
        s->data  = data;
        s->bytes = (int32_t)bytes;
        return true;
    } else { // malloc()/realloc() failed
        return false;
    }
}

static bool str_append(struct str * s, const char* a) {
    const size_t n = strlen(a);
    const int32_t was = s->bytes;
    if (!str_grow(s, s->bytes + n + 1)) { return false; }
    char* d = s->data;
    if (was > 0) { d += was - 1; } // NUL termination
    memcpy(d, a, n + 1); // including NUL char
    return true;
}

static bool is_batch_allocated(struct llama_batch * b, int32_t tc) {
    if (LLAMA_BATCH_GET_ONE) {
        return b->n_tokens == tc && b->token != nullptr;
    } else {
        bool allocated = // everything is malloc() != NULL?
            b->n_tokens == tc && (b->embd || b->token) &&
            b->pos && b->n_seq_id && b->seq_id && b->logits;
        if (allocated) {
            for (int i = 0; i < tc && allocated; i++) {
                allocated = b->seq_id[i];
            }
            assert(b->seq_id[tc] == nullptr);
        }
        return allocated;
    }
}

static llama_batch batch_new(const llama_memory_t mem,
                             llama_token * tokens, int32_t tc) {
    if (LLAMA_BATCH_GET_ONE) {
        return llama_batch_get_one(tokens, tc);
    } else {
        struct llama_batch b = llama_batch_init(tc, 0, 1);
        assert(b.embd == nullptr);
        b.n_tokens = tc;
        assert(is_batch_allocated(&b, tc));
        llama_pos mem_pos = llama_memory_seq_pos_max(mem, seq) + 1;
        for (int32_t i = 0; i < tc; i++) {
            b.token[i]     = tokens[i];
            b.pos[i]       = mem_pos + i;
            b.n_seq_id[i]  = 1;    // number of seq ids:
            b.seq_id[i][0] = seq;
            b.logits[i]    = (i == tc - 1); // output last logit
        }
        return b;
    }
}

static llama_batch batch_view(struct llama_batch * base, int32_t i, int32_t tc) {
    if (LLAMA_BATCH_GET_ONE) {
        llama_batch b = { .n_tokens = tc, .token = base->token + i };
        return b;
    } else {
        assert(base->embd == nullptr);
        assert(is_batch_allocated(base, base->n_tokens));
        llama_batch b = { // sub batch view
            .n_tokens = tc,
            .token    = base->token    + i,
            .embd     = nullptr,
            .pos      = base->pos      + i,
            .n_seq_id = base->n_seq_id + i,
            .seq_id   = base->seq_id   + i,
            .logits   = base->logits   + i,
        };
        return b;
    }
}

static void batch_free(struct llama_batch * b) {
    if (LLAMA_BATCH_GET_ONE) {
        (void)b; // unused
    } else {
        llama_batch_free(*b);
        memset(b, 0, sizeof(*b));
    }
}

static bool token_to_piece(const struct llama_vocab * v,
                           llama_token t, struct str * s) {
    int32_t bytes = s->bytes <= 0 ? 0 : s->bytes - 1;
    bytes = llama_token_to_piece(v, t, s->data, bytes, 0, true);
    if (bytes >= 0) {
        assert(bytes < s->bytes);
        s->data[bytes] = 0; // NUL terminate
    } else {
        bytes = -bytes + 1;
        assert(bytes > s->bytes);
        if (str_grow(s, bytes)) {
            bytes = llama_token_to_piece(v, t, s->data, bytes - 1, 0, true);
            assert(bytes == s->bytes - 1);
            s->data[bytes] = 0; // NUL terminate
        } else { // malloc()/realloc() failed
            str_free(s);
        }
    }
    return s->data != nullptr;
}

static struct str test(struct llama_model * m) {
    #define return_on_error(call) do {                  \
        fprintf(stderr, "error: " call " failed\n");    \
        assert(!call);                                  \
        str_free(&out);                                 \
        return out;                                     \
    } while (1)
    #define return_and_free_on_error(call) do {         \
        batch_free(&b);                                 \
        str_free(&s);                                   \
        return_on_error(call);                          \
    } while (1)
    struct str out = {}; // output
    assert(seq == 0);
    assert(predict > 0);
    const struct llama_vocab * v = llama_model_get_vocab(m);
    int32_t n = (int32_t)strlen(prompt);
    assert(n > 0);
    const int32_t tc = -llama_tokenize(v, prompt, n, NULL, 0, true, true);
    llama_token * tokens = (llama_token *)alloca(sizeof(llama_token) * tc);
    if (llama_tokenize(v, prompt, n, tokens, tc, true, true) < 0) {
        return_on_error("llama_tokenize()");
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = tc + predict - 1; // exact
    cp.n_batch  = n_batch;
    cp.n_ubatch = n_ubatch;
    cp.no_perf  = false; // enable performance counters
    struct llama_context * c = llama_init_from_model(m, cp);
    if (!c) { return_on_error("llama_init_from_model()"); }
    const llama_memory_t mem = llama_get_memory(c);
    if (!m) { return_on_error("llama_get_memory()"); } // encoder only model
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = false; // enable performance counters
    struct llama_sampler * sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
    llama_token t = LLAMA_TOKEN_NULL;
    struct str s = {};   // token piece
    for (int32_t i = 0; i < tc; i++) {
        t = tokens[i];
        if (!token_to_piece(v, t, &s)) {
            str_free(&s);
            return_on_error("llama_token_to_piece()");
        }
        printf("%.*s", s.bytes, s.data); // s.data is also NUL terminated
    }
    printf("\n");
    struct llama_batch b = batch_new(mem, tokens, tc);
    if (llama_model_has_encoder(m)) {
        if (llama_encode(c, b)) {
            return_and_free_on_error("llama_encode()");
        }
        t = llama_model_decoder_start_token(m);
        if (t == LLAMA_TOKEN_NULL) { t = llama_vocab_bos(v); }
        batch_free(&b);
        b = batch_new(mem, &t, 1);
        if (!is_batch_allocated(&b, 1)) {
            return_and_free_on_error("batch_new()");
        }
    }
    const int64_t t_main_start = ggml_time_us();
    int generated = 0;
    int32_t pos = 0;
    while (pos + b.n_tokens < tc + predict) {
        assert(b.n_tokens > 0);
        llama_pos mem_pos = llama_memory_seq_pos_max(mem, seq) + 1;
        assert(mem_pos >= 0);
        for (int32_t i = 0; i < b.n_tokens; i += n_batch) {
            // the number of tokens left in this batch to proceed:
            int32_t vtc = b.n_tokens - i; // view token count
            if (vtc > n_batch) { vtc = n_batch; }
            llama_batch view = batch_view(&b, i, vtc);
            int32_t r = llama_decode(c, view);
            if (r != 0) { return_and_free_on_error("llama_decode()"); }
        }
        pos += b.n_tokens;
        assert(llama_memory_seq_pos_max(mem, seq) + 1 - mem_pos == b.n_tokens);
        t = llama_sampler_sample(sampler, c, -1);
        if (llama_vocab_is_eog(v, t)) { break; } // end of generation?
        if (!token_to_piece(v, t, &s)) {
            return_and_free_on_error("llama_token_to_piece()");
        }
        printf("%.*s", s.bytes, s.data); // s.data is also NUL terminated
        if (!str_append(&out, s.data)) {
            return_and_free_on_error("str_append()");
        }
        fflush(stdout); // intentional flush
        batch_free(&b);
        b = batch_new(mem, &t, 1);
        if (!is_batch_allocated(&b, 1)) {
            return_and_free_on_error("batch_new()");
        }
        generated++; // count only generated (sampled) tokens
    }
    str_free(&s);
    batch_free(&b);
    printf("\n");
    const int64_t t_main_end = ggml_time_us();
    fprintf(stderr, "decoded %d tokens in %.2f s, speed: %.2f t/s\n\n",
            generated,   (t_main_end - t_main_start) / 1000000.0f,
            generated / ((t_main_end - t_main_start) / 1000000.0f));
    llama_perf_sampler_print(sampler);
    llama_perf_context_print(c);
    fprintf(stderr, "\n");
    llama_sampler_free(sampler);
    llama_free(c);
    return out;
}

void llama_test_batch_decode(struct llama_model * p) {
    struct llama_model * m = p;
    LLAMA_BATCH_GET_ONE = true;
    struct str s1 = test(m);
    LLAMA_BATCH_GET_ONE = false;
    struct str s2 = test(m);
    bool same = s1.data && s2.data && strcmp(s1.data, s2.data) == 0;
    str_free(&s1);
    str_free(&s2);
    if (!same) {
        fprintf(stderr, "llama_test_batch_decode() failed\n");
    } else {
        printf("llama_test_batch_decode() succeeded\n");
    }
}

void print_usage(const char* program_name) {
    std::cout << "\nUsage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH      Path to the GGUF model file (required)\n";
    std::cout << "  -p, --prompt TEXT     Text prompt for generation (default: \"Hello, my name is\")\n";
    std::cout << "  -n, --n-predict N     Number of tokens to generate (default: 32)\n";
    std::cout << "  -ngl, --n-gpu-layers N  Number of layers to offload to GPU (default: 99)\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " -m model.gguf -p \"The quick brown fox\" -n 50\n";
    std::cout << std::endl;
}
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
int main(int argc, char** argv) {
    // Configuration parameters
    std::string model_path;
    std::string prompt = "Hello, my name is";
    int n_predict = 32;
    int n_gpu_layers = 99;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Model: " << model_path << "\n";
    std::cout << "  Prompt: \"" << prompt << "\"\n";
    std::cout << "  Tokens to generate: " << n_predict << "\n";
    std::cout << "  GPU layers: " << n_gpu_layers << "\n" << std::endl;

    // Step 1: Load dynamic backends (enables GPU support)
    ggml_backend_load_all();
    std::cout << "Loaded backends" << std::endl;

    // Step 2: Initialize and load the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    std::cout << "Loading model from: " << model_path << std::endl;
    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params));

    if (!model) {
        std::cerr << "Error: Failed to load model from " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;
    llama_test_batch_decode(model.get());
}
/*
 * NEW BATCH API EXPLANATION (post-continuous batching, b2000+)
 *
 * The new llama_batch API supports:
 * - CONTINUOUS BATCHING: Append tokens to KV cache without full re-decode.
 * - MICRO-BATCHING: llama_decode() internally splits large batches into <= n_ubatch
 *   chunks if KV slots are limited, retrying with llama_memory_update().
 * - MULTI-SEQUENCE: Batches can mix multiple independent sequences (e.g., parallel users).
 * - SPARSE/DENSE: Tokens can belong to 0+ sequences via n_seq_id[i] and seq_id[i][j].
 * - EMBEDDINGS-ONLY MODE: Optional b.embd[] input instead of b.token[].
 *
 * KEY FIELDS:
 * - b.pos[n_tokens]: Absolute position in sequence for RoPE/positional embeddings.
 *   - If NULL: AUTO-COMPUTED as llama_memory_seq_pos_max(mem, DEFAULT_SEQ_ID) + 1 + k
 *     for k-th token. DEFAULT_SEQ_ID=0 unless b.all_seq_id set.
 *   - Must be CONTINUOUS and INCREASING across batch tokens for a sequence.
 * - b.n_seq_id[n_tokens]: Number of sequences this token belongs to (usually 1).
 * - b.seq_id[n_tokens][n_seq_id[i]]: Array of seq_ids for this token.
 *   - If NULL: Defaults to b.all_seq_id (single seq_id for whole batch).
 * - b.logits[n_tokens]: bool array; true=compute logits at this pos.
 *
 * llama_memory_seq_pos_max(mem, seq_id):
 * - Returns CURRENT END POSITION of seq_id in KV cache (max pos already stored).
 * - For NEW TOKENS in seq_id: Start positions at max_pos + 1.
 * - Updated AFTER each successful llama_decode() for affected seqs.
 * - Enables CHUNKED DECODING: Decode prompt in n_batch-sized views;
 *   each view's pos starts from UPDATED max_pos + 1 (auto or explicit).
 *
 * llama_batch_get_one(tokens, n_tokens):
 * - Returns STACK-ALLOCATED struct (NO FREE NEEDED!).
 * - Sets: b.token = tokens, b.n_tokens = n_tokens, ALL OTHER FIELDS = NULL.
 * - AUTO-FILLS in llama_decode():
 *   - pos[] = sequential from llama_memory_seq_pos_max(mem, 0) + 1
 *   - seq_id = {0}, n_seq_id=1 for all tokens (single seq).
 *   - logits[] = {false, ..., true} (only last token).
 * - IDEAL FOR: Simple single-seq batches (prompts, 1-token generation).
 *
 * EXPLICIT ALLOCATION via llama_batch_init(n_tokens, n_seq_max, n_embd):
 * - Allocates: pos[], n_seq_id[], seq_id[][] (jagged), logits[].
 * - MUST SET:
 *   - pos[i] = llama_memory_seq_pos_max(mem, seq) + 1 + i  (CONTINUOUS!)
 *   - n_seq_id[i] = 1
 *   - seq_id[i][0] = seq  (seq=0 for single)
 *   - logits[tc-1] = true
 * - Use llama_batch_free(b) to free.
 * - Matches EXACTLY what get_one auto-does.
 *
 * CHUNKING EXAMPLE (large prompt tc > n_batch):
 * for (i=0; i<tc; i += n_batch) {
 *     vtc = min(n_batch, tc - i);
 *     view = batch_view(&b, i, vtc);  // shallow offsets
 *     r = llama_decode(c, view);
 *     // AFTER: mem seq_pos_max += vtc
 * }
 * Next view's pos auto-starts at UPDATED max_pos + 1.
 *
 * VS OLD API (pre-b2000):
 * - NO continuous batching: Full prompt decode ONCE, then 1-token loop.
 * - NO llama_memory: Direct lctx->seq_id_pos[] management.
 * - Positions: Hardcoded b.pos[i] = i  (ALWAYS from 0!).
 * - Single seq ONLY: No seq_id arrays; implicit seq=0.
 * - llama_batch_get_one(tokens, n): OLD sig (2 args); set pos=i implicitly.
 * - NO chunking: n_batch=512 limit; large prompts FAILED.
 * - Generation: Append 1 token, llama_decode(1), sample, repeat.
 * - Memory: Full re-alloc KV per decode for new tokens.
 *
 * WHY THIS CODE WORKS FOR BOTH:
 * - get_one branch: Relies on AUTO pos/seq_id fill.
 * - init branch: Explicitly matches auto values.
 * - Views: Preserve offsets; decode updates mem incrementally.
 * - Single seq=0: Simplest case.
 * - Encoder-decoder (has_encoder): Separate encode(), then decoder batch starts fresh.
 *
 * RULES:
 * - ALWAYS use llama_memory_seq_pos_max() for start pos.
 * - NEVER overlap positions; decode fails with GGML_FAIL.
 * - Multi-seq: Set b.all_seq_id or per-token seq_id[].
 * - Free ONLY init batches; get_one is stack/no-op.
 * - Check r = llama_decode() !=0 for errors (KV full, invalid pos).
 *
 * See: examples/simple/simple.cpp (old), examples/parallel/parallel.cpp (new).
 */

