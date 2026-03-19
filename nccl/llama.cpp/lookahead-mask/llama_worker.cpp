#include "llama_worker.hpp"
#include "llama_utils.hpp"

#include <llama.h>
#include <common.h>
#include <fstream>
#include <string>
#include <cmath>
#include <stdexcept>

// Black magic from the llama.cpp main app
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

LlamaWorkerState::LlamaWorkerState()
{
    initialized = false;
    ctx = nullptr;
    // Has not encoded anything yet
    last_decoded_token_index = -1;
}
LlamaWorkerState::LlamaWorkerState(llama_model *model, gpt_params *params) : LlamaWorkerState()
{
    init_ctx(model, params);
}
LlamaWorkerState::~LlamaWorkerState()
{
    llama_free(ctx);
}

void LlamaWorkerState::init_ctx(llama_context *ctx)
{
    initialized = true;
    this->ctx = ctx;
}
void LlamaWorkerState::init_ctx(llama_model *model, gpt_params *params)
{
    initialized = true;
    auto cparams = llama_context_params_from_gpt_params(*params);
    ctx = llama_new_context_with_model(model, cparams);
}
LlamaWorkerState *LlamaWorkerState::clone(
    const LlamaWorkerState *source, llama_model *model, gpt_params *params)
{
    llama_context *ctx_source = source->ctx;
    size_t source_context_size = llama_state_get_size(ctx_source);

    // Copy context
    uint8_t *ctx_data = (uint8_t *)malloc(source_context_size);
    llama_state_get_data(ctx_source, ctx_data);

    auto cparams = llama_context_params_from_gpt_params(*params);
    llama_context *ctx_target = llama_new_context_with_model(model, cparams);
    llama_state_set_data(ctx_target, ctx_data);
    free(ctx_data);

    // Copy tokens
    std::vector<llama_token> cloned_tokens(source->tokens);
    LOG("cloned tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_source, cloned_tokens).c_str());

    // Create the new object
    LlamaWorkerState *cloned = new LlamaWorkerState();
    cloned->init_ctx(ctx_target);
    cloned->tokens = cloned_tokens;
    cloned->last_decoded_token_index = source->last_decoded_token_index;

    return cloned;
}

LlamaWorker::LlamaWorker(
    llama_model *loaded_model,
    gpt_params *locked_params)
{
    // We want to load or create our own context
    model = loaded_model;
    params = locked_params;
    state = new LlamaWorkerState(model, params);

    append_bos = true;
    output_eos = true;
    output_bos = false;
    should_yield = false;
    // Default on_new_token that does absolutely nothing
    on_new_token = [this](std::string token) {
    };
}
LlamaWorker::~LlamaWorker()
{
    if (state != nullptr)
        delete state;
}

void LlamaWorker::stop()
{
    should_yield = true;
}

void LlamaWorker::use_state(const LlamaWorkerState *new_state)
{
    if (state != nullptr)
    {
        delete state;
        state = nullptr;
    }
    state = LlamaWorkerState::clone(new_state, model, params);
}

// Initialize or cache a state for a prompt
LlamaWorkerState *LlamaWorker::create_state_from_prompt(const std::string prompt)
{
    auto state = new LlamaWorkerState(model, params);
    state->tokens = ::llama_tokenize(model, prompt, true, true);
    // Assume we will have evaluated basically the whole prompt
    state->last_decoded_token_index = state->tokens.size() - 1;

    // New context and sampling context
    llama_context *ctx = state->ctx;
    std::vector<llama_token> token_list = state->tokens;

    // Initialize sampling context
    llama_sampling_context *ctx_sampling = llama_sampling_init(params->sparams);
    for (int token_index = 0; token_index < token_list.size(); token_index++)
    {
        auto token = token_list[token_index];
        llama_sampling_accept(ctx_sampling, ctx, token, false);
    }

    // Begin decoding
    batch_decode_tokens(
        params->n_batch,
        ctx,
        token_list,
        0,
        params->n_parallel);
    return state;
}

void LlamaWorker::ensure_state_initialized()
{
    // Check state
    if (state == nullptr)
    {
        LOG("No initial state provided, creating a blank\n");
        state = new LlamaWorkerState(model, params);
    }
    else
    {
        LOG("Initial state provided. Using state\n");
    }
    if (!state->initialized)
    {
        state->init_ctx(model, params);
    }
    if (state->ctx == nullptr)
    {
        LOG("State does not have a context. Aborting\n");
        throw std::runtime_error("State does not have a context initialized\n");
    }
}

void LlamaWorker::init_state_for_token(std::vector<llama_token> token_list)
{
    ensure_state_initialized();

    // check the cache to see how many tokens we can use (if applicable)
    size_t n_matching_session_tokens = 0;
    if (!state->tokens.empty())
    {
        for (llama_token id : state->tokens)
        {
            if (n_matching_session_tokens >= token_list.size() || id != token_list[n_matching_session_tokens])
                break;
            n_matching_session_tokens++;
        }

        if (n_matching_session_tokens >= token_list.size())
        {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (token_list.size() / 2))
        {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, token_list.size());
        }
        else
        {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, token_list.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(state->ctx, -1, n_matching_session_tokens, -1);
    }
}

// This long function is direct implementation from the main.cpp
std::string LlamaWorker::run(std::vector<llama_token> input_tokens)
{
#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log start\n");
#endif // LOG_DISABLE_LOGS

    // NOTE: the comments contains my version of what the hell is going on
    // Append the prompt
    std::string generated_text = "";
    auto runtime_start = ggml_time_us();

    ensure_state_initialized();

    // Needed llama_context
    llama_sampling_params &sparams = (*params).sparams;
    llama_context *ctx_main = state->ctx;
    llama_context *ctx_guidance = NULL;

    // If some parameters are not supposed to be defined
    if ((*params).logits_all)
        throw std::runtime_error(std::string(__func__) + ": please use the 'perplexity' tool for perplexity calculations");
    if ((*params).embedding)
        throw std::runtime_error(std::string(__func__) + ": please use the 'embedding' tool for embedding calculations");

    // Parameter checks
    if ((*params).n_ctx != 0 && (*params).n_ctx < 8)
    {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        (*params).n_ctx = 8;
    }
    if ((*params).rope_freq_base != 0.0)
    {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, (*params).rope_freq_base);
    }
    if ((*params).rope_freq_scale != 0.0)
    {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, (*params).rope_freq_scale);
    }
    if ((*params).seed == LLAMA_DEFAULT_SEED)
    {
        (*params).seed = time(NULL);
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    if (sparams.cfg_scale > 1.f)
    {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(*params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx_main);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train)
        LOG_TEE(
            "%s: warning: model was trained on only %d context tokens (%d specified)\n",
            __func__, n_ctx_train, n_ctx);

    // print system information
    LOG_TEE("\n");
    LOG_TEE("%s\n", gpt_params_get_system_info((*params)).c_str());

    // does the model require a bos_token for starting generation?
    const bool add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG("add_bos: %d\n", add_bos);

    // construct the prompt tokens
    std::vector<llama_token> token_list = merge_token_list(&state->tokens, &input_tokens, llama_token_bos(model), append_bos);
    if (token_list.size() <= 0)
        token_list.emplace(token_list.begin(), llama_token_bos(model));
    init_state_for_token(token_list);

    // Note: (n_ctx - 4) here is to match the logic for command line prompt handling via
    // --prompt or --file which uses the same value.
    int max_token_size = n_ctx - 4;
    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if (token_list.size() > max_token_size)
    {
        const int skipped_tokens = token_list.size() - max_token_size;
        token_list.resize(max_token_size);
        LOG("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
    }

    // LOG("prompt: \"%s\"\n", log_tostr((*params).prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());

    if ((int)token_list.size() > n_ctx - 4)
    {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)token_list.size(), n_ctx - 4);
        throw std::runtime_error(std::string(__func__) + ": error: prompt is too long (" + std::to_string((int)token_list.size()) + " tokens, max " + std::to_string(n_ctx - 4) + ")");
    }

    // number of tokens to keep when resetting context
    int n_keep = (*params).n_keep;
    if (n_keep < 0 || n_keep > (int)token_list.size())
    {
        n_keep = (int)token_list.size();
    }
    else
    {
        // always keep the BOS token
        n_keep += add_bos;
    }

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, (*params).n_batch, (*params).n_predict, (*params).n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if (*params).grp_attn_n > 1)
    int ga_i = 0;
    const int ga_n = (*params).grp_attn_n;
    const int ga_w = (*params).grp_attn_w;
    if (ga_n != 1)
    {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");                         // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
                                                                                        // GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        // GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    // evaluate initial prompt
    int consumed_index = state->last_decoded_token_index;
    LOG("embd_inp.size(): %d, n_consumed: %d\n", (int)token_list.size(), consumed_index);

    int n_batch = (*params).n_batch;
    batch_decode_tokens(
        n_batch,
        ctx_main,
        token_list,
        // Decode from the new token and skip all processed token
        state->last_decoded_token_index + 1,
        params->n_parallel);

    struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
    if (!ctx_sampling)
    {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        throw std::runtime_error(std::string(__func__) + ": failed to initialize sampling subsystem");
    }
    // push the prompt in the sampling context in order to apply repetition penalties later
    // for the prompt, we don't apply grammar rules
    for (int token_index = 0; token_index < token_list.size(); token_index++)
    {
        auto token = token_list[token_index];
        // should accept from the context. But we're not applying grammar so it's fine
        llama_sampling_accept(ctx_sampling, ctx_main, token, false);
        LOG_TEE(
            "build: sampling context accept '%s' at %d\n",
            llama_token_to_piece(ctx_main, token).c_str(),
            token_index);
    }

    // prepare for Guidance (if enabled)
    int guidance_offset = 0; // Needed for shifting context
    if (ctx_guidance)
    {
        int prompt_size = token_list.size();
        std::vector<llama_token> guidance_tokens;
        guidance_tokens = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
        guidance_offset = guidance_tokens.size() - prompt_size;

        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_tokens).c_str());
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, input_tokens).c_str());
        LOG("original_prompt_len: %s", log_tostr(prompt_size));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));

        int input_size = 0;
        llama_token *input_buf = NULL;

        // Guidance context should have the same data with these modifications:
        //
        // * Replace the initial prompt
        // * Shift everything by guidance_offset
        if (token_list.begin() + prompt_size < token_list.end())
        {
            guidance_tokens.insert(
                guidance_tokens.end(),
                token_list.begin() + prompt_size,
                token_list.end());
        }

        LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, guidance_tokens).c_str());

        batch_decode_tokens(
            n_batch,
            ctx_guidance,
            guidance_tokens,
            0,
            params->n_parallel);
        guidance_tokens.clear();
    }

    int last_evaluated_token = token_list.size() - 1;
    int last_token_pos = last_evaluated_token;
    int remaining = (*params).n_predict;
    int guidance_token_pos = 0;

    // prediction start
    llama_batch predict_batch = llama_batch_init((*params).n_batch, 0, 1); // Should only have 1 at a time
    while (!should_yield && (remaining != 0))                              // lower than zero means infinite generation
    {
        // clear for next prediction
        llama_batch_clear(predict_batch);

        const llama_token sampled_id = llama_sampling_sample(ctx_sampling, ctx_main, ctx_guidance);
        llama_sampling_accept(ctx_sampling, ctx_main, sampled_id, true);
        LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, ctx_sampling->prev).c_str());

        --remaining;
        LOG("n_remain: %d\n", remaining);

        // "decode" and process the sampled token
        const std::string token_str = llama_token_to_piece(ctx_main, sampled_id, !(*params).conversation);
        bool is_bos = (sampled_id == llama_token_bos(model));
        bool is_eos = (sampled_id == llama_token_eos(model));
        if ((!is_bos || output_bos) && (!is_eos || output_eos))
        {
            generated_text.append(token_str);
            on_new_token(token_str);
        }

        // if generation finished, no need fo further decode for next iteration
        if (llama_token_is_eog(model, sampled_id))
        {
            LOG(" [end of text]\n");
            break;
        }

        // prepare the next logit for sampling
        int token_pos = last_token_pos + 1; // one more than last token
        last_token_pos = token_pos;
        // Decode logit for next sampling
        llama_batch_add(predict_batch, sampled_id, token_pos, {0}, true);
        if (llama_decode(ctx_main, predict_batch))
        {
            LOG_TEE("%s : failed to eval\n", __func__);
            throw std::runtime_error(std::string(__func__) + ": failed to eval");
        }

        LOG("n_past = %d\n", token_pos);
        if ((*params).n_print > 0 && token_pos % (*params).n_print == 0)
            LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", token_pos, n_ctx);

        // Handle context extension here
        if (ga_n == 1)
        {
            // infinite text generation via context shifting
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (token_pos + std::max<int>(0, guidance_offset) >= n_ctx)
            {
                if ((*params).n_predict == -2)
                {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, (*params).n_predict);
                    break;
                }

                const int n_left = token_pos - n_keep;
                const int n_discard = n_left / 2;

                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    token_pos, n_left, n_ctx, n_keep, n_discard);

                llama_kv_cache_seq_rm(ctx_main, 0, n_keep, n_keep + n_discard);
                llama_kv_cache_seq_add(ctx_main, 0, n_keep + n_discard, token_pos, -n_discard);

                token_pos -= n_discard; // NOTE: guidance offset used to be affected

                // LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);
                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());
            }
        }
        else
        {
            // context extension via Self-Extend
            while (token_pos >= ga_i + ga_w)
            {
                const int ib = (ga_n * ga_i) / ga_w;
                const int bd = (ga_w / ga_n) * (ga_n - 1);
                const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                LOG("\n");
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, token_pos, ib * bd, ga_i + ib * bd, token_pos + ib * bd);
                LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, token_pos + ib * bd, dd, ga_i + ib * bd + ga_w + dd, token_pos + ib * bd + dd);

                llama_kv_cache_seq_add(ctx_main, 0, ga_i, token_pos, ib * bd);
                llama_kv_cache_seq_div(ctx_main, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                llama_kv_cache_seq_add(ctx_main, 0, ga_i + ib * bd + ga_w, token_pos + ib * bd, dd);

                token_pos -= bd;
                ga_i += ga_w / ga_n;
                LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", token_pos + bd, token_pos, ga_i);
            }
        }
    }

    LOG("prediction completed with %d tokens remaining\n", remaining);

    auto runtime_end = ggml_time_us(); // Context can be saved, so the timing may not be accurate
    LOG_TEE("total runtime: %8.3f seconds\n", (runtime_end - runtime_start) / 1e6f);

    llama_log_timings(ctx_main);

    // Free all the used context here
    // Assume that state is handled immutably
    delete state;
    state = nullptr;

    llama_sampling_free(ctx_sampling);
    if (ctx_guidance)
        llama_free(ctx_guidance);

    return generated_text;
}

std::string LlamaWorker::run_with_lookahead(std::vector<llama_token> input_tokens, lookahead_params *lookahead_params)
{
    std::string generated_result = "";

    const auto runtime_start = ggml_time_us();

    ensure_state_initialized();
    llama_context *ctx_main = state->ctx;

    // Construct the full token first
    std::vector<llama_token> token_list = merge_token_list(&state->tokens, &input_tokens, llama_token_bos(model), false);
    if (token_list.size() <= 0)
        token_list.emplace(token_list.begin(), llama_token_bos(model));
    init_state_for_token(token_list);

    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_main, token_list).c_str());

    // Input and result
    std::vector<llama_token> all_tokens = token_list;

    const int window_size = lookahead_params->window_size; // W
    const int ngram_count = lookahead_params->ngram_size;  // N
    const int max_ngram_verify =                           // G
        lookahead_params->max_ngram_verify > -1 ? lookahead_params->max_ngram_verify : window_size;

    const int batch_size = params->n_batch;
    const int input_size = token_list.size();

    const int max_context_size = llama_n_ctx(ctx_main);
    const int max_token_list_size = max_context_size - 4;
    // Lookahead does not support context shifting for now, unlike autoregressive
    const int max_tokens = params->n_predict > -1 ? params->n_predict : max_token_list_size;

    if (input_size > max_token_list_size)
    {
        throw std::runtime_error(
            std::string(__func__) + ": error: prompt too long (" +
            std::to_string(token_list.size()) + " tokens, " +
            std::to_string(max_token_list_size) + "max" + ")");
    }

    // NOTE: think of windows like levels
    // https://lmsys.org/blog/2023-11-21-lookahead-decoding/
    // How much sequence we need for lookahead + verification (will be used in batch)
    const int total_branch_size = window_size + max_ngram_verify + 1;

    // evaluate the prompt and copy the evaluation to other sequences for batch processing
    const auto t_enc_start = ggml_time_us();

    // for each decoded batch, we have at most W + G + 1 distinct sequences:
    // seq_id == 0           : the current input token
    // seq_id [1, W]         : tokens from the past N - 1 Jacobi iterations (used for generation I think)
    // seq_id [W + 1, W + G] : verification n-grams
    // therefore this batch is seq with id of zero to represent the input
    batch_decode_tokens(batch_size, ctx_main, token_list, state->last_decoded_token_index + 1, (*params).n_parallel);

    // Copy the decoded result
    for (int seq_id = 1; seq_id < total_branch_size; ++seq_id)
        llama_kv_cache_seq_cp(ctx_main, 0, seq_id, -1, -1);

    const auto t_enc_end = ggml_time_us();
    LOG("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", input_size, (t_enc_end - t_enc_start) / 1e6f, input_size / ((t_enc_end - t_enc_start) / 1e6f));

    // Verification n-grams
    LOG("initializing verification branch with %d ngrams\n", max_ngram_verify);
    std::vector<ngram_data> pending_verification_ngrams(max_ngram_verify);

    // These are the lookahead branch (windows_size + (ngram_size - 1))
    // Tokens for the past N - 1 Jacobi iterations
    std::vector<llama_token> last_level(window_size);
    std::vector<std::vector<llama_token>> ngram_levels(ngram_count - 1);

    // NOTE: because we are doing this in batch, ngram_count is counted VERTICALLY
    // Initial lookahead windows is empty, but we have to fill it
    // for parallel batch processing
    for (int level = 0; level < ngram_count - 1; level++)
    {
        ngram_levels[level].resize(window_size);
        for (int token_pos = 0; token_pos < window_size; token_pos++)
            // initialize randomly from the prompt tokens
            ngram_levels[level][token_pos] = all_tokens[1 + rand() % (all_tokens.size() - 1)];
    }

    std::vector<llama_seq_id> lookahead_seq_ids;
    std::vector<llama_seq_id> all_seq_ids(total_branch_size);
    for (int i = 0; i < total_branch_size; i++)
        all_seq_ids[i] = i;
    ngram_pool pool(llama_n_vocab(model), ngram_count, max_ngram_verify);

    const auto t_dec_start = ggml_time_us();

    // prepare sampling
    const int context_size = params->n_ctx;
    llama_batch batch = llama_batch_init(context_size, 0, total_branch_size);
    llama_sampling_context *ctx_sampling = llama_sampling_init(params->sparams);

    int total_predicted_tokens = 0;
    int total_accepted_tokens = 0;
    int n_past = token_list.size();
    // used to determine end of generation
    bool has_eos = false;

    // sample first token as input (do not sample from 0 because the logit from batch API is -1
    // instead of the usual zero due to not llama_batch_get_one)
    // https://github.com/ggerganov/llama.cpp/issues/6475#issuecomment-2036861535
    llama_token curr_input_token = llama_sampling_sample(ctx_sampling, ctx_main, NULL);
    llama_sampling_accept(ctx_sampling, ctx_main, curr_input_token, true);

    // yes, this is a duplicate, exactly the same as the one in the lookahead decoding
    // part. but this is the easiest way to get rid of the initial sampled token
    {
        // generated token here, if v = 0 it is the verified (accepted tokens)
        const std::string token_str = llama_token_to_piece(ctx_main, curr_input_token);
        bool is_bos = (curr_input_token == llama_token_bos(model));
        bool is_eos = (curr_input_token == llama_token_eos(model));
        if ((!is_bos || output_bos) && (!is_eos || output_eos))
        {
            generated_result.append(token_str);
            on_new_token(token_str);
        }
        if (llama_token_is_eog(model, curr_input_token))
            has_eos = true;

        all_tokens.push_back(curr_input_token);
    }

    LOG("generation start with token '%s'\n", llama_token_to_piece(ctx_main, curr_input_token).c_str());
    // prediction start
    while (true && !should_yield)
    {
        // build the mask from https://lmsys.org/blog/2023-11-21-lookahead-decoding/
        //
        // Example for W = 5, N = 4, G = 2:
        // (I = input, L = lookahead, V = verification)
        //
        // Batch:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
        // T:        -2 -2 -2 -2 -1 -1 -1 -1 -1  0  0  0  0  0  0
        // Info:   I  L  L  L  L  L  L  L  L  L  L  L  L  L  L  V  V  V  V  V  V
        // Pos:    0  1  2  3  4  1  2  3  4  5  2  3  4  5  6  1  2  3  1  2  3   (+ n_past)
        // Logits: 1  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1
        // ---------------------------------------------------------------------
        // Seq:    0
        //         1              1              1
        //         2  2              2              2
        //         3  3  3              3              3
        //         4  4  4  4              4              4
        //         5  5  5  5  5              5              5
        //         6                                            6  6  6
        //         7                                                     7  7  7
        // ---------------------------------------------------------------------
        //                                       |  |  |  |  |  |  |  |  |  |  |
        //                                       V  V  V  V  V  |  |  |  |  |  |
        //                                         j_tokens     |  |  |  |  |  |
        //                                                      V  V  V  V  V  V
        //

        llama_batch_clear(batch);
        // current token - first token of the first level (input)
        llama_batch_add(batch, curr_input_token, n_past, all_seq_ids, true);

        // ngrams starting with `input_token`
        const int input_ngrams_count = pool.count[curr_input_token];
        pending_verification_ngrams.resize(input_ngrams_count);

        // verification n-grams - queue this before the lookahead tokens for less KV cache fragmentation
        for (int token_pos = 0; token_pos < input_ngrams_count; token_pos++) // Initialize the windows
        {
            auto &ngram_verification = pending_verification_ngrams[token_pos];
            ngram_verification.active = true;
            ngram_verification.tokens.resize(ngram_count);
            ngram_verification.batch_index.resize(ngram_count);
            ngram_verification.seq_id = window_size + 1 + token_pos; // see earlier comments about seq_id allocation
            ngram_verification.batch_index[0] = 0;                   // labelled zero to indicate current input (see blog)
            ngram_verification.tokens[0] = curr_input_token;
        }
        // loop through each windows and fill the ngrams with an n-th level token
        for (int level = 0; level < ngram_count - 1; level++)
            // iterate over each potential ngrams
            for (int token_pos = 0; token_pos < input_ngrams_count; token_pos++)
            {
                const int idx =
                    // calculate the index at the ring buffer, see the ring buffer comment
                    curr_input_token * (ngram_count - 1) * max_ngram_verify +
                    token_pos * (ngram_count - 1); // because this is the g-th "window"
                const llama_token t = pool.tokens[idx + level];

                // NOTE: +1 to skip the input
                auto &verification_window = pending_verification_ngrams[token_pos];
                // Distribute the tokens into their appropiate batch/window
                verification_window.tokens[level + 1] = t;
                verification_window.batch_index[level + 1] = batch.n_tokens;

                llama_batch_add(batch, t, n_past + level + 1, {window_size + 1 + token_pos}, true);
            }

        // fill the remaining W - 1 tokens for the first level
        // exclude 1 because of the input token already allocated
        for (int token_pos = 1; token_pos < window_size; token_pos++)
        {
            // The reach the current token has
            auto max_seq_id = window_size - token_pos;
            lookahead_seq_ids.resize(max_seq_id);
            // update sequence ids for each respective token on each window
            for (int seq_id = 0; seq_id < max_seq_id; seq_id++)
                lookahead_seq_ids[seq_id] = token_pos + seq_id + 1;
            /// Queue the token from input (as basically "input")
            llama_batch_add(batch, ngram_levels[0][token_pos], n_past + token_pos, lookahead_seq_ids, false);
        }

        // fill the rest of the levels (queue the lookahead)
        // NOTE: window == ngram_count - 2 to enable to logit the last token of each ngrams
        for (int level = 1; level < ngram_count - 1; level++)
            for (int token_pos = 0; token_pos < window_size; token_pos++)
                llama_batch_add(batch, ngram_levels[level][token_pos], n_past + level + token_pos, {token_pos + 1}, level == ngram_count - 2);

        if (llama_decode(ctx_main, batch) != 0)
        {
            LOG_TEE("%s : failed to eval, increase kv cache\n", __func__);
            throw std::runtime_error(std::string(__func__) + ": failed to eval increase kv_cache");
        }

        int seq_id_best = 0;
        // Traverse through the resulting decoded ngrams one step at a time
        for (int level = 0; level < ngram_count; ++level)
        {
            int batch_index = 0;
            // try to search for a matching ngram for lookahead windows
            // if no active ngrams are left, it means the sampled token does not pass the verification
            if (level > 0)
            {
                for (int token_pos = 0; token_pos < pending_verification_ngrams.size(); token_pos++)
                {
                    auto &verification_win = pending_verification_ngrams[token_pos];
                    // Accept token (already passed the verification)
                    if (verification_win.active)
                    {
                        batch_index = verification_win.batch_index[level];
                        seq_id_best = verification_win.seq_id;
                        ++total_accepted_tokens;
                        break;
                    }
                }
                // no more matches -> create a new batch
                if (batch_index == 0)
                    break;
            }

            // sample the next token
            curr_input_token = llama_sampling_sample(ctx_sampling, ctx_main, NULL, batch_index);
            llama_sampling_accept(ctx_sampling, ctx_main, curr_input_token, true);
            LOG("sampled token: '%s'\n", llama_token_to_piece(ctx_main, curr_input_token, true).c_str());
            LOG("level: %d, from_pool: %d\n", level, level > 0);

            {
                // generated token here, if v = 0 it is the verified (accepted tokens)
                const std::string token_str = llama_token_to_piece(ctx_main, curr_input_token);
                bool is_bos = (curr_input_token == llama_token_bos(model));
                bool is_eos = (curr_input_token == llama_token_eos(model));

                if ((!is_bos || output_bos) && (!is_eos || output_eos))
                {
                    generated_result.append(token_str);
                    on_new_token(token_str);
                }
                if (llama_token_is_eog(model, curr_input_token))
                    has_eos = true;

                all_tokens.push_back(curr_input_token);
            }

            ++total_predicted_tokens;
            ++n_past;

            // end of generation
            LOG("max: %d, predicted: %d\n", max_tokens, total_predicted_tokens);
            if ((max_tokens >= 0 && total_predicted_tokens > max_tokens) || has_eos)
            {
                LOG("max has been reached, aborting\n");
                break;
            }

            // verify across active n-grams
            for (int win_index = 0; win_index < pending_verification_ngrams.size(); win_index++)
            {
                auto &verification_window = pending_verification_ngrams[win_index];
                if (verification_window.active)
                {
                    // If the window is already too old
                    if (level == ngram_count - 1)
                        verification_window.active = false;
                    // (check 1 token forward since we validate future token future token)
                    // see the figure for more info
                    else if (curr_input_token != verification_window.tokens[level + 1])
                        verification_window.active = false;
                }
            }

            // print known n-grams starting with token id (debug)
            if (level == 0)
            {
                if (pool.count[curr_input_token] > 0)
                    LOG(
                        "%d n-grams starting with '%s'\n",
                        pool.count[curr_input_token],
                        llama_token_to_piece(ctx_main, curr_input_token).c_str());
                for (int i = 0; i < pool.count[curr_input_token]; i++)
                {
                    // TODO: should we turn this into like a formula
                    std::stringstream matches;
                    const int idx = curr_input_token * (ngram_count - 1) * max_ngram_verify + i * (ngram_count - 1);
                    for (int j = 0; j < ngram_count - 1; j++)
                    {
                        const std::string token_str = llama_token_to_piece(ctx_main, pool.tokens[idx + j]);
                        matches << token_str << ", ";
                    }
                    LOG("   - ngram %2d: %s \n", i, matches.str().c_str());
                }
            }

            // update lookahead tokens
            {
                // move the 'last" window to the last window
                for (int token_pos = 0; token_pos < window_size; token_pos++)
                    last_level[token_pos] = ngram_levels[0][token_pos];

                // shift the windows back by 1 level each
                // exclude the last one because we are shifting it (last element = ngram_count - 1)
                for (int level = 0; level < ngram_count - 2; level++)
                    ngram_levels[level] = ngram_levels[level + 1];

                // sample from the last level (ngram_count - 2)
                // ngram_count - 1 = last element + win_size
                if (level == 0)
                {
                    for (int token_pos = 0; token_pos < window_size; token_pos++)
                    {
                        auto idx = pending_verification_ngrams.size() * (ngram_count - 1) +
                                   window_size * (ngram_count - 2) + token_pos;
                        ngram_levels[ngram_count - 2][token_pos] =
                            llama_sampling_sample(ctx_sampling, ctx_main, NULL, idx);
                    }
                }
                else
                {
                    // If not the last level then just reinitialize em
                    for (int i = 0; i < window_size; i++) // Random init
                        ngram_levels[ngram_count - 2][i] = all_tokens[1 + rand() % (all_tokens.size() - 1)];
                }
            }

            // update ngram pool
            // basically append the ngrams we created during the process
            if (level == 0)
            {
                // the first token of the n-gram is determined by the index in the container so it is not stored
                std::vector<llama_token> ngram(ngram_count - 1);

                // n-gram generation
                // ref: https://github.com/hao-ai-lab/LookaheadDecoding/issues/14#issuecomment-1826198518
                for (int token_pos = 0; token_pos < window_size; ++token_pos)
                {
                    const int first_token = last_level[token_pos]; // first token of the n-gram
                    for (int level = 0; level < ngram_count - 1; ++level)
                        ngram[level] = ngram_levels[level][token_pos];

                    // filter-out repeating n-grams
                    bool is_unique = true;
                    for (int token_pos = 0; token_pos < pool.count[first_token]; ++token_pos)
                    {
                        const int idx =
                            first_token * (ngram_count - 1) * max_ngram_verify + token_pos * (ngram_count - 1);

                        // Check here
                        bool is_match = true;
                        for (int token_pos = 0; token_pos < ngram_count - 1; ++token_pos)
                            if (pool.tokens[idx + token_pos] != ngram[token_pos])
                            {
                                is_match = false;
                                break;
                            }

                        if (is_match)
                        {
                            is_unique = false;
                            break;
                        }
                    }
                    if (!is_unique)
                        continue;

                    const int head = pool.head[first_token];
                    const int idx = first_token * (ngram_count - 1) * max_ngram_verify + head * (ngram_count - 1);

                    // Copy the tokens in the ngrams we build to the pool
                    for (int token_pos = 0; token_pos < ngram_count - 1; token_pos++)
                        pool.tokens[idx + token_pos] = ngram[token_pos];

                    // Record the start index (head)
                    pool.count[first_token] = std::min(max_ngram_verify, pool.count[first_token] + 1);
                    pool.head[first_token] = (head + 1) % max_ngram_verify;
                    pool.n_total++;
                }
            }
        }

        if ((max_tokens >= 0 && total_predicted_tokens > max_tokens) || has_eos)
        {
            LOG("max token predicted has been reached. generation completed\n");
            break;
        }

        // KV cache management
        // if no verification token matched, we simply remove all cells from this batch -> no fragmentation
        llama_kv_cache_seq_rm(ctx_main, -1, n_past, -1);
        if (seq_id_best != 0)
        {
            // if a verification token matched, we keep the best sequence and remove the rest
            // this leads to some KV cache fragmentation
            llama_kv_cache_seq_keep(ctx_main, seq_id_best);
            llama_kv_cache_seq_cp(ctx_main, seq_id_best, 0, -1, -1);
            llama_kv_cache_seq_rm(ctx_main, seq_id_best, -1, -1);
            for (int s = 1; s < total_branch_size; ++s)
                llama_kv_cache_seq_cp(ctx_main, 0, s, -1, -1);
        }
    }

    auto t_dec_end = ggml_time_us();
    auto runtime_end = ggml_time_us();
    // Context can sometimes be sved, so the timing may not be accurate
    LOG_TEE("total runtime: %8.3f seconds\n", (runtime_end - runtime_start) / 1e6f);

    LOG_TEE("\n\n");
    LOG_TEE("W = %2d\n", window_size);
    LOG_TEE("N = %2d\n", ngram_count);
    LOG_TEE("G = %2d\n", max_ngram_verify);
    LOG_TEE("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", input_size, (t_enc_end - t_enc_start) / 1e6f, input_size / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_TEE("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", total_predicted_tokens, (t_dec_end - t_dec_start) / 1e6f, total_predicted_tokens / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_TEE("\n");
    LOG_TEE("\n");
    LOG_TEE("n_predict = %d\n", total_predicted_tokens);
    LOG_TEE("n_accept  = %d\n", total_accepted_tokens);

    llama_log_timings(ctx_main);

    // Free only the local variables (like the model should not be freed)
    llama_sampling_free(ctx_sampling);
    llama_batch_free(batch);

    return generated_result;
}