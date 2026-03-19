#include "Init.hpp"
#include "Session.hpp"
#include "Common.hpp"
#include "PromptComposer.hpp"

Session::Session(const std::string& modelPath,
    int nGpuLayers,
    int nCtxSize,
    int nBatchSize,
    const std::string& systemPrompt,
    PromptFormat format)
    : Composer(systemPrompt, "", format), nCtxSize_(nCtxSize), currentPos_(0)
{
    std::cout << "[SESSION] Loading model...\n";
    model = LoadModel(modelPath, nGpuLayers);
    if (!model)
        throw std::runtime_error("Model is null after initialization.");

    std::cout << "[SESSION] Creating context...\n";
    context = CreateContext(model, nCtxSize, nBatchSize);
    if (!context)
        throw std::runtime_error("Context is null after initialization.");

    std::cout << "[SESSION] Creating sampler...\n";
    sampler = CreateSampler(context);
    if (!sampler)
        throw std::runtime_error("Sampler is null after initialization.");
}

std::string Session::Ask(const std::string& userPrompt, int maxNew)
{
   //  std::cout << "\n[ASK] ========================================\n";
   // std::cout << "[ASK] Starting Ask() at position " << currentPos_ << "\n";

    const llama_vocab* vocab = llama_model_get_vocab(model);
    Composer.SetUser(userPrompt);
    const std::string sanitizedPrompt = Composer.Build();

  //  std::cout << "[ASK] Prompt length: " << sanitizedPrompt.length() << " chars\n";
  //  std::cout << "[ASK] Prompt:\n" << sanitizedPrompt << "\n";

    int32_t text_len = static_cast<int32_t>(sanitizedPrompt.length());

    /*   tokenize */
   // std::cout << "[ASK] Tokenizing (pass 1)...\n";
    llama_token buf[4096];
    int n = llama_tokenize(vocab,
        sanitizedPrompt.c_str(),
        text_len,
        buf, 4096,
        true,   // add_special
        true);  // parse_special

    if (n < 0)
    {
        std::cout << "[ASK] First pass failed (" << n << "), retrying without special parsing...\n";
        n = llama_tokenize(vocab,
            sanitizedPrompt.c_str(),
            text_len,
            buf, 4096,
            true,   // add_special
            false); // parse_special
    }
    if (n < 0)
        throw std::runtime_error("[ASK] Tokenization failed with code: " + std::to_string(n));
    if (n == 0 || n > nCtxSize_)
        throw std::runtime_error("[ASK] Invalid token count: " + std::to_string(n));

    std::cout << "[ASK] Tokens: " << n << "\n";
    std::vector<llama_token> tokens(buf, buf + n);

    /*  Get memory handle for sliding window management */
    llama_memory_t mem = llama_get_memory(context);

    // Fixed, slide before new tokens to avoid overflow
    if (currentPos_ + n + maxNew >= nCtxSize_)  // Now predict overflow
    {
        int keep = nCtxSize_ / 2;
        std::cout << "[ASK] Pre-emptive sliding window! Keeping last "
            << keep << " tokens\n";
        llama_memory_seq_rm(mem, 0, 0, currentPos_ - keep);
        llama_memory_seq_add(mem, 0, 0, -1, -(currentPos_ - keep));
        currentPos_ = keep;
    }

    /*  prompt batch */
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) 
    {
        batch.token[i] = tokens[i];
        batch.pos[i] = currentPos_ + i;  // Continue from current position
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1);
    }

    batch.n_tokens = n;

   // std::cout << "[ASK] Decoding prompt batch...\n";
    if (llama_decode(context, batch) != 0)
    {
        llama_batch_free(batch);
        throw std::runtime_error("[ASK] Prompt decode failed");
    }
    llama_batch_free(batch);

    currentPos_ += n;  // Update position after prompt

    /*  generation loop */
    if (!llama_memory_can_shift(mem))
        throw std::runtime_error("[DEBUG] Memory doesn't support shifting");

    llama_batch single = llama_batch_init(1, 0, 1);
    single.n_seq_id[0] = 1;
    single.seq_id[0][0] = 0;
    single.logits[0] = true;
    single.n_tokens = 1;

    std::string out;
    out.reserve(maxNew * 4);

    std::cout << "Freyja is thinking...\n\n";
    for (int i = 0; i < maxNew; ++i)
    {
        if (i && i % 20 == 0)
           // std::cout << "[ASK] Generated " << i << " tokens...\n";

        /* sliding window during generation */
        if (currentPos_ >= nCtxSize_)
        {
            int keep = nCtxSize_ / 2;
            std::cout << "[ASK] Sliding window during generation!\n";
            llama_memory_seq_rm(mem, 0, 0, currentPos_ - keep);
            llama_memory_seq_add(mem, 0, 0, -1, -(currentPos_ - keep));
            currentPos_ = keep;
        }

        /* sample */
        llama_token id = llama_sampler_sample(sampler, context, -1);
        if (id == llama_vocab_eos(vocab)) break;

        /* token to text */
        char piece[256];
        int len = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, false);
        out.append(piece, len);

        /* decode single token */
        single.token[0] = id;
        single.pos[0] = currentPos_++;
        if (llama_decode(context, single) != 0)
        {
            llama_batch_free(single);
            throw std::runtime_error("[ASK] Token decode failed");
        }
    }

    llama_batch_free(single);
   // std::cout << "[ASK] Generation complete! (" << out.size() << " chars)\n";
    std::cout << "[DEBUG] Current position in context: " << currentPos_ << "/" << nCtxSize_ << "\n\n";

    return out;
}

Session::~Session()
{
    std::cout << "[SESSION] Destructor called\n";

    if (sampler)
    {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (context)
    {
        llama_free(context);
        context = nullptr;
    }
    if (model)
    {
        llama_model_free(model);
        model = nullptr;
    }

    // Backend free is done in main() for now.

    std::cout << "[SESSION] Cleanup complete\n";
}