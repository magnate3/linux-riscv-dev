#pragma once
#include "llama.h"
#include <string>
#include <vector>
#include <chrono> // NEW: For Timing

// Data structure for research metrics
struct PerfStats {
    double loadTimeMs = 0.0;
    double generationTimeMs = 0.0;
    int tokensGenerated = 0;
};

class LLMEngine {
public:
    LLMEngine();
    ~LLMEngine();

    bool loadModel(const std::string& modelPath);
    void unloadModel();
    std::string query(const std::string& prompt, int max_tokens = 2048);
    bool isLoaded() const;

    // NEW: Function to get stats
    PerfStats getLastStats() const { return stats; }
    llama_context* getCtx() const {return ctx;}
    llama_model * getModel() const {return model;}
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    
    // NEW: Stats storage
    PerfStats stats;

    void batch_add_seq(llama_batch &batch, llama_token token, int pos, int32_t seq_id, bool logits);
};
