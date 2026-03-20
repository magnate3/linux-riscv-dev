#pragma once
//#include "gpu_llm_worker.h"
#include "llama.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <deque>
#include <cstdint>
#include <functional>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_token_data;
struct llama_sampler;
struct llama_batch;

namespace ai_scheduler {
// LLM模型配置
struct LLMModelConfig {
    std::string modelPath{};         
    std::string modelType{"mock"}; 
    std::string quantization{};      
    int gpuDeviceId{0};              
    int nGpuLayers{-1};              
    size_t maxContextSize{4096};     
    size_t maxBatchSize{512};        
    float temperature{0.7f};         
    int topK{40};                    
    float topP{0.95f};               
    float repetitionPenalty{1.1f};   
    bool enableCache{true};          
    size_t cacheSize{16};
    std::string draftModelPath{};    // Speculative Decoding Draft Model
    int draftGpuDeviceId{-1};        // -1 for CPU, >=0 for GPU
    int draftContextSize{512};

    bool enableKvSwap{false};
    std::string kvSwapDir{};
    size_t kvSwapTriggerTokens{2048};
};

// LLM推理请求
struct LLMInferenceRequest {
    std::string prompt{};
    std::string conversationId{};
    size_t maxTokens{0};
    float temperature{0.0f};
    int topK{0};
    float topP{0.0f};
    float repetitionPenalty{0.0f};
    bool streamOutput{false};
    std::function<void(const std::string&)> onTokenGenerated{};
    std::function<bool()> shouldStop{};
};

// LLM推理响应
struct LLMInferenceResponse {
    std::string generatedText;     // 生成的文本
    size_t generatedTokens;        // 生成的token数
    float inferenceTime;           // 推理时间（秒）
    bool success;                  // 是否成功
    std::string errorMessage;      // 错误信息
};

// LLM模型接口
class ILLMModel {
public:
    virtual ~ILLMModel() = default;
    virtual bool initialize(const LLMModelConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual LLMInferenceResponse generate(const LLMInferenceRequest& request) = 0;
    virtual std::string getModelInfo() const = 0;
    virtual size_t getMemoryUsage() const = 0;
    virtual bool isReady() const = 0;
};

// Forward declaration
class LlamaCppModel : public ILLMModel {
public:
    LlamaCppModel();
    ~LlamaCppModel() override;

    bool initialize(const LLMModelConfig& config) override;
    void shutdown() override;
    LLMInferenceResponse generate(const LLMInferenceRequest& request) override;
    std::string getModelInfo() const override;
    size_t getMemoryUsage() const override;
    bool isReady() const override;

private:
    // Internal helper methods
    std::vector<int> tokenize(const std::string& text, bool add_bos);
    std::string detokenize(const std::vector<int>& tokens);

    int32_t getSeqId(const std::string& conversationId);
    void touchSeq(int32_t seqId);
    void evictIfNeeded();

    bool loadTokensFromSwap(const std::string& conversationId, std::vector<int>& outTokens);
    bool saveTokensToSwap(const std::string& conversationId, const std::vector<int>& tokens);
    void swapOutConversationIfNeeded(const std::string& conversationId, int32_t seqId);
    bool ensureSeqKvLoaded(llama_seq_id seqId, const std::vector<int>& tokens);
    std::string kvSwapPathForConversation(const std::string& conversationId) const;
    
    // Llama.cpp context
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    
    // Draft model for speculative decoding
    llama_model* draft_model_ = nullptr;
    llama_context* draft_ctx_ = nullptr;

    // llama_sampler* sampler_ = nullptr; // Removed for compatibility with older llama.cpp
    
    // Configuration
    LLMModelConfig config_;
    bool ready_ = false;

    size_t max_sessions_ = 0;
    std::unordered_map<std::string, int32_t> conv_to_seq_;
    std::unordered_map<int32_t, std::string> seq_to_conv_;
    std::unordered_map<int32_t, std::vector<int>> seq_tokens_;
    std::unordered_map<std::string, std::string> conv_to_swap_path_;
    std::unordered_map<int32_t, bool> seq_kv_loaded_;
    std::deque<int32_t> lru_;
    
    // Buffers
    std::vector<int> prompt_tokens_;
    std::vector<int> generated_tokens_;
};

} // namespace ai_scheduler
