#include "llama_model_impl.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <filesystem>
#include <fstream>
#include <future>
#include <chrono>
#include "llama.h"

namespace ai_scheduler {

// 带超时的llama_decode包装函数
// 如果decode在指定时间内未完成，返回-1表示超时
static int llama_decode_with_timeout(
    llama_context* ctx,
    llama_batch& batch,
    int timeout_ms = 30000,  // 默认30秒超时
    const std::function<bool()>& shouldStop = nullptr
) {
    // 使用异步方式执行decode，主线程监控超时
    std::promise<int> result_promise;
    std::future<int> result_future = result_promise.get_future();
    
    // 在单独线程中执行decode
    std::thread decode_thread([&]() {
        int result = llama_decode(ctx, batch);
        result_promise.set_value(result);
    });
    
    // 等待结果或超时
    auto status = result_future.wait_for(std::chrono::milliseconds(timeout_ms));
    
    if (status == std::future_status::timeout) {
        // 超时了，但无法真正终止llama_decode（它可能在GPU上卡死）
        // 我们只能记录日志并返回超时错误
        std::cerr << "llama_decode超时（" << timeout_ms << "ms），GPU可能卡死" << std::endl;
        
        // 分离线程（无法安全终止）
        decode_thread.detach();
        
        // 检查是否应该停止
        if (shouldStop && shouldStop()) {
            return -2;  // 被取消
        }
        return -1;  // 超时
    }
    
    // 正常完成
    decode_thread.join();
    return result_future.get();
}

// Helper functions for llama_batch
static void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

static void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}

static llama_token sample_from_logits(
    const float* logits,
    int n_vocab,
    int top_k,
    float top_p,
    float temperature
) {
    if (!logits || n_vocab <= 0) {
        return 0;
    }

    if (temperature <= 0.0f) {
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return (llama_token)best;
    }

    int k = top_k > 0 ? top_k : 40;
    if (k > n_vocab) {
        k = n_vocab;
    }
    float p = top_p > 0.0f ? top_p : 1.0f;
    if (p > 1.0f) {
        p = 1.0f;
    }

    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        idx[i] = i;
    }

    std::nth_element(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [logits](int a, int b) { return logits[a] > logits[b]; }
    );
    idx.resize(k);
    std::sort(
        idx.begin(),
        idx.end(),
        [logits](int a, int b) { return logits[a] > logits[b]; }
    );

    float max_logit = logits[idx[0]] / temperature;
    std::vector<float> probs;
    probs.reserve(idx.size());
    float sum = 0.0f;
    for (int id : idx) {
        float v = logits[id] / temperature;
        float ev = std::exp(v - max_logit);
        probs.push_back(ev);
        sum += ev;
    }
    if (sum <= 0.0f) {
        return (llama_token)idx[0];
    }
    for (float& v : probs) {
        v /= sum;
    }

    if (p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (size_t i = 0; i < probs.size(); i++) {
            cumulative += probs[i];
            keep++;
            if (cumulative >= p) {
                break;
            }
        }
        if (keep < probs.size()) {
            probs.resize(keep);
            idx.resize(keep);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs.begin(), probs.end());
    int chosen_idx = d(gen);
    return (llama_token)idx[chosen_idx];
}

// Helper to check valid UTF-8 length
static size_t get_valid_utf8_len(const std::string& s) {
    size_t i = 0;
    size_t last_valid = 0;
    while (i < s.size()) {
        size_t len = 0;
        unsigned char c = (unsigned char)s[i];
        if ((c & 0x80) == 0) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else len = 1; // Invalid byte, skip

        if (i + len > s.size()) {
            break; // Incomplete
        }
        i += len;
        last_valid = i;
    }
    return last_valid;
}

static bool g_llama_backend_initialized = false;
static bool g_llama_log_installed = false;

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    if (!text) {
        return;
    }
    if (std::strstr(text, "n_ctx_per_seq") != nullptr) {
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::cerr << text;
        return;
    }
    std::cout << text;
}

LlamaCppModel::LlamaCppModel() {
    if (!g_llama_backend_initialized) {
        llama_backend_init();
        if (!g_llama_log_installed) {
            llama_log_set(llama_log_callback, nullptr);
            g_llama_log_installed = true;
        }
        g_llama_backend_initialized = true;
    }
}

LlamaCppModel::~LlamaCppModel() {
    shutdown();
}

bool LlamaCppModel::initialize(const LLMModelConfig& config) {
    if (ready_) return true;
    
    config_ = config;
    std::cout << "Loading llama model from: " << config_.modelPath << std::endl;

    // 设置日志回调，屏蔽特定的上下文大小警告
    llama_log_set([](ggml_log_level level, const char* text, void* user_data) {
        if (text && std::strstr(text, "n_ctx_per_seq") != nullptr) {
            return; // 屏蔽这个烦人的警告
        }
        if (text) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.nGpuLayers >= 0 ? config.nGpuLayers : 999;
    model_params.main_gpu = config.gpuDeviceId >= 0 ? config.gpuDeviceId : 0;

    // Load model
    model_ = llama_model_load_from_file(config.modelPath.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load llama model: " << config_.modelPath << std::endl;
        return false;
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.maxContextSize > 0 ? (uint32_t)config_.maxContextSize : 4096;
    ctx_params.n_batch = config_.maxBatchSize > 0 ? (uint32_t)config_.maxBatchSize : 512;
    uint32_t safe_batch_limit = ctx_params.n_ctx > 1 ? (ctx_params.n_ctx / 2) : 1;
    if (safe_batch_limit < 1) {
        safe_batch_limit = 1;
    }
    if (ctx_params.n_batch > safe_batch_limit) {
        ctx_params.n_batch = safe_batch_limit;
    }
    if (ctx_params.n_batch < 1) {
        ctx_params.n_batch = 1;
    }
    if (config_.maxBatchSize <= 0 || config_.maxBatchSize != (int32_t)ctx_params.n_batch) {
        config_.maxBatchSize = (int32_t)ctx_params.n_batch;
    }

    max_sessions_ = 0;
    conv_to_seq_.clear();
    seq_to_conv_.clear();
    seq_tokens_.clear();
    conv_to_swap_path_.clear();
    seq_kv_loaded_.clear();
    lru_.clear();
    if (config_.enableCache && config_.cacheSize > 0) {
        max_sessions_ = config_.cacheSize;
    }

    uint32_t n_seq_max = 1;
    if (max_sessions_ > 0) {
        n_seq_max = (uint32_t)(max_sessions_ + 1);
        if (n_seq_max < 2) n_seq_max = 2;
        if (n_seq_max > 128) n_seq_max = 128;
    }
    uint32_t max_seq_by_batch = 1;
    if (ctx_params.n_batch > 0) {
        max_seq_by_batch = (uint32_t)std::max<uint32_t>(1, ctx_params.n_ctx / ctx_params.n_batch);
    }
    if (n_seq_max > max_seq_by_batch) {
        n_seq_max = max_seq_by_batch;
    }
    if (n_seq_max < 1) {
        n_seq_max = 1;
    }
    if (max_sessions_ > 0) {
        int32_t allowed_sessions = (int32_t)n_seq_max - 1;
        if (allowed_sessions < 0) {
            allowed_sessions = 0;
        }
        if ((int32_t)max_sessions_ > allowed_sessions) {
            max_sessions_ = (size_t)allowed_sessions;
        }
    }
    ctx_params.n_seq_max = n_seq_max;
    
    // Create context
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "Failed to create llama context" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    // Load Draft Model if configured
    if (!config_.draftModelPath.empty()) {
        std::cout << "Loading draft model from: " << config_.draftModelPath << std::endl;
        llama_model_params draft_params = llama_model_default_params();
        draft_params.n_gpu_layers = config_.draftGpuDeviceId >= 0 ? 999 : 0;
        draft_params.main_gpu = config_.draftGpuDeviceId >= 0 ? config_.draftGpuDeviceId : 0;
        
        draft_model_ = llama_model_load_from_file(config_.draftModelPath.c_str(), draft_params);
        if (draft_model_) {
            llama_context_params draft_ctx_params = llama_context_default_params();
            draft_ctx_params.n_ctx = config_.draftContextSize > 0 ? (uint32_t)config_.draftContextSize : 512;
            draft_ctx_params.n_batch = config_.maxBatchSize > 0 ? (uint32_t)config_.maxBatchSize : 512;
            draft_ctx_params.n_seq_max = 1; // Draft model typically handles 1 sequence at a time for speculation
            
            draft_ctx_ = llama_init_from_model(draft_model_, draft_ctx_params);
            if (!draft_ctx_) {
                std::cerr << "Failed to create draft context" << std::endl;
                llama_model_free(draft_model_);
                draft_model_ = nullptr;
            } else {
                std::cout << "Draft model loaded successfully." << std::endl;
            }
        } else {
            std::cerr << "Failed to load draft model: " << config_.draftModelPath << std::endl;
        }
    }

    ready_ = true;
    std::cout << "Llama model loaded successfully." << std::endl;
    return true;
}

std::string LlamaCppModel::kvSwapPathForConversation(const std::string& conversationId) const {
    const std::string base = config_.kvSwapDir;
    if (base.empty()) {
        return std::string();
    }
    std::filesystem::path p(base);
    std::string safe = conversationId;
    for (char& c : safe) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_') {
            continue;
        }
        c = '_';
    }
    p /= (safe + ".kvswap");
    return p.string();
}

bool LlamaCppModel::saveTokensToSwap(const std::string& conversationId, const std::vector<int>& tokens) {
    if (!config_.enableKvSwap || tokens.empty()) {
        return false;
    }
    const std::string path = kvSwapPathForConversation(conversationId);
    if (path.empty()) {
        return false;
    }
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    } catch (...) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    const uint32_t n = (uint32_t)tokens.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(tokens.data()), sizeof(int) * tokens.size());
    if (!out.good()) {
        return false;
    }
    out.close();
    conv_to_swap_path_[conversationId] = path;
    return true;
}

bool LlamaCppModel::loadTokensFromSwap(const std::string& conversationId, std::vector<int>& outTokens) {
    if (!config_.enableKvSwap) {
        return false;
    }
    std::string path;
    auto it = conv_to_swap_path_.find(conversationId);
    if (it != conv_to_swap_path_.end()) {
        path = it->second;
    } else {
        path = kvSwapPathForConversation(conversationId);
    }
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    if (!in.good() || n == 0) {
        return false;
    }
    outTokens.resize((size_t)n);
    in.read(reinterpret_cast<char*>(outTokens.data()), sizeof(int) * outTokens.size());
    if (!in.good()) {
        outTokens.clear();
        return false;
    }
    conv_to_swap_path_[conversationId] = path;
    return true;
}

bool LlamaCppModel::ensureSeqKvLoaded(llama_seq_id seqId, const std::vector<int>& tokens) {
    if (!ctx_) {
        return false;
    }
    auto it = seq_kv_loaded_.find((int32_t)seqId);
    const bool already = (it != seq_kv_loaded_.end() && it->second);
    if (already) {
        return true;
    }
    if (tokens.empty()) {
        seq_kv_loaded_[(int32_t)seqId] = true;
        return true;
    }

    llama_memory_seq_rm(llama_get_memory(ctx_), seqId, 0, -1);
    const int config_batch_size = (int)(config_.maxBatchSize > 0 ? config_.maxBatchSize : 512);
    int ctx_size = llama_n_ctx(ctx_);
    if (ctx_size <= 0) {
        ctx_size = (int)(config_.maxContextSize > 0 ? config_.maxContextSize : 4096);
    }
    int seq_max = llama_n_seq_max(ctx_);
    if (seq_max <= 0) {
        seq_max = 1;
    }
    int slot_size = ctx_size / seq_max;
    if (slot_size < 1) {
        slot_size = 1;
    }
    if (config_.enableCache && config_.cacheSize > 0) {
        int cache_seq_limit = (int)config_.cacheSize + 1;
        if (cache_seq_limit < 1) {
            cache_seq_limit = 1;
        }
        int cache_slot_size = ctx_size / cache_seq_limit;
        if (cache_slot_size > 0 && cache_slot_size < slot_size) {
            slot_size = cache_slot_size;
        }
    }
    if (slot_size > 0) {
        int max_tokens_in_slot = slot_size - 1;
        if (max_tokens_in_slot < 1) {
            return false;
        }
        if (tokens.size() > (size_t)max_tokens_in_slot) {
            return false;
        }
    }
    int batch_size = config_batch_size;
    if (slot_size > 0) {
        int safe_limit = slot_size - 1;
        int half_ctx = ctx_size > 1 ? (ctx_size / 2) : 1;
        if (half_ctx < 1) {
            half_ctx = 1;
        }
        if (safe_limit > half_ctx) {
            safe_limit = half_ctx;
        }
        if (safe_limit < 1) {
            safe_limit = 1;
        }
        if (batch_size > safe_limit) {
            batch_size = safe_limit;
        }
    }
    if (batch_size < 1) {
        batch_size = 1;
    }
    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    for (size_t i = 0; i < tokens.size();) {
        llama_batch_clear(batch);
        const size_t remaining = tokens.size() - i;
        int slot_remaining = slot_size - (int)i - 1;
        if (slot_remaining < 1) {
            llama_batch_free(batch);
            return false;
        }
        size_t chunk = remaining;
        if (chunk > (size_t)batch_size) {
            chunk = (size_t)batch_size;
        }
        if (chunk > (size_t)slot_remaining) {
            chunk = (size_t)slot_remaining;
        }
        for (size_t j = 0; j < chunk; j++) {
            const size_t pos = i + j;
            llama_batch_add(batch, (llama_token)tokens[pos], (llama_pos)pos, { seqId }, false);
        }
        if (i + chunk == tokens.size() && batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }
        int decode_result = llama_decode_with_timeout(ctx_, batch, 30000, nullptr);
        if (decode_result != 0) {
            llama_batch_free(batch);
            if (decode_result == -1) {
                std::cerr << "ensureSeqKvLoaded: llama_decode超时" << std::endl;
            }
            return false;
        }
        i += chunk;
    }
    llama_batch_free(batch);
    seq_kv_loaded_[(int32_t)seqId] = true;
    return true;
}

void LlamaCppModel::swapOutConversationIfNeeded(const std::string& conversationId, int32_t seqId) {
    if (!config_.enableKvSwap) {
        return;
    }
    if (conversationId.empty()) {
        return;
    }
    if (ctx_ == nullptr) {
        return;
    }
    auto tokIt = seq_tokens_.find(seqId);
    if (tokIt == seq_tokens_.end()) {
        return;
    }
    const size_t trigger = config_.kvSwapTriggerTokens > 0 ? config_.kvSwapTriggerTokens : 2048;
    if (tokIt->second.size() < trigger) {
        return;
    }
    if (!saveTokensToSwap(conversationId, tokIt->second)) {
        return;
    }
    llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id)seqId, 0, -1);
    tokIt->second.clear();
    tokIt->second.shrink_to_fit();
    seq_kv_loaded_[seqId] = false;
}

void LlamaCppModel::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    
    if (draft_ctx_) {
        llama_free(draft_ctx_);
        draft_ctx_ = nullptr;
    }
    if (draft_model_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
    }

    ready_ = false;

    max_sessions_ = 0;
    conv_to_seq_.clear();
    seq_to_conv_.clear();
    seq_tokens_.clear();
    lru_.clear();
}

std::vector<int> LlamaCppModel::tokenize(const std::string& text, bool add_bos) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    // Calculate required size
    int n_tokens = text.length() + 2; // approximation
    std::vector<llama_token> tokens(n_tokens);
    
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_bos, false);
    
    if (n < 0) {
        // Resize and try again
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_bos, false);
    }
    
    std::vector<int> result;
    if (n > 0) {
        result.reserve(n);
        for (int i = 0; i < n; i++) {
            result.push_back(tokens[i]);
        }
    }
    return result;
}

std::string LlamaCppModel::detokenize(const std::vector<int>& tokens) {
    if (tokens.empty()) return "";
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string result;
    for (int token : tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            // Buffer too small?
            n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        }
        if (n > 0) {
            result.append(buf, n);
        }
    }
    return result;
}

int32_t LlamaCppModel::getSeqId(const std::string& conversationId) {
    auto it = conv_to_seq_.find(conversationId);
    if (it != conv_to_seq_.end()) {
        return it->second;
    }

    if (max_sessions_ > 0) {
        while (conv_to_seq_.size() >= max_sessions_ && !lru_.empty()) {
            int32_t evict_id = lru_.front();
            lru_.pop_front();

            auto conv_it = seq_to_conv_.find(evict_id);
            if (conv_it != seq_to_conv_.end()) {
                const std::string evict_conv = conv_it->second;
                auto tok_it = seq_tokens_.find(evict_id);
                if (tok_it != seq_tokens_.end()) {
                    saveTokensToSwap(evict_conv, tok_it->second);
                }
                conv_to_seq_.erase(conv_it->second);
                seq_to_conv_.erase(conv_it);
            }
            seq_tokens_.erase(evict_id);
            seq_kv_loaded_.erase(evict_id);

            if (ctx_) {
                llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id)evict_id, 0, -1);
            }
        }
    }

    // Generate a small sequence ID between 0 and n_seq_max - 1
    int32_t max_seq = (int32_t)llama_n_seq_max(ctx_);
    if (max_seq <= 0) max_seq = 1;

    int32_t seq = -1;
    // Find an existing empty slot or use a simple counter-based approach
    for (int32_t i = 0; i < max_seq; i++) {
        if (seq_to_conv_.find(i) == seq_to_conv_.end()) {
            seq = i;
            break;
        }
    }

    // If no slot found, we should have evicted already, but as a fallback:
    if (seq == -1) {
        seq = 0; 
    }

    conv_to_seq_[conversationId] = seq;
    seq_to_conv_[seq] = conversationId;
    lru_.push_back(seq);
    seq_kv_loaded_[seq] = false;
    return seq;
}

void LlamaCppModel::touchSeq(int32_t seqId) {
    for (auto it = lru_.begin(); it != lru_.end();) {
        if (*it == seqId) {
            it = lru_.erase(it);
            continue;
        }
        ++it;
    }
    lru_.push_back(seqId);
}

void LlamaCppModel::evictIfNeeded() {
    if (max_sessions_ == 0) {
        return;
    }

    while (conv_to_seq_.size() > max_sessions_ && !lru_.empty()) {
        int32_t evict_id = lru_.front();
        lru_.pop_front();

        auto conv_it = seq_to_conv_.find(evict_id);
        if (conv_it != seq_to_conv_.end()) {
            const std::string evict_conv = conv_it->second;
            auto tok_it = seq_tokens_.find(evict_id);
            if (tok_it != seq_tokens_.end()) {
                saveTokensToSwap(evict_conv, tok_it->second);
            }
            conv_to_seq_.erase(conv_it->second);
            seq_to_conv_.erase(conv_it);
        }
        seq_tokens_.erase(evict_id);
        seq_kv_loaded_.erase(evict_id);

        if (ctx_) {
            llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id)evict_id, 0, -1);
        }
    }
}

LLMInferenceResponse LlamaCppModel::generate(const LLMInferenceRequest& request) {
    std::string pending_stream_buffer;
    if (!ready_) {
        return { "", 0, 0.0f, false, "Model not ready" };
    }

    if (request.shouldStop && request.shouldStop()) {
        return { "", 0, 0.0f, false, "Cancelled" };
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const bool use_cache =
        max_sessions_ > 0 && config_.enableCache && !request.conversationId.empty();

    llama_seq_id seq_id = 0;
    std::vector<int>* cached_tokens = nullptr;
    if (use_cache) {
        int32_t sid = getSeqId(request.conversationId);
        seq_id = (llama_seq_id)sid;
        touchSeq(sid);
        cached_tokens = &seq_tokens_[sid];

        if (config_.enableKvSwap && cached_tokens && cached_tokens->empty()) {
            std::vector<int> restored;
            if (loadTokensFromSwap(request.conversationId, restored)) {
                *cached_tokens = std::move(restored);
                seq_kv_loaded_[sid] = false;
            }
        }
        if (cached_tokens && !cached_tokens->empty()) {
            if (!ensureSeqKvLoaded(seq_id, *cached_tokens)) {
                // If we failed to ensure KV is loaded (e.g. OOM or batch size issues),
                // we must invalidate the cache for this sequence to avoid "Y = X + 1" position errors.
                // The model state might be inconsistent, so we clear it.
                std::cerr << "Warning: ensureSeqKvLoaded failed for seq " << seq_id 
                          << ". Clearing cache and forcing full re-evaluation." << std::endl;
                
                // Remove from VRAM explicitly to be safe
                llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, 0, -1);
                
                // Clear local cache reference so we start with n_past = 0
                cached_tokens->clear();
                seq_kv_loaded_[sid] = false;
            }
        }
    } else {
        seq_id = 0;
        llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, 0, -1);
    }

    prompt_tokens_ = tokenize(request.prompt, true);

    int n_ctx = llama_n_ctx(ctx_);
    int ctx_size = n_ctx > 0 ? n_ctx : (config_.maxContextSize > 0 ? (int)config_.maxContextSize : 4096);
    int seq_max = llama_n_seq_max(ctx_);
    if (seq_max <= 0) {
        seq_max = 1;
    }
    int slot_size = ctx_size / seq_max;
    if (slot_size < 1) {
        slot_size = 1;
    }
    if (config_.enableCache && config_.cacheSize > 0) {
        int cache_seq_limit = (int)config_.cacheSize + 1;
        if (cache_seq_limit < 1) {
            cache_seq_limit = 1;
        }
        int cache_slot_size = ctx_size / cache_seq_limit;
        if (cache_slot_size > 0 && cache_slot_size < slot_size) {
            slot_size = cache_slot_size;
        }
    }

    int slot_ctx_limit = ctx_size - 4;
    if (slot_size > 0) {
        int slot_limit = slot_size - 4;
        if (slot_limit < slot_ctx_limit) {
            slot_ctx_limit = slot_limit;
        }
    }
    if (slot_ctx_limit < 1) {
        slot_ctx_limit = 1;
    }
    if ((int)prompt_tokens_.size() > slot_ctx_limit) {
        return { "", 0, 0.0f, false, "Prompt too long for context" };
    }

    const size_t max_tokens = request.maxTokens > 0 ? request.maxTokens : 512;
    const float temperature = request.temperature > 0.0f ? request.temperature : config_.temperature;
    const int top_k = request.topK > 0 ? request.topK : config_.topK;
    const float top_p = request.topP > 0.0f ? request.topP : config_.topP;

    int batch_size = (int)(config_.maxBatchSize > 0 ? config_.maxBatchSize : 512);
    if (slot_size > 0) {
        int safe_limit = slot_size - 1;
        int half_ctx = ctx_size > 1 ? (ctx_size / 2) : 1;
        if (half_ctx < 1) {
            half_ctx = 1;
        }
        if (safe_limit > half_ctx) {
            safe_limit = half_ctx;
        }
        if (safe_limit < 1) {
            safe_limit = 1;
        }
        if (batch_size > safe_limit) {
            batch_size = safe_limit;
        }
    }
    if (batch_size < 1) {
        batch_size = 1;
    }
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    
    llama_batch draft_batch = llama_batch_init(batch_size, 0, 1);

    // Draft model check
    bool use_speculative = (draft_ctx_ != nullptr) && (temperature < 0.3f); 
    
    // For speculative, we use a fixed draft count for now
    const int n_draft = 5;

    int n_past = 0;
    if (use_cache && cached_tokens) {
        size_t common = 0;
        const size_t max_common = (std::min)(cached_tokens->size(), prompt_tokens_.size());
        while (common < max_common && (*cached_tokens)[common] == prompt_tokens_[common]) {
            common += 1;
        }
        if (common < cached_tokens->size()) {
            llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, (llama_pos)common, -1);
            cached_tokens->resize(common);
        }
        n_past = (int)common;
        
        // Draft cache management
        if (use_speculative) {
             // For simplicity, we just clear draft cache on context switch or non-match
             // A real implementation would track draft cache too
             llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, 0, -1);
        }
    } else {
        if (use_speculative) {
             llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, 0, -1);
        }
    }

    // Prefill Target Model
    for (size_t i = (size_t)n_past; i < prompt_tokens_.size();) {
        if (request.shouldStop && request.shouldStop()) {
            llama_batch_free(batch);
            llama_batch_free(draft_batch);
            return { "", 0, 0.0f, false, "Cancelled" };
        }
        llama_batch_clear(batch);
        const size_t remaining = prompt_tokens_.size() - i;
        int slot_remaining = slot_size - (int)i - 1;
        if (slot_remaining < 1) {
            llama_batch_free(batch);
            llama_batch_free(draft_batch);
            return { "", 0, 0.0f, false, "Prompt too long for context" };
        }
        size_t chunk = remaining;
        if (chunk > (size_t)batch_size) {
            chunk = (size_t)batch_size;
        }
        if (chunk > (size_t)slot_remaining) {
            chunk = (size_t)slot_remaining;
        }
        for (size_t j = 0; j < chunk; j++) {
            const size_t pos = i + j;
            llama_batch_add(batch, (llama_token)prompt_tokens_[pos], (llama_pos)pos, { seq_id }, false);
        }
        if (i + chunk == prompt_tokens_.size() && batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        int decode_result = llama_decode_with_timeout(ctx_, batch, 30000, request.shouldStop);
        if (decode_result != 0) {
            llama_batch_free(batch);
            llama_batch_free(draft_batch);
            if (decode_result == -1) {
                return { "", 0, 0.0f, false, "GPU推理超时（llama_decode卡死）" };
            } else if (decode_result == -2) {
                return { "", 0, 0.0f, false, "Cancelled" };
            }
            return { "", 0, 0.0f, false, "Failed to decode prompt" };
        }
        if (request.shouldStop && request.shouldStop()) {
            llama_batch_free(batch);
            llama_batch_free(draft_batch);
            return { "", 0, 0.0f, false, "Cancelled" };
        }
        
        // Prefill Draft Model
        if (use_speculative) {
             llama_batch_clear(draft_batch);
             for (size_t j = 0; j < chunk; j++) {
                 const size_t pos = i + j; // Draft uses same pos?
                 // Draft model typically uses seq_id 0
                 // Note: Draft context might have different n_past if not synced.
                 // Here we assume we feed same tokens.
                 llama_batch_add(draft_batch, (llama_token)prompt_tokens_[pos], (llama_pos)pos, { 0 }, j == chunk - 1);
             }
             if (llama_decode(draft_ctx_, draft_batch) != 0) {
                 use_speculative = false; // Disable if draft fails
             }
        }

        i += chunk;
    }

    int n_cur = (int)prompt_tokens_.size();
    if (use_cache && cached_tokens) {
        cached_tokens->assign(prompt_tokens_.begin(), prompt_tokens_.end());
    }

    int n_decode = 0;
    std::string generated_text;
    std::vector<int> output_tokens;
    
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const llama_vocab* draft_vocab = use_speculative ? llama_model_get_vocab(draft_model_) : nullptr;

    const size_t max_tokens_by_ctx =
        n_ctx > n_cur + 1 ? (size_t)(n_ctx - n_cur - 1) : (size_t)0;
    const size_t max_tokens_effective = (std::min)(max_tokens, max_tokens_by_ctx);

    // Generation loop
    while ((size_t)n_decode < max_tokens_effective) { 
        if (request.shouldStop && request.shouldStop()) {
            break;
        }

        if (use_speculative) {
            // 1. Generate K draft tokens
            std::vector<llama_token> drafts;
            drafts.reserve(n_draft);
            
            for (int i = 0; i < n_draft; ++i) {
                if (request.shouldStop && request.shouldStop()) {
                    break;
                }
                auto* logits = llama_get_logits(draft_ctx_);
                int n_vocab_draft = llama_vocab_n_tokens(draft_vocab);
                llama_token draft_token = sample_from_logits(logits, n_vocab_draft, top_k, top_p, temperature);
                
                // Safety check: ensure token is valid for target model if we have it
                int n_vocab_target = llama_vocab_n_tokens(vocab);
                if (draft_token >= n_vocab_target) {
                    // Incompatible vocabulary, stop speculative for this step
                    use_speculative = false;
                    break;
                }

                drafts.push_back(draft_token);
                
                llama_batch_clear(draft_batch);
                llama_batch_add(draft_batch, draft_token, n_cur + i, { 0 }, true);
                if (llama_decode(draft_ctx_, draft_batch) != 0) {
                    use_speculative = false;
                    break;
                }
            }
            if (request.shouldStop && request.shouldStop()) {
                break;
            }
            
            if (!use_speculative) continue; // Fallback to normal
            
            // 2. Verify with target model
            llama_batch_clear(batch);
            // Add draft tokens to target batch
            // We need to verify drafts[0]...drafts[K-1]
            // Input to target for drafts[0] is current state (already processed) -> we just need to decode drafts[0] to see if it produces next?
            // Wait, standard speculative:
            // Input to target: [last_token] -> predicts T1
            // We already ran target for last_token.
            // So we need to feed drafts[0] to target, it predicts T2.
            // But we need to check if T1 == drafts[0].
            // We have logits from previous step of target.
            
            // Current state of target: processed up to n_cur-1. Logits available for n_cur.
            auto* logits = llama_get_logits(ctx_);
            int n_vocab_target = llama_vocab_n_tokens(vocab);
            
            // Sample T1 from target logits
            llama_token target_token = sample_from_logits(logits, n_vocab_target, top_k, top_p, temperature);
            
            std::vector<llama_token> accepted;
            bool mismatched = false;
            
            // Check first token
            if (target_token == drafts[0]) {
                accepted.push_back(target_token);
                // Now we need to verify drafts[1]...
                // We need to run target on drafts[0] to get logits for T2.
                // We can batch this!
                
                // Construct batch: drafts[0], drafts[1], ... drafts[K-1]
                // Their output logits will predict T2, T3...
                
                llama_batch_clear(batch);
                for (size_t i = 0; i < drafts.size(); ++i) {
                     // We input drafts[i] at pos n_cur + i
                     // We want logits for all of them
                     llama_batch_add(batch, drafts[i], n_cur + i, { seq_id }, true);
                }
                
                int decode_result = llama_decode_with_timeout(ctx_, batch, 30000, request.shouldStop);
                if (decode_result != 0) {
                     // Error or timeout
                     if (decode_result == -1) {
                         std::cerr << "Speculative decode: llama_decode超时" << std::endl;
                     }
                     break; 
                }
                
                // Now check results
                for (size_t i = 0; i < drafts.size(); ++i) {
                     if (request.shouldStop && request.shouldStop()) {
                         break;
                     }
                     // Logic:
                     // i=0: Input drafts[0], Output Logits -> Predict T2. Check if T2 == drafts[1].
                     // (If i is last, Predict T_{K+1})
                     
                     // Get logits for the i-th token in batch
                     // In llama_batch, we can find where the logits are.
                     // batch.logits[i] is true, so logits are stored sequentially?
                     // llama_get_logits_ith(ctx, i)
                     
                     float* ith_logits = llama_get_logits_ith(ctx_, (int32_t)i);
                     llama_token next_pred = sample_from_logits(ith_logits, n_vocab_target, top_k, top_p, temperature);
                     
                     if (i < drafts.size() - 1) {
                         if (next_pred == drafts[i+1]) {
                             accepted.push_back(next_pred);
                         } else {
                             // Mismatch at i+1
                             // drafts[i+1] is wrong. next_pred is the correct one.
                             // We accept up to drafts[i] (which is accepted[i])
                             // And we add next_pred as the corrected token.
                             accepted.push_back(next_pred); // This is the first rejected token corrected
                             mismatched = true;
                             
                             // We need to rollback draft KV to n_cur + i + 1
                             llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_cur + i + 1, -1);
                             // Also need to rollback target KV?
                             // We fed [0...K-1]. 
                             // We accepted 0...i. (i+1 tokens total).
                             // The batch processing added KV for 0...K-1.
                             // We need to remove KV for i+1...K-1.
                             llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, n_cur + i + 1, -1);
                             
                             break;
                         }
                     } else {
                         // Last draft token. next_pred is the extra token!
                         accepted.push_back(next_pred);
                     }
                }
                if (request.shouldStop && request.shouldStop()) {
                    break;
                }
                
            } else {
                // First token mismatch
                accepted.push_back(target_token);
                mismatched = true;
                // Rollback draft
                llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_cur, -1);
                // Target KV is fine (we didn't feed anything yet)
            }
            
            // Process accepted tokens
            for (llama_token t : accepted) {
                output_tokens.push_back(t);
                 char buf[256];
                int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
                if (n > 0) {
                    generated_text.append(buf, n);
                    if (request.streamOutput && request.onTokenGenerated) {
                        pending_stream_buffer.append(buf, n);
                        size_t valid_len = get_valid_utf8_len(pending_stream_buffer);
                        if (valid_len > 0) {
                            request.onTokenGenerated(pending_stream_buffer.substr(0, valid_len));
                            pending_stream_buffer = pending_stream_buffer.substr(valid_len);
                        }
                    }
                }
                
                // Check EOS
                if (llama_vocab_is_eog(vocab, t) || t == llama_vocab_eos(vocab)) {
                    n_decode = max_tokens_effective; // Force break outer
                    break;
                }
            }
            
            n_cur += accepted.size();
            n_decode += accepted.size();
            
            // Prepare for next loop
            // If we ended with mismatch, we already have the corrected token in KV?
            // Wait, if mismatch at first token:
            // We accepted target_token. We need to feed it to target to update logits for next loop.
            // If mismatch later:
            // We accepted drafts[0...i] and next_pred.
            // We fed drafts[0...i]. We did NOT feed next_pred (it was the output of drafts[i]).
            // So we need to feed the LAST accepted token to get logits for next step.
            
            llama_token last_accepted = accepted.back();
            llama_batch_clear(batch);
            llama_batch_add(batch, last_accepted, n_cur - 1, { seq_id }, true);
            int decode_result = llama_decode_with_timeout(ctx_, batch, 30000, request.shouldStop);
            if (decode_result != 0) {
                if (decode_result == -1) {
                    std::cerr << "Speculative decode (sync): llama_decode超时" << std::endl;
                }
                break;
            }
            
            // Sync draft with last accepted
            // Draft KV is at n_cur - 1. We need to feed last_accepted to it.
            llama_batch_clear(draft_batch);
            llama_batch_add(draft_batch, last_accepted, n_cur - 1, { 0 }, true);
            llama_decode(draft_ctx_, draft_batch);
            
        } else {
            // Normal decoding
            auto logits = llama_get_logits(ctx_);
            int n_vocab_target = llama_vocab_n_tokens(vocab);

            llama_token new_token_id = sample_from_logits(logits, n_vocab_target, top_k, top_p, temperature);
            
            if (llama_vocab_is_eog(vocab, new_token_id) || new_token_id == llama_vocab_eos(vocab)) {
                break;
            }

            output_tokens.push_back(new_token_id);
            
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                generated_text.append(buf, n);
                if (request.streamOutput && request.onTokenGenerated) {
                    pending_stream_buffer.append(buf, n);
                    size_t valid_len = get_valid_utf8_len(pending_stream_buffer);
                    if (valid_len > 0) {
                        request.onTokenGenerated(pending_stream_buffer.substr(0, valid_len));
                        pending_stream_buffer = pending_stream_buffer.substr(valid_len);
                    }
                }
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, new_token_id, n_cur, { seq_id }, true);
            
            n_cur++;
            n_decode++;

            int decode_result = llama_decode_with_timeout(ctx_, batch, 30000, request.shouldStop);
            if (decode_result != 0) {
                if (decode_result == -1) {
                    std::cerr << "Normal decode: llama_decode超时" << std::endl;
                }
                break;
            }
        }
    }

    llama_batch_free(batch);
    llama_batch_free(draft_batch);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end_time - start_time;

    if (use_cache && cached_tokens) {
        if (!output_tokens.empty()) {
            cached_tokens->insert(cached_tokens->end(), output_tokens.begin(), output_tokens.end());
        }
        touchSeq((int32_t)seq_id);
        evictIfNeeded();

        swapOutConversationIfNeeded(request.conversationId, (int32_t)seq_id);
    }

    // Ensure generated_text is valid UTF-8 before returning to Python
    {
        size_t valid_len = get_valid_utf8_len(generated_text);
        if (valid_len < generated_text.size()) {
            generated_text.resize(valid_len);
        }
    }

    if (request.shouldStop && request.shouldStop()) {
        return { generated_text, (size_t)n_decode, duration.count(), false, "Cancelled" };
    }

    return { generated_text, (size_t)n_decode, duration.count(), true, "" };
}

std::string LlamaCppModel::getModelInfo() const {
    char buf[128];
    llama_model_desc(model_, buf, sizeof(buf));
    return std::string("LlamaCppModel: ") + buf;
}

size_t LlamaCppModel::getMemoryUsage() const {
    return llama_model_size(model_);
}

bool LlamaCppModel::isReady() const {
    return ready_;
}

} // namespace ai_scheduler
