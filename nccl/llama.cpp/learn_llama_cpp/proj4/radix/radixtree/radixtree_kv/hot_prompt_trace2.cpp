#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <iomanip>

// 模拟 llama.cpp 的 Token 常量
const int32_t IM_START_TOKEN = 128001; // <|im_start|>
const int32_t SYSTEM_TOKEN   = 15043;  // system
const int32_t USER_TOKEN     = 2183;   // user

struct KVBlock {
    int32_t page_id;
    bool is_hot = false;
};

class ImStartHotCache {
private:
    struct DATNode {
        int32_t base = -1;
        int32_t check = -1;
        KVBlock* block = nullptr;
    };

    std::vector<DATNode> nodes;

    // 针对海量 Token ID 的映射函数 (Alphabet Reduction)
    // 仅映射出现在热点模板中的 Token
    uint32_t map_token(int32_t token) {
        switch(token) {
            case IM_START_TOKEN: return 1;
            case SYSTEM_TOKEN:   return 2;
            case USER_TOKEN:     return 3;
            default:             return 0; // 0 表示非热点
        }
    }

public:
    ImStartHotCache() {
        nodes.resize(1024); // 热点层通常很小，可完全放入 L1 Cache
        nodes[0].base = 1;  // 根节点状态
    }

    // 预注入热点路径：<|im_start|>system
    void pin_system_prompt(KVBlock* b) {
        // 状态 0 --(IM_START)--> 状态 t1
        int s = 0;
        uint32_t c1 = map_token(IM_START_TOKEN);
        uint32_t t1 = nodes[s].base + c1;
        nodes[t1].check = s;
        nodes[t1].base = 10; // 分配一个偏移

        // 状态 t1 --(SYSTEM)--> 状态 t2
        uint32_t c2 = map_token(SYSTEM_TOKEN);
        uint32_t t2 = nodes[t1].base + c2;
        nodes[t2].check = t1;
        nodes[t2].block = b; // 绑定 KV Cache 块
        b->is_hot = true;
    }

    // 极速匹配接口
    KVBlock* match(const int32_t* tokens, size_t len, size_t& matched) {
        int s = 0;
        matched = 0;
        for (size_t i = 0; i < len; ++i) {
            uint32_t c = map_token(tokens[i]);
            if (c == 0) break; // 遇到非热点 Token，退出 DAT 层

            uint32_t t = nodes[s].base + c;
            if (t < nodes.size() && nodes[t].check == s) {
                s = t;
                if (nodes[s].block) {
                    matched = i + 1;
                    return nodes[s].block;
                }
            } else break;
        }
        return nullptr;
    }
};

// --- 测试程序 ---
int main() {
    ImStartHotCache hot_cache;
    KVBlock sys_block = { 777,false };

    // 模拟启动阶段注入
    hot_cache.pin_system_prompt(&sys_block);

    // 模拟输入 Prompt: <|im_start|> system ...
    int32_t input_tokens[] = { 128001, 15043, 200, 300 };
    size_t matched_len = 0;

    auto start = std::chrono::high_resolution_clock::now();
    
    // 模拟高频调用
    KVBlock* result = hot_cache.match(input_tokens, 4, matched_len);

    auto end = std::chrono::high_resolution_clock::now();

    if (result && result->is_hot) {
        std::cout << "DAT Hit! Matched Length: " << matched_len << " Page: " << result->page_id << std::endl;
        std::cout << "Latency: " << std::chrono::duration<double, std::nano>(end - start).count() << " ns" << std::endl;
    }

    return 0;
}

