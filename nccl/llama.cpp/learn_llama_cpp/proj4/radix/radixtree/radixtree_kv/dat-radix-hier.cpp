#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstring>

typedef int32_t llama_token;

// 模拟 KV Cache 物理块
struct KVBlock {
    int32_t page_id;
    int32_t ref_count = 0;
};

class LlamaPrefixIndexer {
private:
    // --- L1: Double-Array Trie 节点 (针对热点前缀) ---
    struct DATNode {
        int32_t base = -1;
        int32_t check = -1;
        KVBlock* block = nullptr;
    };

    // --- L2: Radix Tree 节点 (针对长路径压缩) ---
    struct RadixNode {
        std::vector<llama_token> prefix; // 路径压缩的 Token 序列
        std::unordered_map<llama_token, std::unique_ptr<RadixNode>> children;
        KVBlock* block = nullptr;
    };

    std::vector<DATNode> dat_l1;
    std::unique_ptr<RadixNode> radix_l2;
    
    // Token 映射：将 128k 词表映射到 DAT 的小空间 (如 0-1023)
    std::unordered_map<llama_token, uint32_t> hot_token_map;
    uint32_t next_hot_id = 1;

public:
    LlamaPrefixIndexer() {
        dat_l1.resize(65536);
        dat_l1[0].base = 1; 
        radix_l2 = std::make_unique<RadixNode>();
    }

    // 1. 注册热点 Token (例如模板字符)
    uint32_t register_hot_token(llama_token t) {
        if (hot_token_map.find(t) == hot_token_map.end()) {
            hot_token_map[t] = next_hot_id++;
        }
        return hot_token_map[t];
    }

    // 2. 插入热点前缀到 DAT (L1)
    void insert_hot_prefix(const std::vector<llama_token>& tokens, KVBlock* b) {
        int s = 0;
        for (auto t : tokens) {
            uint32_t code = register_hot_token(t);
            uint32_t target = dat_l1[s].base + code;
            if (target >= dat_l1.size()) dat_l1.resize(target * 2);
            
            dat_l1[target].check = s;
            if (dat_l1[target].base == -1) dat_l1[target].base = target + 1;
            s = target;
        }
        dat_l1[s].block = b;
    }

    // 3. 核心接口：分层检索
    KVBlock* find_longest_prefix(const std::vector<llama_token>& input, size_t& matched_len) {
        matched_len = 0;
        int s = 0;
        KVBlock* best_hit = nullptr;

        // 优先匹配 DAT (L1)
        size_t i = 0;
        for (; i < input.size(); ++i) {
            auto it = hot_token_map.find(input[i]);
            if (it == hot_token_map.end()) break;

            uint32_t t = dat_l1[s].base + it->second;
            if (t < dat_l1.size() && dat_l1[t].check == s) {
                s = t;
                if (dat_l1[s].block) {
                    best_hit = dat_l1[s].block;
                    matched_len = i + 1;
                }
            } else break;
        }

        // 如果 DAT 匹配中断，进入 Radix Tree (L2)
        if (i < input.size()) {
            RadixNode* curr = radix_l2.get();
            // 此处省略复杂的路径压缩 Split/Match 逻辑，仅展示层级衔接
            // curr = curr->children[input[i]]; ...
        }

        return best_hit;
    }
};

// --- 测试程序 ---
int main() {
    LlamaPrefixIndexer indexer;
    
    // 模拟热点：<|im_start|>system (Token IDs: 128001, 15043)
    std::vector<llama_token> sys_prefix = {128001, 15043};
    KVBlock sys_kv = { 1024,0 };
    
    indexer.insert_hot_prefix(sys_prefix, &sys_kv);

    // 检索请求
    std::vector<llama_token> query = {128001, 15043, 29871, 304};
    size_t matched = 0;
    KVBlock* result = indexer.find_longest_prefix(query, matched);

    if (result) {
        std::cout << "Successfully matched " << matched << " tokens in L1 (DAT).\n";
        std::cout << "Physical Page ID: " << result->page_id << "\n";
    }

    return 0;
}

