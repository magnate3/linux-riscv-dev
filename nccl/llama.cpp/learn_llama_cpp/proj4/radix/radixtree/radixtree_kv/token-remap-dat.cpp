#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <chrono>

typedef int32_t llama_token;

// --- DAT 节点结构 (对齐以优化缓存) ---
struct alignas(32) DATNode {
    int32_t base = -1;
    int32_t check = -1;
    int32_t value = -1;           // 关联的 KV Cache Page ID
    uint64_t mask[4] = {0, 0, 0, 0}; // 256-bit 子节点位图

    void set_bit(uint8_t c) { mask[c >> 6] |= (1ULL << (c & 0x3F)); }
    void clear_bit(uint8_t c) { mask[c >> 6] &= ~(1ULL << (c & 0x3F)); }
};

class LlamaDATManager {
private:
    std::vector<DATNode> dat;
    
    // --- Token Remapping 核心逻辑 ---
    // 将原始的 llama_token (0~128k) 映射到紧凑的 8-bit ID (0~255)
    // 针对前缀热点（如模板字符、System Prompt 常用词）进行优化
    std::unordered_map<llama_token, uint8_t> token_map;
    uint8_t next_internal_id = 0;

    uint8_t get_internal_id(llama_token tok, bool create = false) {
        auto it = token_map.find(tok);
        if (it != token_map.end()) return it->second;
        if (create && next_internal_id < 255) {
            token_map[tok] = next_internal_id++;
            return token_map[tok];
        }
        return 255; // 预留 255 作为 Unknown/Fallback ID
    }

    // 利用 __builtin_ctzll 快速提取子节点
    std::vector<uint8_t> get_active_children(int s) {
        std::vector<uint8_t> children;
        for (int i = 0; i < 4; ++i) {
            uint64_t m = dat[s].mask[i];
            while (m) {
                int bit = __builtin_ctzll(m);
                children.push_back((uint8_t)((i << 6) | bit));
                m &= (m - 1);
            }
        }
        return children;
    }

    int find_safe_base(const std::vector<uint8_t>& children) {
        for (int b = 1; ; ++b) {
            bool ok = true;
            for (uint8_t c : children) {
                int target = b + c;
                if (target < (int)dat.size() && dat[target].check != -1) {
                    ok = false; break;
                }
            }
            if (ok) return b;
        }
    }

    void relocate(int s, uint8_t collision_char) {
        auto children = get_active_children(s);
        int old_base = dat[s].base;
        std::vector<uint8_t> new_children = children;
        new_children.push_back(collision_char);

        int new_base = find_safe_base(new_children);
        if (new_base + 256 >= (int)dat.size()) dat.resize(new_base + 1024);

        for (uint8_t c : children) {
            int old_pos = old_base + c;
            int new_pos = new_base + c;

            dat[new_pos].base = dat[old_pos].base;
            dat[new_pos].check = s;
            dat[new_pos].value = dat[old_pos].value;
            std::copy(std::begin(dat[old_pos].mask), std::end(dat[old_pos].mask), std::begin(dat[new_pos].mask));

            auto grand_children = get_active_children(new_pos);
            for (uint8_t gc : grand_children) {
                dat[dat[new_pos].base + gc].check = new_pos;
            }
            dat[old_pos] = DATNode();
        }
        dat[s].base = new_base;
    }

public:
    LlamaDATManager(size_t size = 100000) {
        dat.resize(size);
        dat[0].base = 1; 
    }

    // 插入 Token 序列
    void insert(const std::vector<llama_token>& tokens, int page_id) {
        int s = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            uint8_t c = get_internal_id(tokens[i], true); // 映射 Token
            int t = dat[s].base + c;

            if (t >= (int)dat.size()) dat.resize(t + 1024);

            if (dat[t].check != -1 && dat[t].check != s) {
                relocate(s, c);
                t = dat[s].base + c;
            }

            dat[t].check = s;
            dat[s].set_bit(c);
            if (dat[t].base == -1) dat[t].base = t + 1;
            
            if (i == tokens.size() - 1) dat[t].value = page_id;
            s = t;
        }
    }

    // 查找 Token 序列
    int search(const std::vector<llama_token>& tokens) {
        int s = 0;
        for (auto tok : tokens) {
            auto it = token_map.find(tok);
            if (it == token_map.end()) return -1; // 该词从未出现在热点映射中

            uint8_t c = it->second;
            int t = dat[s].base + c;
            if (t >= (int)dat.size() || dat[t].check != s) return -1;
            s = t;
        }
        return dat[s].value;
    }
};

// --- 测试程序 ---
int main() {
    LlamaDATManager manager(10000);

    // 模拟 Llama-3 常用热点 Token
    llama_token t_begin = 128000; // <|begin_of_text|>
    llama_token t_start = 128006; // <|start_header_id|>
    llama_token t_sys   = 9125;   // system
    
    std::vector<llama_token> prefix = {t_begin, t_start, t_sys};

    std::cout << "[Step 1] 插入热点前缀并执行 Token Remapping...\n";
    manager.insert(prefix, 42); // 关联到物理页 42

    std::cout << "[Step 2] 执行检索性能测试 (1,000,000 次)...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    int dummy = 0;
    for (int i = 0; i < 1000000; ++i) {
        dummy += manager.search(prefix);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "匹配结果 Page ID: " << manager.search(prefix) << "\n";
    std::cout << "总耗时: " << elapsed.count() << " ms\n";
    std::cout << "平均单次延迟: " << (elapsed.count() * 1000 / 1000000.0) << " ns\n";

    return 0;
}

