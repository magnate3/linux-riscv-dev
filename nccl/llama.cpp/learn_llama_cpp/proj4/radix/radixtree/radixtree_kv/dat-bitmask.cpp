#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iomanip>

// 模拟 llama.cpp 的 Token ID (int32_t)
typedef int32_t llama_token;

// --- 核心结构：带位图的 DAT 节点 ---
struct alignas(32) DATNode {
    int32_t base = -1;
    int32_t check = -1;
    int32_t value = -1;  // 关联的 KV Cache Page ID
    uint64_t mask[4] = {0, 0, 0, 0}; // 256-bit 子节点位图

    void set_bit(uint8_t c) { mask[c >> 6] |= (1ULL << (c & 0x3F)); }
    void clear_bit(uint8_t c) { mask[c >> 6] &= ~(1ULL << (c & 0x3F)); }
    bool has_bit(uint8_t c) const { return (mask[c >> 6] & (1ULL << (c & 0x3F))) != 0; }
};

class LlamaDATManager {
private:
    std::vector<DATNode> dat;
    std::vector<uint64_t> global_used; // 全局占用位图 (1位代表1个节点被check占用)

    void set_global_used(int pos) {
        if (pos / 64 >= global_used.size()) global_used.resize(pos / 64 + 1024, 0);
        global_used[pos >> 6] |= (1ULL << (pos & 0x3F));
    }

    // 获取节点的所有子字符 (利用 __builtin_ctzll 加速)
    std::vector<uint8_t> get_active_children(int s) {
        std::vector<uint8_t> children;
        for (int i = 0; i < 4; ++i) {
            uint64_t m = dat[s].mask[i];
            while (m) {
                int bit = __builtin_ctzll(m);
                children.push_back((uint8_t)((i << 6) | bit));
                m &= (m - 1); // 清除最低位
            }
        }
        return children;
    }

    // 寻找不冲突的 base 偏移
    int find_safe_base(const std::vector<uint8_t>& children) {
        for (int b = 1; ; ++b) {
            bool ok = true;
            for (uint8_t c : children) {
                int target = b + c;
                if (target < dat.size() && dat[target].check != -1) {
                    ok = false; break;
                }
            }
            if (ok) return b;
        }
    }

    // 重定位逻辑 (核心)
    void relocate(int s, uint8_t collision_char) {
        auto children = get_active_children(s);
        int old_base = dat[s].base;
        
        // 包含引起冲突的新字符一起寻找新家
        std::vector<uint8_t> new_children = children;
        new_children.push_back(collision_char);
        int new_base = find_safe_base(new_children);

        if (new_base + 255 >= dat.size()) dat.resize(new_base + 512);

        for (uint8_t c : children) {
            int old_pos = old_base + c;
            int new_pos = new_base + c;

            // 迁移数据
            dat[new_pos].base = dat[old_pos].base;
            dat[new_pos].check = s;
            dat[new_pos].value = dat[old_pos].value;
            memcpy(dat[new_pos].mask, dat[old_pos].mask, sizeof(uint64_t) * 4);

            // 修正孙子节点的 check 指针
            auto grand_children = get_active_children(new_pos);
            for (uint8_t gc : grand_children) {
                dat[dat[new_pos].base + gc].check = new_pos;
            }

            // 清理旧节点
            dat[old_pos] = DATNode();
        }
        dat[s].base = new_base;
    }

public:
    LlamaDATManager(size_t size = 100000) {
        dat.resize(size);
        dat[0].base = 1; // 根节点
    }

    // 插入 Token 序列 (简化映射：Token % 256)
    void insert(const std::vector<llama_token>& tokens, int page_id) {
        int s = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            uint8_t c = (uint8_t)(tokens[i] & 0xFF);
            int t = dat[s].base + c;

            if (t >= dat.size()) dat.resize(t + 1024);

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

    // 极速匹配
    int search(const std::vector<llama_token>& tokens) {
        int s = 0;
        for (auto tok : tokens) {
            uint8_t c = (uint8_t)(tok & 0xFF);
            int t = dat[s].base + c;
            if (t >= dat.size() || dat[t].check != s) return -1;
            s = t;
        }
        return dat[s].value;
    }
};

void run_test() {
    LlamaDATManager manager(1000000);

    // 1. 模拟热点前缀：ChatML 模板 <|im_start|>system
    std::vector<llama_token> sys_prompt = {128001, 15043, 29871};
    manager.insert(sys_prompt, 1024); // 关联到 Page 1024

    // 2. 模拟碰撞：插入一个会导致 Relocate 的前缀
    // 故意构造一个可能落在同一 base 区域的 token
    std::vector<llama_token> user_prompt = {128001, 15043, 305}; 
    manager.insert(user_prompt, 2048);

    std::cout << "--- DAT Search Test ---" << std::endl;

    // 3. 验证检索正确性
    int res1 = manager.search(sys_prompt);
    int res2 = manager.search(user_prompt);

    std::cout << "Test 1 (System Prompt): " << (res1 == 1024 ? "PASS" : "FAIL") 
              << " (Page: " << res1 << ")" << std::endl;
    std::cout << "Test 2 (User Prompt):   " << (res2 == 2048 ? "PASS" : "FAIL") 
              << " (Page: " << res2 << ")" << std::endl;

    // 4. 性能压力测试
    std::cout << "\n--- Performance Benchmarking (100k searches) ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    int dummy = 0;
    for (int i = 0; i < 100000; ++i) {
        dummy += manager.search(sys_prompt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "Total Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "Average Latency: " << (elapsed.count() / 100000.0) << " us" << std::endl;
}

int main() {
    try {
        run_test();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

