#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cmath>

typedef int32_t llama_token;

// --- 核心组件 1: Count-Min Sketch (轻量级频率估算) ---
class TokenSketch {
    static constexpr int DEPTH = 4;
    static constexpr int WIDTH = 2048;
    uint16_t table[DEPTH][WIDTH] = {0};

public:
    void add(const llama_token* t, size_t len) {
        for (int i = 0; i < DEPTH; ++i) {
            uint32_t h = hash(t, len, i);
            if (table[i][h % WIDTH] < 0xFFFF) table[i][h % WIDTH]++;
        }
    }

    uint16_t estimate(const llama_token* t, size_t len) {
        uint16_t min_val = 0xFFFF;
        for (int i = 0; i < DEPTH; ++i) {
            uint32_t h = hash(t, len, i);
            min_val = std::min(min_val, table[i][h % WIDTH]);
        }
        return min_val;
    }

    void decay() {
        for (int i = 0; i < DEPTH; ++i)
            for (int j = 0; j < WIDTH; ++j) table[i][j] >>= 1; // 频率减半
    }

private:
    uint32_t hash(const llama_token* t, size_t len, int seed) {
        uint32_t h = seed;
        for (size_t i = 0; i < len; ++i) h = h * 31 + t[i];
        return h;
    }
};

// --- 核心组件 2: 高性能 DAT 管理器 (带 Token 映射与衰减) ---
class AgingRadixManager {
private:
    struct DATNode {
        int32_t base = -1, check = -1, page_id = -1;
        uint16_t freq = 0;
        uint64_t mask[4] = {0}; // 256-bit bitmask
    };

    std::vector<DATNode> dat;
    TokenSketch sketch;
    int16_t token_map[131072]; // 静态 Token 映射 (O(1))
    uint8_t next_id = 0;

    uint32_t access_count = 0;
    const uint32_t WINDOW_SIZE = 1000; // 每 1000 次操作触发一次衰减
    const uint16_t PROMOTE_THRESHOLD = 5; // 命中 5 次以上晋升 DAT

public:
    AgingRadixManager() {
        dat.resize(100000);
        dat[0].base = 1; 
        std::fill(std::begin(token_map), std::end(token_map), -1);
    }

    // --- 核心接口: 记录并检索 ---
    int search_and_update(const std::vector<llama_token>& tokens, int new_page_id) {
        access_count++;
        
        // 1. 更新频率统计
        sketch.add(tokens.data(), tokens.size());
        uint16_t est = sketch.estimate(tokens.data(), tokens.size());

        // 2. 检查衰减窗口
        if (access_count >= WINDOW_SIZE) {
            sketch.decay();
            apply_dat_aging();
            access_count = 0;
        }

        // 3. DAT 极速检索 (L1)
        int s = 0;
        bool miss = false;
        for (auto t : tokens) {
            int16_t c = token_map[t];
            if (c == -1) { miss = true; break; }
            int target = dat[s].base + c;
            if (target >= (int)dat.size() || dat[target].check != s) { miss = true; break; }
            s = target;
        }

        if (!miss && dat[s].page_id != -1) {
            dat[s].freq++; // 增加热度
            return dat[s].page_id;
        }

        // 4. 动态晋升 (L2 -> L1)
        if (est >= PROMOTE_THRESHOLD) {
            insert_to_dat(tokens, new_page_id, est);
        }

        return -1; // 缓存未命中或正在构建
    }

private:
    void apply_dat_aging() {
        for (auto& node : dat) {
            node.freq >>= 1;
            if (node.freq == 0) node.page_id = -1; // 逻辑删除过期热点
        }
    }

    void insert_to_dat(const std::vector<llama_token>& tokens, int page, uint16_t f) {
        int s = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (token_map[tokens[i]] == -1) {
                if (next_id == 255) return;
                token_map[tokens[i]] = next_id++;
            }
            uint8_t c = (uint8_t)token_map[tokens[i]];
            int t = dat[s].base + c;
            if (t >= (int)dat.size()) dat.resize(t + 1024);

            if (dat[t].check != -1 && dat[t].check != s) return; // 简化：碰撞不处理

            dat[t].check = s;
            dat[s].mask[c >> 6] |= (1ULL << (c & 0x3F));
            if (dat[t].base == -1) dat[t].base = t + 1;
            if (i == tokens.size() - 1) {
                dat[t].page_id = page;
                dat[t].freq = f;
            }
            s = t;
        }
    }
};

int main() {
    AgingRadixManager manager;

    // 模拟 Llama-3 Token 序列
    std::vector<llama_token> sys_prompt = {128000, 128006, 9125}; // <|begin|>...system
    std::vector<llama_token> user_prompt = {128001, 5634, 267};   // <|start|>...user

    std::cout << "--- Step 1: 模拟冷启动 (逐渐累加频率) ---" << std::endl;
    for (int i = 0; i < 4; ++i) {
        manager.search_and_update(sys_prompt, 101);
    }
    
    int res = manager.search_and_update(sys_prompt, 101);
    std::cout << "第5次访问结果 (应晋升): " << (res == 101 ? "Hit" : "Miss") << std::endl;

    std::cout << "\n--- Step 2: 模拟时间衰减 (Aging) ---" << std::endl;
    // 强制触发 1000 次空访问导致衰减
    std::vector<llama_token> dummy = {999};
    for (int i = 0; i < 1000; ++i) manager.search_and_update(dummy, 0);

    res = manager.search_and_update(sys_prompt, 101);
    std::cout << "衰减后访问 (频率归零被淘汰): " << (res == -1 ? "Expired (Success)" : "Still Hit (Fail)") << std::endl;

    std::cout << "\n--- Step 3: 性能基准测试 ---" << std::endl;
    // 重新让它变热
    for (int i = 0; i < 10; ++i) manager.search_and_update(sys_prompt, 101);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        manager.search_and_update(sys_prompt, 101);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "100w 次检索耗时: " << ms.count() << " ms" << std::endl;
    std::cout << "平均延迟: " << (ms.count() * 1000 / 1000000.0) << " ns" << std::endl;

    return 0;
}

