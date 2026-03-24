#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <iomanip>

typedef int32_t llama_token;

class LlamaTokenDATTracker {
private:
    struct DATNode {
        int32_t base = -1;
        int32_t check = -1;
        uint32_t frequency = 0; // 核心统计位：存储该前缀出现的频次
    };

    std::vector<DATNode> nodes;
    
    // 为了防止数组过大，通常对 Token ID 进行映射
    // llama_token 可能达到 128,000，直接作为下标会导致空洞过多
    static uint32_t map_token(llama_token id) {
        // 示例：简单截断或使用热点映射表
        return (uint32_t)id % 10000; 
    }

public:
    LlamaTokenDATTracker(size_t size = 1000000) {
        nodes.resize(size);
        nodes[0].base = 1; // 根节点
    }

    // 统计前缀出现的频率
    void record_prefix(const std::vector<llama_token>& tokens) {
        int s = 0; // 起始状态
        for (llama_token t_id : tokens) {
            uint32_t c = map_token(t_id);
            uint32_t t = nodes[s].base + c;

            // 动态扩容
            if (t >= nodes.size()) nodes.resize(t * 2);

            // 冲突处理与状态转移
            if (nodes[t].check == -1 || nodes[t].check == s) {
                nodes[t].check = s;
                if (nodes[t].base == -1) nodes[t].base = t + 1; // 简单分配 base
                s = t;
            } else {
                // 发生碰撞，需执行 Relocate 操作（DAT 构建的难点）
                // relocate(s, b_new); 
                break; 
            }
        }
        // 在最终状态节点增加计数
        nodes[s].frequency++;
    }

    uint32_t get_frequency(const std::vector<llama_token>& tokens) {
        int s = 0;
        for (llama_token t_id : tokens) {
            uint32_t c = map_token(t_id);
            uint32_t t = nodes[s].base + c;
            if (t >= nodes.size() || nodes[t].check != s) return 0;
            s = t;
        }
        return nodes[s].frequency;
    }
};
// 引用之前定义的 LlamaTokenDATTracker 类
// 确保包含前文中的代码逻辑

void print_test_result(const std::string& label, bool success) {
    std::cout << std::left << std::setw(40) << label 
              << "[" << (success ? "PASS" : "FAIL") << "]" << std::endl;
}

int main() {
    // 1. 初始化 Tracker (预分配空间)
    LlamaTokenDATTracker tracker(1000000);

    // 2. 模拟常用的 Token 序列 (N-grams)
    // 示例: "<|begin_of_text|><|start_header_id|>system"
    std::vector<int32_t> system_prefix = {128000, 128006, 9125};
    // 示例: "You are a helpful"
    std::vector<int32_t> help_prefix = {2675, 527, 263, 11143};
    // 示例: 随机短前缀
    std::vector<int32_t> random_prefix = {100, 200};

    std::cout << "--- Starting LlamaTokenDATTracker Test ---" << std::endl;

    // 3. 测试插入与频率统计
    tracker.record_prefix(system_prefix);
    tracker.record_prefix(system_prefix); // 再次插入，频率应为 2
    tracker.record_prefix(help_prefix);   // 插入 1 次
    
    // 4. 验证频率获取
    uint32_t freq_sys = tracker.get_frequency(system_prefix);
    uint32_t freq_help = tracker.get_frequency(help_prefix);
    uint32_t freq_none = tracker.get_frequency(random_prefix);

    print_test_result("Check System Prefix Frequency (Exp: 2)", freq_sys == 2);
    print_test_result("Check Help Prefix Frequency (Exp: 1)", freq_help == 1);
    print_test_result("Check Non-existent Frequency (Exp: 0)", freq_none == 0);

    // 5. 性能测试：海量重复查询
    std::cout << "\n--- Performance Benchmarking ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    const int iterations = 100000;
    uint32_t total_freq = 0;
    for (int i = 0; i < iterations; ++i) {
        total_freq += tracker.get_frequency(system_prefix);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Total Elapsed Time: " << elapsed.count() << " us" << std::endl;
    std::cout << "Average Latency per Query: " << (elapsed.count() / iterations) << " us" << std::endl;

    // 6. 前缀子集验证
    // 插入了 [128000, 128006, 9125]，查询 [128000, 128006] 应能找到路径但频率不同
    std::vector<int32_t> sub_prefix = {128000, 128006};
    uint32_t freq_sub = tracker.get_frequency(sub_prefix);
    std::cout << "\nSub-prefix [128000, 128006] Frequency: " << freq_sub << " (Expected: 0 if not recorded separately)" << std::endl;

    std::cout << "------------------------------------------" << std::endl;

    return 0;
}
