#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <cstring>

// 模拟 llama.cpp 的数据结构
struct llama_token_data {
    int32_t id;
    float   logit;
};

struct llama_token_data_array {
    llama_token_data * data;
    size_t size;
};

class LlamaRadixSampler {
public:
    // 将 IEEE 754 浮点数映射为可直接按位比较大小的无符号整数
    // 关键技巧：处理正负号，使映射后的 uint32 保持原始 float 的大小顺序
    static inline uint32_t float_to_ordered_uint(float f) {
        union { float f; uint32_t i; } u;
        u.f = f;
        return (u.i & 0x80000000u) ? ~u.i : (u.i | 0x80000000u);
    }

    // 分层 Radix-Select 采样核心函数
    static void apply_top_k_radix(llama_token_data_array * candidates, int k) {
        if (k <= 0 || candidates->size <= (size_t)k) return;

        const size_t n = candidates->size;
        std::vector<uint32_t> u_logits(n);

        // 1. 预处理：映射 (O(N))
        for (size_t i = 0; i < n; ++i) {
            u_logits[i] = float_to_ordered_uint(candidates->data[i].logit);
        }

        uint32_t threshold = 0;
        uint32_t mask = 0;

        // 2. 分层统计 (从高位到低位，每层 4-bit)
        // 4-bit 步长平衡了层数(8层)与统计桶大小(16个)
        for (int shift = 28; shift >= 0; shift -= 4) {
            uint32_t counts[16] = {0};
            uint32_t current_bit_mask = (0xFu << shift);
            
            // 统计当前层 16 个桶的分布
            for (size_t i = 0; i < n; ++i) {
                if ((u_logits[i] & mask) == threshold) {
                    uint32_t bucket = (u_logits[i] >> shift) & 0xF;
                    counts[bucket]++;
                }
            }

            // 寻找第 K 个元素落在哪一个桶中
            uint32_t accum = 0;
            int target_bucket = 0;
            for (int i = 15; i >= 0; --i) {
                accum += counts[i];
                if (accum >= (uint32_t)k) {
                    target_bucket = i;
                    break;
                }
            }

            // 更新阈值前缀
            threshold |= (uint32_t(target_bucket) << shift);
            mask |= current_bit_mask;
        }

        // 3. 原地过滤 (Partition)
        // 此时 threshold 就是第 K 大（或稍大一点点，处理并列值）的界限
        size_t write_idx = 0;
        for (size_t i = 0; i < n; ++i) {
            if (u_logits[i] >= threshold) {
                std::swap(candidates->data[write_idx++], candidates->data[i]);
                if (write_idx >= (size_t)k) break;
            }
        }

        candidates->size = write_idx;
    }
};

// 使用示例
int main() {
    std::vector<llama_token_data> tokens = {
        {0, 1.5f}, {1, 10.2f}, {2, -1.0f}, {3, 8.5f}, {4, 10.2f}, {5, 0.1f}
    };
    llama_token_data_array arr = { tokens.data(), tokens.size() };

    std::cout << "Before Radix Top-K (K=2):" << std::endl;
    LlamaRadixSampler::apply_top_k_radix(&arr, 2);

    for (size_t i = 0; i < arr.size; ++i) {
        std::cout << "Token: " << arr.data[i].id << " Logit: " << arr.data[i].logit << std::endl;
    }

    return 0;
}

