#include <iostream>
#include <vector>
#include <queue>
#include <bitset>
#include <algorithm>
#include <cmath>

// 模拟向量与 Top-K 结果
struct VectorResult {
    uint64_t signature; // 量化后的特征签名
    float score;        // 相似度得分 (例如与 Query 的汉明距离或余弦值)
    
    bool operator>(const VectorResult& other) const { return score > other.score; }
};

class EmbeddingRadixTree {
private:
    struct Node {
        uint64_t prefix = 0;      // 路径压缩存储的公共位
        int prefix_len = 0;       // 公共位长度
        int count = 0;            // 子树包含的向量总数
        Node* children[2] = {nullptr, nullptr}; // 二叉分支
        bool is_leaf = false;

        ~Node() { delete children[0]; delete children[1]; }
    };

    Node* root = nullptr;
    const int MAX_BITS = 64; // 假设量化签名为 64 位

public:
    EmbeddingRadixTree() { root = new Node(); }
    ~EmbeddingRadixTree() { delete root; }

    // 1. 插入量化后的 Embedding 签名
    void Insert(uint64_t signature) {
        root = insert(root, signature, MAX_BITS - 1);
    }

    // 2. 检索 Top-K (基于汉明距离相似度)
    std::vector<VectorResult> GetTopK(uint64_t query_sig, int k) {
        std::priority_queue<VectorResult, std::vector<VectorResult>, std::greater<VectorResult>> heap;
        search(root, query_sig, MAX_BITS - 1, k, heap);

        std::vector<VectorResult> results;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

private:
    // 计算两个签名之间的公共前缀位数
    int get_common_prefix(uint64_t s1, uint64_t s2, int start_bit) {
        uint64_t diff = s1 ^ s2;
        if (start_bit < 63) {
            uint64_t mask = ~((1ULL << (start_bit + 1)) - 1);
            diff |= mask; // 忽略高位
        }
        int lz = __builtin_clzll(diff);
        int common = 63 - lz; 
        return std::max(0, start_bit - common + 1);
    }

    Node* insert(Node* n, uint64_t sig, int bit_pos) {
        if (!n) {
            Node* leaf = new Node();
            leaf->prefix = sig;
            leaf->prefix_len = bit_pos + 1;
            leaf->is_leaf = true;
            leaf->count = 1;
            return leaf;
        }

        // 路径压缩逻辑：检查当前 sig 与节点 prefix 的匹配程度
        int match = 0;
        if (n->prefix_len > 0) {
            uint64_t mask = (n->prefix_len == 64) ? ~0ULL : (1ULL << n->prefix_len) - 1;
            uint64_t n_bits = (n->prefix >> (bit_pos - n->prefix_len + 1)) & mask;
            uint64_t s_bits = (sig >> (bit_pos - n->prefix_len + 1)) & mask;
            
            if (n_bits != s_bits) {
                // 需要分裂节点
                int diff_at = 63 - __builtin_clzll(n_bits ^ s_bits);
                int common = n->prefix_len - (diff_at + 1);

                Node* parent = new Node();
                parent->prefix = n->prefix;
                parent->prefix_len = common;
                
                int n_dir = (n->prefix >> diff_at) & 1;
                n->prefix_len -= (common + 1);
                parent->children[n_dir] = n;
                
                int s_dir = (sig >> diff_at) & 1;
                parent->children[s_dir] = insert(nullptr, sig, diff_at - 1);
                parent->count = parent->children[0]->count + (parent->children[1] ? parent->children[1]->count : 0);
                return parent;
            }
        }

        // 继续向深层递归
        if (n->is_leaf) {
            n->count++; // 简单处理重复 Key
        } else {
            int dir = (sig >> (bit_pos - n->prefix_len)) & 1;
            n->children[dir] = insert(n->children[dir], sig, bit_pos - n->prefix_len - 1);
            n->count++;
        }
        return n;
    }

    void search(Node* n, uint64_t query, int bit_pos, int k, auto& heap) {
        if (!n) return;

        // 剪枝逻辑：估算当前分支的汉明距离下界
        // 在量化搜索中，如果该分支即便剩下位全匹配也进不了 Top-K，则跳过
        if (n->is_leaf) {
            float dist = (float)__builtin_popcountll(n->prefix ^ query);
            if (heap.size() < k) {
                heap.push({n->prefix, dist});
            } else if (dist < heap.top().score) {
                heap.pop();
                heap.push({n->prefix, dist});
            }
            return;
        }

        // 启发式搜索：优先走与 Query 位匹配的分支
        int best_dir = (query >> (bit_pos - n->prefix_len)) & 1;
        search(n->children[best_dir], query, bit_pos - n->prefix_len - 1, k, heap);
        search(n->children[1 - best_dir], query, bit_pos - n->prefix_len - 1, k, heap);
    }
};

int main() {
    EmbeddingRadixTree tree;
    // 插入一些模拟的 Embedding 签名 (量化后)
    tree.Insert(0b10110011);
    tree.Insert(0b10110100);
    tree.Insert(0b11001100);
    tree.Insert(0b00110011);

    uint64_t query = 0b10110000;
    auto topk = tree.GetTopK(query, 2);

    std::cout << "Top-2 Results for Query " << std::bitset<8>(query) << ":\n";
    for (auto& res : topk) {
        std::cout << "Sig: " << std::bitset<8>(res.signature) << " | Hamming Dist: " << res.score << "\n";
    }
    return 0;
}

