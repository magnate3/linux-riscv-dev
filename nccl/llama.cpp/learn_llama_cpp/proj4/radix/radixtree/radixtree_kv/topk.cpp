#include <iostream>
#include <vector>
#include <queue>
#include <stdint.h>

struct Element {
    uint32_t key;
    int frequency;
    // 最小堆比较：频率最小的在堆顶，方便弹出
    bool operator>(const Element& other) const {
        return frequency > other.frequency;
    }
};

class RadixTopK {
private:
    static constexpr int BITS = 8; // 每层 8 位 (256 个分支)
    static constexpr int SIZE = 1 << BITS;
    static constexpr int MASK = SIZE - 1;

    struct Node {
        int max_freq = 0;      // 子树中的最大频率，用于剪枝
        Node* children[SIZE] = {nullptr};
        int leaf_freq = 0;     // 仅叶子节点有效
    };

    Node* root = new Node();

public:
    // 1. 更新频率（模拟流式输入）
    void Insert(uint32_t key, int count) {
        Node* curr = root;
        for (int shift = 24; shift >= 0; shift -= BITS) {
            int idx = (key >> shift) & MASK;
            if (!curr->children[idx]) curr->children[idx] = new Node();
            curr = curr->children[idx];
            // 更新路径上的最大值
            if (count > curr->max_freq) curr->max_freq = count;
        }
        curr->leaf_freq = count;
    }

    // 2. 检索 Top-K
    void GetTopK(int k, std::vector<Element>& result) {
        // 使用最小堆维护当前的 Top-K 候选者
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> min_heap;

        // 深度优先搜索 + 剪枝
        search(root, 0, 24, k, min_heap);

        while (!min_heap.empty()) {
            result.push_back(min_heap.top());
            min_heap.pop();
        }
    }

private:
    void search(Node* node, uint32_t current_key, int shift, int k, 
                std::priority_queue<Element, std::vector<Element>, std::greater<Element>>& heap) {
        
        // --- 核心剪枝逻辑 ---
        // 如果当前子树的最大频率还不如堆顶元素（Top-K 中的最小值），直接剪枝
        if (heap.size() == k && node->max_freq <= heap.top().frequency) {
            return;
        }

        // 到达叶子节点
        if (shift < 0) {
            Element e = {current_key, node->leaf_freq};
            if (heap.size() < k) {
                heap.push(e);
            } else if (e.frequency > heap.top().frequency) {
                heap.pop();
                heap.push(e);
            }
            return;
        }

        // 递归子节点
        for (int i = 0; i < SIZE; ++i) {
            if (node->children[i]) {
                search(node->children[i], current_key | (i << shift), shift - BITS, k, heap);
            }
        }
    }
};

int main() 
{
    return 0;
}
