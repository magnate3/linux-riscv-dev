#include <iostream>
#include <vector>
#include <queue>

struct Element {
    uint32_t key;
    int freq;
    bool operator>(const Element& o) const { return freq > o.freq; }
};

class CompressedRadixTree {
    struct Node {
        uint32_t prefix = 0;    // 压缩存储的位路径
        int prefix_len = 0;     // 路径长度（比特数）
        int max_freq = 0;       // 用于 Top-K 剪枝
        int leaf_freq = 0;      // 若为叶子节点则 > 0
        Node* children[2] = {nullptr}; // 二进制分支 (可扩展为多叉)

        bool is_leaf() const { return children[0] == nullptr && children[1] == nullptr; }
    };

    Node* root = nullptr;

public:
    // 插入逻辑：涉及路径分裂 (Split)
    void Insert(uint32_t key, int count) {
        root = insert(root, key, 31, count);
    }

    // Top-K 检索
    void GetTopK(int k, std::vector<Element>& res) {
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> heap;
        search(root, 0, 31, k, heap);
        while (!heap.empty()) {
            res.push_back(heap.top());
            heap.pop();
        }
    }

private:
    // 路径分裂逻辑（简化版二叉基数树）
    Node* insert(Node* n, uint32_t key, int bit_pos, int count) {
        if (!n) {
            Node* leaf = new Node();
            leaf->prefix = key; // 剩余位全部压缩
            leaf->prefix_len = bit_pos + 1;
            leaf->max_freq = leaf->leaf_freq = count;
            return leaf;
        }

        // 1. 计算当前 key 与节点 prefix 的公共位数 (Common Prefix)
        int diff_bit = get_first_diff_bit(key, n->prefix, bit_pos, n->prefix_len);
        
        if (diff_bit < bit_pos - n->prefix_len + 1) {
            // 2. 需要分裂节点 (Split)
            Node* parent = new Node();
            int split_pos = diff_bit; 
            parent->prefix = n->prefix;
            parent->prefix_len = bit_pos - split_pos;
            
            // 将旧节点挂在新节点下
            int old_dir = (n->prefix >> split_pos) & 1;
            parent->children[old_dir] = n;
            n->prefix_len -= (parent->prefix_len + 1); // 缩短旧节点路径

            // 插入新分支
            int new_dir = (key >> split_pos) & 1;
            parent->children[new_dir] = insert(nullptr, key, split_pos - 1, count);
            
            parent->max_freq = std::max(n->max_freq, parent->children[new_dir]->max_freq);
            return parent;
        }

        // 3. 路径完全匹配，继续向深层走或更新叶子
        if (n->is_leaf()) {
            n->leaf_freq = count;
        } else {
            int dir = (key >> (bit_pos - n->prefix_len)) & 1;
            n->children[dir] = insert(n->children[dir], key, bit_pos - n->prefix_len - 1, count);
        }
        n->max_freq = std::max(n->max_freq, count);
        return n;
    }

    int get_first_diff_bit(uint32_t k1, uint32_t k2, int start, int len) {
        // 使用 CLZ (Count Leading Zeros) 指令加速位比较
        uint32_t mask = (len == 32) ? 0xFFFFFFFF : ((1U << len) - 1) << (start - len + 1);
        uint32_t diff = (k1 ^ k2) & mask;
        if (diff == 0) return start - len; // 无差异
        return 31 - __builtin_clz(diff);
    }

    void search(Node* n, uint32_t key, int bit_pos, int k, auto& heap) {
        if (!n || (heap.size() == k && n->max_freq <= heap.top().freq)) return;

        if (n->is_leaf()) {
            Element e = {n->prefix, n->leaf_freq};
            if (heap.size() < k) heap.push(e);
            else if (e.freq > heap.top().freq) { heap.pop(); heap.push(e); }
            return;
        }

        // 优先递归 max_freq 更大的分支，以更快提升 Top-K 阈值
        int first = (n->children[1] && (!n->children[0] || n->children[1]->max_freq > n->children[0]->max_freq)) ? 1 : 0;
        search(n->children[first], key, bit_pos - n->prefix_len - 1, k, heap);
        search(n->children[1 - first], key, bit_pos - n->prefix_len - 1, k, heap);
    }
};

int main()
{
    return 0;
}
