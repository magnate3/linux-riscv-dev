#include "llama.h"
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <list>
#include <string>
#define N_SEQ_MAX 8
struct Stream;
void process_decode(llama_batch & batch,llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_ctx);
// Radix 树节点：代表一段 KV Cache 中的 Token 序列
struct RadixNode {
    std::vector<llama_token> tokens; // 存储的 Token 片段
    int slot_id;                     // 对应 KV Cache 中的 Sequence ID (源 Slot)
    int end_pos;                     // 该节点在 KV Cache 中的结束位置 (n_past)
    
    int ref_count = 0;
    RadixNode* parent = nullptr;
    // LRU 辅助：指向自己在管理器链表中的位置
    std::list<RadixNode*>::iterator lru_it;
    bool in_lru = false;       // 标记是否在可回收队列中
    std::map<llama_token, std::unique_ptr<RadixNode>> children;

    RadixNode(int s_id, int pos) : slot_id(s_id), end_pos(pos) {}

    int length() const { return (int)tokens.size(); }
};

class RadixManager {
private:
    std::unique_ptr<RadixNode> root;
    llama_context* ctx;
    int n_parallel;
    std::list<RadixNode*> lru_list;
public:
    RadixManager(llama_context* c, int parallel) : ctx(c), n_parallel(parallel) {
        // 初始化根节点，绑定到公共前缀槽 Slot 0
        root = std::make_unique<RadixNode>(0, 0);
    }

    // 核心函数：匹配最长前缀，并执行 KV Cache 克隆
    int match_and_apply(int user_slot, const std::vector<llama_token>& prompt) {
        RadixNode* curr = root.get();
        int prompt_idx = 0;
        int total_matched = 0;

        while (prompt_idx < (int)prompt.size()) {
            auto it = curr->children.find(prompt[prompt_idx]);
            if (it == curr->children.end()) break;

            RadixNode* next = it->second.get();
            int i = 0;
            // 逐个 Token 比对节点内容
            while (i < next->length() && (prompt_idx + i) < (int)prompt.size() && 
                   next->tokens[i] == prompt[prompt_idx + i]) {
                i++;
            }

            if (i < next->length()) {
                // 【场景：节点分裂】 新 Prompt 在当前节点中途分叉
                split_node(next, i);
                total_matched += i;
                curr = next;
                break;
            } else {
                // 完全匹配该节点，继续向下钻取
                total_matched += i;
                prompt_idx += i;
                curr = next;
            }
        }

        if (total_matched > 0) {
            // 【核心修复】：物理/逻辑克隆 KV Cache
            // 1. 彻底清除用户 Slot 旧状态，规避 Y = X + 1 报错
            //llama_kv_cache_seq_rm(ctx, user_slot, 0, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), user_slot, -1, -1);
            // 2. 执行克隆：从匹配到的节点 Slot 继承数据
            // 解决 "is_full" 报错：确保 n_ctx 足够大或使用 Paged KV
            //llama_kv_cache_seq_cp(ctx, curr->slot_id, user_slot, 0, total_matched);
            llama_memory_seq_cp(llama_get_memory(ctx), curr->slot_id, user_slot, -1, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), user_slot, total_matched+1, -1);
        }
        // 1. 查找最长匹配 (此处省略具体的树遍历和节点分裂逻辑...)
    // curr 最终指向匹配到的最深节点
    
        if (total_matched> 0 && curr != root.get()) {
            // --- 核心操作：递归增加引用计数 ---
            RadixNode* temp = curr;
            while (temp && temp != root.get()) {
                // 如果该节点原本在 LRU 队列中（即之前没人用），现在被命中，必须移出
                if (temp->ref_count == 0) {
                    remove_from_lru(temp); 
                }
                
                temp->ref_count++; // 增加引用
                temp = temp->parent; // 向上递归到根
            }
        }
        return total_matched;
    }

    // 节点分裂逻辑：将一个长节点拆分为 [前缀] -> [后缀]
    void split_node(RadixNode* node, int split_at) {
        auto suffix = std::make_unique<RadixNode>(node->slot_id, node->end_pos);
        suffix->tokens.assign(node->tokens.begin() + split_at, node->tokens.end());
        suffix->children = std::move(node->children);
        suffix->parent = node;

        // 更新原节点
        node->tokens.resize(split_at);
        node->end_pos = node->end_pos - (int)suffix->tokens.size();
        node->children[suffix->tokens[0]] = std::move(suffix);
    }
#if 0
    // 将新处理完的请求沉淀为树节点（供后续请求复用）
    void insert_new_path(int slot_id, int start_pos, const std::vector<llama_token>& new_tokens) {
        // 简化逻辑：这里演示如何将用户生成的私有内容挂载到树上
        // 生产环境需结合 LRU 回收旧节点
        RadixNode* curr = root.get(); // 实际应找到匹配后的末尾节点
        auto newNode = std::make_unique<RadixNode>(slot_id, start_pos + (int)new_tokens.size());
        newNode->tokens = new_tokens;
        newNode->parent = curr;
        curr->children[new_tokens[0]] = std::move(newNode);
    }
#else
    void insert_new_path(int slot_id, int start_pos, const std::vector<llama_token>& new_tokens) {
    if (new_tokens.empty()) return;

    // 1. 寻找当前 Slot 在树中的挂载点 (通常是上一次 match_and_apply 留下的节点)
    // 如果是全新开始，则从 root 开始
    RadixNode* curr = find_node_by_slot_and_pos(slot_id, start_pos); 
    if (!curr) curr = root.get();

    // 2. 创建新节点，存储新增的 Token 段
    int end_pos = start_pos + (int)new_tokens.size();
    auto newNode = std::make_unique<RadixNode>(slot_id, end_pos);
    newNode->tokens = new_tokens;
    newNode->parent = curr;
    
    // 3. 初始状态设置
    // 新沉淀的节点目前没有被其他 Slot 引用，因此 ref_count = 0
    newNode->ref_count = 0;
#if 0 
    // 4. 【核心】：加入 LRU 链表
    // 只有加入 LRU，evict_oldest() 才能在空间不足时回收它
    lru_list.push_front(newNode.get());
    newNode->lru_it = lru_list.begin();
    newNode->in_lru = true;
#endif
    // 5. 挂载到树结构中
    llama_token first_token = new_tokens[0];
    curr->children[first_token] = std::move(newNode);

    printf("Radix Insert: Slot %d 的新路径已沉淀 (pos %d-%d)\n", slot_id, start_pos, end_pos);
    }
#endif
    RadixNode* find_node_by_slot_and_pos(int slot_id, int target_pos) {
    // 调用递归辅助函数
    return find_recursive(root.get(), slot_id, target_pos);
    }

    RadixNode* find_recursive(RadixNode* curr, int slot_id, int target_pos) {
    // 1. 检查当前节点是否匹配
    // 如果 Slot ID 一致，且该节点的结束位置刚好是我们要找的位置
    if (curr->slot_id == slot_id && curr->end_pos == target_pos) {
        return curr;
    }

    // 2. 递归查找子节点
    for (auto& it : curr->children) {
        RadixNode* found = find_recursive(it.second.get(), slot_id, target_pos);
        if (found) return found;
    }

        return nullptr; // 未找到
    }
    void merge_node(RadixNode* parent, RadixNode* child, llama_context* ctx) {
    // 1. 安全检查：只有当父节点仅有一个子节点时才能合并
    if (parent->children.size() != 1 || parent->children.begin()->second.get() != child) {
        return;
    }

    // 2. 合并 Token 序列
    // 将子节点的 tokens 追加到父节点末尾
    parent->tokens.insert(parent->tokens.end(), child->tokens.begin(), child->tokens.end());

    // 3. 继承子树
    // 父节点接管子节点的所有子节点
    parent->children = std::move(child->children);
    
    // 更新所有孙子节点的父指针
    for (auto& it : parent->children) {
        it.second->parent = parent;
    }

    // 4. 更新位置元数据
    // 父节点的起始位置 offset 不变，但 end_pos 变为子节点的 end_pos
    parent->end_pos = child->end_pos;

    // 5. 注意：子节点 child 会在 unique_ptr 作用域结束时自动释放
    printf("Radix Merge: 节点已合并，当前长度: %zu\n", parent->tokens.size());
   }
   //触发合并的时机：LRU 释放与路径缩减
   //在动态管理中，合并通常发生在 回收（Eviction） 或 任务结束（Finish Slot） 阶段。
   void release_slot_reference(RadixNode* leaf_node) {
    RadixNode* curr = leaf_node;
    
    while (curr && curr->parent) {
        curr->ref_count--;
         // 如果引用归零，加入 LRU 缓存池
        if (curr->ref_count == 0) {
                //lru_list.push_front(curr); // 最新访问的放头部
                //curr->lru_it = lru_list.begin();
                add_to_lru(curr);
        }
        // 如果引用归零，且该节点只有一个子节点，尝试向上合并
        if (curr->ref_count == 0 && curr->parent->children.size() == 1) {
            merge_node(curr->parent, curr, ctx);
        }
        
        curr = curr->parent;
    }
}
    // --- 辅助：释放引用并进入 LRU ---
    void release_node(RadixNode * node) {
        while (node && node->parent) { // 根节点通常常驻
            node->ref_count--;
            if (node->ref_count == 0 && node->children.empty()) {
                // 进入 LRU 待命
                if (!node->in_lru) {
                    lru_list.push_front(node);
                    node->lru_it = lru_list.begin();
                    node->in_lru = true;
                }
            }
            node = node->parent;
        }
    }
//在 RadixManager 中删除一个 RadixNode 必须遵循 “引用计数归零 -> 从 LRU 移除 -> 从树结构解绑 -> 物理清理 KV Cache” 的严谨顺序。
void delete_node(RadixNode* node) {
    if (!node || node == root.get()) return; // 根节点不可删除

    // 1. 【安全检查】只有引用计数为 0 的节点才能删除
    if (node->ref_count > 0) {
        fprintf(stderr, "警告：尝试删除正在被使用的节点 (ref_count: %d)\n", node->ref_count);
        return;
    }

    // 2. 【递归删除子节点】（可选，通常叶子节点先被 LRU 选中）
    // 如果直接删除中间节点，必须递归清理其所有子孙
    for (auto it = node->children.begin(); it != node->children.end(); ) {
        delete_node(it->second.get());
        it = node->children.erase(it);
    }

    // 3. 【物理清理 KV Cache】
    // 解决 "is_full" 报错：释放该段在物理 Buffer 中的空间
    int start_pos = node->end_pos - (int)node->tokens.size();
   // llama_kv_cache_seq_rm(ctx, node->slot_id, start_pos, node->end_pos);

    llama_memory_seq_rm(llama_get_memory(ctx), node->slot_id, start_pos, node->end_pos);
    // 4. 【从 LRU 链表抹除】
    if (node->in_lru) {
        lru_list.erase(node->lru_it);
        node->in_lru = false;
    }

    // 5. 【从父节点解绑】
    if (node->parent) {
        // 注意：这会触发 unique_ptr 的析构，彻底释放内存
        node->parent->children.erase(node->tokens[0]); 
    }
}

    // 将节点移出 LRU (当它被重新引用时)
    void remove_from_lru(RadixNode* node) {
        if (node->in_lru) {
            lru_list.erase(node->lru_it);
            node->in_lru = false;
        }
    }

    // 将节点加入 LRU (当引用归零时)
    void add_to_lru(RadixNode* node) {
        if (!node->in_lru && node != root.get()) {
            lru_list.push_front(node); // 最新释放的放头部
            node->lru_it = lru_list.begin();
            node->in_lru = true;
        }
    }
    // --- 释放 Slot 逻辑 ---
    void release_slot(RadixNode* leaf) {
        RadixNode* curr = leaf;
        while (curr && curr != root.get()) {
            curr->ref_count--;
            if (curr->ref_count == 0) {
                add_to_lru(curr); // 引用归零：进入待回收队列
            }
            curr = curr->parent;
        }
    }

#if 0
    // --- 强制驱逐逻辑 (解决 is_full 报错) ---
    void evict_to_make_room(int needed_tokens) {
        while (!lru_list.empty()) {
            int used = llama_get_kv_cache_used_cells(ctx);
            if (used + needed_tokens <= llama_n_ctx(ctx)) break;

            // 驱逐最久未使用的节点 (从尾部取)
            RadixNode* victim = lru_list.back();
            lru_list.pop_back();

            // 物理清理 KV Cache，腾出空间
            llama_memory_seq_rm(llama_get_memory(ctx), victim->slot_id, 
                                  victim->end_pos - victim->tokens.size(), 
                                  victim->end_pos);

            // 从树结构中彻底移除
            if (victim->parent) {
                victim->parent->children.erase(victim->tokens[0]);
            }
        }
    }
#endif
     // --- 核心：安全驱逐函数 ---
    bool evict_oldest() {
        if (lru_list.empty()) return false;

        // 1. 获取 LRU 尾部的“牺牲者”
        RadixNode* victim = lru_list.back();
        
        // 安全检查：严禁删除有子节点的中间节点（除非递归删除）
        // 或者是引用计数不为 0 的节点
        if (!victim->children.empty() || victim->ref_count > 0) {
            // 这种情况下通常将其移到 LRU 头部或报错，正常逻辑不应发生
            return false; 
        }

        // 2. 【物理删除】：释放 KV Cache 空间
        // 计算该节点在物理 Buffer 中的范围 [start, end)
        //int start_pos = victim->end_pos - victim->tokens.size();
        //llama_memory_seq_rm(llama_get_memory(ctx), victim->slot_id, start_pos, victim->end_pos);
        // 这一步解决了 "is_full" 报错，腾出了线性空间
        llama_memory_seq_rm(llama_get_memory(ctx), victim->slot_id, 
                                  victim->end_pos - victim->tokens.size(), 
                                  victim->end_pos);

        // 3. 【逻辑删除】：从树结构中移除
        lru_list.pop_back();
        if (victim->parent) {
            // 触发 unique_ptr 析构，释放 C++ 内存
            victim->parent->children.erase(victim->tokens[0]); 
        }

        return true;
    }
};
#if 0
struct Stream {
    // --- 基础状态 ---
    int slot_id;                // 唯一的槽位 ID (对应 KV Cache 的 Sequence ID)
    bool active = false;        // 当前流是否正在处理
    bool is_prefill = true;     // 是否处于首轮预填充阶段
    
    // --- Token 缓冲区 ---
    std::vector<llama_token> prompt_tokens;  // 原始输入的所有 Token
    std::vector<llama_token> generated_tokens; // 已生成的 Token 序列
    llama_token last_token;                 // 上一个生成的 Token (用于增量 decode)

    // --- KV Cache 进度管理 ---
    int n_past = 0;             // 当前 Slot 在 KV Cache 中的逻辑长度 (pos)
    int n_matched = 0;          // 在 Radix Tree 中匹配到的前缀长度

    // --- Radix Tree 绑定 ---
    // 指向 Radix Tree 中该流当前关联的最深节点
    // 用于在任务结束时执行 ref_count--
    RadixNode* matched_node = nullptr; 

    // --- 采样参数 (可选) ---
    // float temp = 0.8f;
    // int32_t top_k = 40;

    Stream(int id) : slot_id(id) {}
};
#else
struct Stream {
    int id;                         // 流 ID
    int slot_id;                    // 对应的 llama seq_id
    int n_past = 0;                 // 当前流在 KV Cache 中的进度
    std::vector<llama_token> pending_tokens; // 待处理的 Token (Pre-fill 或新生成的词)
    std::vector<llama_token> prompt_tokens;  // 原始输入的所有 Token
    //std::vector<llama_token> generated_tokens; // 已生成的 Token 序列
    bool is_prefill = true; // 标记是否是该流的第一轮（需要耦合前缀）
    bool active = false;            // 是否激活
    bool completed = false;         // 是否生成结束
};
    //Stream(int id) : slot_id(id) {}
#endif
static void
llama_batch_clear(struct llama_batch &batch)
{
    batch.n_tokens = 0;
}
#if 1
static void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}
#endif
std::string token_to_piece(const struct llama_vocab* vocab, llama_token token) {
    std::vector<char> result(32);
    const int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        llama_token_to_piece(vocab, token, result.data(), result.size(), 0, false);
    } else {
        result.resize(n_tokens);
    }
    return std::string(result.data(), result.size());
}
void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens]    = id;
    batch.pos[batch.n_tokens]      = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;

    batch.n_tokens++;
}
void process_stream_prefill(llama_context* ctx,const struct llama_vocab* vocab, RadixManager& radix, Stream& s,int n_ctx) {
    // 1. 调用 Radix 管理器查找并应用最长前缀
    int matched_len = radix.match_and_apply(s.slot_id, s.prompt_tokens);

    // 2. 构造增量 Batch (跳过已复用的 matched_len)
    llama_batch batch = llama_batch_init(2048, 0, 1);
    for (size_t i = matched_len; i < s.prompt_tokens.size(); ++i) {
        bool is_last = (i == s.prompt_tokens.size() - 1);
        
        // 关键点：pos 必须紧接在 matched_len 之后
        // 只传 {s.slot_id}，彻底解决 "coupled/diverged" 报错
        common_batch_add(batch, s.prompt_tokens[i], (int)i, { s.slot_id }, is_last);
    }

    // 3. 执行推理
    if (batch.n_tokens > 0) {
        if (llama_decode(ctx, batch) != 0) {
            printf("Slot %d: Decode failed! 检查 n_ctx 是否足够。\n", s.slot_id);
        }
        else
        {
            //process_decode(batch,ctx,vocab,{s},n_ctx);
        } 
    }

    // 4. 更新流状态
    s.n_past = s.prompt_tokens.size();
    s.is_prefill = false;
    llama_batch_free(batch);
}
void process_decode(llama_batch & batch,llama_context * ctx,const struct llama_vocab* vocab,std::vector<Stream> & streams, int n_ctx) {
#if 1
        // 遍历 batch 找到需要采样的 token
        for (int i = 0; i < batch.n_tokens; ++i) {
            // 只有标记了 logits 为 true 的位置才能调用 llama_get_logits_ith
            if (batch.logits[i]) {
                int slot_id = batch.seq_id[i][0]; // 获取该 logits 属于哪个流
                struct Stream & s = streams[slot_id -1];
                float * logits = llama_get_logits_ith(ctx, i);
                
                 // 贪婪采样
                 llama_token next_token = 0;
                 float max_p = -1e10;
                 for (int v = 0; v < llama_vocab_n_tokens(vocab); ++v) {
                     if (logits[v] > max_p) { max_p = logits[v]; next_token = v; }
                 }

                 // 转换并输出
                 std::string piece = token_to_piece(vocab, next_token);
                 printf("[Stream %d]: %s\n", s.id, piece.c_str());
                 fflush(stdout);

                 // 检查结束条件
                 if (next_token == llama_vocab_eos(vocab) || s.n_past >= n_ctx) {
                     s.active = false;
                     s.completed = true;
                     // 释放该槽位的 KV 缓存，让出显存
                     //llama_kv_cache_seq_rm(ctx, s.slot_id, -1, -1);
                     llama_memory_seq_rm (llama_get_memory(ctx),s.slot_id, -1, -1);
                     printf("[Stream %d]: generation finsh and kv cache free\n", s.id);
                 } else {
                     // 将新生成的 Token 放入下一轮处理队列
                     s.pending_tokens.push_back(next_token);
                 }
            }
        }
#endif
}
int main() {
    // 1. 初始化模型和上下文
    llama_model_params mparams = llama_model_default_params();
    std::string model_path = "/workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf";
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    int  N_CTX = 8196;
    // 明确关闭 Flash Attention
    //cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; 
    cparams.n_ctx = N_CTX; // 确保上下文足够容纳前缀 + 输出
    //开启 KV 缓存量化 (Q8_0) 以节省高并发下的显存
    //cparams.type_k = GGML_TYPE_Q8_0; 
    //cparams.type_v = GGML_TYPE_Q8_0;
    // 2. 关键修复：允许每个 Token 同时关联多个 Sequence ID
    // 如果你有关联 {0, slot_id} 的操作，这里至少设为 2
    cparams.n_seq_max = N_SEQ_MAX; // 建议设为你的最大并发 Slot 数 + 1
    const llama_vocab * vocab = llama_model_get_vocab(model);
    //llama_context * ctx = llama_new_context_with_model(model, cparams);
    llama_context * ctx = llama_init_from_model(model, cparams);
    auto tokenize = [&](std::string text, bool add_bos) {
        std::vector<llama_token> tokens(text.size() + 3);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
        tokens.resize(n);
        return tokens;
    };
        // 3. 处理【公共前缀】(seq_id = 0)
    std::string prefix_text = "you are a perfect expert";
    std::vector<llama_token> prefix_tokens(prefix_text.size() + 2);
    int n_prefix = llama_tokenize(vocab, prefix_text.c_str(), prefix_text.size(), prefix_tokens.data(), prefix_tokens.size(), true, true);
    prefix_tokens.resize(n_prefix);

    llama_batch batch = llama_batch_init(cparams.n_ctx, 0, 1);
    //llama_batch batch = llama_batch_init(cparams.n_ctx, 0, N_SEQ_MAX);
    for (int i = 0; i < n_prefix; ++i) {
        llama_batch_add(batch, prefix_tokens[i], i, {0}, false);
    }
    if (llama_decode(ctx, batch) != 0) { printf("Prefix decode failed\n"); return 1; }
    printf("[System] common prefix : %d tokens\n", n_prefix);
    // 4. 初始化多个并发流 (模拟两个用户同时提问)
    std::vector<Stream> streams(2);
    std::string queries[] = {"you are a perfect expert, talk a story about Los Angeles Lakers", "who is the best superstar of lakers?"};

    for (int i = 0; i < 2; ++i) {
        streams[i].id = i;
        streams[i].slot_id = i + 1; // seq_id 0 被前缀占用
        streams[i].active = true;
        streams[i].n_past = n_prefix;
        // 将用户问题转为 Token
        std::vector<llama_token> q_tokens(queries[i].size() + 2);
        int n = llama_tokenize(vocab, queries[i].c_str(), queries[i].size(), q_tokens.data(), q_tokens.size(), false, true);
        q_tokens.resize(n);
        streams[i].prompt_tokens= q_tokens;
        //streams[i].pending_tokens = q_tokens;
    }
#if 0
#else
      RadixManager manager(ctx,N_SEQ_MAX);
      manager.insert_new_path(0,0,prefix_tokens); 
      process_stream_prefill(ctx,vocab,manager,streams[0],cparams.n_ctx);
      process_stream_prefill(ctx,vocab,manager,streams[1],cparams.n_ctx);
     //process_first_turn(ctx,prefix_tokens,streams,n_prefix); 
     //process_first_turn(ctx,streams,n_prefix); 
     // *************************
     //process_first_turn(ctx,vocab,streams,n_prefix,cparams.n_ctx); 
     //process_first_turn_2(ctx,vocab,streams,n_prefix,cparams.n_ctx); 
     //process_first_turn(batch,ctx,streams,n_prefix); 
     // 遍历 batch 找到需要采样的 token
#if 0
    // 5. 【主循环】连续批处理 (Continuous Batching)
    while (true) {
        llama_batch_clear(batch);
        bool has_work = false;

        // 收集所有流待处理的 Token (不管是 Pre-fill 还是上一轮生成的)
        for (auto & s : streams) {
            if (!s.active) continue;
            has_work = true;
            for (size_t i = 0; i < s.pending_tokens.size(); ++i) {
                // 只有每个流输入序列的最后一个 Token 才需要计算 Logits 用于推理
                bool is_last = (i == s.pending_tokens.size() - 1);
                    llama_batch_add(batch, s.pending_tokens[i], s.n_past + i, {s.slot_id}, is_last);
            }
            s.n_past += s.pending_tokens.size();
            s.pending_tokens.clear();
        }

        if (!has_work || batch.n_tokens == 0) break;

         if (batch.n_tokens > 0) {
             // 执行单次批处理推理
             if (llama_decode(ctx, batch) != 0) {
                  printf("decode stream failed\n");
                 break;
             }
         }
         process_decode(batch,ctx,vocab,streams,cparams.n_ctx); 
      }
#endif
#endif
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
