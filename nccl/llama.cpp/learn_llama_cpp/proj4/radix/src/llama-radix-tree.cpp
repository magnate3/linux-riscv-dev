#include "llama-radix-tree.h"
#include <cassert>
#include <algorithm>

std::pair<llama_radix_node *, uint32_t> llama_radix_tree::find_prefix(const std::vector<llama_token> & tokens) const {
    if (tokens.empty()) {
        return {root.get(), 0};
    }

    // FIX: Need to cast away const for the return value
    // This is safe because we're only reading, not modifying
    llama_radix_node * current = const_cast<llama_radix_node *>(root.get());
    uint32_t matched_length = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = current->children.find(tokens[i]);
        if (it == current->children.end()) {
            break;
        }
        current = it->second.get();
        matched_length++;
    }

    return {current, matched_length};
}

llama_radix_node * llama_radix_tree::insert_sequence(
    const std::vector<llama_token> & tokens,
    const std::vector<uint32_t> & cache_slots) {

    assert(cache_slots.size() == n_layers);

    llama_radix_node * current = root.get();

    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = current->children.find(tokens[i]);

        if (it == current->children.end()) {
            // Create new node
            auto new_node = std::make_unique<llama_radix_node>();
            new_node->token = tokens[i];
            new_node->parent = current;
            current->children[tokens[i]] = std::move(new_node);
        }
        current = current->children[tokens[i]].get();
    }

    // Update cache slots for the final node
    current->cache_slots = cache_slots;
    current->inc_ref();
    current->touch(tick());

    return current;
}

void llama_radix_tree::remove_sequence(const std::vector<llama_token> & tokens) {
    if (tokens.empty()) return;

    auto [node, match_len] = find_prefix(tokens);

    if (match_len == tokens.size()) {
        node->dec_ref();

        // Clean up nodes with zero references
        while (node && node->parent && node->can_evict() && node->children.empty()) {
            llama_radix_node * parent = node->parent;
            parent->children.erase(node->token);
            node = parent;
        }
    }
}

std::vector<std::vector<uint32_t>> llama_radix_tree::evict_lru(uint32_t max_evict_count) {
    std::vector<llama_radix_node *> evictable;
    collect_evictable_nodes(root.get(), evictable);

    // Sort by last access time (LRU first)
    std::sort(evictable.begin(), evictable.end(),
        [](const llama_radix_node * a, const llama_radix_node * b) {
            return a->last_access_time < b->last_access_time;
        });

    std::vector<std::vector<uint32_t>> freed_slots;

    for (uint32_t i = 0; i < std::min(max_evict_count, (uint32_t)evictable.size()); ++i) {
        auto * node = evictable[i];
        freed_slots.push_back(node->cache_slots);

        // Remove from parent
        if (node->parent) {
            node->parent->children.erase(node->token);
        }
    }

    return freed_slots;
}

size_t llama_radix_tree::get_node_count() const {
    return count_nodes(root.get());
}

size_t llama_radix_tree::get_cached_token_count() const {
    return get_node_count() - 1; // Exclude root node
}

size_t llama_radix_tree::count_nodes(const llama_radix_node * node) const {
    if (!node) return 0;

    size_t count = 1;
    for (const auto & [token, child] : node->children) {
        count += count_nodes(child.get());
    }
    return count;
}

void llama_radix_tree::collect_evictable_nodes(
    llama_radix_node * node,
    std::vector<llama_radix_node *> & nodes) {

    if (!node || node == root.get()) return;

    if (node->can_evict()) {
        nodes.push_back(node);
    }

    for (auto & [token, child] : node->children) {
        collect_evictable_nodes(child.get(), nodes);
    }
}

void llama_radix_tree::increment_node_ref(llama_radix_node * node) {
    if (!node || node == root.get()) {
        return;
    }
    
    node->inc_ref();
    node->touch(tick());
    
    LLAMA_LOG_INFO("%s: incremented ref count for node (depth=%u, ref_count=%u)\n",
        __func__, node->depth, node->ref_count);
}