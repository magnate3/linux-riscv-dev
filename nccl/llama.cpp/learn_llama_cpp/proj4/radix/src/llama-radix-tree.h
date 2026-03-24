#pragma once

#include "llama.h"
#include "llama-impl.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>

// Forward declarations
//
// Radix Tree Node for KV Cache Reuse
//
class llama_kv_cache;

struct llama_radix_node {
    // The token at this node (root has special value -1)
    llama_token token = -1;
    
    // KV cache slot information
    // Which cells in the KV cache contain the KV pairs for this prefix
    std::vector<uint32_t> cache_slots;  // Per-layer cache slot indices
    
    // Reference count - how many sequences are using this prefix
    uint32_t ref_count = 0;
    
    // Last access time (for LRU eviction)
    uint64_t last_access_time = 0;
    
    // Parent node (nullptr for root)
    llama_radix_node * parent = nullptr;
    
    // Children nodes - map from token to child node
    std::unordered_map<llama_token, std::unique_ptr<llama_radix_node>> children;
    
    // Depth in the tree (number of tokens from root)
    uint32_t depth = 0;
    
    // Lock status - true if this node's cache slots are immutable
    bool locked = false;
    
    // Constructor - fixed initialization order
    llama_radix_node() : token(-1), ref_count(1), last_access_time(0), parent(nullptr) {}
    
    // Get the full token sequence from root to this node
    std::vector<llama_token> get_token_sequence() const {
        std::vector<llama_token> sequence;
        const llama_radix_node * node = this;
        
        while (node && node->token != -1) {
            sequence.push_back(node->token);
            node = node->parent;
        }
        
        std::reverse(sequence.begin(), sequence.end());
        return sequence;
    }
    
    // Increment reference count
    void inc_ref() {
        ref_count++;
    }
    
    // Decrement reference count
    void dec_ref() {
        if (ref_count > 0) {
            ref_count--;
        }
    }
    
    // Check if node can be evicted
    bool can_evict() const {
        return ref_count == 0 && !locked;
    }
    
    // Update last access time
    void touch(uint64_t time) {
        last_access_time = time;
    }
};

//
// Radix Tree for managing shared prefixes
//
// Current limitations:
// - Single-stream mode only (unified KV cache)
// - All cache slots belong to stream 0
//
// Future enhancements for multi-stream support:
// - Per-stream radix trees
// - Stream-aware cache slot allocation
// - Cross-stream prefix sharing
//
class llama_radix_tree {
public:
    explicit llama_radix_tree(uint32_t n_layers) : n_layers(n_layers), current_time(0) {
        root = std::make_unique<llama_radix_node>();
        root->token = -1; // Special marker for root
    }

    // Find the longest matching prefix for a given token sequence
    // Returns the node representing the longest matching prefix
    std::pair<llama_radix_node *, uint32_t> find_prefix(const std::vector<llama_token> & tokens) const;
    
    // Insert a new sequence into the tree
    // Returns the newly created or existing node for this sequence
    llama_radix_node * insert_sequence(
        const std::vector<llama_token> & tokens,
        const std::vector<uint32_t> & cache_slots);
    
    // Phase 3.2: Increment reference on existing node (for seq_cp)
    void increment_node_ref(llama_radix_node * node);
    
    // Remove a sequence from the tree (decrement ref counts)
    void remove_sequence(const std::vector<llama_token> & tokens);
    
    // Evict unused nodes to free cache slots (LRU policy)
    // Returns the cache slots that were freed
    std::vector<std::vector<uint32_t>> evict_lru(uint32_t max_evict_count);
    
    // Get root node
    llama_radix_node * get_root() { return root.get(); }
    
    // Increment global time counter
    uint64_t tick() { return ++current_time; }
    
    // Phase 3.3: Multi-stream support (future)
    // Currently returns 0 (single stream), can be extended later
    uint32_t get_stream_for_node(const llama_radix_node * node) const {
        GGML_UNUSED(node);
        return 0; // Always stream 0 in unified mode
    }
    
    // Get total number of nodes in the tree
    size_t get_node_count() const;
    
    // Get total number of tokens cached
    size_t get_cached_token_count() const;
    
private:
    std::unique_ptr<llama_radix_node> root;
    uint32_t n_layers; // Number of layers in the model (used for cache_slots sizing)
    uint64_t current_time;
    
    // Phase 3.3: Future multi-stream support
    // std::vector<std::unique_ptr<llama_radix_node>> stream_roots; // One root per stream
    // std::unordered_map<llama_radix_node *, uint32_t> node_to_stream; // Track node ownership
    
    // Helper function to recursively count nodes
    size_t count_nodes(const llama_radix_node * node) const;
    
    // Helper function to collect evictable nodes
    void collect_evictable_nodes(
        llama_radix_node * node,
        std::vector<llama_radix_node *> & nodes);
};