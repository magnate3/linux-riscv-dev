
```c
// llama.cpp KV cache slot management
struct llama_kv_cell {
    llama_pos pos;           // Position in sequence
    llama_pos delta;         // RoPE position delta
    int32_t src;             // Source cell for copy
    int32_t seq_id;          // Sequence ID for batching
    llama_seq_id seq_ids[];  // Multiple sequences can share cells
};

struct llama_kv_cache {
    llama_kv_cell * cells;   // Per-position cells
    ggml_tensor * k;         // [n_ctx, n_embd] key cache
    ggml_tensor * v;         // [n_ctx, n_embd] value cache (transposed!)
    // ... defragmentation support
};
```

```
// find how many cells are currently in use
static int32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size - 1; i > 0; --i) {
        if (cache.cells[i].pos >= 0 && !cache.cells[i].seq_id.empty()) {
            return i + 1;
        }
    }

    return 0;
}

```


```
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_ctx    = cache.size;
    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens > n_ctx) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > n_ctx) {
            n_tested += n_ctx - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= n_ctx) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }
```


```
data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(lctx.kv_self.cells[i].pos, ubatch.pos[j], hparams.n_rel_attn_bkts, lctx.is_encoding);
```


```
// Update cache with new tokens
static void llama_kv_cache_update(
    struct llama_context * ctx,
    const llama_batch & batch,
    int32_t layer_idx
) {
    struct llama_kv_cache & cache = ctx->kv_self;

    // For each token in batch
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        llama_pos pos = batch.pos[i];
        llama_seq_id seq_id = batch.seq_id[i][0];

        // Find cache cell for this position
        struct llama_kv_cell & cell = cache.cells[pos];

        // Update cell metadata
        cell.pos = pos;
        cell.seq_id.insert(seq_id);

        // K and V tensors are updated by the attention operation
        // (written to cache.k_l[layer_idx] and cache.v_l[layer_idx])
    }

    // Update cache usage
    cache.used = std::max(cache.used, (uint32_t)(batch.pos[batch.n_tokens - 1] + 1));
}
```

# Advanced Cache Management

```c++
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans   = true;  // the value tensor is transposed
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)
```

Support multiple independent sequences in parallel:

```
// Sequence-aware cache
struct llama_kv_cell {
    llama_pos pos = -1;
    std::set<llama_seq_id> seq_id;  // Multiple sequences can use this cell
};

// Example: 2 independent conversations
// Sequence 0: "Tell me about AI"
// Sequence 1: "What is quantum computing?"

// Cache cells:
// pos=0: seq_id={0}      "Tell"
// pos=1: seq_id={0}      "me"
// pos=2: seq_id={0}      "about"
// pos=3: seq_id={0}      "AI"
// pos=4: seq_id={1}      "What"
// pos=5: seq_id={1}      "is"
// pos=6: seq_id={1}      "quantum"
// pos=7: seq_id={1}      "computing"
```

>  ##  Rolling Buffer Cache


```
// Ring buffer implementation
uint32_t cache_pos = token_pos % cache.size;

// Overwrite oldest entry
cache.cells[cache_pos].pos = token_pos;
cache.cells[cache_pos].seq_id = {seq_id};

// Copy new K/V into cache at cache_pos
// (overwrites old data)
```

> ## Cache Defragmentation
  When cache becomes fragmented:
  
  
```
void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    // Move all active cells to the front
    uint32_t new_head = 0;

    for (uint32_t i = 0; i < cache.size; i++) {
        if (cache.cells[i].pos >= 0) {
            if (i != new_head) {
                // Copy cell
                cache.cells[new_head] = cache.cells[i];

                // Copy K/V tensors (expensive!)
                for (uint32_t il = 0; il < n_layer; il++) {
                    copy_kv_cell(cache, il, i, new_head);
                }
            }
            new_head++;
        }
    }

    cache.used = new_head;
    cache.head = new_head;
}
```