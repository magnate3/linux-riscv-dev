# KV Cache Storage by Layer Example

This document demonstrates through actual code how KV cache is stored by layer in llama.cpp, and how `cpy_k` and `get_k` are used.

## 1. Data Structure Review

### Independent K/V Tensors Per Layer
```cpp
// Location: src/llama-kv-cache.h:206-216
struct kv_layer {
    uint32_t il;  // Layer index in model
    
    ggml_tensor * k;  // K cache: [n_embd_k_gqa, kv_size, n_stream]
    ggml_tensor * v;  // V cache: [n_embd_v_gqa, kv_size, n_stream]
    
    std::vector<ggml_tensor *> k_stream;  // View for each stream
    std::vector<ggml_tensor *> v_stream;
};

std::vector<kv_layer> layers;  // Key: One independent tensor per layer
```

**Key Points:**
- `layers[0].k` stores all K values for layer 0
- `layers[1].k` stores all K values for layer 1
- ...and so on, **separated storage by layer**

## 2. get_k() - Read K Cache for a Specific Layer

### Function Signature and Implementation
```cpp
// Location: src/llama-kv-cache.cpp:1008-1027
ggml_tensor * llama_kv_cache::get_k(
    ggml_context * ctx,
    int32_t il,              // Input: layer index
    uint32_t n_kv,           // Number of cells to read
    const slot_info & sinfo  // Cell mapping information
) const {
    // Step 1: Find corresponding kv_layer through layer index
    const int32_t ikv = map_layer_ids.at(il);
    
    // Step 2: Get K tensor for this layer
    auto * k = layers[ikv].k;  // This is the K cache for layer il
    
    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];
    
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    
    // Step 3: Create a 4D view for attention computation
    // Return shape: [n_embd_head_k, n_head_kv, n_kv, n_stream]
    return ggml_view_4d(ctx, k,
        hparams.n_embd_head_k, hparams.n_head_kv(il), n_kv, ns,
        ggml_row_size(k->type, hparams.n_embd_head_k),      // stride for heads
        ggml_row_size(k->type, n_embd_k_gqa),               // stride for tokens
        ggml_row_size(k->type, n_embd_k_gqa*kv_size),       // stride for streams
        ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}
```

**Key reflection of storage by layer:**
```cpp
auto * k = layers[ikv].k;  // Directly index to specific layer's tensor
```

### Usage in Actual Attention Computation
```cpp
// Location: src/llama-graph.cpp:1676
// When building attention graph for layer il

ggml_tensor * k = mctx_cur->get_k(ctx0, il);  // Get K for layer il
ggml_tensor * v = mctx_cur->get_v(ctx0, il);  // Get V for layer il

// k now contains all K values for layer il, shape: [head_dim, n_heads, n_kv, n_stream]
// Used to compute QK^T
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);  // Q @ K^T
```

## 3. cpy_k() - Write to K Cache for a Specific Layer

### Function Signature and Implementation
```cpp
// Location: src/llama-kv-cache.cpp:1060-1093
ggml_tensor * llama_kv_cache::cpy_k(
    ggml_context * ctx,
    ggml_tensor * k_cur,     // Input: current computed K [n_embd_head_k, n_head_k, n_tokens]
    ggml_tensor * k_idxs,    // Input: cell indices [n_tokens]
    int32_t il,              // Input: layer index
    const slot_info & sinfo
) const {
    // Step 1: Find corresponding kv_layer through layer index
    const int32_t ikv = map_layer_ids.at(il);
    
    // Step 2: Get K tensor for this layer (destination)
    ggml_tensor * k = layers[ikv].k;  // This is the K cache for layer il
    
    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];
    
    const int64_t n_embd_gqa = n_embd_head*n_head;
    
    // Step 3: Reshape k_cur to 2D: [n_embd_gqa, n_tokens]
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    
    const int64_t n_stream = k->ne[2];
    
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        // Merge all streams into one large 2D tensor
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }
    
    // Step 4: Use set_rows to write by index
    // k[k_idxs[i]] = k_cur[i] for each token i
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

**Key reflection of storage by layer:**
```cpp
ggml_tensor * k = layers[ikv].k;  // Write to specific layer's tensor
```

### Usage in Actual Computation
```cpp
// Location: src/llama-graph.cpp:1669
// During forward propagation of layer il

// Compute K values for current batch
ggml_tensor * k_cur = ggml_mul_mat(ctx0, wk, cur);  // [head_dim, n_heads, n_tokens]

// Prepare cell indices
const auto & k_idxs = inp->get_k_idxs();  // [n_tokens], each value is a cell index

// Write k_cur to K cache for layer il
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
```

## 4. Complete Flow Example

Assume processing a 2-layer model, batch contains 3 tokens:

### Initialization
```cpp
// Create tensors for each layer during construction
layers[0].k = ggml_new_tensor_3d(ctx, type_k, 4096, 512, 1);  // Layer 0
layers[1].k = ggml_new_tensor_3d(ctx, type_k, 4096, 512, 1);  // Layer 1
```

### Layer 0 Processing
```cpp
// 1. Compute K values for Layer 0
ggml_tensor * k_cur_l0 = compute_k_layer_0(input);  // [128, 32, 3]
                                                     // [head_dim, n_heads, n_tokens]

// 2. Allocate cells (assume allocated to cells 5, 6, 7)
k_idxs->data = {5, 6, 7};

// 3. Write to Layer 0's K cache
cpy_k(ctx, k_cur_l0, k_idxs, il=0);
// Executes: layers[0].k[..., 5] = k_cur_l0[..., 0]
//           layers[0].k[..., 6] = k_cur_l0[..., 1]
//           layers[0].k[..., 7] = k_cur_l0[..., 2]

// 4. Read Layer 0's K cache (assume already has 10 tokens)
ggml_tensor * k_l0 = get_k(ctx, il=0, n_kv=10);
// Returns: layers[0].k[..., 0:10]  // Read only from Layer 0
```

### Layer 1 Processing
```cpp
// 1. Compute K values for Layer 1
ggml_tensor * k_cur_l1 = compute_k_layer_1(hidden);  // [128, 32, 3]

// 2. Use same cell indices {5, 6, 7}
k_idxs->data = {5, 6, 7};

// 3. Write to Layer 1's K cache (note: different tensor)
cpy_k(ctx, k_cur_l1, k_idxs, il=1);
// Executes: layers[1].k[..., 5] = k_cur_l1[..., 0]  // Write to Layer 1's tensor
//           layers[1].k[..., 6] = k_cur_l1[..., 1]
//           layers[1].k[..., 7] = k_cur_l1[..., 2]

// 4. Read Layer 1's K cache
ggml_tensor * k_l1 = get_k(ctx, il=1, n_kv=10);
// Returns: layers[1].k[..., 0:10]  // Read only from Layer 1
```

## 5. Key Summary

### Cell Management (token dimension)
- **Globally shared**: All layers use same cell allocation
- `v_cells` records which cells are occupied and their token positions
- Cell indices (like 5, 6, 7) have same meaning across all layers

### Data Storage (layer dimension)
- **Separated by layer**: Each layer has independent `layers[il].k` and `layers[il].v` tensors
- Cell 5 stores **different data** in Layer 0 and Layer 1:
  - `layers[0].k[..., 5]` stores Layer 0's K value for a token
  - `layers[1].k[..., 5]` stores Layer 1's K value for the **same token**

### Read/Write Operations
```
Write (cpy_k):
  token → cell index → write in specific layer's tensor
  
Read (get_k):
  specify layer → return view of that layer's tensor → use for that layer's attention

Key: Each cpy_k/get_k specifies layer via il parameter,
     thus operating on layers[il].k, an **independent** tensor
```

## 6. Memory Layout Visualization

```
KV Cache Memory Structure:

layers[0].k:  [embedding_dim, kv_size, n_stream]
              ┌─────────────────────────────────┐
    Cell 0 →  │ Layer 0, K for token at pos=0   │
    Cell 1 →  │ Layer 0, K for token at pos=1   │
    Cell 2 →  │ Layer 0, K for token at pos=2   │
              │            ...                  │
              └─────────────────────────────────┘

layers[1].k:  [embedding_dim, kv_size, n_stream]
              ┌─────────────────────────────────┐
    Cell 0 →  │ Layer 1, K for token at pos=0   │
    Cell 1 →  │ Layer 1, K for token at pos=1   │
    Cell 2 →  │ Layer 1, K for token at pos=2   │
              │            ...                  │
              └─────────────────────────────────┘

v_cells:      Unified management of cell allocation for all layers
              ┌────────┬──────┬────────────┐
    Cell 0 →  │ pos=0  │ used │ seq_id=0   │
    Cell 1 →  │ pos=1  │ used │ seq_id=0   │
    Cell 2 →  │ pos=2  │ used │ seq_id=0   │
              │  ...   │      │            │
              └────────┴──────┴────────────┘
```

**Key design reflections:**
1. Cell management (`v_cells`) is global, across all layers
2. Data storage (`layers[il].k/v`) is layered, each layer independent
3. Select which layer's tensor to operate on via `il` parameter
4. Cell indices are consistent across all layers, but stored data differs
