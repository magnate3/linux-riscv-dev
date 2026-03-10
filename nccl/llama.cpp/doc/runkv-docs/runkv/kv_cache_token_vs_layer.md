# KV Cache: Token Metadata Management vs Layer Data Storage

## Question: Why is metadata managed by Token, but storage separated by Layer?

### Short Answer
**Cell management** and **data storage** are two different dimensions:
- **Cell metadata**: Records "which token positions are occupied" (globally unified)
- **Actual data**: Each layer's processing result for the same token differs (separated by layer)

---

## Detailed Explanation

### 1. A Token Has Different Representations in Different Layers

In Transformer models, the same token produces different hidden states across layers:

```python
# Pseudo code example
token = "Hello"

# Layer 0 processing
hidden_0 = layer_0(token_embedding)
K_0, V_0 = compute_kv(hidden_0)  # Layer 0's K/V

# Layer 1 processing
hidden_1 = layer_1(hidden_0)
K_1, V_1 = compute_kv(hidden_1)  # Layer 1's K/V (different from Layer 0!)

# Layer 2 processing
hidden_2 = layer_2(hidden_1)
K_2, V_2 = compute_kv(hidden_2)  # Layer 2's K/V (different again!)
```

**Key Point: The same token "Hello" has different K/V values at each layer**

### 2. Role of Cell Index: Cross-Layer "Address"

```cpp
// Suppose we want to store token "Hello" (position = 5)
// v_cells allocates it to Cell 3

v_cells[stream].pos[3] = 5;      // Cell 3 stores token at position=5
v_cells[stream].seq[3].set(0);   // Belongs to sequence 0

// Now use Cell 3 in every layer, but store different data:
layers[0].k[..., 3] = K_layer0_token5;  // Layer 0's K for token "Hello"
layers[1].k[..., 3] = K_layer1_token5;  // Layer 1's K for token "Hello"
layers[2].k[..., 3] = K_layer2_token5;  // Layer 2's K for token "Hello"
```

### 3. Concrete Example: Processing 3 Tokens

Assume input: "How are you"

```
Token sequence:
  Token 0: "How"  (position = 0)
  Token 1: "are"  (position = 1)
  Token 2: "you"  (position = 2)

Step 1: Cell allocation (metadata management - by Token)
  v_cells.pos[0] = 0   // Cell 0 stores position=0
  v_cells.pos[1] = 1   // Cell 1 stores position=1
  v_cells.pos[2] = 2   // Cell 2 stores position=2
  
  k_idxs = [0, 1, 2]   // Cell indices for 3 tokens

Step 2: Layer 0 forward propagation
  // Compute Layer 0's K/V
  k_cur_l0 = WK_0 @ hidden_0  // shape: [128, 32, 3]
  
  // Write to Layer 0's cache
  cpy_k(k_cur_l0, k_idxs, il=0)
  
  Execution result:
    layers[0].k[..., 0] = k_cur_l0[..., 0]  // "How" at Layer 0's K
    layers[0].k[..., 1] = k_cur_l0[..., 1]  // "are" at Layer 0's K
    layers[0].k[..., 2] = k_cur_l0[..., 2]  // "you" at Layer 0's K

Step 3: Layer 1 forward propagation
  // Compute Layer 1's K/V (based on Layer 0's output)
  k_cur_l1 = WK_1 @ hidden_1  // shape: [128, 32, 3]
  
  // Write to Layer 1's cache (using same cell indices!)
  cpy_k(k_cur_l1, k_idxs, il=1)
  
  Execution result:
    layers[1].k[..., 0] = k_cur_l1[..., 0]  // "How" at Layer 1's K
    layers[1].k[..., 1] = k_cur_l1[..., 1]  // "are" at Layer 1's K
    layers[1].k[..., 2] = k_cur_l1[..., 2]  // "you" at Layer 1's K

Step 4: Layer 2 forward propagation
  // Same as above...
  layers[2].k[..., 0] = k_cur_l2[..., 0]  // "How" at Layer 2's K
  layers[2].k[..., 1] = k_cur_l2[..., 1]  // "are" at Layer 2's K
  layers[2].k[..., 2] = k_cur_l2[..., 2]  // "you" at Layer 2's K
```

### 4. Memory Layout Visualization

```
v_cells (metadata - globally unified):
┌──────┬────────┬─────────┬─────────┐
│ Cell │ pos    │ seq_id  │ used    │
├──────┼────────┼─────────┼─────────┤
│  0   │   0    │  {0}    │  true   │  ← "How"'s address
│  1   │   1    │  {0}    │  true   │  ← "are"'s address
│  2   │   2    │  {0}    │  true   │  ← "you"'s address
│  3   │  -1    │  {}     │  false  │  ← Unused
│  4   │  -1    │  {}     │  false  │  ← Unused
└──────┴────────┴─────────┴─────────┘

layers[0].k (Layer 0 data):
┌──────┬─────────────────────────────┐
│ Cell │ Stored content               │
├──────┼─────────────────────────────┤
│  0   │ [Layer 0's K for "How"]     │
│  1   │ [Layer 0's K for "are"]     │
│  2   │ [Layer 0's K for "you"]     │
│  3   │ [Unused space]               │
│  4   │ [Unused space]               │
└──────┴─────────────────────────────┘

layers[1].k (Layer 1 data - independent memory space):
┌──────┬─────────────────────────────┐
│ Cell │ Stored content               │
├──────┼─────────────────────────────┤
│  0   │ [Layer 1's K for "How"]     │  ← Note: Cell 0 but different content!
│  1   │ [Layer 1's K for "are"]     │
│  2   │ [Layer 1's K for "you"]     │
│  3   │ [Unused space]               │
│  4   │ [Unused space]               │
└──────┴─────────────────────────────┘

layers[2].k (Layer 2 data - another independent space):
┌──────┬─────────────────────────────┐
│ Cell │ Stored content               │
├──────┼─────────────────────────────┤
│  0   │ [Layer 2's K for "How"]     │  ← Still Cell 0, but different again!
│  1   │ [Layer 2's K for "are"]     │
│  2   │ [Layer 2's K for "you"]     │
│  3   │ [Unused space]               │
│  4   │ [Unused space]               │
└──────┴─────────────────────────────┘
```

### 5. Why This Design?

#### Approach A: If Truly Storing by Token
```cpp
// Hypothetical "store by token" approach
struct token_kv_data {
    vector<tensor> k_all_layers;  // K for all layers
    vector<tensor> v_all_layers;  // V for all layers
};

vector<token_kv_data> cache;  // Each cell stores all layer data for one token

// Problems:
// 1. Reading all K for Layer 3 requires traversing all tokens
//    for (int i = 0; i < n_tokens; i++) {
//        k_layer3[i] = cache[i].k_all_layers[3];  // Non-contiguous memory!
//    }
// 2. Cannot utilize ggml's efficient matrix operations
```

#### Approach B: Current "Store by Layer" Approach (Actually Used)
```cpp
// Each layer has independent large tensors
layers[il].k = [n_embd, kv_size, n_stream]  // Contiguous memory

// Advantages:
// 1. Reading all K for Layer 3: directly return layers[3].k
// 2. Memory contiguous, efficient GPU access
// 3. Can be directly used for operations like ggml_mul_mat
```

### 6. Code Implementation

```cpp
// src/llama-graph.cpp:1669-1677
// In each layer's processing:

for (int il = 0; il < n_layers; il++) {
    // 1. Compute current layer's K/V
    ggml_tensor * k_cur = compute_k(il);  // [head_dim, n_heads, n_tokens]
    
    // 2. Write to current layer's cache (using globally unified cell indices)
    cpy_k(k_cur, k_idxs, il);
    //                    ↑ Specify which layer to write to
    
    // 3. Read current layer's cache
    ggml_tensor * k = get_k(il);
    //                      ↑ Specify which layer to read from
    
    // 4. Perform current layer's attention computation
    attn = compute_attention(q, k, v);
}
```

### 7. Analogy: Database Table Design

This is like two approaches in database design:

**Approach A: Wide table (all data together)**
```sql
CREATE TABLE kv_cache (
    token_id INT,
    layer0_k BLOB,
    layer1_k BLOB,
    layer2_k BLOB,
    ...
);
-- Querying all data for Layer 3 requires scanning the entire table
```

**Approach B: Separate tables (by layer)** ← Current implementation
```sql
CREATE TABLE kv_cache_layer0 (
    cell_id INT,
    k_value BLOB
);
CREATE TABLE kv_cache_layer1 (
    cell_id INT,
    k_value BLOB
);
-- Querying all data for Layer 1: directly query one table
```

---

## Summary

### Cell Metadata (by Token)
- **Role**: Records which cells are occupied, corresponding to which token positions
- **Scope**: Globally unified, shared across all layers
- **Analogy**: Room number management system

### Actual Data (by Layer)
- **Role**: Stores each layer's computation result for each token
- **Scope**: Each layer has independent memory space
- **Analogy**: Actual room content on each floor

### Key Understanding
```
Same Cell index (e.g., Cell 5):
  - In v_cells represents: token at position=10
  - In layers[0].k[..., 5] stores: Layer 0's K value for that token
  - In layers[1].k[..., 5] stores: Layer 1's K value for that token
  - In layers[2].k[..., 5] stores: Layer 2's K value for that token

Cell index is a unified "address", but each layer stores different "content"!
```
