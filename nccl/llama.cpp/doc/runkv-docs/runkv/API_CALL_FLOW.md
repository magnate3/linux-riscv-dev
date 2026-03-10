# Complete Inference Request API Call Flow in llama.cpp

This document explains the core functions that need to be called to submit and execute a complete inference request in llama.cpp.

## 1. Initialization Phase (One-time)

### 1. Load Backend
```cpp
ggml_backend_load_all();
```
- **Purpose**: Load all available compute backends (CPU, CUDA, Metal, etc.)
- **When to call**: Once at program startup

### 2. Load Model
```cpp
// Configure model parameters
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 32;  // Number of GPU layers

// Load model
llama_model * model = llama_model_load_from_file(model_path, model_params);
```
- **Purpose**: Load model weights from GGUF file
- **Key parameters**: 
  - `n_gpu_layers`: Number of layers to offload to GPU
  - `vocab_only`: Whether to only load vocabulary

### 3. Create Inference Context
```cpp
// Configure context parameters
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048;      // KV cache size (total cells)
ctx_params.n_batch = 512;     // Logical batch size
ctx_params.n_ubatch = 128;    // Physical ubatch size
ctx_params.n_seq_max = 4;     // Maximum concurrent sequences

// Create context (allocates KV cache)
llama_context * ctx = llama_init_from_model(model, ctx_params);
```
- **Purpose**: Create inference context, allocate KV cache memory
- **Key parameters**:
  - `n_ctx`: Total KV cache cells (affects maximum supported text length)
  - `n_batch`: Maximum tokens processable per batch
  - `n_seq_max`: Maximum concurrent sequences (affects unified vs streaming mode)

### 4. Create Sampler
```cpp
// Create sampler chain
auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);

// Add sampling strategies
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));  // Final sampler
```
- **Purpose**: Configure token sampling strategy
- **Common samplers**:
  - `top_k`: Top-K sampling
  - `top_p`: Top-P (nucleus) sampling
  - `temp`: Temperature scaling
  - `greedy`: Greedy sampling (always select highest probability)
  - `dist`: Sample from distribution

## 2. Single Request Processing (Per Request)

### 5. Tokenize Input Text
```cpp
const llama_vocab * vocab = llama_model_get_vocab(model);

// Method 1: Two-step approach (recommended)
// Step 1: Get token count
int n_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), 
                                NULL, 0, true, true);

// Step 2: Allocate space and tokenize
std::vector<llama_token> tokens(n_tokens);
llama_tokenize(vocab, prompt.c_str(), prompt.size(), 
               tokens.data(), n_tokens, true, true);
```
- **Purpose**: Convert text to token ID sequence
- **Parameter explanation**:
  - `add_special=true`: Add BOS (beginning of sequence) token
  - `parse_special=true`: Parse special tokens (e.g., `<|endoftext|>`)

### 6. Create Batch
```cpp
// Method 1: Use convenience function (single sequence)
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// Method 2: Manual creation (supports multiple sequences)
llama_batch batch = llama_batch_init(n_batch_max, 0, n_seq_max);
```
- **Purpose**: Create batch container
- **Batch structure**:
  - `token[]`: Token ID array
  - `pos[]`: Position of each token
  - `seq_id[][]`: List of sequence IDs each token belongs to
  - `logits[]`: Whether to output logits for this token

### 7. Fill Batch (Multi-sequence/Manual Control)
```cpp
// Use common helper function
for (int i = 0; i < n_tokens; i++) {
    common_batch_add(batch, tokens[i], i, { seq_id }, false);
}
// Last token needs logits output
batch.logits[batch.n_tokens - 1] = true;
```
- **Purpose**: Add tokens to batch
- **Parameter explanation**:
  - `token`: Token ID
  - `pos`: Position (usually incrementing from 0)
  - `seq_ids`: List of sequence IDs this token belongs to (supports one token belonging to multiple sequences)
  - `logits`: Whether to output logits for this token (usually only the last one needs it)

### 8. Execute Inference (Prefill/Decode)
```cpp
int ret = llama_decode(ctx, batch);
if (ret != 0) {
    // Error handling
    // ret=1: KV cache space insufficient
    // ret=2: Aborted
    // ret<0: Fatal error
}
```
- **Purpose**: **Core inference function**, executes transformer forward propagation
- **Internal flow**:
  1. Calls `llama_kv_cache::find_slot()` to find available KV cache cells
  2. Executes computation in ubatches (each ubatch ≤ n_ubatch tokens)
  3. Calls `llama_kv_cache::apply_ubatch()` to update cell metadata
  4. Computes K/V and stores in KV cache
  5. Computes attention and FFN
  6. Outputs logits (for tokens with `logits[i]=true`)
- **Return values**:
  - `0`: Success
  - `1`: KV cache space insufficient
  - `2`: Aborted (via abort callback)
  - `<0`: Fatal error

### 9. Get Logits
```cpp
// Get logits of the last token
float * logits = llama_get_logits_ith(ctx, -1);  // -1 means last token

// Or get logits of the i-th output token
float * logits_i = llama_get_logits_ith(ctx, i);
```
- **Purpose**: Get model output logits (vocabulary probability distribution)
- **Note**: Only tokens with `batch.logits[i]=true` will have output

### 10. Sample Next Token
```cpp
llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
```
- **Purpose**: Select next token based on logits and sampling strategy
- **Parameter explanation**:
  - `ctx`: Context (contains logits)
  - `-1`: Use logits of the last token

### 11. Check End Condition
```cpp
if (llama_vocab_is_eog(vocab, new_token)) {
    // Encountered EOS/EOT, end generation
    break;
}
```
- **Purpose**: Check if end token is encountered

### 12. Convert Token to Text (Optional)
```cpp
char buf[128];
int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
std::string text(buf, n);
printf("%s", text.c_str());
```
- **Purpose**: Convert token ID to text output

### 13. Prepare Next Decode Round
```cpp
// Create new batch (only contains the sampled new token)
batch = llama_batch_get_one(&new_token, 1);

// Continue loop to step 8
```

## 3. Multi-Sequence Management (Concurrent Requests)

### 14. Manage KV Cache Sequences
```cpp
// Get memory object
llama_memory_t mem = llama_get_memory(ctx);

// Remove sequence (release KV cache)
llama_memory_seq_rm(mem, seq_id, -1, -1);

// Copy sequence (for beam search)
llama_memory_seq_cp(mem, src_seq_id, dst_seq_id, -1, -1);

// Keep a range of sequence
llama_memory_seq_keep(mem, seq_id);

// Query position range of sequence
llama_pos min_pos = llama_memory_seq_pos_min(mem, seq_id);
llama_pos max_pos = llama_memory_seq_pos_max(mem, seq_id);
```

## 4. Cleanup Phase

### 15. Release Resources
```cpp
// Release in reverse order
llama_sampler_free(smpl);    // Release sampler
llama_batch_free(batch);     // Release batch
llama_free(ctx);             // Release context (release KV cache)
llama_model_free(model);     // Release model
```

## Complete Call Flow Diagram

```
Initialization Phase:
  ggml_backend_load_all()
       ↓
  llama_model_load_from_file()
       ↓
  llama_init_from_model()  ← Allocate KV cache
       ↓
  llama_sampler_chain_init() + llama_sampler_chain_add()

Request Processing Loop:
  llama_tokenize() ← Text → tokens
       ↓
  llama_batch_get_one() / common_batch_add()
       ↓
  ┌─→ llama_decode() ← ★★★ Core Inference ★★★
  │    ├─ find_slot()      (Find KV cache space)
  │    ├─ apply_ubatch()   (Update cell metadata)
  │    └─ Compute transformer (Write K/V to cache)
  │        ↓
  │   llama_get_logits_ith()
  │        ↓
  │   llama_sampler_sample()
  │        ↓
  │   llama_vocab_is_eog() ─→ End? → Exit
  │        ↓ No
  │   llama_token_to_piece() → Output text
  │        ↓
  │   llama_batch_get_one() (Prepare next token)
  └────┘

Multi-Sequence Management (Optional):
  llama_get_memory()
       ↓
  llama_memory_seq_rm() / _cp() / _keep()

Cleanup Phase:
  llama_sampler_free()
       ↓
  llama_batch_free()
       ↓
  llama_free()
       ↓
  llama_model_free()
```

## Key Function Summary

| Function | Purpose | Call Frequency | Importance |
|----------|---------|----------------|-----------|
| `llama_decode()` | Execute inference | Per token | ★★★ Core |
| `llama_get_logits_ith()` | Get output | Per token | ★★★ Core |
| `llama_sampler_sample()` | Sample token | Per token | ★★★ Core |
| `llama_tokenize()` | Text to tokens | Per request start | ★★ Important |
| `llama_batch_get_one()` | Create batch | Per token | ★★ Important |
| `common_batch_add()` | Fill batch | Multi-seq scenario | ★ Helper |
| `llama_memory_seq_rm()` | Cleanup KV cache | Request end | ★ Management |
| `llama_token_to_piece()` | Token to text | During output | - Optional |

## Important Notes

1. **`llama_decode()` is the only function that executes inference**
   - All other functions prepare data or process results
   - KV cache allocation and updates are completed inside this function

2. **Batch `logits[]` flag is important**
   - Only tokens set to `true` will output logits
   - Prefill phase usually only needs the last token's logits
   - Decode phase has one token per iteration, need to set its logits=true

3. **Position must be correct**
   - Prefill: Increment from 0
   - Decode: Use previous token's position + 1

4. **Multi-sequence scenarios require manual seq_id management**
   - Using `common_batch_add()` is more convenient
   - Different requests use different seq_id
   - Call `llama_memory_seq_rm()` to release space after request ends

5. **Error handling**
   - `llama_decode()` returning 1 indicates insufficient KV cache
   - Need to cleanup old sequences or increase `n_ctx`
