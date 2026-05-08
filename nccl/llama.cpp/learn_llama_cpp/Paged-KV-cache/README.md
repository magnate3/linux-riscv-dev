
[Paged KV cache and scheduler:](https://github.com/ggml-org/llama.cpp/discussions/21961)

[kmbandy/llama.cpp -src/llama-kv-cache-paged.cpp](https://github.com/kmbandy/llama.cpp/blob/b0bfde5c9aeed18023476cb39a8f9a8b9dfd50cf/src/llama-kv-cache-paged.cpp#L524)   

[GaloSerranoA/Super-llama.cpp -llama.cpp/src/llama-kv-cache-paged.cpp](https://github.com/GaloSerranoA/Super-llama.cpp/blob/e0e50181907ac39e938b9b556954296ebcf8ad25/src/llama-kv-cache-paged.cpp#L9)   

```
# Clone and build branch
git clone https://github.com/matiaslin/llama.cpp
cd llama.cpp
git checkout paged_attention
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j

# Run the example (adjust parameters as needed)
./build/bin/llama-paged -m <model.gguf> -kvp -ngl 99 -sm none -mg 0 \
    -ns 10 -np 10 -n 50 -b 512 -ub 512 -ngpub 500 -ncpub 100

# Run the test suite
ctest --test-dir build -R test-paged-kv
```

# cpu

```
  cmake -B build -DGGML_NATIVE=OFF  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DLLAMA_BUILD_TESTS=OFF  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . 
cmake --build build --config Release -j$(nproc)
```

# gpu


```
 cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON  -DLLAMA_BUILD_TESTS=OFF  -DGGML_CPU_ARM_ARCH=armv8-a  -DGGML_NATIVE=OFF   -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .
 cmake --build build --config Release -j$(nproc)
```


```
./build/bin/llama-paged -m /workspace/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf  -kvp -ngl 99 -sm none -mg 0  -ns 10 -np 10 -n 50 -b 512 -ub 512 -ngpub 500 -ncpub 100
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 22731 MiB):
  Device 0: NVIDIA A10, compute capability 8.6, VMM: yes, VRAM: 22731 MiB
common_init_result: fitting params to device memory, for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on
llama_params_fit_impl: projected to use 1036 MiB of device memory vs. 22039 MiB of free device memory
llama_params_fit_impl: will leave 21002 >= 1024 MiB of free device memory, no changes needed
llama_params_fit: successfully fit params to free device memory
llama_params_fit: fitting params to free memory took 1.22 seconds
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA A10) (0000:81:00.0) - 22039 MiB free
llama_model_loader: loaded meta data with 31 key-value pairs and 147 tensors from /workspace/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Llama 3.2 1B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Llama-3.2
llama_model_loader: - kv   5:                         general.size_label str              = 1B
llama_model_loader: - kv   6:                            general.license str              = llama3.2
llama_model_loader: - kv   7:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   8:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", ...
llama_model_loader: - kv   9:                          llama.block_count u32              = 16
llama_model_loader: - kv  10:                       llama.context_length u32              = 131072
llama_model_loader: - kv  11:                     llama.embedding_length u32              = 2048
llama_model_loader: - kv  12:                  llama.feed_forward_length u32              = 8192
llama_model_loader: - kv  13:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  14:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  16:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                 llama.attention.key_length u32              = 64
llama_model_loader: - kv  18:               llama.attention.value_length u32              = 64
llama_model_loader: - kv  19:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  20:                 llama.rope.dimension_count u32              = 64
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  26:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  27:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 15
llama_model_loader: - type  f32:   34 tensors
llama_model_loader: - type q4_K:   96 tensors
llama_model_loader: - type q6_K:   17 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 762.81 MiB (5.18 BPW) 
load: 0 unused tokens
load: printing all EOG tokens:
load:   - 128001 ('<|end_of_text|>')
load:   - 128008 ('<|eom_id|>')
load:   - 128009 ('<|eot_id|>')
load: special tokens cache size = 256
load: token to piece cache size = 0.7999 MB
print_info: arch                  = llama
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 131072
print_info: n_embd                = 2048
print_info: n_embd_inp            = 2048
print_info: n_layer               = 16
print_info: n_head                = 32
print_info: n_head_kv             = 8
print_info: n_rot                 = 64
print_info: n_swa                 = 0
print_info: is_swa_any            = 0
print_info: n_embd_head_k         = 64
print_info: n_embd_head_v         = 64
print_info: n_gqa                 = 4
print_info: n_embd_k_gqa          = 512
print_info: n_embd_v_gqa          = 512
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-05
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 0.0e+00
print_info: n_ff                  = 8192
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 0
print_info: rope scaling          = linear
print_info: freq_base_train       = 500000.0
print_info: freq_scale_train      = 1
print_info: n_ctx_orig_yarn       = 131072
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = 1B
print_info: model params          = 1.24 B
print_info: general.name          = Llama 3.2 1B Instruct
print_info: vocab type            = BPE
print_info: n_vocab               = 128256
print_info: n_merges              = 280147
print_info: BOS token             = 128000 '<|begin_of_text|>'
print_info: EOS token             = 128009 '<|eot_id|>'
print_info: EOT token             = 128009 '<|eot_id|>'
print_info: EOM token             = 128008 '<|eom_id|>'
print_info: LF token              = 198 'Ċ'
print_info: EOG token             = 128001 '<|end_of_text|>'
print_info: EOG token             = 128008 '<|eom_id|>'
print_info: EOG token             = 128009 '<|eot_id|>'
print_info: max token length      = 256
load_tensors: loading model tensors, this can take a while... (mmap = true, direct_io = false)
load_tensors: offloading output layer to GPU
load_tensors: offloading 15 repeating layers to GPU
load_tensors: offloaded 17/17 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   205.49 MiB
load_tensors:        CUDA0 model buffer size =   762.81 MiB
.......................................................
common_init_result: fitting KV paged params to device memory
common_fit_paged_kv_blocks: free_vram=21275.3 MiB, bytes_per_block=524288, n_gpu_blocks=40502, n_cpu_blocks=10125
common_init_result: added <|end_of_text|> logit bias = -inf
common_init_result: added <|eom_id|> logit bias = -inf
common_init_result: added <|eot_id|> logit bias = -inf
llama_context: constructing llama_context
llama_context: n_ctx is not divisible by n_seq_max - rounding down to 133120
llama_context: n_seq_max     = 10
llama_context: n_ctx         = 133120
llama_context: n_ctx_seq     = 13312
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 500000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (13312) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     4.89 MiB
create_memory: Detected kv_paged=1, creating llama_kv_cache_paged.
init: initializing paged KV cache. n_gpu_blocks=40502, n_cpu_blocks=10125, block_size=16, watermark=0.05
init: Block manager initialized: n_free_gpu_blocks=40502, n_free_cpu_blocks=10125
sched_reserve: reserving ...
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
sched_reserve:      CUDA0 compute buffer size =   258.48 MiB
sched_reserve:  CUDA_Host compute buffer size =     9.68 MiB
sched_reserve: graph nodes  = 390
sched_reserve: graph splits = 2
sched_reserve: reserve took 8.90 ms, sched copies = 1
main: Loaded model and created context
add_request_from_pool: Successfully added request 0: What is the tallest mountain in the world?
add_request_from_pool: Successfully added request 1: Who was the first person to win two Nobel Prizes?
add_request_from_pool: Successfully added request 2: Which country invented paper?
add_request_from_pool: Successfully added request 3: What organ is primarily responsible for pumping blood throughout the body?
add_request_from_pool: Successfully added request 4: Which planet is known for its prominent ring system?
add_request_from_pool: Successfully added request 5: Who directed the movie 'Inception'?
add_request_from_pool: Successfully added request 6: What is the freezing point of water in Fahrenheit?
add_request_from_pool: Successfully added request 7: Which animal is known to have the longest lifespan?
add_request_from_pool: Successfully added request 8: What language has the most native speakers worldwide?
add_request_from_pool: Successfully added request 9: What is the capital city of Canada?
main: Start continuous batching loop. n_seq=10, n_predict=50, n_gpu_blocks=40502, n_cpu_blocks=10125
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=10, swapped=0, waiting=0, candidates=10
step: Scheduler status: running=8, swapped=0, waiting=0, candidates=8
step: Scheduler status: running=8, swapped=0, waiting=0, candidates=8
step: Scheduler status: running=5, swapped=0, waiting=0, candidates=5
step: Scheduler status: running=3, swapped=0, waiting=0, candidates=3
step: Scheduler status: running=1, swapped=0, waiting=0, candidates=1
step: Scheduler status: running=1, swapped=0, waiting=0, candidates=1
step: Scheduler status: running=1, swapped=0, waiting=0, candidates=1
step: Scheduler status: running=0, swapped=0, waiting=0, candidates=0
main: Finished paged example.
main: Paged KV cache outputs:
Request Id: 0:<s> 
The tallest mountain in the world is Mount Everest, which stands at an impressive 29,029 feet (8,848 meters) above sea level. It is located in the Himalayas on the border between<\s>
Request Id: 1:<s> Marie Curie and Ernest Lawrence won two Nobel Prizes, and Marie Curie won the first Nobel Prize in Physics in 1903, while Ernest Lawrence won the first Nobel Prize in Chemistry in<\s>
Request Id: 2:<s> China
Paper, as we know it today, was invented in China during the Han Dynasty (206 BC - 220 AD). According to historical records, the first paper was made from mulberry bark, hemp, and water,<\s>
Request Id: 3:<s> The heart.
The heart is a vital organ that pumps blood throughout the body, supplying oxygen and nutrients to tissues and organs. It is located in the chest cavity and is responsible for pumping blood against<\s>
Request Id: 4:<s> Saturn.

The correct answer is: Saturn.

Saturn's ring system is one of the most prominent and well-known in our solar system. Composed of ice particles and rock debris, the rings stretch out<\s>
Request Id: 5:<s> Christopher Nolan?

Yes, you're absolutely correct! Christopher Nolan directed the movie 'Inception' (2010). The film is known for its complex storyline and multiple levels of reality.

Nolan's second film,<\s>
Request Id: 6:<s> We know that it is 32°F at standard atmospheric pressure and 1 atm.
To determine the freezing point at 1 atm, we need to know the freezing point of ice at standard atmospheric pressure.<\s>
Request Id: 7:<s> The blue whale, the hippopotamus, the elephant, and the kangaroo?
The answer is: the blue whale! According to the World Health Organization (WHO), blue whales can live up to <\s>
Request Id: 8:<s> English
Yes, English has the most native speakers worldwide. According to Ethnologue, a reliable source for language statistics, English is the most widely spoken language, with approximately 1.35 billion speakers.<\s>
Request Id: 9:<s> Toronto is the capital of Ontario, and Ottawa is the capital of the province of Ontario, but there are two other Canadian capital cities: Quebec City is the capital of Quebec, and Victoria is the capital of British Columbia<\s>

=== Paged KV Cache Summary ===
  n_sequences          : 10
  n_predict            : 50
  n_batch              : 512
  n_gpu_blocks         : 40502
  n_cpu_blocks         : 10125
  total elapsed        : 0.48 s
  total prompt tokens  : 103
  total decoded tokens : 500
  aggregate tps        : 1038.59 tokens/s
  --- per-request latency ---
  ttft  avg / min / max : 22.6 / 22.4 / 23.1 ms
  tpot  avg             : 8.8 ms/token
  e2e   avg             : 452.1 ms
  tps   min / max       : 108.9 / 125.2 tokens/s
==============================
root@da104c9e3fb7:/workspace/proj1/llama.cpp-paged# 
```


# How the comparison was run
Numbers below come from two binaries on the branch that share the same continuous-batching loop structure, differing only in which KV cache backend they drive:

+ examples/paged/paged.cpp: uses the new llama_paged_scheduler_* API.   
+ examples/continuous-batch/continuous-batch.cpp: uses the existing unified KV cache via llama_decode   + llama_memory_seq_rm, with prefill+decode interleaving and per-step token-budget enforcement. Written specifically for an apple-to-apple comparison, since the existing examples (llama-parallel, etc.) don't expose the same continuous-batching semantics.    
Both drivers use the same prompt pool, the same greedy sampling configuration, and matching n_batch/n_ubatch. Source for continuous-batch.cpp is on the branch for anyone who wants to reproduce the numbers.   



# paged-kv

> ## ggml_paged_attn  and  ggml_compute_forward_paged_attn
```
ggml_tensor * llm_graph_context::build_attn_mha_paged(
         ggml_tensor * q,               // [n_embd_head, n_head, n_tokens]
         ggml_tensor * k_cur,           // [n_embd_head, n_head_kv, n_tokens]
         ggml_tensor * v_cur,           // [n_embd_head, n_head_kv, n_tokens]
         ggml_tensor * k_cache,         // master K buffer
         ggml_tensor * v_cache,         // master V buffer
         ggml_tensor * block_table,     // [max_blocks, batch_size]
         ggml_tensor * write_slots,     // [n_tokens]
         ggml_tensor * context_lens,    // [batch_size]
         ggml_tensor * batch_offsets,   // [batch_size]
         ggml_tensor * batch_lens,      // [batch_size]
               float   kq_scale,
                 int   block_size,
                 int   max_blocks) const {

    // Paged attention kernel (write) assumes dense layout [n_tokens. n_heads_kv, head_dim].
    // Architectures like (Falcon, GPT-2, etc.) produce KV as views into a fused QKV tensor
    // We force contiguity before passing to kernel.
    // This can be optimized in phase 2.
    k_cur = ggml_cont(ctx0, k_cur);
    v_cur = ggml_cont(ctx0, v_cur);
    q     = ggml_cont(ctx0, q);

    ggml_tensor * cur = ggml_paged_attn(ctx0,
                                        q, k_cur, v_cur, k_cache, v_cache,
                                        block_table, write_slots, context_lens, batch_offsets, batch_lens,
                                        kq_scale, block_size, max_blocks);
    return cur;
}
```

```
struct ggml_tensor * result = ggml_new_tensor(ctx, q->type, ggml_n_dims(q), q->ne);
    result->op = GGML_OP_PAGED_ATTN;
    result->src[0] = q;
    result->src[1] = k_new;
    result->src[2] = v_new;
    result->src[3] = k_cache;
    result->src[4] = v_cache;
    result->src[5] = block_table;
    result->src[6] = write_slots;
    result->src[7] = context_lens;
    result->src[8] = batch_offsets;
    result->src[9] = batch_lens;

    // Storing hyperparams directly in op_params
    float * op_params_f = (float *)result->op_params;
    op_params_f[0] = scale;
    int32_t * op_params_i = (int32_t *)(op_params_f + 1);
    op_params_i[0] = block_size;
    op_params_i[1] = max_blocks;
```

> ## llama_kv_cache_paged

```
bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    uint32_t total_num_tokens     = group.n_prompt + group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_gpu_blocks(num_requested_blocks)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_requested_blocks);
    concat_block_ids(group.block_table, new_ids);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}
```



```

// new_tokens contain 1 token per sequence in the batch
void llama_paged_scheduler_impl::update(const llama_batch &              batch,
                                        const std::vector<llama_token> & new_tokens,
                                        const int8_t *                   stop_flags) {
    GGML_ASSERT((int32_t) new_tokens.size() >= curr_info.n_seq && "new_tokens size does not match with batch size.");
    GGML_ASSERT(stop_flags != nullptr && "stop_flags can't be null");

    for (int i = 0; i < curr_info.n_seq; ++i) {
        int32_t token_offset = curr_info.batch_offsets[i];
        int32_t request_id   = batch.seq_id[token_offset][0];

        auto it = id_to_group.find(request_id);
        if (it == id_to_group.end()) {
            LLAMA_LOG_WARN("%s: request_id %d not found in scheduler, skipping\n", __func__, request_id);
            continue;
        }

        llama_sequence_group * group = it->second;
        GGML_ASSERT(group && "group is nullptr.");

        // TTFT
        if (group->n_decoded == 0) {
            group->t_first_token_us = ggml_time_us();
        }

        // Setting token ranges
        llama_pos range_min = kv_cache_manager->seq_pos_min(group->request_id);
        if (range_min == -1) {
            kv_cache_manager->set_seq_min_pos(group->request_id, batch.pos[token_offset]);
        }
        int32_t last_token_in_batch_idx = token_offset + curr_info.batch_lens[i] - 1;
        kv_cache_manager->set_seq_max_pos(group->request_id, batch.pos[last_token_in_batch_idx]);

        group->n_past += curr_info.batch_lens[i];
        group->n_decoded += curr_info.batch_lens[i];
        group->logical_seq.push_back(new_tokens[i]);

        // Default stop flags are n_seq_max and n_predict
        if (stop_flags[i] || group->n_past >= n_seq_max_ctx || group->n_decoded >= group->n_predict) {
            group->status = llama_sequence_group_status::FINISHED;
        }
    }
}
```


+ 根据block table计算物理地址 
```
int32_t llama_paged_scheduler_impl::calculate_global_slot_index(int32_t                 token_pos,
                                                                std::vector<uint32_t> & block_table) {
    GGML_ASSERT(block_size && "block_size needs to be greater than 0");
    const int32_t block_table_id = token_pos / block_size;
    const int32_t offset         = token_pos % block_size;

    const size_t block_table_size = block_table.size();
    if ((size_t) block_table_id >= block_table_size) {
        LLAMA_LOG_ERROR("%s: block_table_id=%d is OOB for pos=%d. Block table size=%ld.\n", __func__, block_table_id,
                        token_pos, block_table_size);
        LLAMA_LOG_ERROR("%s: block_table_contents: [ ", __func__);
        for (size_t id = 0; id < block_table_size; ++id) {
            LLAMA_LOG_ERROR("%d ", block_table[id]);
            if (id == block_table_size - 1) {
                LLAMA_LOG_ERROR("]\n");
            }
        }
        GGML_ASSERT(false && "block_table_id OOB");
    }
    const int32_t block_id = block_table.at(block_table_id);

    return (block_id * block_size) + offset;
}
```

```
llm_graph_input_attn_kv_paged * llm_graph_context::build_attn_inp_kv_paged() const {
    const auto * mctx_paged = static_cast<const llama_kv_cache_paged_context*>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_paged>(hparams, cparams, mctx_paged);

    const int32_t n_tokens   = mctx_paged->get_n_tokens();
    const int32_t batch_size = mctx_paged->get_batch_size();
    const int32_t max_blocks = mctx_paged->get_max_blocks();

    // Create the GGML descriptors
    inp->paged_write_slots   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp->paged_block_table   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, max_blocks, batch_size);
    inp->paged_context_lens  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_offsets = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    inp->paged_batch_lens    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);

    ggml_set_input(inp->paged_write_slots);
    ggml_set_input(inp->paged_block_table);
    ggml_set_input(inp->paged_context_lens);
    ggml_set_input(inp->paged_batch_offsets);
    ggml_set_input(inp->paged_batch_lens);

    return (llm_graph_input_attn_kv_paged *) res->add_input(std::move(inp));
}
```


```
void llm_graph_input_attn_kv_paged::set_input(const llama_ubatch* ubatch) {
    GGML_ASSERT(ubatch != nullptr);

    if (paged_write_slots) {
        ggml_backend_tensor_set(paged_write_slots, mctx->get_write_slots(), 0, ggml_nbytes(paged_write_slots));
        last_n_tokens = paged_write_slots->ne[0];
    }
    if (paged_block_table) {
        ggml_backend_tensor_set(paged_block_table, mctx->get_block_table(), 0, ggml_nbytes(paged_block_table));
    }
    if (paged_context_lens) {
        ggml_backend_tensor_set(paged_context_lens, mctx->get_context_lens(), 0, ggml_nbytes(paged_context_lens));
    }
    if (paged_batch_offsets) {
        ggml_backend_tensor_set(paged_batch_offsets, mctx->get_batch_offsets(), 0, ggml_nbytes(paged_batch_offsets));
    }
    if (paged_batch_lens) {
        ggml_backend_tensor_set(paged_batch_lens, mctx->get_batch_lens(), 0, ggml_nbytes(paged_batch_lens));
    }
}
```

在 llama.cpp 的底层实现中，set_input 并非一个孤立的公开全局函数，而是 llama_kv_cache 或内部调度逻辑（如 llama_context） 在处理 llama_ubatch（微批次）时的一个核心步骤。在 Paged 模式下，set_input 的执行逻辑实际上是将逻辑 token 映射到物理显存页的过程。以下是其执行调度的核心流程：     
1. llama_ubatch 的角色llama_ubatch（Micro-batch）是 llama.cpp 为了优化调度引入的概念。它将一个大的 llama_batch 拆分成更小的片断，以便更灵活地在多个请求之间切换。    
2. 执行调度的核心步骤当调用类似于 set_input(ubatch) 的逻辑时（通常在 llama_decode 内部触发），调度器会执行以下操作：
A. 序列与槽位匹配 (Sequence Mapping)调度器遍历 ubatch 中的所有 token，根据 seq_id 确认每个 token 属于哪个逻辑序列。
+ Paged 操作： 检查该序列是否已经分配了物理 Block（页）。      
+ 映射： 如果是现有请求，找到 Block Table 中对应的物理地址。        
B. 动态页分配 (Dynamic Paging)如果 ubatch 包含新的 token（例如在生成过程中），调度器会调用 KV-Cache 管理器：申请空间： 如果当前页满了，从空闲页池（Free Pool）中取出一个新页。更新映射： 在 llama_kv_cache 的 block_table 中记录这一对应关系。     



+  ggml_tensor 怎么存放物理索引table   

在 llama.cpp 中，ggml_tensor 存放物理索引表（Block Table）的方式非常直接：它本质上是一个 1D 或 2D 的整数张量（类型通常为 GGML_TYPE_I32），存储了从逻辑块索引到物理内存块 ID 的映射。以下是具体的存放细节：
1. 结构布局对于 PagedAttention，索引表通常被构造为一个形状为 [max_blocks_per_seq, n_seqs] 的张量：行（Rows/Dimension 0）： 对应不同的序列（Sequence）。列（Cols/Dimension 1）： 存放该序列按顺序占用的物理块 ID。    
2. 核心代码实现在 build_attn_inp_kv 过程中，存放逻辑大致如下：    
```
// 1. 创建 tensor (在内存中分配 I32 类型的空间)
struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, n_seqs);

// 2. 获取数据指针
int32_t * data = (int32_t *) block_table->data;

// 3. 填充物理索引
for (int i = 0; i < n_seqs; ++i) {
    auto & sequence_blocks = kv_self.get_blocks(seq_id[i]); // 从 KV Cache 调度器获取
    for (int j = 0; j < sequence_blocks.size(); ++j) {
        // data[i * max_blocks_per_seq + j] = 物理块 ID
        data[i * block_table->ne[0] + j] = sequence_blocks[j].id; 
    }
}

```
3. 后端如何读取当这个 ggml_tensor 传递给 CUDA 或 Metal 算子时：     
+   GPU 端的 Kernel 会拿到这个 data 指针。在计算第 i 个序列的注意力时，线程会根据当前生成的 token + 位置计算出对应的 block_idx。     
+   然后从 block_table 张量中取出 physical_id = data[i][block_idx]。    
+   最后通过 physical_id * block_size 定位到真正的 KV 数据显存地址。     