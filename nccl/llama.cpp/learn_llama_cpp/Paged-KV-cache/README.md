
[Paged KV cache and scheduler:](https://github.com/ggml-org/llama.cpp/discussions/21961)



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