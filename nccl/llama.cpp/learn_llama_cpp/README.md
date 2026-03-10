 #   run

```
cmake -B build
cmake --build build
```
or
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```



# proj1

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
```
 cmake --build build
```



#  fundamentals-llama.cpp
```
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# make simple-prompt-multi
g++ src/simple-prompt-multi.cpp -o simple-prompt-multi -std=c++17 -g -Wall -I/workspace/llama.cpp/include -I/workspace/llama.cpp/ggml/include -I/workspace/llama.cpp/common -I/workspace/llama.cpp/src -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -Wl,-rpath,/bin


```


```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# ./simple-prompt-multi 
```


# proj2

```
 export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
```


```
cmake .. -DLLAMA_CPP_DIR=/workspace/llama.cpp
cmake -S . -B build -DLLAMA_CPP_DIR=/workspace/llama.cpp
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# cmake --build build
```

```
root@centos7:/workspace/Let_us_learn_llama_cpp/code-examples/cpp# ./build/01-simple-inference  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.ggu
```

> ##  llama_test_batch_decode
[llama_test_batch_decode](https://github.com/ggml-org/llama.cpp/discussions/17680)   
```
./build/04-llama_test_batch_decode  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
```
> ## passkey


```
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_USE_HTTPLIB -I/workspace/llama.cpp/common/. -I/workspace/llama.cpp/common/../vendor -I/workspace/llama.cpp/src/../include -I/workspace/llama.cpp/ggml/src/../include -O3 -DNDEBUG -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -o  passkey.o  -c 05-passkey.cpp
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp#  c++ -O3 -DNDEBUG passkey.o -o llama-passkey  /workspace/llama.cpp/build/common/libcommon.a /workspace/llama.cpp/build/bin/libllama.so.0.0.7941 /workspace/llama.cpp/build/bin/libggml.so.0.9.5  /workspace/llama.cpp/build/bin/libggml-cpu.so.0.9.5  /workspace/llama.cpp/build/bin/libggml-base.so.0.9.5  /workspace/llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a /usr/lib/aarch64-linux-gnu/libssl.so /usr/lib/aarch64-linux-gnu/libcrypto.so
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# 
```

```
 ./build/05-passkey  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
```

> ##   fill the KV cache
```
// fill the KV cache
    for (int i = 0; i < n_ctx; i += n_batch) {
        if (i > 0 && n_grp > 1) {
            // if SelfExtend is enabled, we compress the position from the last batch by a factor of n_grp
            const int ib = i/n_batch - 1;
            const int bd = n_batch_grp*(n_grp - 1);

            llama_memory_seq_add(mem, 0, n_past - n_batch,         n_past,         ib*bd);
            llama_memory_seq_div(mem, 0, n_past - n_batch + ib*bd, n_past + ib*bd, n_grp);

            n_past = llama_memory_seq_pos_max(mem, 0) + 1;
        }

        common_batch_clear(batch);

        for (int j = 0; j < n_batch && i + j < n_tokens_all; j++) {
            common_batch_add(batch, tokens_list[i + j], n_past++, { 0 }, false);
        }

        if (i + n_batch >= n_tokens_all) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_INF("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        LOG_INF("%s: processed: [%6d, %6d)\n", __func__, i, std::min(i + n_batch, n_tokens_all));

        if (i + n_batch >= n_tokens_all) {
            break;
        }
    }
```

#  Principle:Ggml org Llama cpp Context Window Management

[Principle:Ggml org Llama cpp Context Window Management](https://leeroopedia.com/index.php/Principle:Ggml_org_Llama_cpp_Context_Window_Management) 


#  llama-simple-chat



```
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# /workspace/llama.cpp/build/bin/llama-simple-chat -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf -c  64
..........................................................
> where are you from
<think>
Okay, the user is asking "where are you from?" I need to respond to this question. First, I should acknowledge their question and explain that I'm a language model. It's important to clarify that I don't have a physical presence or location, but I can help with various tasks like answering questions, providing information, or assisting with other interactions. I should make sure my response is friendly and helpful, and explain that I don't have a physical location. Also, I should keep the tone conversational and not too formal.
</think>

I'm a language model, and I don't have a physical location. However, I can help you with questions, provide information, or assist with other tasks! Let me know how I can help!
> what can you do
<think>
Okay, the user just asked, "what can you do?" I need to respond appropriately. Let me start by confirming that I can assist with various tasks. I should mention my capabilities, like answering questions, providing information, or helping with other interactions. It's important to stay friendly and open to further questions. I should keep the tone positive and helpful. Maybe add something about being available
context size exceeded
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# 
``` 

`context size exceeded` 


```
where are you from
what can you do
where is japan
Who is your parent?
what do you like?
```


#  parallel 


```
./build/09-parallel   -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf 
build: 7941 (11fb327bf) with GNU 11.4.0 for Linux aarch64
common_init_result: fitting params to device memory, for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on
llama_params_fit_impl: no devices with dedicated memory found
```


#  10-avllm_cli
```
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# ./build/10-avllm_cli   -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
```

```
system >where are you from
<think>
Okay, the user asked where I am from. I need to answer that. First, I should mention my origin, which is a bit tricky. I'm from the United States, right? But maybe I should add more details to be specific. Let me make sure I'm honest and not making it too simple. I can say I'm from the United States. If they want more info, I can explain further. Keep the response friendly and straightforward.
</think>

I am from the United States. Let me know if you'd like more details about me!
```

# server

+ json   
```
不需要 apt-get install nlohmann-json3-dev,采用vendor/nlohmann/ 
```
    

+ 准备loading.html.hpp   
```
cp ./build/tools/server/loading.html.hpp server/
```
+ make   
```
root@centos7:/workspace/llama.cpp/server# cmake -S . -B  build
root@centos7:/workspace/llama.cpp/server# cmake --build build
Consolidate compiler generated dependencies of target server-context
[ 45%] Built target server-context
Consolidate compiler generated dependencies of target llama-server
[100%] Built target llama-server
root@centos7:/workspace/llama.cpp/server# 
```
+ run   
```
root@centos7:/workspace/llama.cpp/server# ./build/llama-server  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf -c 2048
main: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
```

#  llama_decode

```
(gdb) bt
#0  0x0000ffffbe5a3d24 in llama_kv_cache::find_slot(llama_ubatch const&, bool) const@plt () from /workspace/llama.cpp/build/bin/libllama.so.0
#1  0x0000ffffbe616930 in llama_kv_cache::prepare(std::vector<llama_ubatch, std::allocator<llama_ubatch> > const&) () from /workspace/llama.cpp/build/bin/libllama.so.0
#2  0x0000ffffbe61827c in llama_kv_cache::init_batch(llama_batch_allocr&, unsigned int, bool) () from /workspace/llama.cpp/build/bin/libllama.so.0
#3  0x0000ffffbe5d81ec in llama_context::decode(llama_batch const&) () from /workspace/llama.cpp/build/bin/libllama.so.0
#4  0x0000ffffbe5d9720 in llama_decode () from /workspace/llama.cpp/build/bin/libllama.so.0
#5  0x0000aaaaaaaa30bc in main ()
(gdb) 
```
build_attn --> llama_kv_cache_context::get_k    
```
(gdb) bt
#0  0x0000ffffbe5a0c24 in llama_kv_cache_context::get_k(ggml_context*, int) const@plt () from /workspace/llama.cpp/build/bin/libllama.so.0
#1  0x0000ffffbe6048d0 in llm_graph_context::build_attn(llm_graph_input_attn_kv*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*, float, int) const () from /workspace/llama.cpp/build/bin/libllama.so.0
#2  0x0000ffffbe704588 in llm_build_qwen3::llm_build_qwen3(llama_model const&, llm_graph_params const&) () from /workspace/llama.cpp/build/bin/libllama.so.0
#3  0x0000ffffbe64bf14 in llama_model::build_graph(llm_graph_params const&) const () from /workspace/llama.cpp/build/bin/libllama.so.0
#4  0x0000ffffbe5d0070 in llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) () from /workspace/llama.cpp/build/bin/libllama.so.0
#5  0x0000ffffbe5d82cc in llama_context::decode(llama_batch const&) () from /workspace/llama.cpp/build/bin/libllama.so.0
#6  0x0000ffffbe5d9720 in llama_decode () from /workspace/llama.cpp/build/bin/libllama.so.0
#7  0x0000aaaaaaab3a8c in main ()
```

## UBatch拆分算法
llama.cpp提供了三种UBatch拆分策略：     
1. 简单拆分（Simple Split）
```
llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch) {
    std::vector<int32_t> idxs;
    uint32_t cur_idx = 0;
    
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }
    
    while (idxs.size() < n_ubatch && cur_idx < used.size()) {
        idxs.push_back(cur_idx);
        used[cur_idx] = true;
        ++n_used;
        ++cur_idx;
    }
    
    return ubatch_add(idxs, idxs.size(), false);
}
```

2. 均衡拆分（Equal Split）    
```
llama_ubatch llama_batch_allocr::split_equal(uint32_t n_ubatch, bool sequential) {
    std::vector<seq_set_t> cur_seq_set;
    
    // 确定参与此ubatch的非重叠序列集
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (used[i]) continue;
        
        bool add = true;
        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {
            if (!(cur_seq_set[s] & seq_set[i]).none()) {
                add = false;
                break;
            }
        }
        
        if (add) {
            cur_seq_set.push_back(seq_set[i]);
            if (cur_seq_set.size() > n_ubatch) break;
        }
    }
    
    // 处理每个序列集的token
    std::vector<idx_vec_t> idxs_per_seq(cur_seq_set.size());
    // ... 具体实现
}
```

3. 序列拆分（Sequence Split）    
```
llama_ubatch llama_batch_allocr::split_seq(uint32_t n_ubatch) {
    uint32_t cur_idx = 0;
    while (cur_idx < used.size() && used[cur_idx]) {
        ++cur_idx;
    }
    
    auto cur_seq_set = seq_set[cur_idx];
    std::vector<int32_t> idxs;
    
    while (idxs.size() < n_ubatch) {
        idxs.push_back(cur_idx);
        used[cur_idx] = true;
        ++n_used;
        
        // 查找下一个符合条件的token
        do {
            ++cur_idx;
        } while (cur_idx < get_n_tokens() && 
                (used[cur_idx] || 
                 ((cur_seq_set & seq_set[cur_idx]) != seq_set[cur_idx])));
        
        if (cur_idx == get_n_tokens()) break;
        cur_seq_set = seq_set[cur_idx];
    }
    
    return ubatch_add(idxs, 1, true);
}
```
llama.cpp通过智能内存管理实现连续推理：
```
llama_memory_context_ptr llama_memory_hybrid::init_batch(
    llama_batch_allocr & balloc, 
    uint32_t n_ubatch, 
    bool embd_all) {
    
    std::vector<llama_ubatch> ubatches;
    
    while (true) {
        llama_ubatch ubatch;
        if (embd_all) {
            ubatch = balloc.split_seq(n_ubatch);
        } else {
            ubatch = balloc.split_equal(n_ubatch, false);
        }
        
        if (ubatch.n_tokens == 0) break;
        ubatches.push_back(std::move(ubatch));
    }
    
    // 准备循环和注意力ubatches
    if (!mem_recr->prepare(ubatches)) {
        LLAMA_LOG_ERROR("Failed to prepare recurrent ubatches");
        return nullptr;
    }
    
    auto heads_attn = mem_attn->prepare(ubatches);
    if (!heads_attn) {
        LLAMA_LOG_ERROR("Failed to prepare attention ubatches");
        return nullptr;
    }
    
    return std::make_shared<llama_memory_hybrid_context>(
        this, std::move(heads_attn), std::move(ubatches));
}
```