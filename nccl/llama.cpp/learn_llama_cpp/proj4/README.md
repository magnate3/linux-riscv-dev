

```
c++ -std=c++11 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src batch.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama 

```

```
 c++ -std=c++17 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src -I /workspace/llama.cpp/common  batch2.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -lggml-base  -L/workspace/llama.cpp/build/common -lcommon -lz

```

```
c++ -std=c++17 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src -I /workspace/llama.cpp/common  batch3.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -lggml-base  -L/workspace/llama.cpp/build/common -lcommon -lz

```

```
cmake --build build -j64
```

```
./build/simple-chat -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  -c 2028
```

#  main调试
+ log   

```
 const char * log_level = getenv("LLGUIDANCE_LOG_LEVEL");
```
采用LOG_INF会有输出   

+ 参数解析采用LLAMA_EXAMPLE_COMPLETION
```
common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)
```

```

./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   -p "where are you from ? what can you do ?" -c 1024 --grp-attn-n  1
```


+  ontext full and context shift is disabled    

```
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   -p "where are you from ? what can you do ?" -c 256 --grp-attn-n  1
```

```
</think>

I am

main: context full and context shift is disabled => stopping
 an
```

+ 设置params.ctx_shift=1


```
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   -p "where are you from ? what can you do ?" -c 256 --grp-attn-n  1
```


```
where is the capital of France
where is the center of the world
```
输出

```
> where is the center of the world
after swap: n_past = 127
embd: [ '
':198, '<|im_start|>':151644, 'user':872, '
':198, 'where':2870, ' is':374, ' the':279, ' center':4126, ' of':315, ' the':279, ' world':1879, '<|im_end|>':151645, '
':198, '<|im_start|>':151644, 'assistant':77091, '
':198 ]
clear session path

```


```
# First run (computes and saves cache)
 


```
 ./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   --prompt-cache my_cache.bin -p "where are you from?"
```

# Subsequent run (loads cache and continues)
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf -p "where are you from ?what can you do ?" 



main: attempting to load saved session from 'my_cache.bin'
main: loaded a session with prompt size of 12 tokens
main: session file has low similarity to prompt (7 / 18 tokens); will mostly be reevaluated

```


```
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   --prompt-cache my_cache.bin -p "where are you from ?what can you do ?" --file new.bin
error while handling argument "--file": error: failed to open file 'new.bin'


usage:
-f,    --file FNAME                     a file containing the prompt (default: none)

```


> ##  n_past   n_keep
- n_past：      
记录“已经有多少token在KV缓存里”，每次生成+1，满了会被裁剪重置。     
- n_keep：    
上下文满了要丢旧内容时，前 n_keep 个token绝对不删，用来保住系统提示、人设prompt。   

```
 n_ctx = 20 （上下文窗口总长度）
 n_keep = 5 （前5个token永远不丢）
初始prompt一共 15个token
1. 初始状态
 
- 输入prompt： [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O] （共15个token）
-  n_ctx = 20 
-  n_keep = 5 
- 初始处理后：
-  n_past = 15 
- KV缓存里存了这15个token
 
此时窗口还剩：20 - 15 = 5个位置

2. 开始生成token
 
每生成1个token：
 
- 新token加入上下文
-  n_past += 1 
 
第1轮生成
 
生成： P 
 
- 上下文： A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P 
-  n_past = 16 
 
第2轮生成
 
生成： Q 
 
-  n_past = 17 
 
第3轮生成
 
生成： R 
 
-  n_past = 18 
 
第4轮生成
 
生成： S 
 
-  n_past = 19 
 
第5轮生成
 
生成： T 
 
-  n_past = 20 
 
现在上下文刚好满了：
 A B C D E F G H I J K L M N O P Q R S T （20个）
 

3. 再生成第6个token，触发裁剪
 
现在还要继续生成  U ，但  n_ctx=20  满了，必须丢旧token。
 
规则：
 
1. 前 n_keep=5 个 token 绝对不动： A B C D E  保留
2. 剩下的旧token从最老的开始丢，直到腾出位置
 
当前完整上下文：
 [A B C D E] [F G H I J K L M N O P Q R S T] 
 
需要腾出1个位置放新token  U ：
 
- 丢掉最老的非保留token： F 
- 把后面整体往前挪
- 最后加上新token  U 
 
裁剪后上下文变成：
 A B C D E G H I J K L M N O P Q R S T U 
 
同时：
 
-  n_past  不会变成21，而是被重置/调整
实际llama.cpp里会设为：
 n_past = n_ctx - 1  或类似，保证下次继续从缓存续推

4. 如果 n_keep = -1（保留整个初始prompt）
 
假设初始prompt还是15token：
 A B C D E F G H I J K L M N O 
 
-  n_keep = -1  → 等价于  n_keep = 15 
- 上下文满了要裁剪时：
- 前15个token不动
- 只裁剪后面生成的内容

```


```
// 上下文满了，开始裁剪
if (n_past >= n_ctx) {
    // 保留前 n_keep 个token
    keep_count = n_keep;
    if (n_keep == -1) keep_count = prompt_tokens_count;

    // 把 [keep_count ... n_past-1] 往前移
    // 丢掉中间一段旧token
    shift_kv_cache(keep_count);

    // n_past 调整到合适位置
    n_past = n_ctx - 1;
}
```

#  shift

```
// 1. 定义平移参数
int n_prefix = 100;           // 永远保留的系统提示词长度
int n_keep   = 50;            // 除了前缀，额外保留的最近对话轮数 (tokens)
int n_discard = 100;          // 本次需要丢弃的旧 Token 数量

// 2. 从 KV Cache 中删除最早的旧对话 (不包括前缀)
// 范围：从 n_prefix 开始，删除 n_discard 个 token
// [Prefix] [---Discard---] [---Keep---]
llama_kv_cache_seq_rm(ctx, slot_id, n_prefix, n_prefix + n_discard);

// 3. 将留在缓存里的 [Keep] 部分向前平移，填补 [Discard] 的空位
// p0: 起始位置 (n_prefix + n_discard)
// p1: 结束位置 (-1 表示到最后)
// delta: 平移距离 (负数表示向前移动)
llama_kv_cache_seq_add(ctx, slot_id, n_prefix + n_discard, -1, -n_discard);

// 4. 更新你的 Slot 计数器
my_slot.n_past -= n_discard;

printf("平移完成：删除了 %d tokens，Slot %d 当前长度: %d\n", 
        n_discard, slot_id, my_slot.n_past);

// 5. 接下来的推理
// 新 Token 的起始位置 (pos) 应该是 n_prefix + my_slot.n_past

```

```
关键点解析：
llama_kv_cache_seq_rm:
它只是在 KV 缓存管理逻辑中将这些位置标记为“无效”。
注意：不要删除 seq_id = 0 的部分，否则所有用户共享的公共前缀都会失效。

llama_kv_cache_seq_add (魔法函数):
它的作用是修改 KV 缓存中 Token 的 pos（位置索引）。
如果不做这一步，模型在计算时会发现 pos 序列是不连续的（例如从 100 跳到了 200），导致 RoPE（旋转位置编码） 计算错误，模型输出会直接变成乱码。

```

# slot 和 prefix 复用

```
// 假设分配到了 slot_id = 5
int n_prefix = 100; // 公共前缀长度（seq_id 0）
Slot& my_slot = slot_pool[5];

// 如果是新用户，需要处理全部 Prompt
// 如果是老用户（n_past > 0），只需要处理新增的 Token
int start_pos = n_prefix + my_slot.n_past; 

llama_batch batch = llama_batch_init(tokens_to_decode.size(), 0, 1);

for (size_t i = 0; i < tokens_to_decode.size(); ++i) {
    // 关键点：每个 Token 同时属于 公共序列(0) 和 用户私有序列(slot_id)
    llama_batch_add(batch, tokens_to_decode[i], start_pos + i, {0, my_slot.id}, true);
}

llama_decode(ctx, batch);

// 更新槽位状态
my_slot.n_past += tokens_to_decode.size();
my_slot.last_used_us = ggml_time_us();
```

# 计算similarity


```
cosine_similarity
 LCP similarity
 calculate_slot_similarity
 get_cached_tokens_similarity
```

#  GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")
```
当你开启了 量化 KV 缓存（如 Q8_0 或 Q4_0）或者使用了 Flash Attention 时，KV 缓存的存储方式不再是简单的线性数组，而是分块（Block）存储。在这种情况下，seq_cp 只能复制“完整的、未被裁剪过的” Block。
根本原因与解决方法
1. 禁用 Flash Attention (最直接的修复)
Flash Attention 极大地改变了 KV 缓存的物理布局。如果你在初始化时开启了它，seq_cp 往往会因为缓存不“完整”（非线性）而触发断言。
修复：在 llama_context_params 中确保 f_offload_kqv 为 false（默认通常是 true，这会触发硬件优化）。或者在命令行启动时不要使用 -fa。
2. 避免对量化缓存进行“部分复制”
如果你的 type_k 或 type_v 不是 F16 或 F32，而是量化格式，llama.cpp 不允许你复制“前 100 个 Token”这种操作，因为它无法在量化块内部进行精确切割。
修复：
将 type_k 和 type_v 设为 GGML_TYPE_F16。
如果必须用量化，请确保复制的是整个序列（即 p0=0, p1=-1），而不是中间的一段。
3. 放弃 seq_cp，改用“逻辑共享” (最高效、最稳健)
```


既然 seq_cp 物理拷贝限制这么多（容易 Core Dump，又容易断言失败），最工业化的做法是完全不用它。
通过在 llama_batch 中给每个 Token 贴上两个标签，模型会自动在计算时把它们拼在一起，且不需要任何物理拷贝。


逻辑共享（不使用 seq_cp）的完整实现逻辑：
```
// 1. 初始化：允许 Token 关联多个序列
cparams.n_seq_max = 16; 

// 2. 预处理公共前缀 (ID 0)
// 这里不需要做任何特殊操作，正常 decode 即可
for (int i = 0; i < n_prefix; ++i) {
    llama_batch_add(batch, prefix_tokens[i], i, {0}, false);
}
llama_decode(ctx, batch); 

// 3. 用户流 A (Slot 1) 接入
// 【核心】：不需要调用 seq_cp！直接在 batch 中引用 ID 0
for (size_t i = 0; i < user_a_tokens.size(); ++i) {
    // 每一个新 Token 同时属于 {0, 1}
    // 0 代表它“继承”了 ID 0 的所有历史
    // 1 代表它自己属于 Slot 1
    llama_batch_add(batch, user_a_tokens[i], n_prefix + i, {0, 1}, true);
}
llama_decode(ctx, batch);

// 4. 用户流 B (Slot 2) 接入 (同样操作)
for (size_t i = 0; i < user_b_tokens.size(); ++i) {
    // 同样“继承” ID 0，但私有部分属于 Slot 2
    llama_batch_add(batch, user_b_tokens[i], n_prefix + i, {0, 2}, true);
}
llama_decode(ctx, batch);
```

#   build kv
调用llm_build_kv_store存储,再调用llm_build_kqv获取输出