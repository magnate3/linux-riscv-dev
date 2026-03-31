

```
./build/bin/llama-speculative  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  --model-draft /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  -p "Simple python quicksort function:\n" -n 200 -e  --temp 1
```


```
 ./build/bin/llama-speculative -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  --model-draft /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  -p "Simple python quicksort function:\n" -n 200  -c 4096    --draft 5  

```
  --draft 5                          # 每次让小模型先猜 5 个 Token    
  
# llama

```
pip3 install huggingface-hub
python3 -m pip install -U "huggingface_hub[cli]"
```

```
root@centos7:/workspace/qwen/models# python3 -m pip install -U "huggingface_hub[cli]"
Requirement already satisfied: huggingface_hub[cli] in /usr/local/lib/python3.10/dist-packages (1.8.0)
WARNING: huggingface-hub 1.8.0 does not provide the extra 'cli'
```
下载脚本   
```
from huggingface_hub import hf_hub_download

repo_id = "QuantFactory/AMD-Llama-135m-GGUF"
filename = "AMD-Llama-135m.Q8_0.gguf"

print(f"download: {filename}...")

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="./",
    local_dir_use_symlinks=False
)
print(f"download {model_path} finish")
```

```
huggingface-cli download QuantFactory/AMD-Llama-135m-GGUF \
  --include "AMD-Llama-135m.Q8_0.gguf" \
  --local-dir ./ \
  --local-dir-use-symlinks False
```



```
huggingface-cli download QuantFactory/Meta-Llama-3-8B-GGUF --include "Meta-Llama-3-8B.Q4_K_M.gguf" --local-dir ./local-llama3-gguf

```

> ## modelscope
```
pip install modelscope
```
下载完整模型库

+ 大模型
```
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```


```
oot@centos7:/workspace/qwen/models# modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

 _   .-')                _ .-') _     ('-.             .-')                              _ (`-.    ('-.
( '.( OO )_             ( (  OO) )  _(  OO)           ( OO ).                           ( (OO  ) _(  OO)
 ,--.   ,--.).-'),-----. \     .'_ (,------.,--.     (_)---\_)   .-----.  .-'),-----.  _.`     \(,------.
 |   `.'   |( OO'  .-.  ',`'--..._) |  .---'|  |.-') /    _ |   '  .--./ ( OO'  .-.  '(__...--'' |  .---'
 |         |/   |  | |  ||  |  \  ' |  |    |  | OO )\  :` `.   |  |('-. /   |  | |  | |  /  | | |  |
 |  |'.'|  |\_) |  |\|  ||  |   ' |(|  '--. |  |`-' | '..`''.) /_) |OO  )\_) |  |\|  | |  |_.' |(|  '--.
 |  |   |  |  \ |  | |  ||  |   / : |  .--'(|  '---.'.-._)   \ ||  |`-'|   \ |  | |  | |  .___.' |  .--'
 |  |   |  |   `'  '-'  '|  '--'  / |  `---.|      | \       /(_'  '--'\    `'  '-'  ' |  |      |  `---.
 `--'   `--'     `-----' `-------'  `------'`------'  `-----'    `-----'      `-----'  `--'      `------'

Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct-GGUF

Successfully Downloaded from model LLM-Research/Meta-Llama-3-8B-Instruct-GGUF.
```

下载单个文件到指定本地文件夹（以下载README.md到当前路径下“dir”目录为例）     
```

modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct-GGUF README.md --local_dir ./dir
```

+ 小模型 

```

modelscope download --model second-state/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
  --local_dir ./

```

> ## 词表对齐
Meta-Llama-3-8B-Instruct-Q4_K_M.gguf(大)   
Llama-3.2-1B-Instruct-Q4_K_M.gguf（小）   
```
n_vocab_tgt 128256,n_vocab_dft 128256 
Target ID for 'Hello': 9906
Draft ID for 'Hello':  9906
```

# 大小模型前缀复用优化


```
前缀复用：大模型先算一遍 RAG 文档的 KV Cache，小模型直接“白嫖”这些计算好的 Hot Tokens。
```



# 测试

```
cmake -S . -B build -DLLAMA_CPP_DIR=/workspace/llama.cpp/
```

```
./build/big_small_co2
```

```
n_vocab_tgt 128256,n_vocab_dft 32000 
Target ID for 'Hello': 9906
Draft ID for 'Hello':  15043
vocab mismatch 
Accept part: 1/4 | Total: 11
Accept part: 3/4 | Total: 18
Accept part: 3/4 | Total: 25
Accept part: 3/4 | Total: 32
Accept part: 1/4 | Total: 39
Accept part: 1/4 | Total: 41
Accept part: 1/4 | Total: 43
Accept part: 3/4 | Total: 53
Accept part: 3/4 | Total: 60
Accept part: 3/4 | Total: 67
Accept part: 3/4 | Total: 75
init: invalid token[0] = 62224
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
Draft decode failed at pos 77
~llama_context:        CPU compute buffer size is 274.5020 MiB, matches expectation of 274.5020 MiB
~llama_context:        CPU compute buffer size is  79.0020 MiB, matches expectation of  79.0020 MiB
```
AMD-Llama-135m.Q8_0.gguf 和Meta-Llama-3-8B-Instruct-Q4_K_M.gguf 词表大小不一样   


大小模型词表不一致   
```
llama.cpp 提供了一种不需要小模型的投机采样方案：Prompt Lookup。它通过在已有的 Prompt（Hot Tokens）中寻找重复模式来预测。
实现逻辑：将 draft_model 替换为从输入文本中提取的 N-gram。
C++ 接口：调用 llama_sample_token_greedy 配合自定义的查找逻辑（common 库中有实现）。
```


> ##  加上自回归推理（Autoregressive Inference）



```
if (n_accept == n_draft) {
             llama_batch b_dft = llama_batch_init(1, 0, 1);
             b_dft.n_tokens = 1;
             b_dft.token[0] = next_token;
             b_dft.pos[0]   = n_past-1;
             b_dft.n_seq_id[0] = 1;
             b_dft.seq_id[0][0] = 0;
             b_dft.logits[0] = true;

            if (llama_decode(ctx_dft, b_dft) != 0) {
                fprintf(stderr, "Draft decode failed at pos %d\n", b_dft.pos[0]);
                llama_batch_free(b_dft);
                goto fail1;
            }
            llama_batch_free(b_dft);
             llama_batch b_tgt = llama_batch_init(1, 0, 1);
             b_tgt.n_tokens = 1;
             b_tgt.token[0] = next_token;
             b_tgt.pos[0]   = n_past-1;
             b_tgt.n_seq_id[0] = 1;
             b_tgt.seq_id[0][0] = 0;
             b_tgt.logits[0] = true;

            if (llama_decode(ctx_tgt, b_tgt) != 0) {
                fprintf(stderr, "Target decode failed at pos %d\n", b_tgt.pos[0]);
                llama_batch_free(b_tgt);
                goto fail1;
            }
            llama_batch_free(b_tgt);
        }
```

```
Accept part: 1/4 | Total: 7
Accept part: 3/4 | Total: 11
Accept part: 1/4 | Total: 13
Accept ALL : 4/4 | Total: 18
Accept ALL : 4/4 | Total: 24
Accept part: 2/4 | Total: 27
Accept ALL : 4/4 | Total: 33
Accept part: 3/4 | Total: 37
Accept ALL : 4/4 | Total: 42
Accept ALL : 4/4 | Total: 47
Accept ALL : 4/4 | Total: 52
Accept ALL : 4/4 | Total: 58
Accept ALL : 4/4 | Total: 63
Accept ALL : 4/4 | Total: 68
Accept ALL : 4/4 | Total: 73
Accept ALL : 4/4 | Total: 78
Accept ALL : 4/4 | Total: 83
Accept ALL : 4/4 | Total: 88
Accept ALL : 4/4 | Total: 93
Accept ALL : 4/4 | Total: 98
Accept ALL : 4/4 | Total: 103
The capital of France is Paris.
The capital of Germany is Berlin.
The capital of of Italy is Rome.
TheThe capital of of Japan is Tokyo.
TheThe capital of of China is Beijing.
TheThe capital of of India India is New Delhi.
```


#  采用贪梦算法生成了token后怎么填充kv cache


 
在 llama.cpp 中，采用贪婪算法（Greedy Search）生成一个 Token 后，填充 KV Cache 的过程实际上是自回归推理（Autoregressive Inference）的标准步骤。
与 Prefill 阶段（一次性处理整个 Prompt）不同，Decode 阶段（生成阶段）每次**只处理一个新生成的 Token**，并将其对应的 KV 数据追加到缓存池中。     

以下是实现这一过程的完整逻辑和代码步骤：    
1. 基本原理    
输入：仅包含上一步生成的最后一个 Token。  
位置 (pos)：该 Token 在序列中的逻辑位置，即 n_past（已处理的 Token 总数）。
缓存更新：llama_decode 会自动计算该 Token 的 Key 和 Value，并根据你指定的 pos 将其存入 KV Cache 的对应 Cell 中。       

2. 核心代码实现   
假设你已经通过贪婪采样得到了 new_token_id：
```
// 1. 记录当前已在缓存中的 token 数量 (n_past)
// 初始时 n_past = prompt_tokens.size()
int n_past = ...; 

// 2. 准备只包含一个新 token 的 batch
llama_batch batch = llama_batch_init(1, 0, 1);

// 3. 填充 batch 信息
batch.token[0]    = new_token_id; // 刚才生成的 token
batch.pos[0]      = n_past;       // 紧跟在之前的序列后面
batch.n_seq_id[0] = 1;
batch.seq_id[0][0]= 0;            // 假设使用 sequence ID 0
batch.logits[0]   = true;         // 下一步还需要生成 logits 才能继续采样

// 4. 执行推理
// 此函数会自动将该 token 的 KV 数据写入缓存中对应的 pos 位置
if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "推理失败\n");
    return;
}

// 5. 更新计数，准备下一次循环
n_past += 1;

```
填充后的关键点   
+  RoPE 旋转位置编码：llama.cpp 内部会根据 batch.pos[0] 对 KV 数据应用 RoPE 旋转。这意味着你指定的 pos 必须准确，否则模型会“忘记”上下文顺序。   
+  KV Cache 复用：llama_decode 内部逻辑是增量式的。它会检索之前 [0, n_past-1] 位置的所有 KV 数据，与当前新计算的 KV 拼接，参与 Attention 计算。  


#  speculative  



```
./build/speculative -m /workspace/qwen/models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf   --model-draft //workspace/qwen/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf   -p "where is the capital of france \n" -n 200  -c 4096    --draft 5  
```  

## Multi-candidate Speculative Decoding
