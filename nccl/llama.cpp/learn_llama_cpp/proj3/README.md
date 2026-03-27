

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

