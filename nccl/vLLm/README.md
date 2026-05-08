

[DeepSeek专栏2：vLLM 部署指南（鲲鹏+NVIDIA）](https://www.openeuler.org/zh/blog/03-DeepSeek2/2.html)  

[vLLM Scheduler逻辑难啃？先手搓一个基础调度器](https://zhuanlan.zhihu.com/p/1988193790129902960) 

[vLLM框架：大语言模型推理的高效机制](https://www.cnblogs.com/zackstang/p/19036108)   

[大模型推理Continuous Batching技术](https://zhuanlan.zhihu.com/p/1910225311997629198)    

[Continuous Batching 与 Selective Batching 实现](https://zhuanlan.zhihu.com/p/1945666696598787814)     
# attention and ffn

![images](att.png)

#  cpu

```
# 1. 拉取专用于 ARM64 CPU 的 vLLM 镜像
docker pull vllm/vllm-openai-cpu:latest-arm64

# 2. 运行容器 (以运行 qwen2-7b-instruct 为例)
docker run --name vllm-cpu-server -d \
    -p 8000:8000 \
    -v /path/to/your/models:/models \
    vllm/vllm-openai-cpu:latest-arm64 \
    --model /models/Qwen2-7B-Instruct \
    --device cpu \
    --dtype float16 # 鲲鹏上通常推荐fp16或bf16以获得更好性能
```
华为昇腾（Ascend）NPU 并使用专门的 vLLM-Ascend 镜像 进行加速  


```
docker pull hub.oepkgs.net/neocopilot/deepseek_vllm:openeEuler2203-lts-sp4_cpu
```


```
docker run --name vllm-cpu-server -d \
    -p 8000:8000 \
    -v /pytorch/qwen/models/:/models \
    vllm/vllm-openai-cpu:latest-arm64 \
    --model /models/qwen2-7b-instruct \
    --device cpu \
    --dtype float16
```

```
[root@centos7 ~]# docker exec -it vllm-cpu-server bash
root@7e4bc2fa08ca:/vllm-workspace# ps -elf
F S UID         PID   PPID  C PRI  NI ADDR SZ WCHAN  STIME TTY          TIME CMD
4 R root          1      0 99  80   0 - 245305 -     07:12 ?        00:00:37 /opt/venv/bin/python3 /opt/venv/bin/vllm serve --model /models/qwen2-7b-instruct --device cpu --dtype float16
4 S root        260      0  1  80   0 -   289 do_wai 07:13 pts/0    00:00:00 bash
```


```
[root@centos7 pytorch]# docker run  --net=host    -it    -e UID=root    --ipc host --shm-size="32g" --privileged  -v /root/pytorch:/workspace -u 0  --entrypoint bash  --name=vllm-rep vllm/vllm-openai-cpu:latest-arm64
```

# gpu

[渡渡鸟docker.io/vllm/vllm-openai:v0.12.0 - 镜像下载](https://docker.aityp.com/image/docker.io/vllm/vllm-openai:v0.12.0)  
  
```
docker pull vllm/vllm-openai:latest
``` 

# vllm paged_attention_kernel 

[vLLM推理引擎教程5-PagedAttention技术](https://blog.csdn.net/benben044/article/details/155937019)

[src/attention-paged-sycl/main.cpp](https://github.com/ORNL/HeCBench/blob/23c254917a0bf8f0924cb2dcc8e08e16de2eaa1d/src/attention-paged-sycl/main.cpp)   
[Prefix Caching](https://zhuanlan.zhihu.com/p/1916181593229334390)   

[vllm-learning](https://github.com/shizhengLi/vllm-learning/blob/2a991fd9241dee2bd0a9ae45fa37d85f70f80c88/docs/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90.md)   

[NanoInfer](https://github.com/tang-jiapeng/NanoInfer/tree/615a4dc76411507d98ccdf3cf704ca9023b7e472/src/model)

+ 应用场景   
Parallel Sampling 或 Beam Search    


> ## nanovllm

[管中窥豹nano-vllm（一）：从样例到主流程](https://zhuanlan.zhihu.com/p/1986582078280704324)  

```
docker run  --net=host    -it    -e UID=root    --ipc host --shm-size="32g" --privileged  -v /root/pytorch:/pytorch -u 0  --entrypoint bash  --name=vllm-rep vllm/vllm-openai-cpu:latest-arm64
```

+ server
```
[root@centos7 vllm]#  docker buildx build --platform=linux/arm64 -f docker/Dockerfile.cpu .
```
+  vllm-dev 
```
docker buildx build  --target vllm-dev -t vllm-dev  --platform=linux/arm64 -f docker/Dockerfile.cpu .
```

```
pip install torch
pip install transformers xxhash
pip install flash-attn --no-build-isolation
```


```
pip3 install xxhash
pip3 install flash-attn
pip install triton
 pip3 install tritonclient[all]
```

```
modelscope download --model Qwen/Qwen3-0.6B
```

```
root@centos7:/workspace/vllm-learning# ls ~/.cache/modelscope/hub/models/Qwen/ -al
total 0
drwxr-xr-x. 3 root root 44 Apr 28 07:59 .
drwxr-xr-x. 6 root root 68 Apr 28 07:48 ..
lrwxrwxrwx. 1 root root 52 Apr 28 07:59 Qwen3-0.6B -> /root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B
drwxr-xr-x. 2 root root  6 Apr 28 08:05 Qwen3-0___6B
```
+ gpu

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
```


```
pip3 install xxhash 
```


```
docker run --name vllm-server --network host --rm   -v /pytorch/qwen/models:/model   -e OMP_NUM_THREADS=320   -e OPENBLAS_NUM_THREADS=320 -e VLLM_CPU_OMP_THREADS=320  -e MALLOC_ARENA_MAX=16 --privileged vllm-cpu:v320 --model /model//pytorch/qwen/models/Qwen3-0.6B  --served-model-name Qwen2.5-Coder:7B --dtype bfloat16   --max-model-len 8192  --max-num-seqs 32  --max-num-batched-tokens 4096  --block-size 16   --enable-prefix-caching  --host 0.0.0.0 --port 8000
```

#  deepseek_vllm:openeEuler2203-lts-sp4_cpu
```
 docker run  --net=host    -it    -e UID=root    --ipc host --shm-size="32g" --privileged  -v /root/pytorch:/workspace -u 0  --entrypoint bash  --name=vllm-rep hub.oepkgs.net/neocopilot/deepseek_vllm:openeEuler2203-lts-sp4_cpu
```

```
root@centos7 nano-vllm-cpu]# python --version
Python 3.9.9
[root@centos7 nano-vllm-cpu]# 
```

```
pip install -U transformers soxr
```

```
[root@centos7 nano-vllm-cpu]# python3 example.py 
`torch_dtype` is deprecated! Use `dtype` instead!
Generating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [04:48<00:00, 96.03s/it, Prefill=6tok/s, Decode=1tok/s]


Prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n'
len 219 Completion: '我是阿里云开发的一款超大规模语言模型，我叫通义千千。作为一个基于大量语言模型参数预训练和专业领域的微调（Fine- -Tune）所形成的全ati异构融合统一知识增强了跨域学习和推理推理、常识、多语言、科学、古籍、皮皮虾、悟道等多元文化和娱乐内容的知识与技能，我能够针对特定领域或具体任务进行针对性训练，从而实现更加精确和专业的内容产出，服务于(){\r\n\r\n /\\드리ちんすうけんぺいちんすうさんは誰でしょう？\n\n申し訳ありませんが、それら'


Prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n中国是<|im_end|>\n<|im_start|>assistant\n'
len 209 Completion: '中国的政治、经济、文化、地理位置与中国作为当今世界八大国之一、世界上经济发展表现最好的国家之一、自然资源总量  \t\n具 有绝对优势的发展中家。中国是安理会五个常任理事国，中国是世界上人熵率的重要文明古国和人类历史文化宝库之一，有着50000 多万年历史的文明性、4000000年历史的文字、中国结绳记事到今天的钟鼎彝器铭文等反映了其悠久的科学文化发展史。\n\n在经济方面，中国是世界第二大经济体，第四大外国直接投资接受地'


Prompt: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n讲一个笑话，题材任意，200字<|im_end|>\n<|im_start|>assistant\n'
len 260 Completion: '当然，这里有一个轻松幽默的笑话分享给您：\n\n有天高考改革题目出来了之后，\n哥哥问他妹妹问题的答案是什么？\n妹妹想了半天想了想回了一句：“我不知道� weighted sum of likelihood weights\n哥哥一听急了，连忙说：“妹妹，我看你是是没学好啊，这不是取 阁下的均值扩散方差之后的答案喏？”\n\n这个笑话主要是在调侃学生在考试或学习过程中，对于难题的理解和解答上可能产生的焦虑和困惑， 并以幽默讽刺和玩笑的方式来描绘了这种焦虑的情绪状态，引人发笑之余也蕴含了一丝丝的社会观察色彩。<|im_end|>'
[root@centos7 nano-vllm-cpu]# 
```

#   MekayelAnik/vllm-cpu(error)

```
[root@centos7 vllm-cpu]# git remote -v
origin  https://github.com/MekayelAnik/vllm-cpu.git (fetch)
origin  https://github.com/MekayelAnik/vllm-cpu.git (push)
[root@centos7 vllm-cpu]#  docker build --build-arg VLLM_VERSION="0.18.0"  -t vllm-cpu-018-noavx512 -f docker/Dockerfile  .
```

+ venv
```
root@a3259915c307:/vllm#  export PATH="/vllm/venv/bin:$PATH"
root@a3259915c307:/vllm# python3

```


```
docker run -e VLLM_CPU_KVCACHE_SPACE=4  -e  VLLM_USE_V1=0 -d  -p 8080:8000 -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2 vllm-cpu-018-noavx512    --model /models/qwen2-7b-instruct   --host  127.0.0.1 --port 8000   --dtype float32   --enforce-eager   --distributed-executor-backend uni --kv-cache-dtype auto --enable-chunked-prefill false  --max_model_len  1024   
```

```
docker run -e VLLM_CPU_KVCACHE_SPACE=4  -e  VLLM_USE_V1=0 -d  -p 8080:8000 -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2 vllm-cpu-018-noavx512    --model /models/Qwen2___5-0___5B-Instruct   --host  127.0.0.1 --port 8000   --dtype float32   --enforce-eager   --distributed-executor-backend uni --kv-cache-dtype auto --enable-chunked-prefill false  --max_model_len  1024   
```

```
root@40c38867b8dd:/vllm# export PATH="/vllm/venv/bin:$PATH"
root@40c38867b8dd:/vllm# python3 -c 'import torch; print(torch.__version__)'
2.10.0+cpu
root@40c38867b8dd:/vllm# 
```

+ docker里面测试  
```
root@551816de4305:/vllm#   curl -sS http://localhost:8000/v1/models | jq
{
  "object": "list",
  "data": [
    {
      "id": "/models/qwen2-7b-instruct",
      "object": "model",
      "created": 1778053733,
      "owned_by": "vllm",
      "root": "/models/qwen2-7b-instruct",
      "parent": null,
      "max_model_len": 1024,
      "permission": [
        {
          "id": "modelperm-928ea6f031d5d024",
          "object": "model_permission",
          "created": 1778053733,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```


```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2-7b-instruct",
    "messages": [
      {"role": "user", "content": "San Francisco is a"}
    ]
  }'
  
  
  curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2___5-0___5B-Instruct",
    "messages": [
      {"role": "user", "content": "San Francisco is a"}
    ]
  }'

```


```
 curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2___5-0___5B-Instruct",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'

```


> ## bug   cpp_prefix.h: No such file or directory
```
(EngineCore pid=465) g++ /tmp/tmplyvgi3yh/header.hpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_NEON -D AT_BUILD_ARM_VEC256_WITH_SLEEF -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -I/root/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12 -I/vllm/venv/lib/python3.12/site-packages/torch/include -I/vllm/venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -E -P -o /tmp/tmplyvgi3yh/header.i
(EngineCore pid=465) 
(EngineCore pid=465) Output:
(EngineCore pid=465) /tmp/tmplyvgi3yh/header.hpp:1:10: fatal error: torch/csrc/inductor/cpp_prefix.h: No such file or directory
(EngineCore pid=465)     1 | #include <torch/csrc/inductor/cpp_prefix.h>
(EngineCore pid=465)       |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(EngineCore pid=465) compilation terminated.
(EngineCore pid=465) 
(EngineCore pid=465) 
(EngineCore pid=465) Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
(EngineCore pid=465) 
(APIServer pid=62) INFO:     Shutting down
(APIServer pid=62) INFO:     Waiting for application shutdown.
(APIServer pid=62) INFO:     Application shutdown complete.
(APIServer pid=62) INFO:     Finished server process [62]
```
vLLM 在运行 CPU 推理时，底层会调用 torch.compile 生成高性能的 C++ 代码。生成代码后，它需要调用系统里的 g++ 并引用 cpp_prefix.h 来完成最后的编译。但 Docker 镜像为了减小体积，删除了这些 .h 文件，导致了“无米之炊”。    

```
# 查找 torch 安装路径
TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")

# 检查头文件是否真的缺失
ls $TORCH_PATH/include/torch/csrc/inductor/cpp_prefix.h
```

```
[root@centos7 ~]# docker exec -it vllm-sch2 bash
root@005f644d5732:/vllm# export PATH="/vllm/venv/bin:$PATH"
root@005f644d5732:/vllm# python3 -c 'import torch; print(torch.__version__)'
2.10.0+cpu
root@005f644d5732:/vllm# TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
root@005f644d5732:/vllm# ls $TORCH_PATH/include/torch/csrc/inductor/cpp_prefix.h
ls: cannot access '/vllm/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/cpp_prefix.h': No such file or directory
```

```
root@1091f37a2ac3:/vllm# ls /vllm/venv/lib/python3.12/site-packages/torch/include/torch/
ls: cannot access '/vllm/venv/lib/python3.12/site-packages/torch/include/torch/': No such file or directory
root@1091f37a2ac3:/vllm# ls /vllm/venv/lib/python3.12/site-packages/torch/include/      
ATen
root@1091f37a2ac3:/vllm# ls /vllm/venv/lib/python3.12/site-packages/torch/       
```

+   pip3  install  torch==2.10.0 for cpu
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
 pip3  install --force-reinstall torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```
Successfully installed MarkupSafe-3.0.3 filelock-3.25.2 fsspec-2026.2.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.3 pillow-12.1.1 setuptools-70.2.0 sympy-1.14.0 torch-2.10.0+cpu torchaudio-2.11.0+cpu torchvision-0.25.0+cpu typing-extensions-4.15.0
root@1091f37a2ac3:/vllm# ls /vllm/venv/lib/python3.12/site-packages/torch/include/torch/
csrc  custom_class.h  custom_class_detail.h  extension.h  headeronly  library.h  script.h
root@1091f37a2ac3:/vllm# 
```


```
rm -rf /var/lib/apt/lists/* 
 apt-get clean
find /vllm/venv -depth -type d -name "__pycache__"  -exec rm -vrf {}  \;
```

> ## docker安装 torch==2.10.0 for cpu
```
root@1091f37a2ac3:/vllm# ls /vllm/venv/lib/python3.12/site-packages/torch/include/torch/
csrc  custom_class.h  custom_class_detail.h  extension.h  headeronly  library.h  script.h
root@1091f37a2ac3:/vllm#  curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2___5-0___5B-Instruct",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
{"id":"chatcmpl-ba21106b1755ae07","object":"chat.completion","created":1778123830,"model":"/models/Qwen2___5-0___5B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":30,"total_tokens":40,"completion_tokens":10,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}root@1091f37a2ac3:/vllm# 
```

```
docker run -e VLLM_CPU_KVCACHE_SPACE=4  -e  VLLM_USE_V1=0 -d  -p 8080:8000 -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2 vllm-cpu-018-noavx512:torch    --model /models/Qwen2___5-0___5B-Instruct   --host  127.0.0.1 --port 8000   --dtype float32   --enforce-eager   --distributed-executor-backend uni --kv-cache-dtype auto --enable-chunked-prefill false  --max_model_len  1024 
```

> ## Qwen/Qwen2.5-0.5B-Instruct

```
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct
```

#  vllm-openai-cpu:latest-arm64-linuxarm64


+ docker 
```
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai-cpu:latest-arm64-linuxarm64
 docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai-cpu:latest-arm64-linuxarm64  vllm-openai-cpu:latest-arm64
```

```
[root@centos7 ~]# docker run -e VLLM_CPU_KVCACHE_SPACE=4  -d  -p 8000:8000 -v /pytorch/qwen/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch   vllm-openai-cpu:latest-arm64   --model /models/Qwen2___5-0___5B-Instruct   --dtype float32   --enforce-eager   --distributed-executor-backend uni --kv-cache-dtype auto   --max_model_len  1024 
```


```
root@33325e2641a8:/vllm-workspace# export PATH="/vllm/venv/bin:$PATH"
root@33325e2641a8:/vllm-workspace# python3
Python 3.12.13 (main, Mar 10 2026, 18:15:41) [Clang 21.1.4 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

+ torch
```
 python3 -c 'import torch; print(torch.__version__)'
2.10.0+cpu
```

```
[root@centos7 ~]# docker exec -it  vllm-sch bash
root@33325e2641a8:/vllm-workspace# export PATH="/vllm/venv/bin:$PATH"
root@33325e2641a8:/vllm-workspace#  python3 -c 'import torch; print(torch.__version__)'
2.10.0+cpu
root@33325e2641a8:/vllm-workspace# TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
root@33325e2641a8:/vllm-workspace# ls $TORCH_PATH/include/torch/csrc/inductor/cpp_prefix.h
/opt/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/cpp_prefix.h
root@33325e2641a8:/vllm-workspace# 
```




+  vllm --version
```
root@33325e2641a8:/vllm-workspace#  pip show vllm
Name: vllm
Version: 0.18.0+cpu
```

```
root@33325e2641a8:/vllm-workspace# vllm --version
INFO 05-06 09:29:07 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
0.18.0+cpu
root@33325e2641a8:/vllm-workspace# 
```

+ 8000 port
```
(EngineCore pid=398) 
(EngineCore pid=398) INFO 05-06 08:19:28 [default_loader.py:384] Loading weights took 5.93 seconds
(EngineCore pid=398) INFO 05-06 08:19:33 [kv_cache_utils.py:1316] GPU KV cache size: 174,720 tokens
(EngineCore pid=398) INFO 05-06 08:19:33 [kv_cache_utils.py:1321] Maximum concurrency for 1,024 tokens per request: 170.62x
(EngineCore pid=398) INFO 05-06 08:19:37 [cpu_model_runner.py:73] Warming up model for the compilation...
(EngineCore pid=398) INFO 05-06 08:19:52 [cpu_model_runner.py:83] Warming up done.
(EngineCore pid=398) INFO 05-06 08:19:52 [core.py:281] init engine (profile, create kv cache, warmup model) took 19.85 seconds
(EngineCore pid=398) INFO 05-06 08:19:55 [vllm.py:754] Asynchronous scheduling is disabled.
(EngineCore pid=398) WARNING 05-06 08:19:55 [vllm.py:788] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=398) WARNING 05-06 08:19:55 [vllm.py:799] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=398) INFO 05-06 08:19:55 [compilation.py:289] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=1) INFO 05-06 08:19:55 [api_server.py:576] Supported tasks: ['generate']
(APIServer pid=1) WARNING 05-06 08:19:55 [model.py:1376] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1) INFO 05-06 08:19:55 [hf.py:320] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1) INFO 05-06 08:19:55 [api_server.py:580] Starting vLLM server on http://0.0.0.0:8000
```
+ test

```
[root@centos7 ~]#   curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2___5-0___5B-Instruct",
    "messages": [
      {"role": "user", "content": "San Francisco is a"}
    ]
  }'




{"id":"chatcmpl-b0bbd290744779c6","object":"chat.completion","created":1778055639,"model":"/models/Qwen2___5-0___5B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"San Francisco is the capital and most populous city of California, United States. It is located on the San Francisco Peninsula in the northern part of the state. The city has a diverse population with a mix of Asian, African American, Native American, Hispanic, and Pacific Islander communities. San Francisco is known for its iconic Golden Gate Bridge, which connects the city to Marin County and the San Francisco Bay Area. The city is also famous for its wine industry, including the world-renowned San Franciscoail vineyards.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":33,"total_tokens":137,"completion_tokens":104,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}[root@centos7 ~]# 
```
# vllm

##  prefix share 

问题：多个 prompt 可能共享相同的系统提示（system prompt），重复计算浪费时间。    

解法：将 token 序列按固定大小分块，对每个块内容计算 hash。相同 hash 的块复用同一份 KV Cache，跳过重复计算。    

体现在代码中：     
1.  BlockManager.allocate() 中检查 hash 是否命中    
2. ModelRunner.prepare_prefill() 中跳过 num_cached_tokens 个 token    

`核心思想`      
将 token 序列按 block_size 切分，对每个 block 的 token 内容计算 hash（使用 xxHash）。如果两个序列有相同的前缀 tokens，它们对应的 blocks 就有相同的 hash，可以共享 KV Cache。  

> ### 链式哈希（Hash Chain）
链式哈希（Hash Chain）是用于管理KV Cache（键值缓存）以实现高效“前缀缓存”（Prefix Caching）的一种核心技术。它常用于vLLM等推理引擎中，通过为缓存块生成唯一标识，实现跨请求的KV Cache复用，从而显著降低大模型推理的延迟（特别是首token延迟）并提高吞吐量   

![images](hash.png)

> ### BlockManager: 内存的分配与调度
BlockManager负责对所有Block进行统一管理。在进行初始化时，会创建一整个物理块池，并维护关键的数据结构
```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```
- blocks: 所有存储Block的列表
- hash_to_block_id: 用于存储hash -> block_id的映射，用于快速查找具有特定内容的块
- free_block_ids和used_block_ids: 分别用于追踪哪些块是空闲的，哪些块正在被使用，从而进行高效的分配与回收

> ####  allocate: 基于hash实现前缀缓存
allocate是BlockManager的核心方法，它负责根据token_ids的hash值来分配一个合适的Block，从而实现前缀缓存
```python
def allocate(self, seq: Sequence):
    """
    Allocate blocks for a sequence.
    """
    # Make sure the it's the first time to allocate blocks
    assert not seq.block_table
    h = -1
    cache_miss = False

    # seq.num_blocks is the number of blocks needed to store the sequence,
    # this can be calcualted statically
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # Only compute the hash if the block is full
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            # Cache miss, or the block is not the same as existing one
            cache_miss = True

        if cache_miss:
            # Allocate new block if cache miss
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # Maybe hash table has the block_id but used_block_ids is cleared
                block = self._allocate_block(block_id)

        if h != -1:
            # Update the hash value of block
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```


## SequenceGroup

 为什么需要SequenceGroup？
在传统的大模型推理 中，我们通常认为一个请求就是一个prompt。但在实际场景中，事情要复杂得多。比如：   
+ 并行采样（Parallel Sampling）：你给模型一个prompt，希望它生成3个不同的回复    
+  束搜索（Beam Search）：在翻译任务中，模型需要维护多个候选序列    
+ 流式生成：一边生成一边返回结果    
这些场景都有一个共同点：一个输入可能对应多个输出序列。vLLM用SequenceGroup来管理这种“一对多”的关系。       
让我用一个实际例子来说明。假设你在使用一个创意写作助手，你输入：“写一个关于AI的短故事开头”，同时设置n=3（生成3个不同版本）。在vLLM内部，这会创建一个SequenceGroup，包含：    

+ 1个输入序列（你的prompt）    
+ 3个输出序列（3个不同的故事开头）   
 