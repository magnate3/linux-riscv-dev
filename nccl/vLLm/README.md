

[DeepSeek专栏2：vLLM 部署指南（鲲鹏+NVIDIA）](https://www.openeuler.org/zh/blog/03-DeepSeek2/2.html)  

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

# gpu

[渡渡鸟docker.io/vllm/vllm-openai:v0.12.0 - 镜像下载](https://docker.aityp.com/image/docker.io/vllm/vllm-openai:v0.12.0)  
  
```
docker pull vllm/vllm-openai:latest
``` 

# vllm paged_attention_kernel 

[vLLM推理引擎教程5-PagedAttention技术](https://blog.csdn.net/benben044/article/details/155937019)

[src/attention-paged-sycl/main.cpp](https://github.com/ORNL/HeCBench/blob/23c254917a0bf8f0924cb2dcc8e08e16de2eaa1d/src/attention-paged-sycl/main.cpp)   
[Prefix Caching](https://zhuanlan.zhihu.com/p/1916181593229334390)   

+ 应用场景   
Parallel Sampling 或 Beam Search    