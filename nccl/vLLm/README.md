

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