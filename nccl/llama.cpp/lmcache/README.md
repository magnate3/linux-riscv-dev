sudo docker run -it --rm --net=host    --gpus=all     -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0 -d  -p 8000:8000 -v /pytorch/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2   vllm-openai:latest   --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 --trust-remote-code --max-model-len 16384





```
sudo docker logs -f vllm-sch2
```

 
# vllm-openai:latest（未跑通）


```
sudo docker run -it --rm --net=host    --gpus=all     -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0 -d  -p 8000:8000 -v /pytorch/models/:/models -v /pytorch:/workspace --shm-size=4g  --name  cachegen    --entrypoint "/bin/bash"  vllm-openai:latest 
```

```
pip install lmcache
pip install lmcache_vllm
pip show torch transformers vllm
pip install "transformers>=4.44.0,<4.46.0" --force-reinstall
# 升级到较新版本的 vllm（会自动带入正确的 kv-transfer 命令行参数支持）
pip install --upgrade vllm torch transformers --upgrade-strategy eager
pip install ninja
```

```
vllm serve /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1/     --skip-tokenizer-init     --no-enable-prefix-caching     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

#  lmcache/vllm-openai:v0.4.2(lmcache和cuda不兼容)


```
echo "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}" > my_mistral.jinja

```


```
sudo docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmcache/vllm-openai:v0.4.2
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmcache/vllm-openai:v0.3.14-lightweight
```


```
sudo docker run -it --rm --net=host    --gpus=all     -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0 -d  -p 8000:8000 -v /pytorch/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2   lmcache-vllm-openai:v0.4.2   --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 --trust-remote-code --max-model-len 16384  --chat-template /workspace/lmcache/my_mistral.jinja
```

```
curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1","object":"model","created":1784597231,"owned_by":"vllm","root":"/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1","parent":null,"max_model_len":16384,"permission":[{"id":"modelperm-9428585c610cbb05","object":"model_permission","created":1784597231,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```


```
 curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1",
        "prompt": "San Francisco is a",
        "max_tokens": 15,
        "temperature": 0
    }'
{"id":"cmpl-bcc58ceab551e0da","object":"text_completion","created":1784597270,"model":"/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1","choices":[{"index":0,"text":" city that is known for its beautiful scenery, its rich history,","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":20,"completion_tokens":15,"prompt_tokens_details":null},"kv_transfer_params":null
```

+ 对话模式
```
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ],
        "max_tokens": 20
    }'
```





> ## lmcache

```
sudo docker run -it --rm --net=host    --gpus=all     -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0 -d  -p 8000:8000 -v /pytorch/models/:/models -v /pytorch:/workspace --shm-size=4g  --name vllm-sch2  --entrypoint "/bin/bash"    lmcache-vllm-openai:v0.4.2  
 
```

```
vllm --version
0.17.1
/opt/venv/bin/python3 -c "import vllm; print('vLLM 版本:', vllm.__version__)")"
vLLM 版本: 0.17.1
```


```
python3 -c "import torch; print(torch.__version__)"  
2.10.0+cu128
```

```
which python3
/opt/venv/bin/python3
which pip    
/usr/bin/pip
python3 -m pip uninstall -y lmcache lmcache_vllm
/opt/venv/bin/python3: No module named pip
```

```
 ls /opt/venv/bin/lmcache_*
/opt/venv/bin/lmcache_controller  /opt/venv/bin/lmcache_server  /opt/venv/bin/lmcache_v0_server

```

```
/opt/venv/bin/python3 -c "
import sys
from types import ModuleType

# 1. 内存伪造，阻断二进制报错
fake_c_ops = ModuleType('lmcache.c_ops')
sys.modules['lmcache.c_ops'] = fake_c_ops

# 2. 加载入口
from lmcache.v1.server.__main__ import main
import sys as os_sys

# 3. 传入匹配 [host] [port] [storage] 的正确参数
os_sys.argv = ['lmcache_server', '127.0.0.1', '5555', 'cpu']
print('🚀 [热补丁] 符号注入成功，正在拉起 LMCache 独立服务端...')
main()
"
```

后台运行

```
nohup /opt/venv/bin/python3 -c "
import sys
from types import ModuleType
fake_c_ops = ModuleType('lmcache.c_ops')
sys.modules['lmcache.c_ops'] = fake_c_ops
from lmcache.v1.server.__main__ import main
import sys as os_sys
os_sys.argv = ['lmcache_server', '127.0.0.1', '5555', 'cpu']
main()
" > lmcache_server.log 2>&1 &

```


```
# 1. 强行删除 /opt/venv 中残留的损坏 lmcache 文件夹（防止旧符号干扰）
rm -rf /opt/venv/lib/python3.12/site-packages/lmcache*

# 2. 进入克隆好的 LMCache 源码目录
cd LMCache

# 3. 借用系统 pip 进行现场编译，并强行注入到虚拟环境的库目录中
/usr/bin/pip install . \
    --break-system-packages \
    --target=/opt/venv/lib/python3.12/site-packages

# 4. 同样的方式安装 vLLM 适配器
/usr/bin/pip install lmcache_vllm \
    --break-system-packages \
    --target=/opt/venv/lib/python3.12/site-packages

```



```
export LMCACHE_USE_EXPERIMENTAL=True  
export LMCACHE_CHUNK_SIZE=256  
export LMCACHE_LOCAL_CPU=True  
export LMCACHE_MAX_LOCAL_CPU_SIZE=16.0 
```


```
vllm serve /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1     --skip-tokenizer-init     --no-enable-prefix-caching     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

参数及环境变量关键解析：    
--no-enable-prefix-caching：必须关闭 vLLM 自带的 Prefix 缓存，将整个 KV 缓存的复用与淘汰交由 LMCache 接管。    
kv_connector: LMCacheConnectorV1：告诉 vLLM 推理引擎在发生 Cache Miss 时向 LMCache 驱动去请求获取或存储 KV 块。   



#  lmcache/vllm-openai:latest
```
sudo docker run -it --rm --net=host \
    --gpus=all \
    -e UID=root \
    --ipc host \
    --privileged \
    -u 0 -d \
    -p 8000:8000 \
    -v /pytorch/models/:/models \
    -v /pytorch:/workspace \
    --shm-size=32g \
    --name vllm-sch2 \
    -e LMCACHE_USE_EXPERIMENTAL=True \
    -e LMCACHE_CHUNK_SIZE=256 \
    -e LMCACHE_LOCAL_CPU=True \
    -e LMCACHE_MAX_LOCAL_CPU_SIZE=16.0 \
    lmcache/vllm-openai:latest  \
    --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 \
    --skip-tokenizer-init \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

 

```
sudo docker stop vllm-sch2 && sudo docker rm vllm-sch2
sudo docker logs --tail 50 vllm-sch2
sudo docker logs -f vllm-sch2
```


+ chat-template(去掉--skip-tokenizer-init )


```
sudo docker run -it --rm --net=host \
    --gpus=all \
    -e UID=root \
    --ipc host \
    --privileged \
    -u 0 -d \
    -p 8000:8000 \
    -v /pytorch/models/:/models \
    -v /pytorch:/workspace \
    --shm-size=32g \
    --name vllm-sch2 \
    -e LMCACHE_USE_EXPERIMENTAL=True \
    -e LMCACHE_CHUNK_SIZE=256 \
    -e LMCACHE_LOCAL_CPU=True \
    -e LMCACHE_MAX_LOCAL_CPU_SIZE=16.0 \
    lmcache/vllm-openai:latest  \
    --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'\
	 --chat-template /workspace/lmcache/my_mistral.jinja
```


```
# 1. 强制杀死正在报错的旧容器
sudo docker stop vllm-sch2 && sudo docker rm vllm-sch2

# 2. 重新启动 
sudo docker run -it --rm --net=host \
    --gpus=all \
    -e UID=root \
    --ipc=host \
    --privileged \
    -u 0 -d \
    -p 8000:8000 \
    -v /pytorch/models/:/models \
    -v /pytorch:/workspace \
    --shm-size=32g \
    --name vllm-sch2 \
    -e LMCACHE_USE_EXPERIMENTAL=True \
    -e LMCACHE_CHUNK_SIZE=256 \
    -e LMCACHE_LOCAL_CPU=True \
    -e LMCACHE_MAX_LOCAL_CPU_SIZE=16.0 \
    lmcache/vllm-openai:latest \
    --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --chat-template /workspace/lmcache/my_mistral.jinja

```


```
curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1","object":"model","created":1784615535,"owned_by":"vllm","root":"/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-b4e5ee20e49b82b8","object":"model_permission","created":1784615535,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```



```
python3 test_cache.py
正在发送 第 1 次请求（冷启动） ...
💬 模型实际续写内容为: [ Deep learning is a subset of machine learning, which is essentially a neural network]
⏱️ 第 1 次请求耗时: 1.6579 秒

等待 3 秒中...
正在发送 第 2 次请求（预期命中 LMCache） ...
💬 模型实际续写内容为: [ Deep learning is a subset of machine learning, which is essentially a neural network]
⏱️ 第 2 次请求耗时: 0.4449 秒

🚀 === LMCache 节省了: 1.2130 秒 ===
```


第 1 次请求（冷启动）：1.6579 秒GPU 必须对几千个 Token 的英文上下文执行完整的 Prefill（预填充）计算，这消耗了绝大部分的时间。计算完成后，LMCache 自动在后台将这批 KV 缓存异步 offload 到了系统的 CPU 内存中。
第 2 次请求（热启动）：0.4449 秒由于上下文内容完全一致，vLLM 完美命中了 LMCache 缓存。LMCache 以超过 20 GB/s 的极高带宽将 0.65 GB 的 KV 缓存秒传回 GPU，直接跳过了耗时的 Prefill 阶段，让大模型直接开始 Decode 吐字。

> ## lmcache ping kvcache

```
lmcache ping kvcache
[2026-07-21 07:22:02,959] LMCache INFO:  torch_dev=<module 'torch.cuda' from '/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py'>, torch_device_type=cuda (__init__.py:63:lmcache)
[2026-07-21 07:22:02,970] LMCache INFO: CudaPinMemoryBackend: using torch cudart (pin_memory.py:89:lmcache.v1.platform.cuda.pin_memory)
[2026-07-21 07:22:03,056] LMCache INFO: Skipping backend lmcache.v1.platform.musa.ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:22:03,057] LMCache INFO: Skipping backend lmcache.xpu_ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:22:03,058] LMCache INFO: Using backend: lmcache.c_ops (__init__.py:132:lmcache)

 _     __  __    ____           _          
| |   |  \/  |  / ___|__ _  ___| |__   ___      LMCache v0.5.1 (g979719d7)
| |   | |\/| | | |   / _` |/ __| '_ \ / _ \     Website:  https://lmcache.ai/
| |___| |  | | | |__| (_| | (__| | | |  __/     Recipes:  https://docs.lmcache.ai/recipes
|_____|_|  |_|  \____\__,_|\___|_| |_|\___|     LinkedIn: https://www.linkedin.com/company/lmcache-lab
Set LMCACHE_DISABLE_BANNER=1 to hide this banner.

[2026-07-21 07:22:03,397] LMCache INFO: multi_layer_block_kv_transfer mode: ptr (base.py:94:lmcache.v1.multiprocess.transfer_context.base)
======= Ping KV Cache ========
Status:                   FAIL
Round trip time (ms):    11.89
==============================
Cannot connect to http://localhost:8080/healthcheck: [Errno 111] Connection refused
```
LMCache 作为组件**内嵌（In-Process 模式）**在 vLLM 的 8000 端口进程里运行

+ 对于 In-Process 模式ping  
```
lmcache ping engine --url http://localhost:8000
[2026-07-21 07:26:47,053] LMCache INFO:  torch_dev=<module 'torch.cuda' from '/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py'>, torch_device_type=cuda (__init__.py:63:lmcache)
[2026-07-21 07:26:47,065] LMCache INFO: CudaPinMemoryBackend: using torch cudart (pin_memory.py:89:lmcache.v1.platform.cuda.pin_memory)
[2026-07-21 07:26:47,153] LMCache INFO: Skipping backend lmcache.v1.platform.musa.ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:26:47,153] LMCache INFO: Skipping backend lmcache.xpu_ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:26:47,155] LMCache INFO: Using backend: lmcache.c_ops (__init__.py:132:lmcache)

 _     __  __    ____           _          
| |   |  \/  |  / ___|__ _  ___| |__   ___      LMCache v0.5.1 (g979719d7)
| |   | |\/| | | |   / _` |/ __| '_ \ / _ \     Website:  https://lmcache.ai/
| |___| |  | | | |__| (_| | (__| | | |  __/     Recipes:  https://docs.lmcache.ai/recipes
|_____|_|  |_|  \____\__,_|\___|_| |_|\___|     LinkedIn: https://www.linkedin.com/company/lmcache-lab
Set LMCACHE_DISABLE_BANNER=1 to hide this banner.

[2026-07-21 07:26:47,502] LMCache INFO: multi_layer_block_kv_transfer mode: ptr (base.py:94:lmcache.v1.multiprocess.transfer_context.base)
======== Ping Engine =========
Status:                     OK
Round trip time (ms):    22.24
==============================
```

> ## 内存




```
sudo docker logs vllm-sch2 | grep -E "Stored|Retrieved"
sudo docker stats vllm-sch2 --no-stream
sudo docker exec -it vllm-sch2 kill -HUP 1
```


```
netstat -pan | grep 8080
tcp        1      0 172.22.116.89:45866     34.236.19.149:8080      CLOSE_WAIT  149/VLLM::EngineCor 
lmcache kvcache clear --url http://localhost:8080
```


+ 清空LMCache cache

由于 LMCache 在 offload 到 CPU 内存时，数据会经过 Linux 系统的虚拟内存文件系统（Page Cache / Buffer）。我们可以直接在宿主机下达底层内核指令，强制 Linux 操作系统立刻回收所有未被激活的内存缓存页。

下达清理指令后：echo 3 | sudo tee /proc/sys/vm/drop_caches 成功将系统内存中所有处于非活跃状态的缓存页强行剥离。   

```
ubuntu@ubuntu:/pytorch/lmcache$ free -g
               total        used        free      shared  buff/cache   available
Mem:              62          21           2          16          55          40
Swap:              1           0           1
ubuntu@ubuntu:/pytorch/lmcache$ sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
3
ubuntu@ubuntu:/pytorch/lmcache$ free -g
               total        used        free      shared  buff/cache   available
Mem:              62          21          40          16          17          40
Swap:              1           0           1
ubuntu@ubuntu:/pytorch/lmcache$ 
```

执行清空后：系统的 buff/cache 瞬间从 55 GB 暴跌到了 17 GB！腾出来的空间直接转化为了 free（完全空闲内存），从 2 GB 暴涨到了 40 GB！



```
 python3 test_cache.py 
正在发送 第 1 次请求（冷启动） ...
💬 模型实际续写内容为: [ Deep learning is a subset of machine learning, which is essentially a neural network]
⏱️ 第 1 次请求耗时: 0.4890 秒

等待 3 秒中...
正在发送 第 2 次请求（预期命中 LMCache） ...
💬 模型实际续写内容为: [ Deep learning is a subset of machine learning, which is essentially a neural network]
⏱️ 第 2 次请求耗时: 0.4422 秒

🚀 === LMCache 节省了: 0.0468 秒 ===
```

竟然两次都是 0.4 秒左右！这个结果揭示了一个非常核心的技术细节：虽然 drop_caches 成功把系统的 Page Cache 洗掉了（buff/cache 从 55G 跌到 17G），但并没有影响到大模型的响应速度。这说明 LMCache 此时依然在高效运作，而且它的 KV 缓存数据并没有被真正洗掉！🔍 为什么会这样？从数据来看，第一次请求也从原来的 1.6 秒缩短到了 0.4 秒。这代表 vLLM 内部自带的底层前缀缓存（Chunked Prefill 或内部缓存机制）依然常驻在 GPU 显存或容器常驻进程的私有虚拟内存中。
因为：drop_caches 只能回收干净的、未被进程锁定的内核页缓存（Page Cache）。LMCache 的本地 CPU 模式是通过 shm（共享内存）或多进程常驻内存实现的。在你的 Docker 启动命令中，我们配置了 --ipc=host。这意味着容器内的 LMCache 进程直接在系统底层的匿名内存区锁定了这部分虚拟空间。只要 vLLM 容器（vllm-sch2）没有被杀死，这些虚拟内存块就属于“活跃、被锁定状态”，Linux 内核是绝对不会去强行擦除它们的。


> ##  独立的 LMCache Server


vLLM 推理服务：http://localhost:8000（发送大模型对话请求）LMCache     数据通道：lmserver://localhost:8080（vLLM 内部默默传输 KV Cache，使用 ZMQ 协议）
LMCache管理控制：http://localhost:8081（供你使用 curl 或 lmcache 命令行去 ping 或 clear 缓存）   


新增 -e LMCACHE_REMOTE_URL="http://localhost:8080" 配置，通过环境变量 LMCACHE_REMOTE_URL 告诉 vLLM 容器，LMCache Server 运行在哪里



```
sudo docker run -it --rm --net=host    --gpus=all     -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0 -d  -p 8000:8000 -p 8080:8080 -v /pytorch/models/:/models -v /pytorch:/workspace --shm-size=4g  --name  vllm-sch2    --entrypoint "/bin/bash"  lmcache/vllm-openai:latest 
```

```
lmcache server --host 0.0.0.0 --port 8080 --http-port 8081 --l1-size-gb 10 --eviction-policy LRU


lmcache ping kvcache --url http://localhost:8081
lmcache kvcache clear --url  http://localhost:8081
```

```
 curl -s -X POST http://localhost:8081/cache/clear
{"status":"ok","cleared":{"tier":"l1"}}
```


```
lmcache ping kvcache --url http://localhost:8081
[2026-07-21 07:55:29,985] LMCache INFO:  torch_dev=<module 'torch.cuda' from '/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py'>, torch_device_type=cuda (__init__.py:63:lmcache)
[2026-07-21 07:55:29,997] LMCache INFO: CudaPinMemoryBackend: using torch cudart (pin_memory.py:89:lmcache.v1.platform.cuda.pin_memory)
[2026-07-21 07:55:30,092] LMCache INFO: Skipping backend lmcache.v1.platform.musa.ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:55:30,092] LMCache INFO: Skipping backend lmcache.xpu_ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 07:55:30,093] LMCache INFO: Using backend: lmcache.c_ops (__init__.py:132:lmcache)

 _     __  __    ____           _          
| |   |  \/  |  / ___|__ _  ___| |__   ___      LMCache v0.5.1 (g979719d7)
| |   | |\/| | | |   / _` |/ __| '_ \ / _ \     Website:  https://lmcache.ai/
| |___| |  | | | |__| (_| | (__| | | |  __/     Recipes:  https://docs.lmcache.ai/recipes
|_____|_|  |_|  \____\__,_|\___|_| |_|\___|     LinkedIn: https://www.linkedin.com/company/lmcache-lab
Set LMCACHE_DISABLE_BANNER=1 to hide this banner.

[2026-07-21 07:55:30,443] LMCache INFO: multi_layer_block_kv_transfer mode: ptr (base.py:94:lmcache.v1.multiprocess.transfer_context.base)
======= Ping KV Cache ========
Status:                     OK
Round trip time (ms):    14.40
==============================
```

```
lmcache kvcache clear --url  http://localhost:8081
[2026-07-21 08:01:32,563] LMCache INFO:  torch_dev=<module 'torch.cuda' from '/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py'>, torch_device_type=cuda (__init__.py:63:lmcache)
[2026-07-21 08:01:32,574] LMCache INFO: CudaPinMemoryBackend: using torch cudart (pin_memory.py:89:lmcache.v1.platform.cuda.pin_memory)
[2026-07-21 08:01:32,662] LMCache INFO: Skipping backend lmcache.v1.platform.musa.ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 08:01:32,662] LMCache INFO: Skipping backend lmcache.xpu_ops: predicate returned False (__init__.py:114:lmcache)
[2026-07-21 08:01:32,663] LMCache INFO: Using backend: lmcache.c_ops (__init__.py:132:lmcache)

 _     __  __    ____           _          
| |   |  \/  |  / ___|__ _  ___| |__   ___      LMCache v0.5.1 (g979719d7)
| |   | |\/| | | |   / _` |/ __| '_ \ / _ \     Website:  https://lmcache.ai/
| |___| |  | | | |__| (_| | (__| | | |  __/     Recipes:  https://docs.lmcache.ai/recipes
|_____|_|  |_|  \____\__,_|\___|_| |_|\___|     LinkedIn: https://www.linkedin.com/company/lmcache-lab
Set LMCACHE_DISABLE_BANNER=1 to hide this banner.

[2026-07-21 08:01:33,009] LMCache INFO: multi_layer_block_kv_transfer mode: ptr (base.py:94:lmcache.v1.multiprocess.transfer_context.base)
================ KV Cache Clear ================
Status:                                       OK
================================================
```

```
export LMCACHE_USE_EXPERIMENTAL=True 
export LMCACHE_REMOTE_URL="lmserver://localhost:8080"  
vllm serve  --model /models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --chat-template /workspace/lmcache/my_mistral.jinja
```


+ 两个容器

```
sudo docker run -d \
    --name lmcache-server \
    --net=host \
    -p 8080:8080 \
    lmcache/standalone:latest \
    lmcache server --host 0.0.0.0 --port 8080 --l1-size-gb 10

```


```
sudo docker run -it --rm --net=host \
    --gpus=all \
    --ipc=host \
    --privileged \
    -d \
    -p 8000:8000 \
    -v /your/local/path:/models \
    --shm-size=32g \
    --name vllm-lmcache \
    -e LMCACHE_USE_EXPERIMENTAL=True \
    -e LMCACHE_REMOTE_URL="http://localhost:8080" \
    lmcache/vllm-openai:latest \
    --model /models/your-model \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


```
