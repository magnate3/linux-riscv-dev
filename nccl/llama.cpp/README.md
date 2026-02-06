

# qwen

```
 huggingface-cli download bartowski/Qwen_Qwen3-0.6B-GGUF --include "Qwen_Qwen3-0.6B-Q4_K_M.gguf" --local-dir ./models
```

#  llama.cpp

```
docker pull ghcr.io/ggml-org/llama.cpp:server
```

 

```
docker run -v /pytorch/qwen/models:/models -p 8080:8080 llama-cpp-connector:latest -m /models/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 --host 0.0.0.0 -n 512
```

> ## arm64


```
# remove armv9 builds
sed -i '/armv9/d' "ggml/src/CMakeLists.txt"

# start docker build
            docker buildx build \
              --push \
              --platform linux/arm64 \
              --build-arg TARGETARCH=arm64 \
              --target server \
              --file .devops/cpu.Dockerfile \
              --attest type=provenance,disabled=true \
              --tag  llama-server:latest \
              .
```


```
[root@centos7 llama.cpp]# docker run -v /pytorch/qwen/models:/models -p 8080:8080 9d2f72ddb130 -m /models/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 --host 0.0.0.0 -n 512
load_backend: loaded CPU backend from /app/libggml-cpu-armv8.0_1.so
warn: LLAMA_ARG_HOST environment variable is set, but will be overwritten by command line argument --host
main: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
build: 7941 (11fb327bf) with GNU 11.4.0 for Linux aarch64
system info: n_threads = 128, n_threads_batch = 128, total_threads = 128

system_info: n_threads = 128 (n_threads_batch = 128) / 128 | CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 

Running without SSL
```


> ## bash client
```
[root@centos7 llama.cpp]# bash ./tools/server/chat.sh
> hello qwen , how to play basketball
 Hello, Qwen. How can I help you with playing basketball?yes
 Let
```



> ## client

```
curl -X POST "http://localhost:8080/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "your-model-name",
           "messages": [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "Hello, who are you?"}
           ]
         }'
```


+ llama-cpp-api-client   
```
docker run --name llama-cli -itd   --net=host --cap-add=NET_ADMIN --privileged=true -v /root/pytorch/:/workspace python:3.11.14-slim-bookworm
```

```
# install from github
pip install git+https://github.com/ubergarm/llama-cpp-api-client
```
#  单精度评测

llama.cpp提供了perplexity可执行文件来验证模型的PPL精度，这里以wikitext语料来简单测试一下千问14B的性能（通义千问可能更偏向于中文，wikitext-2多数都是英文语料）。需要先下载解压wikitext-2到本地，这里解压到了llama.cpp/wikitext-2-raw/目录下，运行一下命令：   

```
./perplexity -m models/Qwen/14B/ggml-model-Q4_0.gguf -f wikitext-2-raw/wiki.test.raw
```

# dev


```
[root@centos7 docker]# docker run --rm  --name llama.cppdev --net=host    -itd    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0 -v /pytorch:/workspace -p 8088:8080  llamp.cpp:cpudev2 
WARNING: Published ports are discarded when using host network mode
6699a510e446d4f640b9e1e09d674dd0c60741bb4ed9ef522fe9bf617a45e44d
[root@centos7 docker]# docker exec -it llama.cppdev  bash
root@centos7:/#  
```

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CPU_ARM_ARCH=armv8-a
make --build build -j $(nproc)
```

```
root@centos7:/workspace/llama.cpp# ./build/bin/llama-server -m /workspace/qwen/models/  --host 0.0.0.0 --port 8080 -n 512
```