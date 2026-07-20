# usage

## build llama.cpp with cuda backend

### GPU

```bash
cd ../llama.cpp
patch -p1 < ${ROOT_DIR}/gpu-profile.patch
cmake -B cuda_build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DLLAMA_CURL=OFF -DGGML_CPU_REPACK=OFF #-DCMAKE_BUILD_TYPE=Debug
cmake --build cuda_build --config Release -j $(nproc)
```


### CPU

```bash
cd ../llama.cpp
cmake -B cpu_build -DGGML_CUDA=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DLLAMA_CURL=OFF -DGGML_CPU_REPACK=OFF #-DCMAKE_BUILD_TYPE=Debug
cmake --build cpu_build --config Release -j $(nproc)
```

## build and run profiling test

### GPU

```bash
make clean
make
# make layer-gpu-bench
# FLAGS="-g" make
LD_LIBRARY_PATH=../llama.cpp/cuda_build/bin EPSILON=0.032 ./layer-gpu-bench -m <your model path>/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf -l blk.0.attn_q.weight -p 0 -n 128 -t 1
```

| model                          |         size |       params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | -----------: | -----------: | ---------- | --: | ------: | --------------: | -------------------: |
| llama 8B Q2_K - Medium         |  2513.80 MiB |    8030.26 M | CUDA       |  99 |       1 |           tg128 |                179.56 |
| blk.0.attn_q.weight            |     5.25 MiB |      16.78 M | CUDA       |  99 |       1 |           tg128 |              59445.08 |


### CPU

```bash
make clean
FLAGS="-DNOGPU -O3" make layer-cpu-bench
# FLAGS="-DNOGPU -g" make layer-cpu-bench
LD_LIBRARY_PATH=../llama.cpp/cpu_build/bin ./layer-cpu-bench -m <your model path>/Meta-Llama-3.1-8B-Instruct-Q2_K.gguf -l blk.0.attn_q.weight -p 0 -n 64 -t 8 -ngl 0 --no-warmup
```

| model                          |         size |       params | backend    | threads |            test |                  t/s |
| ------------------------------ | -----------: | -----------: | ---------- | ------: | --------------: | -------------------: |
| llama 8B Q2_K - Medium         |  2513.80 MiB |    8030.26 M | CPU        |       8 |            tg64 |                 23.86 |
| blk.0.attn_q.weight            |     5.25 MiB |      16.78 M | CPU        |       8 |            tg64 |              11658.04 |