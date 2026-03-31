




 2026 版推荐：使用 ggml_backend_sched
如果你在 llama.cpp 框架内，通常不需要手动管理同步。
ggml_backend_sched (调度器) 会自动处理：
+ 它会构建一个计算图。    
+ 自动识别哪些张量在 CPU，哪些在 GPU。    
+ 自动插入异步拷贝和同步指令。   
+ 验证方法：开启 LLAMA_LOG_LEVEL=debug，日志中会显示 kv_cache_move: cpu -> gpu 以及对应的同步等待耗时。 



```
cmake -DGGML_USE_CUDA=1 -S . -B build
 docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash
 
  docker run --rm --net=host    --gpus=all -itd    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch f200b8e983c8
```




```
/pytorch/GGML-Tutorial/src/gpu-cpu# ./build/gpu_cpu_memcp 
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 3090)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Core(TM) i9-14900)
ggml_backend_alloc_ctx_tensors_from_buft: all tensors in the context are already allocated
 cpu --> gpu success 
```  
cpu --> gpu success 
