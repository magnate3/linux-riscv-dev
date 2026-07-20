

```
 cmake -S . -B build -DLLAMA_CPP_DIR=/workspace/llama.cpp
```



#  测试

```
 LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);
```

```
[System] common prefix : 5 tokens
Radix Insert: Slot 0 的新路径已沉淀 (pos 0-5)
update: copying KV buffer: stream 0 to stream 1
```



```
g++  topk-prune.cpp -std=c++20 -o topk
g++  hot_prompt_trace.cpp -std=c++20 -o hot
g++  dat-radix-hier.cpp -std=c++20 -o hot
g++  dat-bitmask.cpp -std=c++20 -o hot
g++  token-remap-dat.cpp -std=c++20 -o hot
g++  dat-sketch.cpp -std=c++20 -o hot
```