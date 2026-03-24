

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