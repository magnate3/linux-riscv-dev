


```
cmake -S . -B build  -DLLAMA_CPP_DIR=/workspace/llama.cpp
 cmake --build build -j64
```


#  ./build/15-prefix 
```
 ./build/15-prefix 
```

```
[System] common prefix : 5 tokens
update: copying KV buffer: stream 0 to stream 1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  8, n_outputs =  512
[Stream 0]: 2
update: copying KV buffer: stream 0 to stream 2
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  8, n_outputs =  512
[Stream 1]:  and
[Stream 0]: 0
[Stream 1]:  what
[Stream 0]: 2
[Stream 1]:  is
[Stream 0]: 3
[Stream 1]:  the
[Stream 0]: -
[Stream 1]:  best
[Stream 0]: 2
[Stream 1]:  thing
[Stream 0]: 0
[Stream 1]:  about
[Stream 0]: 2
[Stream 1]:  the
[Stream 0]: 4
[Stream 1]:  l
```