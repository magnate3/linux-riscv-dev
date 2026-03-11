

```
c++ -std=c++11 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src batch.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama 

```

```
 c++ -std=c++17 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src -I /workspace/llama.cpp/common  batch2.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -lggml-base  -L/workspace/llama.cpp/build/common -lcommon -lz

```

```
c++ -std=c++17 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include  -I  /workspace/llama.cpp/src -I /workspace/llama.cpp/common  batch3.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -lggml-base  -L/workspace/llama.cpp/build/common -lcommon -lz

```

```
cmake --build build -j64
```

