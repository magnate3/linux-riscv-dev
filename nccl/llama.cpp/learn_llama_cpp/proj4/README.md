

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

```
./build/simple-chat -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  -c 2028
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   -p "Your very long prompt goes here..." -c 4096 --temp 0.7
```
