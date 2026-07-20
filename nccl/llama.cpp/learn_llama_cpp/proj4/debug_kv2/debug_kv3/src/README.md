
````
cmake -DDEBUG_LLAMA_CPP=ON -S . -B build
#cmake -DDEBUG_KV_IN_CLASS=1 -S . -B build
cmake -DDEBUG_LLAMA_CPP=OFF -S . -B build
cmake --build build -j64
./build/simple-chat -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf  -c 2028
./build/main  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf   -p "Your very long prompt goes here..." -c 4096 --temp 0.7
```
