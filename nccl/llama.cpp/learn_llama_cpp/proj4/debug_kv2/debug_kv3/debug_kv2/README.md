
```
cmake -DDEBUG_LLAMA_CPP=ON -S . -B build
#cmake -DDEBUG_KV_IN_CLASS=1 -S . -B build
cmake -DDEBUG_LLAMA_CPP=OFF -S . -B build
cmake --build build -j64
```
