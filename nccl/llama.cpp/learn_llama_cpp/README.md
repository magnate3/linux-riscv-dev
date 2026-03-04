 #   run

```
cmake -B build
cmake --build build
```
or



# proj1

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
```
 cmake --build build
```



#  fundamentals-llama.cpp
```
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# make simple-prompt-multi
g++ src/simple-prompt-multi.cpp -o simple-prompt-multi -std=c++17 -g -Wall -I/workspace/llama.cpp/include -I/workspace/llama.cpp/ggml/include -I/workspace/llama.cpp/common -I/workspace/llama.cpp/src -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -Wl,-rpath,/bin


```


```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# ./simple-prompt-multi 
```


# proj2

```
 export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
```


```
cmake .. -DLLAMA_CPP_DIR=/workspace/llama.cpp
cmake -S . -B build -DLLAMA_CPP_DIR=/workspace/llama.cpp
```

```
root@centos7:/workspace/Let_us_learn_llama_cpp/code-examples/cpp# ./build/01-simple-inference  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.ggu
```

> ##  llama_test_batch_decode
[llama_test_batch_decode](https://github.com/ggml-org/llama.cpp/discussions/17680)   
```
./build/04-llama_test_batch_decode  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
```
> ## passkey


```
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DLLAMA_SHARED -DLLAMA_USE_HTTPLIB -I/workspace/llama.cpp/common/. -I/workspace/llama.cpp/common/../vendor -I/workspace/llama.cpp/src/../include -I/workspace/llama.cpp/ggml/src/../include -O3 -DNDEBUG -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -o  passkey.o  -c 05-passkey.cpp
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp#  c++ -O3 -DNDEBUG passkey.o -o llama-passkey  /workspace/llama.cpp/build/common/libcommon.a /workspace/llama.cpp/build/bin/libllama.so.0.0.7941 /workspace/llama.cpp/build/bin/libggml.so.0.9.5  /workspace/llama.cpp/build/bin/libggml-cpu.so.0.9.5  /workspace/llama.cpp/build/bin/libggml-base.so.0.9.5  /workspace/llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a /usr/lib/aarch64-linux-gnu/libssl.so /usr/lib/aarch64-linux-gnu/libcrypto.so
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# 
```

#  Principle:Ggml org Llama cpp Context Window Management

[Principle:Ggml org Llama cpp Context Window Management](https://leeroopedia.com/index.php/Principle:Ggml_org_Llama_cpp_Context_Window_Management)   