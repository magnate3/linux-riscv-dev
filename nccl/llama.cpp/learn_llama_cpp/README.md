 #   run

```
cmake -B build
cmake --build build
```
or
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```



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
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# cmake --build build
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

```
 ./build/05-passkey  -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
```

#  Principle:Ggml org Llama cpp Context Window Management

[Principle:Ggml org Llama cpp Context Window Management](https://leeroopedia.com/index.php/Principle:Ggml_org_Llama_cpp_Context_Window_Management) 


#  llama-simple-chat



```
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# /workspace/llama.cpp/build/bin/llama-simple-chat -m /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf -c  64
..........................................................
> where are you from
<think>
Okay, the user is asking "where are you from?" I need to respond to this question. First, I should acknowledge their question and explain that I'm a language model. It's important to clarify that I don't have a physical presence or location, but I can help with various tasks like answering questions, providing information, or assisting with other interactions. I should make sure my response is friendly and helpful, and explain that I don't have a physical location. Also, I should keep the tone conversational and not too formal.
</think>

I'm a language model, and I don't have a physical location. However, I can help you with questions, provide information, or assist with other tasks! Let me know how I can help!
> what can you do
<think>
Okay, the user just asked, "what can you do?" I need to respond appropriately. Let me start by confirming that I can assist with various tasks. I should mention my capabilities, like answering questions, providing information, or helping with other interactions. It's important to stay friendly and open to further questions. I should keep the tone positive and helpful. Maybe add something about being available
context size exceeded
root@centos7:/workspace/Let_us_learn_llama_cpp/proj2/cpp# 
``` 

`context size exceeded` 


```
where are you from
what can you do
where is japan
```