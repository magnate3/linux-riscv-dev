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




#  fundamentals-llama.cpp
```
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# make simple-prompt-multi
g++ src/simple-prompt-multi.cpp -o simple-prompt-multi -std=c++17 -g -Wall -I/workspace/llama.cpp/include -I/workspace/llama.cpp/ggml/include -I/workspace/llama.cpp/common -I/workspace/llama.cpp/src -L/workspace/llama.cpp/build/bin -lllama -lggml -lggml-cpu -Wl,-rpath,/bin


```


```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
root@centos7:/workspace/Let_us_learn_llama_cpp/fundamentals-llama.cpp# ./simple-prompt-multi 
```