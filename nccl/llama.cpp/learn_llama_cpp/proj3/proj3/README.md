# Granite Compiled Inference via llama.cpp
 - Marwan Yassini Chairi El Kamel
## Overview
This repository contains instructions and a simple script allowing for compiled inference of GGUF files utilising the API provided by llama.cpp

## Prerequisites
- CMake
- C++ Compiler (g++, clang)
- GGUF File for Granite (any other model works)
    https://huggingface.co/lmstudio-community/granite-3.2-8b-instruct-GGUF

## Installation
1. Clone this repository
   
2. Clone llama.cpp into the same folder
    ```bash
    git clone https://github.com/ggml-org/llama.cpp
    ```

3. Build the library
    ```bash
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release
    ```
4. Compile the program
    ```bash
        c++ -std=c++11 -I /workspace/llama.cpp/include -I  /workspace/llama.cpp/ggml/include main.cpp   -o test  -L/workspace/llama.cpp/build/bin -lllama
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/workspace/llama.cpp/build/bin:/workspace/llama.cpp/build/common"
    ```

5. Run the program with the following parameter
    ```bash
       ./test    /workspace/qwen/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf
    ```

