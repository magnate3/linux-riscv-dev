#include "llama.h"
#include <iostream>

int main()
{
    // 1. 初始化后端 (必须步骤)
    llama_backend_init();

    // 2. 打印 llama.cpp 的版本信息或简单尝试创建一个参数结构体
    llama_model_params model_params = llama_model_default_params();

    std::cout << "Llama.cpp backend initialized successfully!" << std::endl;
    std::cout << "Main GPU: " << model_params.main_gpu << std::endl;

    // 3. 释放资源
    llama_backend_free();

    return 0;
}