# 深入浅出 llama.cpp（二）：环境搭建与最小可行示例

## 零、前言

在上一章中，我简单介绍了 `llama.cpp` 的整体项目架构，而在这一章中，我将要将其导入我的项目进行集成，并且构建出一个最小可行程序，让它能够成功跑起来。

> 注：本章节适合对 CMake 项目配置、导入和构建不熟悉的人群，如果你已经对这一过程完全了解，可以直接跳过。

## 一、拉取 Git 子模块

为了便于集成，我刻意避免了直接的复制粘贴依赖的方式，而是采用 git 的子模块设计将其导入我的项目：

```bash
git submodule add https://github.com/ggml-org/llama.cpp.git
```

添加完成后，将 `.gitmodules` 文件推送到 Github，可以看到它并没有直接把整个 `llama.cpp` 传在仓库里，而是以链接的形式跳转到 `llama.cpp` 本身的 Github 远程仓库。

## 二、搭建 CMake 项目开发环境

在子模块添加完成后，使用 CMake 将其快捷地集成到我的项目中。

以下是一个最小的 `CMakeLists.txt` 程序：

```cmake
cmake_minimum_required(VERSION 3.18)
project(let_us_learn_llama_cpp VERSION 0.1.0)

# 1. 设置 C++ 标准 (llama.cpp 要求至少 C++11，建议使用 17 或更高)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 引入 llama.cpp 目录
add_subdirectory(llama.cpp)

# common: 官方示例中常用的工具类 (包含采样、加载模型等辅助函数)
# file(GLOB LLAMA_COMMON_SRC llama.cpp/common/*.cpp)

# 3. 添加可执行程序
add_executable(main_app
    src/main.cpp)

# 4. 链接库
# llama: 核心库 (llama.h 对应的实现)
target_link_libraries(main_app PRIVATE llama)

# 5. 如果需要包含 llama.cpp 内部的头文件路径
target_include_directories(main_app PRIVATE
    llama.cpp/include

    # llama.cpp/common
)
```

细心的读者会发现，我在这里将导入 `common/` 的部分刻意注释掉了。这主要是因为 `llama.cpp` 的 C++ 封装会导入如 `nlohmann/json` 等其他第三方依赖，而这并非使用 `llama.cpp` 所必须的。

如果后续真的有必要集成，我才会考虑将其其他依赖也跟着拉取下来，现在就先暂时使用 `llama.cpp` 官方的最小依赖开始使用即可。

## 三、开发最小可执行程序

先创建 `src/` 目录并创建一个文件 `main.cpp`：

```cpp
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
```

这样，就可以利用 CMake 将项目与 `llama.cpp` 集成起来了。

## 四、（可选）配置 VSCode 代码提示添加

在 VSCode 中，如果直接这么集成，是会产生错误波形曲线的，这主要是因为 VSCode 没能识别出 `llama.cpp` 的目录，为了保证它能够识别，需要打开 C++ 的配置并在 `.vscode/c_cpp_properties.json` 中添加一行：

```json
{
    "configurations": [
        {
            "includePath": [
                // ...
                "${workspaceFolder}/llama.cpp/**"
                // 这一行使 VSCode 自动寻找路径内的所有头文件并导入
            ],
            // ...
        },
        // ...
    ],
    // ...
}
```

添加完毕后，再次点开 `src/main.cpp`，就可以看到不会再提示错误了！

## 五、构建整个项目并运行

### 创建构建目录，避免污染原项目

```bash
mkdir build && cd build
```

### 开始构建（默认使用 CPU）

```bash
cmake ..
cmake --build . --config Release
```

### 运行输出

```bash
./build/main_app 
Llama.cpp backend initialized successfully!
Main GPU: 0
```

可以看到，输出的这两行正是 `llama.cpp` 可用的标志！

## 六、结语

这一章主要集中于如何搭建 C++ 项目的 CMake 开发环境。我通过子模块添加、配置 CMake、添加 IDE 提示和构建四个步骤成功地将 `llama.cpp` 集成到了项目之中。

在下一篇博客中，我将从 `examples/simple` 入手，拆解那几百行代码背后的逻辑，让 `llama.cpp` 真正能够在本地跑起来。