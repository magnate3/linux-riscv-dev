cmake_minimum_required(VERSION 3.18)
project(let_us_learn_llama_cpp VERSION 0.1.0)

# 1. 默认使用 Release 模式
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# 2. 设置 C++ 标准 (llama.cpp 要求至少 C++11，建议使用 17 或更高)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. CPU 指令集优化：开启高度优化并启用原生指令集 (AVX2/AVX512等)
if(MSVC)
    # Windows / MSVC 优化
    add_compile_options(/O2 /GL /arch:AVX2) # 开启 O2 优化、全局优化和 AVX2
    add_link_options(/LTCG) # 开启链接时代码生成
else()
    # Linux / macOS (GCC/Clang) 优化
    # -O3: 最高级优化
    # -march=native: 让编译器针对你当前的 CPU 指令集进行全速优化
    # -flto: 开启链接时优化，减小体积并提升跨文件调用性能
    add_compile_options(-O3 -march=native -mtune=native -flto=auto)
    add_link_options(-flto)
endif()

# 4. 引入 llama.cpp 目录并开启内部优化开关
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Enable native optimization" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common library" FORCE)
add_subdirectory(llama.cpp)

# 5. 添加可执行程序
file(GLOB SRC_SUBDIRS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/*
)

function(add_llama_example TARGET SRC)
    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} PRIVATE
        llama
        common
    )
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
    )
endfunction()

foreach(dir ${SRC_SUBDIRS})
    # 只处理目录
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        # 只匹配纯数字目录名
        if(dir MATCHES "^src/[0-9]+$")
            # 提取数字 N
            string(REGEX REPLACE "^src/([0-9]+)$" "\\1" N ${dir})

            set(TARGET_NAME main${N})
            set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/main.cpp)

            if(EXISTS ${MAIN_SRC})
                add_llama_example(${TARGET_NAME} ${MAIN_SRC})
            else()
                message(WARNING "Skip ${dir}: main.cpp not found")
            endif()
        endif()
    endif()
endforeach()
