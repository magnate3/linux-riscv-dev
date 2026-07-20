#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <execinfo.h>

// 运行时库的错误处理
#define RUNTIME_CHECK(call)\
do {\
    cudaError_t error_code = call;\
    if (error_code != cudaSuccess) {\
        printf("CUDA runtime api error:\n");\
        printf("    File:       %s\n", __FILE__);\
        printf("    Line:       %d\n", __LINE__);\
        printf("    Error code: %d\n", error_code);\
        printf("    Error text: %s\n", cudaGetErrorString(error_code));\
        printf("    Stack trace:\n");\
        void* callstack[128];\
        int frames = backtrace(callstack, 128);\
        char** strs = backtrace_symbols(callstack, frames);\
        for (int i = 0; i < frames; ++i) {\
            printf("        %s\n", strs[i]);\
        }\
        free(strs);\
        exit(1);\
    }\
} while(0)  

// 低级驱动库的错误处理
#define DRIVE_CHECK(call)                                           \
do {                                                                \
    CUresult result = call;                                         \
    if (result != CUDA_SUCCESS) {                                   \
        const char *errMsg; cuGetErrorString(result, &errMsg);      \
        printf("CUDA drive api error:\n");                          \
        printf("    File:       %s\n", __FILE__);                   \
        printf("    Line:       %d\n", __LINE__);                   \
        printf("    Error code: %d\n", result);                     \
        printf("    Error text: %s\n", errMsg);                     \
        void* callstack[128];\
        int frames = backtrace(callstack, 128);\
        char** strs = backtrace_symbols(callstack, frames);\
        for (int i = 0; i < frames; ++i) {\
            printf("        %s\n", strs[i]);\
        }\
        free(strs);\
        exit(1);                                                    \
    }                                                               \
} while(0)  
