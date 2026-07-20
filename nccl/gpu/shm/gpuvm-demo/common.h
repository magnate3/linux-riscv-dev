#include <iostream>
#include <string>
#include <cuda.h>

#define GRANULARITY (1024 * 1024 * 2)
#define PAGE_SIZE (1024 * 1024 * 2)
#define ALIGNMENT (1024 * 1024 * 2)

#define CU_ASSERT(x) _cu_assert(x, #x)

inline void _cu_assert(CUresult result, const char* func) {
    if (result != CUDA_SUCCESS) {
        const char* perrorName;
        const char* perrorString;
        cuGetErrorName(result, &perrorName);
        cuGetErrorString(result, &perrorString);
        std::string errorMessage = "Error encountered: " + std::string(func) + " " + std::string(perrorName) + " " + std::string(perrorString);
        throw std::runtime_error(errorMessage);
    }
}