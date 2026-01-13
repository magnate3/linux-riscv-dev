#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include "common.h"
#include "page_allocator.h"


// nvcc gpuvm.cpp -o gpuvm -lcuda
int main(int argc, char** argv) {
    try {
        CUcontext ctx;
        CUdevice device;
        cuInit(0);
        CU_ASSERT(cuDeviceGet(&device, 0));
        CU_ASSERT(cuCtxCreate(&ctx, 0, device));
        CU_ASSERT(cuCtxSetCurrent(ctx));

        const size_t numPages = 64;
        PageAllocator* allocator = new PageAllocator(ctx, numPages);
        
        CUmemGenericAllocationHandle h1 = allocator->allocatePage();
        allocator->freePage(h1);

        delete allocator;
        CU_ASSERT(cuCtxDestroy(ctx));
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
