#pragma once
#include <unordered_map>
#include <queue>
#include <cuda.h>

class PageAllocator {
private:
    const size_t numPages;
    CUdeviceptr pvmRange;
    std::unordered_map<CUmemGenericAllocationHandle, int> pageTable;
    std::queue<int> available;
    void assertPush(const int value);
    const int assertPop();
    void mapUvm(CUmemGenericAllocationHandle halloc);
    void unmapUvm(CUmemGenericAllocationHandle halloc);

public:
    PageAllocator(CUcontext& ctx, const size_t numPages);
    CUmemGenericAllocationHandle allocatePage();
    void freePage(CUmemGenericAllocationHandle ptr);
    ~PageAllocator();
};