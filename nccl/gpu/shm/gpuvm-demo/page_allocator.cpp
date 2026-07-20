#include <stdexcept>
#include "page_allocator.h"
#include "common.h"


PageAllocator::PageAllocator(CUcontext& ctx, const size_t numPages) : numPages(numPages) {
    for (int i = 0; i < numPages; i++) {
        assertPush(i);
    }
    CU_ASSERT(cuMemAddressReserve(&pvmRange, numPages * PAGE_SIZE, ALIGNMENT, 0, 0));
    std::cout << "Reserved virtual memory range of 2MiB from 0x" << std::hex << pvmRange << " to " << pvmRange + numPages * PAGE_SIZE << " this is a device virtual address space" << std::endl;
}

void PageAllocator::assertPush(int value) {
    if (available.size() < numPages) {
        available.push(value);
    } else {
        throw std::runtime_error("Error: invalid number of pages.");
    }
}

const int PageAllocator::assertPop() {
    if (!available.empty()) {
        const int value = available.front();
        available.pop();
        return value;
    } else {
        throw std::runtime_error("Error: out of memory.");
    }
}

void PageAllocator::mapUvm(CUmemGenericAllocationHandle halloc) {
    const int vpn = pageTable[halloc];
    CU_ASSERT(cuMemMap(pvmRange + PAGE_SIZE * vpn, PAGE_SIZE, 0, halloc, 0));
    std::cout << "Mapping successful on vpn " << std::dec << vpn << std::endl;
}

void PageAllocator::unmapUvm(CUmemGenericAllocationHandle halloc) {
    const int vpn = pageTable[halloc];
    CU_ASSERT(cuMemUnmap(pvmRange + PAGE_SIZE * vpn, PAGE_SIZE));
    std::cout << "Unmap successful on vpn " << std::dec << vpn << std::endl;
}

CUmemGenericAllocationHandle PageAllocator::allocatePage() {
    CUmemGenericAllocationHandle halloc;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    const int vpn = assertPop();
    CU_ASSERT(cuMemCreate(&halloc, PAGE_SIZE, &prop, 0));
    pageTable[halloc] = vpn;
    std::cout << "Reserving 1 page of uvm space at 0x" << std::hex << halloc << " this is a uvm address." << std::endl;
    return halloc;
}

void PageAllocator::freePage(CUmemGenericAllocationHandle ptr) {
    assertPush(pageTable[ptr]);
    CU_ASSERT(cuMemRelease(ptr));
    std::cout << "Freed page at 0x" << std::hex << ptr << std::endl;
}

PageAllocator::~PageAllocator() {
    CU_ASSERT(cuMemAddressFree(pvmRange, numPages * PAGE_SIZE));
    std::cout << "Freed device virtual memory range." << std::endl;
}