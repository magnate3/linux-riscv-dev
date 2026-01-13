#include <iostream>
#include <cuda.h>
#include <vector>  // 包含vector头文件

static size_t create_size = 0;  // 成功分配的内存大小

// 错误检查宏
#define CHECK_CU(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errMsg; \
        cuGetErrorString(err, &errMsg); \
        std::cerr << "CUDA error: " << errMsg << " at line " << __LINE__ << std::endl; \
        std::cout << "最大分配内存: " << create_size / 1024 / 1024 << " MB." << "  /  " <<  create_size / 1024 / 1024 / 1024 << " GB." << std::endl; \
        exit(err); \
    } \
}

/* 获取内存分配所需的最小分配粒度 */
size_t getGranularitySize()
{
    static size_t granularity = -1;

    if(granularity == -1) {
        int current_device;
        CHECK_CU(cuCtxGetDevice(&current_device));  // 获取当前上下文的设备ID

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 固定内存
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = current_device;  // 设备ID

        CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    return granularity;
}

struct block {
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle alloc_handle;
    block(CUdeviceptr ptr_, CUmemGenericAllocationHandle alloc_handle_) : ptr(ptr_), alloc_handle(alloc_handle_) {}
};

int main() {
    // 初始化CUDA驱动API
    CHECK_CU(cuInit(0));

    // 获取设备数
    int deviceCount = 0;
    CHECK_CU(cuDeviceGetCount(&deviceCount));
    std::cout << "Found " << deviceCount << " devices." << std::endl;

    // 获取目标设备（GPU 2，索引从0开始，所以是设备2表示第三个设备）
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 2));  // 获取GPU 2

    // 创建CUDA上下文
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    // 在此上下文中执行CUDA任务...
    int current_device;
    cuCtxGetDevice(&current_device);
    std::cout << "Running CUDA operations on GPU" << current_device << " ." << std::endl;

    // ------------------------- 测试 vm-api ---------------------------------------------------------------------

    // 获取显存最小分配粒度
    size_t granularity = getGranularitySize();
    std::cout << "vm-api 显存分配粒度: " << granularity / 1024 / 1024 << " MB." << std::endl;
     
    // 显存信息列表
    std::vector<block> block_list;

    int print_flag = 0;
    // 循环获取
    while(true) {
        print_flag++;
        
        // 获取物理块
        CUmemGenericAllocationHandle alloc_handle;      // 固定内存块句柄
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;      
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;   
        prop.location.id = current_device;  // 设备ID
        CHECK_CU(cuMemCreate(&alloc_handle, granularity, &prop, 0));   // 创建固定内存块

        // reserve一段虚拟内存
        CUdeviceptr ptr;
        CHECK_CU(cuMemAddressReserve(&ptr, granularity, 0, 0, 0));

        // 映射
        CHECK_CU(cuMemMap(ptr, granularity, 0ULL, alloc_handle, 0ULL));

        create_size += granularity;

        // 保存信息，以供释放
        block_list.push_back(block(ptr, alloc_handle));

        //std::cout << "vm-api 成功分配了 " << create_size / 1024 / 1024 << " MB 内存." << std::endl;
        if(print_flag % (1024 * 4) == 0) {
            std::cout << "vm-api 已分配 " << create_size / 1024 / 1024 << " MB 内存" << "  /  " <<  create_size / 1024 / 1024 / 1024 << " GB 内存。" << std::endl;
        }
    }

    // ------------------------- 测试 vm-api ---------------------------------------------------------------------

    // 执行完任务后，销毁上下文
    CHECK_CU(cuCtxDestroy(context));

    return 0;
}
