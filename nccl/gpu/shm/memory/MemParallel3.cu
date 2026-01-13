// 这个文件证明了并发的调用vmm-api并不能真正提高性能。这里的瓶颈在于GPU的调度能力，而非cpu。

#include <cstddef>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "error_handling.h"
#include "KernelForTest.cuh"
#include <chrono>
#include <memory>
#include <vector>
#include <cstddef>
#include <mutex>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#define NUM_KERNELS 30000000

// 定义一个全局互斥锁以确保线程安全（如果您的应用程序是多线程的）
std::mutex log_mutex;

/**
 * @brief 记录当前时间，并根据参数决定输出到控制台或文件
 * 
 * @param label 描述当前时间点的字符串
 * @param output 输出位置标识符：0 表示输出到控制台，1 表示输出到文件
 */
void log_time(const std::string& label, int output) {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为 time_t 以便格式化
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // 转换为本地时间结构
    std::tm* local_tm = std::localtime(&now_time_t);

    // 获取毫秒部分
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    long duration = value.count() % 1000;

    // 格式化时间字符串
    std::ostringstream oss;
    oss << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S") 
        << "." << std::setfill('0') << std::setw(3) << duration;

    std::string formatted_time = oss.str();

    // 构造日志消息
    std::ostringstream log_message;
    log_message << "[" << formatted_time << "] " << label;

    // 使用互斥锁确保线程安全
    std::lock_guard<std::mutex> guard(log_mutex);

    if (output == 0) {
        // 仅打印到控制台
        std::cout << log_message.str() << std::endl;
    }
    else if (output == 1) {
        // 仅打印到日志文件
        std::ofstream log_file("log.txt", std::ios_base::app); // 以追加模式打开
        if (log_file.is_open()) {
            log_file << log_message.str() << std::endl;
            log_file.close();
        } else {
            std::cerr << "无法打开 log.txt 文件进行写入。" << std::endl;
        }
    }
    else {
        // 无效的输出标识符
        std::cerr << "Invalid output identifier: " << output << ". Use 0 for console or 1 for file." << std::endl;
    }
}

CUresult setMemAccess(void* ptr, std::uint64_t size, int current_device_in = -1)
{
    int current_device = 0;
    // if(current_device == -1) {
        // DRIVE_CHECK(cuCtxGetDevice(&current_device));
    // }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = current_device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUresult result = cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1); 
    return result;
}

struct phy_block {
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle alloc_handle;
    phy_block(CUdeviceptr ptr_, CUmemGenericAllocationHandle alloc_handle_) : 
        ptr(ptr_), alloc_handle(alloc_handle_) {}
    ~phy_block(){
        DRIVE_CHECK(cuMemRelease(alloc_handle)); 
    }
};

struct Block{
    CUdeviceptr ptr;
    size_t size;
    std::vector<std::shared_ptr<phy_block>> phy_blocks;

    Block(CUdeviceptr ptr_, size_t size_, std::vector<std::shared_ptr<phy_block>>&& phy_blocks_):
        ptr(ptr_), size(size_), phy_blocks(std::move(phy_blocks_)) {}

    ~Block(){
        // 解除映射
        DRIVE_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        // 释放虚拟地址
        DRIVE_CHECK(cuMemAddressFree(CUdeviceptr(ptr), size));
    }
    
};

/* 获取内存分配所需的最小分配粒度 */
size_t getGranularitySize()
{
    static size_t granularity_ = 0;

    if(granularity_ == 0) {
        // int current_device;
        // DRIVE_CHECK(cuCtxGetDevice(&current_device));  // 获取当前上下文的设备ID

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 固定内存
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;  // 设备ID

        DRIVE_CHECK(cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    return granularity_;
}

std::shared_ptr<Block> getBlock(size_t size, size_t granularity){
    int num_phy = size / granularity;
    // 1. reserve 一段虚拟内存
    CUdeviceptr ptr;
    DRIVE_CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    // 2. 创建物理内存，并将其映射到虚拟内存
    std::vector<std::shared_ptr<phy_block>> tmp_phy_blocks;
    for(int i = 0; i < num_phy; i++){
        CUmemGenericAllocationHandle alloc_handle;      // 固定内存块句柄
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;      
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;   
        prop.location.id = 0;  // 设备ID
        DRIVE_CHECK(cuMemCreate(&alloc_handle, granularity, &prop, 0));   // 创建固定内存块
        std::shared_ptr<phy_block> phy = std::make_shared<phy_block> (ptr + i * granularity, alloc_handle);
        tmp_phy_blocks.push_back(std::move(phy));
        // 3. 映射到虚拟内存
        auto  block_ptr = (void*) ( ((char*)ptr) + (i * granularity));
        CUdeviceptr device_ptr = (CUdeviceptr)block_ptr;
        DRIVE_CHECK(cuMemMap(device_ptr, granularity,0ULL, alloc_handle, 0ULL));
        // 4. 设置权限
        CUresult err = setMemAccess((char*)(ptr) + i * granularity, granularity);
    }
    std::shared_ptr<Block> new_block = std::make_shared<Block>(ptr, size, std::move(tmp_phy_blocks));
    return new_block;
}

void mem_parallel1(){
    // 测试单线程的vmm-api申请与释放内存的代价
    auto t0 = std::chrono::steady_clock::now();
    log_time("Start testing vmm-api calls 1", 0);
    // 显存块大小：5GB，10GB，8GB，16GB，2GB，6GB
    std::vector<std::uint64_t> MemBlockSizes = {
    5ULL * 1024 * 1024 * 1024,
    10ULL * 1024 * 1024 * 1024,
    8ULL * 1024 * 1024 * 1024,
    };
    // 获取显存分配粒度
    size_t granularity = 0;
    granularity = getGranularitySize();
    // 存放Block的池
    std::vector<std::shared_ptr<Block>> pool;
    
    for (auto& MemBlockSize : MemBlockSizes){
        std::shared_ptr<Block> new_block = getBlock(MemBlockSize, granularity);
        pool.push_back(new_block);
        // std::cout << "Allocating " << MemBlockSize << " bytes of memory, " << new_block->phy_blocks.size() << " physical blocks" << std::endl;
    }
    // 释放所有Block
    for (auto& block : pool){
        block.reset();
    }
    auto t1 = std::chrono::steady_clock::now();
    log_time("End testing vmm-api calls 1", 0);
    using Ms = std::chrono::duration<double, std::milli>;
    Ms vmm_api_time = t1 - t0;
    std::cout << "Total time for vmm-api memory allocation and deallocation 1: " << vmm_api_time.count() << " ms" << std::endl;
}

void mem_parallel2(){
    // 测试单线程的vmm-api申请与释放内存的代价
    auto t0 = std::chrono::steady_clock::now();
    log_time("Start testing vmm-api calls 2", 0);
    // 显存块大小：5GB，10GB，8GB，16GB，2GB，6GB
    std::vector<std::uint64_t> MemBlockSizes = {
    16ULL * 1024 * 1024 * 1024,
    2ULL * 1024 * 1024 * 1024,
    6ULL * 1024 * 1024 * 1024
    };
    // 获取显存分配粒度
    size_t granularity = 0;
    granularity = getGranularitySize();
    // 存放Block的池
    std::vector<std::shared_ptr<Block>> pool;
    
    for (auto& MemBlockSize : MemBlockSizes){
        std::shared_ptr<Block> new_block = getBlock(MemBlockSize, granularity);
        pool.push_back(new_block);
        // std::cout << "Allocating " << MemBlockSize << " bytes of memory, " << new_block->phy_blocks.size() << " physical blocks" << std::endl;
    }
    // 释放所有Block
    for (auto& block : pool){
        block.reset();
    }
    auto t1 = std::chrono::steady_clock::now();
    log_time("End testing vmm-api calls 2", 0);
    using Ms = std::chrono::duration<double, std::milli>;
    Ms vmm_api_time = t1 - t0;
    std::cout << "Total time for vmm-api memory allocation and deallocation 2: " << vmm_api_time.count() << " ms" << std::endl;
}

int main(){
    // setup
    int N = 1024;
    float *d_A, *d_B, *d_C;
    // 分配设备端内存
    RUNTIME_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    RUNTIME_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    RUNTIME_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    // 定义 CUDA 线程块和网格大小
    dim3 blockDim(16, 16);  // 每个线程块包含 16x16 个线程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // 测试单线程计算NUN_KERNELS次核函数的代价
    using Ms = std::chrono::duration<double, std::milli>;
    Ms sigle_thread_time_for_100kernel_ = Ms(0);
    auto t0 = std::chrono::steady_clock::now();
    auto tt_pre = std::chrono::steady_clock::now();
    
    std::thread thread1(mem_parallel1);
    std::thread thread2(mem_parallel2);
    

    log_time("Start testing kernel calls", 0);
    for (int i = 0; i < NUM_KERNELS; i++){
        matrixMultiply<<<blockDim, gridDim>>>(d_A, d_B, d_C, N);
        // if (i%100 == 0 && i!= 0){
        //     cudaDeviceSynchronize();
        //     auto tt = std::chrono::steady_clock::now();
        //     std::cout << "Kernel " << i << " finished in " << std::chrono::duration_cast<Ms>(tt - tt_pre).count() << " ms" << std::endl;
        //     tt_pre = tt;
        // }
    }
    cudaDeviceSynchronize();
    log_time("End testing kernel calls", 0);

    auto t1 = std::chrono::steady_clock::now();
    sigle_thread_time_for_100kernel_ = t1 - t0;
    std::cout << "Total time for " << NUM_KERNELS << " kernel calls: " << sigle_thread_time_for_100kernel_.count() << " ms" << std::endl;
    
    thread1.join();
    thread2.join();
    
}
