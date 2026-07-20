#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string.h>
#include <nvtx3/nvToolsExt.h>
#if CUDART_VERSION < 12020
#define CU_MEM_LOCATION_TYPE_HOST_NUMA (CUmemLocationType)0x3
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID (CUdevice_attribute)134
#define CU_MEM_LOCATION_TYPE_HOST (CUmemLocationType)0x2
#endif

void dataCheck(char* c_data, char* d_data, size_t GB, cudaError_t err, int offset){
    //test for result after unmap
    err = cudaMemcpy(c_data+offset, d_data+offset , sizeof(char) * GB, cudaMemcpyDeviceToHost); // check last part
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        free(c_data);
        // cudaFree(d_data);
        return ;
    };
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "c_data[" << i << "] first is h_data = " << static_cast<int>(c_data[i]) << std::endl;
    };
    size_t GB_lo = 1 << 30;
    for (size_t i = sizeof(char) * GB_lo; i < (sizeof(char) * GB_lo + 10); ++i) {
        std::cout << "c_data[" << i << "] after half is d_data = " << static_cast<int>(c_data[i]) << std::endl;
    };

    memset(c_data, 0, sizeof(char) * 2*GB);

};

int main(){
    //allocate a contiguous 2GB buffer where 1 GB resides on GPU 0 and 1 GB resides on the host
    //API requires cuda 12.2, driver bug is fixed with cuda 12.4 / driver 550.54.14 (linux)
    bool checkflag = false;
    
    // constexpr size_t GB = 1 << 30;
    size_t GB = 1 << 30;
    GB /= 8;
    cudaSetDevice(0); //initialize cuda context

    // set param for allocation prop and granularity
    CUresult status = CUDA_SUCCESS;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularityDevice = 0;
    size_t granularityHost = 0;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    status = cuMemGetAllocationGranularity(&granularityDevice, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);

    prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id = 0;
    status = cuMemGetAllocationGranularity(&granularityHost, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);

    size_t granularity = std::max(granularityDevice, granularityHost);
    std::cout << "granularity is " << granularity << "b \n";

    
    const size_t allocationSize = 80*GB;// 10*GB
    std::cout << "total allocation is " << allocationSize / GB << " 1/8 GB \n";
    assert(GB % granularity == 0);
    assert(allocationSize % granularity == 0);
    std::cout << "1.1.init vmm range for device"  << std::endl;
    nvtxRangePushA("1.1.init vmm range for device");
    CUdeviceptr deviceptr = 0; // point addr for the vmm
   // reserve
    status = cuMemAddressReserve(&deviceptr, allocationSize, 0, 0, 0); // reserve allocsize
    assert(status == CUDA_SUCCESS);
    nvtxRangePop();

    nvtxRangePushA("1.2.allocate 10 times");
    // alloc device block physical
    CUmemGenericAllocationHandle allocationHandle;
    // prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; // host numa
    // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // add by 12.4
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    std::vector<CUmemAccessDesc> accessDescriptors(1);
    accessDescriptors[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptors[0].location.id = 0;
    accessDescriptors[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    for (unsigned int i = 0; i <allocationSize / GB; i++){
        size_t offset_alloc = i * GB;
        status = cuMemCreate(&allocationHandle, GB, &prop, 0); // physical at allocationHandle
        assert(status == CUDA_SUCCESS);
        // map at the end of deviceptr
        status = cuMemMap(deviceptr + offset_alloc, GB, 0, allocationHandle, 0);
        assert(status == CUDA_SUCCESS);
        // do not need handle after map
        status = cuMemRelease(allocationHandle);
        assert(status == CUDA_SUCCESS);
        // set access 
        //set access control such that the device chunk is only accessible from the device,
        //and the host chunk is also only accessible from the device  
        status = cuMemSetAccess(deviceptr + offset_alloc, GB, accessDescriptors.data(), 1);
        assert(status == CUDA_SUCCESS);
    }
    nvtxRangePop();
    std::cout << "2.allocate pure data from cudaMalloc"  << std::endl;
    cudaError_t err;
    nvtxRangePushA("2.allocate pure data from cudaMalloc");
    char* g_data[allocationSize / GB]; 
    
    for (unsigned int i = 0; i < allocationSize / GB; i++){
        // size_t offset_alloc = i * GB;
        err = cudaMalloc((void**)&g_data[i], sizeof(char) * GB);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed for block %d: %s\n", i, cudaGetErrorString(err));
            // Free previously allocated blocks in case of failure
            for (unsigned int j = 0; j < i; ++j) {
                cudaFree(g_data[j]);
            }
            return -1;
        }
    }
    nvtxRangePop();
    std::cout << "2.1.allocate data from host cudaMallocHost"  << std::endl;
    nvtxRangePushA("2.1.allocate data from host cudaMallocHost");
    // allocate data
    
    char* h_data; cudaMallocHost(&h_data, sizeof(char) * allocationSize);
    nvtxRangePop();
    for (size_t i = 0; i < sizeof(char) * allocationSize; ) {
        h_data[i] = static_cast<char>(i % 256); // Example modification
        h_data[i+1] = static_cast<char>(i+1 % 256);
        h_data[i+2] = static_cast<char>(i+2 % 256);
        i = i + sizeof(char) *GB;
    }

    // Print the first few values to verify
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << static_cast<int>(h_data[i]) << std::endl;
    };
    

    char* d_data = (char*)deviceptr; // addr for the vmm 
    // data for check on device
    char* c_data;
    c_data = (char *)malloc(sizeof(char) * allocationSize);
    if (c_data == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        cudaFree(c_data);
        return -1;
    }


    std::cout << "2.2.datacpy for init first device Range"  << std::endl;
    nvtxRangePushA("2.2.datacpy for init first device Range");
    // older drivers may report errors on the next lines
    cudaError_t rtstatus = cudaSuccess;
    for (unsigned int i = 0; i <allocationSize / GB; i++){
        size_t offset_alloc = i * GB;
        rtstatus = cudaMemcpy(d_data + offset_alloc, h_data + offset_alloc, GB, cudaMemcpyHostToDevice); // init device memory
        assert(rtstatus == cudaSuccess);
        //std::cout << "cudaMemcpy to device chunk: " << cudaGetErrorString(rtstatus) << "\n";
        cudaGetLastError();
    }
    nvtxRangePop();
    std::cout << "2.3.datacpy for PURE device"  << std::endl;
    nvtxRangePushA("2.3.datacpy for PURE device");
    for (unsigned int i = 0; i <allocationSize / GB; i++){
        size_t offset_alloc = i * GB;
        // older drivers may report errors on the next lines
        rtstatus = cudaMemcpy(g_data[i], h_data + offset_alloc, GB, cudaMemcpyHostToDevice); // init device memory
        //std::cout << "cudaMemcpy to device chunk: " << cudaGetErrorString(rtstatus) << "\n";
        assert(rtstatus == cudaSuccess);
        cudaGetLastError();
    }
    nvtxRangePop();


    std::cout << "---init c_data with d_data  with first part h_data"  << std::endl;
    // test for current data

    if(checkflag){
        std::cout << "d_data check"  << std::endl;
        dataCheck(c_data, d_data, 2*GB, err, 0);
        std::cout << "g_data check"  << std::endl;
        for (unsigned int i = 0; i < 2; i++)
            dataCheck(c_data, g_data[i], GB, err, 0); // half
    }
    

    std::cout << "4.free unmap Range"  << std::endl;
    nvtxRangePushA("4.free unmap Range");
    std::cout << "---unmap first part of vmm"  << std::endl;
    // test for unmap
    for (unsigned int i = 0; i < allocationSize / GB; i++){
        size_t offset_alloc = i * GB;
        CUresult result = cuMemUnmap(deviceptr + offset_alloc, sizeof(char) *GB); // unmap first part
        if (result != CUDA_SUCCESS) {
            const char* errMsg;
            cuGetErrorString(result, &errMsg);
            std::cerr << "cuMemUnmap failed: " << errMsg << std::endl;
            // std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(result) << std::endl;
            return -1;
        }
    }
    nvtxRangePop();
    std::cout << "4.1.free g_data Range"  << std::endl;
    nvtxRangePushA("4.1.free g_data Range");
    for (unsigned int i = 0; i <allocationSize / GB; i++){
        err = cudaFree(g_data[i]);
        if (err != cudaSuccess) {
            printf("cudaFree failed: %s\n", cudaGetErrorString(err));
            return -1;
        
        }
    }
    nvtxRangePop();
   std::cout << "---remap with the same allocationHandle with the first part"  << std::endl;
        nvtxRangePushA("5.remap device Range");
        // remap
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; // update to device
        prop.location.id = 0;
        for (unsigned int i = 0; i < allocationSize / GB; i++){
            size_t offset_alloc = i * GB;
            status = cuMemCreate(&allocationHandle, GB, &prop, 0); //  new physical at allocationHandle
            assert(status == CUDA_SUCCESS);
            // map at the end of deviceptr
            status = cuMemMap(deviceptr + offset_alloc, GB, 0, allocationHandle, 0);
            if (status != CUDA_SUCCESS) {
                const char* errMsg;
                cuGetErrorString(status, &errMsg);
                std::cerr << "cuMemUnmap failed: " << errMsg << std::endl;
                // std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(result) << std::endl;
                return -1;
            }

            status = cuMemRelease(allocationHandle);
            assert(status == CUDA_SUCCESS);

            status = cuMemSetAccess(deviceptr + offset_alloc, GB, accessDescriptors.data(), 1);
            assert(status == CUDA_SUCCESS);
        // status = cuMemSetAccess(deviceptr + GB, GB, accessDescriptors.data(), 1);
        // assert(status == CUDA_SUCCESS);
        }
        nvtxRangePop();

        char* g_data2[allocationSize / GB]; 
        nvtxRangePushA("5.1. pure new device range");
        for (unsigned int i = 0; i < allocationSize / GB; i++){
            //size_t offset_alloc = i * GB;
            err = cudaMalloc((void**)&g_data2[i], sizeof(char) * GB);
            if (err != cudaSuccess) {
                printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
                return -1;
            }
        }
        nvtxRangePop();

        // if(checkflag){
        //     std::cout << "---new physical block"  << std::endl;
        //     std::cout << "d_data check"  << std::endl;
        //     dataCheck(c_data, d_data, GB, err, 0);
        //     std::cout << "g_data check"  << std::endl;
        //     for (unsigned int i = 0; i <2; i++)
        //         dataCheck(c_data, g_data[i], GB, err, 0);
        // }

        std::cout << "---copy  from second to new first after remap"  << std::endl;
        nvtxRangePushA("6.host to new vmm device");
        for (unsigned int i = 0; i < allocationSize / GB; i++){
            size_t offset_alloc = i * GB;
            err = cudaMemcpy(d_data + offset_alloc, h_data + offset_alloc, GB, cudaMemcpyHostToDevice); // first to second
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy memory from device to device: " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_data);
                return -1;
            }
        }
        nvtxRangePop();

        nvtxRangePushA("6.1.host to new pure device");
        for (unsigned int i = 0; i < allocationSize / GB; i++){
            size_t offset_alloc = i * GB;
            err = cudaMemcpy(g_data2[i], h_data + offset_alloc, GB, cudaMemcpyHostToDevice); // first to second
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy memory from device to device: " << cudaGetErrorString(err) << std::endl;
                cudaFree(g_data2[i]);
                return -1;
            }
        }
        nvtxRangePop();
        
        // test for current data
        if(checkflag){
            std::cout << "---get update data  with host - device"  << std::endl;
            std::cout << "d_data check"  << std::endl;
            dataCheck(c_data, d_data, 2*GB, err, 0);
            std::cout << "g_data2 check"  << std::endl;
            for (unsigned int i = 0; i < 2; i++)
                dataCheck(c_data, g_data2[i], GB, err, 0);
        }

        std::cout << "---end"  << std::endl;

        for (unsigned int i = 0; i < allocationSize / GB; ++i) {
            cudaError_t err = cudaFree(g_data2[i]);
            if (err != cudaSuccess) {
                printf("cudaFree failed for block %d: %s\n", i, cudaGetErrorString(err));
                return -1;
            }
        }

        nvtxRangePushA("4.free unmap Range");
        for (unsigned int i = 0; i < allocationSize / GB; i++){
            size_t offset_alloc = i * GB;
            status = cuMemUnmap(deviceptr + offset_alloc, sizeof(char) *GB);
            assert(status == CUDA_SUCCESS);
        }
        nvtxRangePop();


    assert(status == CUDA_SUCCESS);
    status = cuMemAddressFree(deviceptr, allocationSize);
    assert(status == CUDA_SUCCESS);
    err = cudaFreeHost(h_data);
    assert(err == cudaSuccess);
}
