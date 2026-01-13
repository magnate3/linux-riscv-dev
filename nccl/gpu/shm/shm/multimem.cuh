#ifndef MULTIMEM_CUH
#define MULTIMEM_CUH

/*
    Multimem package
    - KittenGang for threadpooling
    - mc_object struct and mc_X functions for MultiCast operations

    Full example usage in 34-multimem-api.cu
*/
#include "multi-gpu.cuh"
enum
{
CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132
}

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

/*
    CUDA-specific ThreadPool

    Example usage

    // Construction
    KittenGang gang(NUM_DEVICES);

    // Dispatch work to all threads (no need to set device)
    gang.execute([&](int dev_idx) {
        int dev;
        CUDACHECK(cudaGetDevice(&dev));
        if (dev != dev_idx) {
            fprintf(stderr, "Device mismatch: expected %d, got %d\n", dev_idx, dev);
            exit(1);
        }
    });
*/
class KittenGang {
public:
    KittenGang(const int *device_ids, const int num_devices);
    ~KittenGang();

    // Dispatches `task` to all threads, and waits for all threads to finish (using cv)
    void execute(std::function<void(int)> task);

private:
    // Condition indicators
    bool stop;
    std::vector<bool> task_available;
    int n_task_done;

    // Threadpool
    std::vector<std::thread> workers;
    
    // Main entry point for each thread
    void worker(int worker_id, int device_id);

    // Used to dispatch work to all threads
    std::function<void(int)> current_task;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};

KittenGang::KittenGang(const int *device_ids, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t i = 0; i < num_devices; ++i) {
        task_available.push_back(false);
        workers.emplace_back([this, i, device_ids] { worker(i, device_ids[i]); });
    }
}

KittenGang::~KittenGang() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void KittenGang::execute(std::function<void(int)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

void KittenGang::worker(int worker_id, int device_id) {
    CUDACHECK(cudaSetDevice(device_id));
    while (true) {
        std::function<void(int)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, worker_id] { return stop || task_available[worker_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[worker_id] = false;
        }
        task(worker_id);
        {
            std::lock_guard<std::mutex> lock(mutex); // adds 10 microseconds overhead
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}

template <typename T>
struct mc_object {
    T *raw_mem_pointer_;
    T *raw_mc_pointer_;

    CUdeviceptr mem_va_;
    CUdeviceptr mc_va_;

    CUmemGenericAllocationHandle mem_handle_;
    CUmemGenericAllocationHandle mc_handle_;

    size_t size_;
    int device_id_;
    int seq_id_; // sequential number for ptr indexing
};

void mc_check(int device_id) {
    // Check if device supports MultiCast Objects
    CUdevice dev;
    CUCHECK(cuDeviceGet(&dev, device_id));

    int device_supports_multicast;
    CUCHECK(cuDeviceGetAttribute(
        &device_supports_multicast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));

    if (!device_supports_multicast) {
        fprintf(stderr, "Device %d does not support Multicast Objects\n", device_id);
        exit(1);
    }
}

template <typename T>
void mc_malloc(mc_object<T> **mc_objects, const int *device_ids, const int num_devices, const size_t size) {

    if (num_devices <= 1) {
        fprintf(stderr, "mc_malloc: num_devices must be greater than 1\n");
        exit(1);
    }

    // cuInit must be called before any Driver API calls, and argument SBZ
    CUCHECK(cuInit(0));

    // Quary device for MultiCast support
    for (int i = 0; i < num_devices; ++i) {
        mc_check(device_ids[i]);
    }

    // Allocate mc objects
    *mc_objects = (mc_object<T> *)malloc(num_devices * sizeof(mc_object<T>));
    if (*mc_objects == NULL) {
        fprintf(stderr, "mc_malloc: failed to allocate memory for mc_objects\n");
        exit(1);
    }

    // Create MC handle props (once for all devices)
    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = num_devices;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
    mc_prop.flags = 0; // SBZ

    // Query for granularities
    size_t granularity = 0;
    CUCHECK(cuMulticastGetGranularity(
        &granularity, &mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (size % granularity != 0) {
        fprintf(stderr, "mc_malloc: size must be a multiple of granularity %lu\n", granularity);
        exit(1);
    }
    mc_prop.size = size;

    // Create MC handle (once for all devices)
    CUmemGenericAllocationHandle mc_handle;
    CUCHECK(cuMulticastCreate(&mc_handle, &mc_prop));

    // Add devices to MC handle
    for (int i = 0; i < num_devices; ++i) {
        CUdevice dev;
        CUCHECK(cuDeviceGet(&dev, device_ids[i]));
        CUCHECK(cuMulticastAddDevice(mc_handle, dev));
    }

    // Main loop
    for (int i = 0; i < num_devices; ++i) {
        CUDACHECK(cudaSetDevice(device_ids[i]));

        // Create memory handle prop
        CUmemAllocationProp mem_prop = {};
        mem_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; // use pinned memory
        mem_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        mem_prop.location.id = device_ids[i];
        mem_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

        // Query for recommended granularity
        size_t mem_granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity( // or CU_MEM_ALLOC_GRANULARITY_MINIMUM
            &mem_granularity, &mem_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        if (size % mem_granularity != 0) {
            fprintf(stderr, "Size must be a multiple of mem granularity %lu\n", granularity);
            exit(1);
        }

        // Allocate physical memory on the device
        CUCHECK(cuMemCreate(&(*mc_objects)[i].mem_handle_, size, &mem_prop, 0));

        // Bind the physical memory to the multicast handle
        CUCHECK(cuMulticastBindMem(mc_handle, 0, (*mc_objects)[i].mem_handle_, 0, size, 0));
        
        // Create VAs for the multicast handle and physical memory
        CUCHECK(cuMemAddressReserve(&(*mc_objects)[i].mc_va_, size, mem_granularity, 0, 0));
        CUCHECK(cuMemAddressReserve(&(*mc_objects)[i].mem_va_, size, mem_granularity, 0, 0));

        // Bind VAs to the multicast handle and physical memory
        CUCHECK(cuMemMap((*mc_objects)[i].mc_va_, size, 0, mc_handle, 0));
        CUCHECK(cuMemMap((*mc_objects)[i].mem_va_, size, 0, (*mc_objects)[i].mem_handle_, 0));

        // Remember to set access AFTER mapping
        CUmemAccessDesc desc[1];
        desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc[0].location.id = device_ids[i];
        desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        CUCHECK(cuMemSetAccess((*mc_objects)[i].mc_va_, size, desc, 1));
        CUCHECK(cuMemSetAccess((*mc_objects)[i].mem_va_, size, desc, 1));

        // Fill up the rest of mc_object for this device
        (*mc_objects)[i].raw_mem_pointer_ = (T *)(*mc_objects)[i].mem_va_;
        (*mc_objects)[i].raw_mc_pointer_ = (T *)(*mc_objects)[i].mc_va_;
        (*mc_objects)[i].mc_handle_ = mc_handle;
        (*mc_objects)[i].device_id_ = device_ids[i];
        (*mc_objects)[i].seq_id_ = i;
        (*mc_objects)[i].size_ = size;
    }
}

template <typename T>
void mc_free(mc_object<T> *mc_objects, const int num_devices) {

    // Free resources
    for (int i = 0; i < num_devices; ++i) {
        CUDACHECK(cudaSetDevice(mc_objects[i].device_id_));

        // Always free the memory in this order
        CUCHECK(cuMemUnmap(mc_objects[i].mem_va_, mc_objects[i].size_));
        CUCHECK(cuMemUnmap(mc_objects[i].mc_va_, mc_objects[i].size_));
        CUCHECK(cuMemAddressFree(mc_objects[i].mem_va_, mc_objects[i].size_));
        CUCHECK(cuMemAddressFree(mc_objects[i].mc_va_, mc_objects[i].size_));
        CUCHECK(cuMemRelease(mc_objects[i].mem_handle_));
    }

    free(mc_objects);
}

constexpr int WARPSIZE = 32;
constexpr int STRIDE = 32;
constexpr int MAX_VEC_SIZE = 16; // CUDA supports up to 16-byte vector ops
constexpr int THREADS_PER_BLOCK = 256;

__global__ void mc_kernel_all_reduce_sum_f32(float *data, const int N);
__global__ void mc_kernel_all_reduce_sum_bf16(__nv_bfloat16 *data, const int N);

void mc_all_reduce_sum_f32(KittenGang &gang, mc_object<float> *mc_objects, const int num_devices) {

    // User must ensure that all mc_objects have the same size and identical mc handle
    int N = mc_objects[0].size_ / sizeof(float);
    int N_per_dev = N / num_devices;
    constexpr int N_per_block = THREADS_PER_BLOCK * STRIDE * (MAX_VEC_SIZE / sizeof(float));

    gang.execute([&](int worker_id) {
        mc_kernel_all_reduce_sum_f32<<<(N_per_dev + N_per_block - 1) / N_per_block, THREADS_PER_BLOCK>>>
            (mc_objects[worker_id].raw_mc_pointer_ + N_per_dev * worker_id, N_per_dev);
        CUDACHECK(cudaDeviceSynchronize());
    });
}

void mc_all_reduce_sum_bf16(KittenGang &gang, mc_object<__nv_bfloat16> *mc_objects, const int num_devices) {

    // User must ensure that all mc_objects have the same size and identical mc handle
    int N = mc_objects[0].size_ / sizeof(__nv_bfloat16);
    int N_per_dev = N / num_devices;
    constexpr int N_per_block = THREADS_PER_BLOCK * STRIDE * (MAX_VEC_SIZE / sizeof(__nv_bfloat16));

    gang.execute([&](int worker_id) {
        mc_kernel_all_reduce_sum_bf16<<<(N_per_dev + N_per_block - 1) / N_per_block, THREADS_PER_BLOCK>>>
            (mc_objects[worker_id].raw_mc_pointer_ + N_per_dev * worker_id, N_per_dev);
        CUDACHECK(cudaDeviceSynchronize());
    });
}

__global__ void mc_kernel_all_reduce_sum_f32(float *data, const int N) {

    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) {
        printf("mc: only 1D grids and blocks should be passed in\n");
        return;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARPSIZE;
    int lane_id = threadIdx.x % WARPSIZE;

    constexpr int N_per_iter = MAX_VEC_SIZE / sizeof(float);
    constexpr int N_per_warp_per_iter = N_per_iter * WARPSIZE;
    constexpr int N_per_warp = STRIDE * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < STRIDE; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < N) {
            volatile float x, y, z, w;
            float *ptr = data + idx;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ptr) : "memory");
            asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
        }
        __syncthreads();
    }
}

__global__ void mc_kernel_all_reduce_sum_bf16(__nv_bfloat16 *data, const int N) {

    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) {
        printf("mc: only 1D grids and blocks should be passed in\n");
        return;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARPSIZE;
    int lane_id = threadIdx.x % WARPSIZE;

    constexpr int N_per_iter = MAX_VEC_SIZE / sizeof(__nv_bfloat16);
    constexpr int N_per_warp_per_iter = N_per_iter * WARPSIZE;
    constexpr int N_per_warp = STRIDE * N_per_warp_per_iter;
    int start_idx = N_per_warp * warp_id;

    for (int i = 0; i < STRIDE; ++i) {
        int idx = start_idx + i * N_per_warp_per_iter + lane_id * N_per_iter;
        if (idx < N) {
            volatile float x, y, z, w; // hacking type to hold 2 bfloat16s
            __nv_bfloat16 *ptr = data + idx;
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ptr) : "memory");
            asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
        }
        __syncthreads();
    }
}

#endif // MULTIMEM_CUH
