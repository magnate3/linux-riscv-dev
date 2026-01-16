#  Streaming Multiprocessors Deep Dive

Streaming Multiprocessors (SMs) are the heart of GPU architecture. Understanding SM design, resource allocation, and scheduling is essential for optimizing CUDA applications and achieving maximum performance.

**[Back to Overview](1_cuda_execution_model.md)** | **Previous: [Warp Execution Guide](3_warp_execution.md)** | **Next: [Synchronization Guide](../03_synchronization/1_synchronization_basics.md)**

---

##  **Table of Contents**

1. [ SM Architecture Overview](#-sm-architecture-overview)
2. [ Resource Management](#-resource-management)
3. [ Warp Scheduling in Detail](#-warp-scheduling-in-detail)
4. [ Memory Subsystem](#-memory-subsystem)
5. [ Instruction Pipeline](#-instruction-pipeline)
6. [ Performance Optimization](#-performance-optimization)
7. [ Profiling and Analysis](#-profiling-and-analysis)
8. [ Advanced Techniques](#-advanced-techniques)

---

##  **SM Architecture Overview**

Each Streaming Multiprocessor (SM) is a complete compute unit capable of executing multiple warps concurrently. Understanding its architecture is key to optimization.

###  **SM Component Breakdown**

| Component | Function | Quantity (varies by arch) | Impact on Performance |
|-----------|----------|---------------------------|----------------------|
| **CUDA Cores** | Integer/Float operations | 64-128 per SM | Compute throughput |
| **Tensor Cores** | AI/ML acceleration | 0-8 per SM | Deep learning performance |
| **Warp Schedulers** | Warp execution control | 2-4 per SM | Instruction throughput |
| **Register File** | Thread-private storage | 32K-65K registers | Occupancy limits |
| **Shared Memory** | Block-level cache | 48KB-164KB | Inter-thread communication |
| **L1 Cache** | Global memory cache | 16KB-128KB | Memory latency hiding |
| **Special Function Units** | Transcendental ops | 4-32 per SM | Math function performance |

###  **Visual SM Architecture**

```
Streaming Multiprocessor (SM)
 Warp Schedulers (2-4 units)
    Instruction Fetch & Decode
    Operand Collection
    Execution Unit Dispatch
 Execution Units
    CUDA Cores (FP32/INT32)
       Core[0-31]
       Core[32-63]
       Core[64-95] (if available)
       Core[96-127] (if available)
    Tensor Cores (FP16/BF16/INT8)
       Tensor[0]
       Tensor[1]
       ...
    Special Function Units (SFU)
       SIN/COS/LOG/EXP
       Square Root
       Reciprocal
    Load/Store Units (LSU)
        L1 Cache Interface
        Shared Memory Interface
        Global Memory Interface
 Memory Hierarchy
    Register File (32K-65K registers)
    Shared Memory (48KB-164KB)
    L1 Cache (16KB-128KB)
    Read-Only Cache (48KB)
 Inter-SM Communication
     L2 Cache Interface
     Memory Controller Interface
```

###  **Architecture Evolution**

#### **Kepler to Ampere Progression:**
```cpp
// Architecture comparison and optimization strategies
struct SMArchitecture {
    const char* name;
    int cuda_cores_per_sm;
    int tensor_cores_per_sm;
    int warp_schedulers;
    int max_warps_per_sm;
    int shared_memory_kb;
    int register_file_size;
    float base_clock_ghz;
};

// Architecture database
SMArchitecture architectures[] = {
    {"Kepler (GTX 780)",    192,  0, 4, 64,  48,  32768, 0.86f},
    {"Maxwell (GTX 980)",   128,  0, 4, 64,  96,  32768, 1.22f},
    {"Pascal (GTX 1080)",   128,  0, 4, 64,  96,  65536, 1.73f},
    {"Volta (V100)",        64,   8, 4, 64, 128,  65536, 1.53f},
    {"Turing (RTX 2080)",   64,   8, 4, 32,  96,  65536, 1.80f},
    {"Ampere (RTX 3080)",   128,  4, 4, 48, 128,  65536, 1.71f},
    {"Ada (RTX 4080)",      128,  4, 4, 48, 128,  65536, 2.21f}
};

__global__ void architecture_aware_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Query current architecture capabilities
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

    if (tid == 0) {
        printf("Running on Compute Capability %d.%d\n", major, minor);

        // Architecture-specific optimizations
        if (major >= 8) {  // Ampere+
            printf("Using Ampere+ optimizations\n");
        } else if (major >= 7) {  // Volta/Turing
            printf("Using Volta/Turing optimizations\n");
        } else {  // Pascal and earlier
            printf("Using legacy optimizations\n");
        }
    }
}
```

---

##  **Resource Management**

SMs have finite resources that must be shared among concurrent blocks and warps. Understanding resource allocation is crucial for maximizing occupancy.

###  **Resource Allocation Model**

#### **Primary Resources:**
```cpp
// Resource calculation utilities
struct SMResources {
    int total_warps;
    int total_threads;
    int total_registers;
    int total_shared_memory;
    int max_blocks;

    // Calculate theoretical occupancy
    float calculate_occupancy(int threads_per_block, int registers_per_thread,
                             int shared_mem_per_block) {
        // Warp limitation
        int warps_per_block = (threads_per_block + 31) / 32;
        int max_blocks_by_warps = total_warps / warps_per_block;

        // Register limitation
        int registers_per_block = threads_per_block * registers_per_thread;
        int max_blocks_by_registers = total_registers / registers_per_block;

        // Shared memory limitation
        int max_blocks_by_shared_mem = total_shared_memory / shared_mem_per_block;

        // Thread limitation
        int max_blocks_by_threads = total_threads / threads_per_block;

        // Block count limitation
        int max_blocks_by_limit = max_blocks;

        // Find bottleneck
        int actual_blocks = min({max_blocks_by_warps, max_blocks_by_registers,
                                max_blocks_by_shared_mem, max_blocks_by_threads,
                                max_blocks_by_limit});

        return (float)(actual_blocks * warps_per_block) / total_warps;
    }
};

// Device-specific resource initialization
SMResources get_sm_resources(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    return {
        .total_warps = prop.maxThreadsPerMultiProcessor / 32,
        .total_threads = prop.maxThreadsPerMultiProcessor,
        .total_registers = prop.regsPerMultiprocessor,
        .total_shared_memory = prop.sharedMemPerMultiprocessor,
        .max_blocks = prop.maxBlocksPerMultiProcessor
    };
}
```

#### **Register Pressure Analysis:**
```cpp
// Analyze register usage impact on occupancy
__global__ void __launch_bounds__(256, 4)  // 256 threads, min 4 blocks per SM
register_pressure_demo(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // High register usage version
        float r1 = data[tid];
        float r2 = r1 * 2.0f;
        float r3 = sin(r2);
        float r4 = cos(r2);
        float r5 = r3 * r4;
        float r6 = sqrt(r5);
        float r7 = log(r6 + 1.0f);
        float r8 = exp(r7);

        // Many intermediate registers reduce occupancy
        data[tid] = r8;
    }
}

// Optimized version with reduced register pressure
__global__ void register_optimized_demo(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Chain operations to reduce intermediate storage
        float value = data[tid];
        value = exp(log(sqrt(sin(value * 2.0f) * cos(value * 2.0f)) + 1.0f));
        data[tid] = value;
    }
}

// Host-side register usage analysis
void analyze_register_usage() {
    // Use CUDA occupancy API
    int min_grid_size, block_size;

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       register_pressure_demo, 0, 0);
    printf("High register kernel optimal block size: %d\n", block_size);

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       register_optimized_demo, 0, 0);
    printf("Optimized kernel optimal block size: %d\n", block_size);
}
```

###  **Shared Memory Banking**

#### **Bank Conflict Analysis:**
```cpp
// Demonstrate shared memory banking system
__global__ void shared_memory_banking_demo() {
    __shared__ float shared_data[256];  // 256 floats = 32 banks * 8 floats/bank

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    //  GOOD: No bank conflicts (stride 1)
    shared_data[tid] = tid;
    __syncthreads();
    float no_conflict = shared_data[tid];

    //  BAD: 2-way bank conflict (stride 2)
    __syncthreads();
    float two_way_conflict = shared_data[tid * 2 % 256];

    //  WORSE: 32-way bank conflict (same bank for all threads)
    __syncthreads();
    float worst_conflict = shared_data[lane_id * 32];

    //  GOOD: Broadcast (all threads read same address)
    __syncthreads();
    float broadcast = shared_data[0];

    if (tid == 0) {
        printf("Banking demo: no_conflict=%.1f, conflict=%.1f, worst=%.1f, broadcast=%.1f\n",
               no_conflict, two_way_conflict, worst_conflict, broadcast);
    }
}

// Optimized matrix transpose using proper banking
__global__ void optimized_transpose(float* input, float* output,
                                   int width, int height) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Load tile from input (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Store transposed tile (coalesced, no bank conflicts)
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

---

##  **Warp Scheduling in Detail**

The SM's warp schedulers are responsible for issuing instructions to execution units. Understanding their behavior helps optimize instruction-level parallelism.

###  **Scheduler Operation**

#### **Multi-Level Scheduling:**
```cpp
// Conceptual warp scheduler implementation
class WarpScheduler {
private:
    struct WarpContext {
        uint32_t warp_id;
        uint64_t program_counter;
        uint32_t active_mask;
        uint32_t predicate_mask;
        int stall_cycles;
        bool waiting_for_memory;
        bool waiting_for_barrier;
        int priority;
    };

    std::vector<WarpContext> active_warps;
    std::vector<WarpContext> stalled_warps;
    int current_cycle;

public:
    WarpContext* select_warp_to_execute() {
        // Priority 1: Ready warps with highest priority
        auto ready_warps = get_ready_warps();
        if (!ready_warps.empty()) {
            return select_by_priority(ready_warps);
        }

        // Priority 2: Warps becoming ready this cycle
        update_stall_counters();
        auto newly_ready = move_ready_warps_from_stalled();
        if (!newly_ready.empty()) {
            return newly_ready[0];
        }

        // Priority 3: Issue NOP if no warps ready
        return nullptr;
    }

    void issue_instruction(WarpContext* warp) {
        auto instruction = fetch_instruction(warp->program_counter);

        switch (instruction.type) {
            case ARITHMETIC:
                schedule_to_cuda_cores(warp, instruction);
                break;
            case MEMORY_LOAD:
                schedule_to_load_store_unit(warp, instruction);
                warp->waiting_for_memory = true;
                warp->stall_cycles = estimate_memory_latency(instruction);
                break;
            case SPECIAL_FUNCTION:
                schedule_to_sfu(warp, instruction);
                break;
            case SYNCHRONIZATION:
                handle_synchronization(warp, instruction);
                break;
        }

        warp->program_counter += instruction.size;
    }
};
```

#### **Instruction Throughput Optimization:**
```cpp
// Demonstrate instruction-level parallelism
__global__ void instruction_parallelism_demo(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Version 1: Sequential dependency chain (poor ILP)
        float x = a[tid];
        x = x + 1.0f;      // Depends on previous
        x = x * 2.0f;      // Depends on previous
        x = sin(x);        // Depends on previous
        c[tid] = x;

        // Version 2: Independent operations (good ILP)
        float x1 = a[tid];
        float x2 = b[tid];
        float y1 = x1 + 1.0f;    // Independent
        float y2 = x2 * 2.0f;    // Independent
        float z1 = sin(y1);      // Independent (after y1)
        float z2 = cos(y2);      // Independent (after y2)
        c[tid] = z1 + z2;        // Combines results
    }
}

// Optimal instruction scheduling for different execution units
__global__ void mixed_workload_optimized(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Interleave different instruction types for better scheduling
        float arith_result = value * 2.0f;           // CUDA cores
        float special_result = __sinf(value);        // Special function unit
        float memory_value = data[(tid + 1) % N];    // Load/store unit

        // More arithmetic while memory loads
        arith_result = fmaf(arith_result, 1.5f, 0.5f);  // CUDA cores

        // Combine results
        data[tid] = arith_result + special_result + memory_value;
    }
}
```

###  **Latency Hiding Strategies**

#### **Memory Latency Hiding:**
```cpp
// Demonstrate effective latency hiding through high occupancy
__global__ void latency_hiding_demo(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Use grid-stride loop to increase work per thread
    // while maintaining high occupancy
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // Memory load (high latency)
        float value = input[i];

        // Computation while other warps handle their memory loads
        float result = 0.0f;
        for (int j = 0; j < 10; j++) {
            result += sin(value + j * 0.1f) * cos(value - j * 0.1f);
        }

        // Memory store
        output[i] = result;
    }
}

// Compare with low-occupancy version
__global__ void __launch_bounds__(64, 1)  // Force low occupancy
low_occupancy_demo(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Same computation, but fewer warps to hide latency
        float value = input[tid];

        float result = 0.0f;
        for (int j = 0; j < 10; j++) {
            result += sin(value + j * 0.1f) * cos(value - j * 0.1f);
        }

        output[tid] = result;
    }
}

// Host code to compare performance
void compare_occupancy_impact() {
    // Setup data
    int N = 1024 * 1024;
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // High occupancy launch
    int high_occ_blocks = (N + 255) / 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    latency_hiding_demo<<<high_occ_blocks, 256>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float high_occ_time;
    cudaEventElapsedTime(&high_occ_time, start, stop);

    // Low occupancy launch
    int low_occ_blocks = (N + 63) / 64;
    cudaEventRecord(start);
    low_occupancy_demo<<<low_occ_blocks, 64>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float low_occ_time;
    cudaEventElapsedTime(&low_occ_time, start, stop);

    printf("High occupancy time: %.2f ms\n", high_occ_time);
    printf("Low occupancy time: %.2f ms\n", low_occ_time);
    printf("Speedup from high occupancy: %.2fx\n", low_occ_time / high_occ_time);

    cudaFree(d_input);
    cudaFree(d_output);
}
```

---

##  **Memory Subsystem**

The SM memory subsystem is complex, with multiple cache levels and specialized memory types. Understanding its behavior is crucial for performance.

###  **Cache Hierarchy**

#### **L1 Cache Behavior:**
```cpp
// Demonstrate L1 cache effects
__global__ void l1_cache_demo(float* data, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Test different access patterns
    if (tid < N / stride) {
        // Strided access pattern
        int index = tid * stride;

        // First access: cache miss
        float value1 = data[index];

        // Nearby access: likely cache hit if stride is small
        float value2 = data[index + 1];

        // Process values
        data[index] = value1 + value2;
    }
}

// Measure cache hit rates with different strides
void measure_cache_performance() {
    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = i;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Test different strides
    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int stride : strides) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int blocks = (N / stride + 255) / 256;

        cudaEventRecord(start);
        l1_cache_demo<<<blocks, 256>>>(d_data, N, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);

        float bandwidth = (2.0f * N / stride * sizeof(float)) / (time_ms * 1e-3) / 1e9;
        printf("Stride %d: %.2f ms, %.2f GB/s\n", stride, time_ms, bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    delete[] h_data;
    cudaFree(d_data);
}
```

#### **Shared Memory vs L1 Cache Configuration:**
```cpp
// Demonstrate configurable shared memory/L1 cache split
__global__ void shared_memory_heavy_kernel(float* global_data, int N) {
    // Large shared memory usage
    __shared__ float shared_buffer[48 * 1024 / sizeof(float)];  // 48KB

    int tid = threadIdx.x;
    int global_tid = tid + blockIdx.x * blockDim.x;

    // Load data into shared memory
    for (int i = tid; i < sizeof(shared_buffer) / sizeof(float); i += blockDim.x) {
        if (global_tid + i < N) {
            shared_buffer[i] = global_data[global_tid + i];
        }
    }

    __syncthreads();

    // Process data in shared memory
    if (tid < sizeof(shared_buffer) / sizeof(float)) {
        shared_buffer[tid] = shared_buffer[tid] * 2.0f + 1.0f;
    }

    __syncthreads();

    // Write back to global memory
    for (int i = tid; i < sizeof(shared_buffer) / sizeof(float); i += blockDim.x) {
        if (global_tid + i < N) {
            global_data[global_tid + i] = shared_buffer[i];
        }
    }
}

__global__ void l1_cache_heavy_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Multiple accesses to global memory - benefits from large L1 cache
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            int index = (tid + i) % N;  // Reuse some cache lines
            sum += data[index];
        }
        data[tid] = sum / 16.0f;
    }
}

// Host code to configure cache preference
void test_cache_configurations() {
    // Prefer shared memory (reduces L1 cache)
    cudaFuncSetCacheConfig(shared_memory_heavy_kernel, cudaFuncCachePreferShared);

    // Prefer L1 cache (reduces shared memory)
    cudaFuncSetCacheConfig(l1_cache_heavy_kernel, cudaFuncCachePreferL1);

    // Equal split
    cudaFuncSetCacheConfig(shared_memory_heavy_kernel, cudaFuncCachePreferEqual);
}
```

---

##  **Instruction Pipeline**

Understanding the SM instruction pipeline helps optimize for throughput and minimize stalls.

###  **Pipeline Stages**

#### **Instruction Flow:**
```cpp
// Demonstrate pipeline-aware programming
__global__ void pipeline_optimized_kernel(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Pipeline stage analysis:

        // 1. Fetch: Load values from global memory
        float val_a = a[tid];        // Load/Store Unit
        float val_b = b[tid];        // Load/Store Unit (can overlap)

        // 2. Decode & Schedule: While loads are in flight
        int computation_steps = 10;

        // 3. Execute: Arithmetic operations
        float result = 0.0f;
        for (int i = 0; i < computation_steps; i++) {
            // CUDA cores can execute while loads complete
            result += val_a * (i + 1) + val_b * (i + 2);
        }

        // 4. Memory: Store result
        c[tid] = result;             // Load/Store Unit
    }
}

// Pipeline stall demonstration
__global__ void pipeline_stall_demo(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Version 1: Sequential dependencies cause pipeline stalls
        float x = data[tid];
        x = sqrt(x);          // Special Function Unit
        x = x + 1.0f;         // CUDA Core (waits for sqrt)
        x = sin(x);           // Special Function Unit (waits for add)
        data[tid] = x;        // Load/Store Unit (waits for sin)

        // Version 2: Interleaved independent operations
        // (This would be in a separate kernel for comparison)
    }
}

__global__ void pipeline_optimized_demo(float* data1, float* data2, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Interleave independent operations to avoid stalls
        float x1 = data1[tid];
        float x2 = data2[tid];

        // Start both sqrt operations
        float y1 = sqrt(x1);      // SFU - first operation
        float y2 = sqrt(x2);      // SFU - second operation (can overlap)

        // Arithmetic while SFU is busy
        float z1 = x1 + 1.0f;     // CUDA Core
        float z2 = x2 + 2.0f;     // CUDA Core

        // More SFU operations
        float w1 = sin(y1);       // SFU
        float w2 = cos(y2);       // SFU (can overlap)

        // Combine results
        data1[tid] = w1 + z1;
        data2[tid] = w2 + z2;
    }
}
```

#### **Resource Contention Analysis:**
```cpp
// Analyze resource usage patterns
__global__ void resource_usage_analysis(float* data, int* usage_stats, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (tid < N) {
        // Track different execution unit usage
        unsigned long long start_time = clock64();

        // CUDA Core intensive
        float cuda_result = 0.0f;
        for (int i = 0; i < 100; i++) {
            cuda_result = fmaf(cuda_result, 1.01f, 0.1f);
        }

        unsigned long long cuda_time = clock64();

        // Special Function Unit intensive
        float sfu_result = 0.0f;
        for (int i = 0; i < 10; i++) {
            sfu_result += sin(data[tid] + i) + cos(data[tid] - i);
        }

        unsigned long long sfu_time = clock64();

        // Load/Store intensive
        float memory_result = 0.0f;
        for (int i = 0; i < 20; i++) {
            memory_result += data[(tid + i) % N];
        }

        unsigned long long memory_time = clock64();

        // Record timing statistics (only lane 0 per warp)
        if (lane_id == 0) {
            usage_stats[warp_id * 3 + 0] = cuda_time - start_time;
            usage_stats[warp_id * 3 + 1] = sfu_time - cuda_time;
            usage_stats[warp_id * 3 + 2] = memory_time - sfu_time;
        }

        // Store combined result
        data[tid] = cuda_result + sfu_result + memory_result;
    }
}
```

---

##  **Performance Optimization**

###  **Occupancy Optimization**

#### **Advanced Occupancy Tuning:**
```cpp
// Comprehensive occupancy optimization framework
class OccupancyOptimizer {
private:
    cudaDeviceProp device_props;

public:
    OccupancyOptimizer(int device_id = 0) {
        cudaGetDeviceProperties(&device_props, device_id);
    }

    struct OptimizationResult {
        int optimal_block_size;
        int optimal_grid_size;
        float achieved_occupancy;
        int limiting_factor;  // 0=warps, 1=registers, 2=shared_mem, 3=blocks
    };

    template<typename KernelFunc>
    OptimizationResult optimize_kernel(KernelFunc kernel, int problem_size,
                                      size_t shared_mem_per_block = 0) {
        OptimizationResult best_result = {0, 0, 0.0f, -1};

        // Test different block sizes
        for (int block_size = 32; block_size <= 1024; block_size += 32) {
            // Calculate occupancy for this configuration
            int max_active_blocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_active_blocks, kernel, block_size, shared_mem_per_block);

            if (max_active_blocks == 0) continue;

            float occupancy = (float)(max_active_blocks * block_size) /
                             device_props.maxThreadsPerMultiProcessor;

            // Calculate grid size needed for problem
            int grid_size = (problem_size + block_size - 1) / block_size;

            // Prefer higher occupancy, but also consider grid utilization
            float grid_efficiency = min(1.0f, (float)grid_size /
                                              (device_props.multiProcessorCount * max_active_blocks));

            float score = occupancy * grid_efficiency;

            if (score > best_result.achieved_occupancy) {
                best_result = {block_size, grid_size, occupancy,
                              analyze_limiting_factor(kernel, block_size, shared_mem_per_block)};
            }
        }

        return best_result;
    }

private:
    template<typename KernelFunc>
    int analyze_limiting_factor(KernelFunc kernel, int block_size, size_t shared_mem) {
        // Analyze what limits occupancy
        int warps_per_block = (block_size + 31) / 32;
        int max_warps = device_props.maxThreadsPerMultiProcessor / 32;
        int max_blocks_by_warps = max_warps / warps_per_block;

        // Register analysis (simplified - would need compiler info)
        int max_blocks_by_registers = device_props.regsPerMultiprocessor /
                                     (block_size * 32);  // Estimate

        int max_blocks_by_shared_mem = device_props.sharedMemPerMultiprocessor /
                                      (shared_mem + 1);  // Avoid division by zero

        int max_blocks_by_limit = device_props.maxBlocksPerMultiProcessor;

        // Find which resource is most limiting
        int bottleneck = min({max_blocks_by_warps, max_blocks_by_registers,
                             max_blocks_by_shared_mem, max_blocks_by_limit});

        if (bottleneck == max_blocks_by_warps) return 0;
        if (bottleneck == max_blocks_by_registers) return 1;
        if (bottleneck == max_blocks_by_shared_mem) return 2;
        return 3;
    }
};

// Usage example
void demonstrate_occupancy_optimization() {
    OccupancyOptimizer optimizer;

    auto result = optimizer.optimize_kernel(pipeline_optimized_kernel, 1024*1024);

    printf("Optimal configuration:\n");
    printf("Block size: %d\n", result.optimal_block_size);
    printf("Grid size: %d\n", result.optimal_grid_size);
    printf("Achieved occupancy: %.2f%%\n", result.achieved_occupancy * 100);

    const char* limiting_factors[] = {"Warps", "Registers", "Shared Memory", "Block Limit"};
    printf("Limited by: %s\n", limiting_factors[result.limiting_factor]);
}
```

###  **Memory Bandwidth Optimization**

#### **Bandwidth Saturation Techniques:**
```cpp
// Maximize memory bandwidth utilization
__global__ void bandwidth_optimized_copy(float4* src, float4* dst, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Use grid-stride loop for better memory access patterns
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // Vectorized memory access (16 bytes per transaction)
        float4 data = src[i];

        // Optional processing without breaking coalescing
        data.x = fmaf(data.x, 1.1f, 0.1f);
        data.y = fmaf(data.y, 1.1f, 0.1f);
        data.z = fmaf(data.z, 1.1f, 0.1f);
        data.w = fmaf(data.w, 1.1f, 0.1f);

        dst[i] = data;
    }
}

// Compare different memory access patterns
void benchmark_memory_patterns() {
    int N = 16 * 1024 * 1024;  // 64MB of float4 data
    float4 *d_src, *d_dst;

    cudaMalloc(&d_src, N * sizeof(float4));
    cudaMalloc(&d_dst, N * sizeof(float4));

    // Test different configurations
    struct Config {
        int block_size;
        int grid_size;
        const char* name;
    };

    Config configs[] = {
        {256, (N + 255) / 256, "Standard"},
        {512, (N + 511) / 512, "Large blocks"},
        {256, min(2048, (N + 255) / 256), "Limited grid"},
        {128, (N + 127) / 128, "Small blocks"}
    };

    for (auto& config : configs) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bandwidth_optimized_copy<<<config.grid_size, config.block_size>>>(d_src, d_dst, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);

        float bandwidth = (2.0f * N * sizeof(float4)) / (time_ms * 1e-3) / 1e9;
        printf("%s: %.2f ms, %.2f GB/s\n", config.name, time_ms, bandwidth);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}
```

---

##  **Profiling and Analysis**

###  **SM-Level Metrics**

#### **Custom Performance Profiler:**
```cpp
// SM utilization profiler
class SMProfiler {
private:
    struct SMMetrics {
        float sm_efficiency;
        float achieved_occupancy;
        float memory_throughput;
        float compute_throughput;
        int active_warps;
        int stalled_warps;
    };

public:
    template<typename KernelFunc>
    SMMetrics profile_kernel(KernelFunc kernel, dim3 grid, dim3 block,
                           void** args, size_t shared_mem = 0) {
        SMMetrics metrics = {};

        // Use CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Launch kernel with profiling
        cudaEventRecord(start);
        cudaLaunchKernel(kernel, grid, block, args, shared_mem, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        // Calculate theoretical metrics
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        int total_threads = grid.x * grid.y * grid.z * block.x * block.y * block.z;
        int total_warps = (total_threads + 31) / 32;

        // Theoretical occupancy
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, kernel, block.x * block.y * block.z, shared_mem);

        metrics.achieved_occupancy = (float)(max_active_blocks * block.x * block.y * block.z) /
                                    prop.maxThreadsPerMultiProcessor;

        // SM efficiency (simplified)
        int active_sms = min(grid.x * grid.y * grid.z, prop.multiProcessorCount);
        metrics.sm_efficiency = (float)active_sms / prop.multiProcessorCount;

        printf("Kernel Performance Profile:\n");
        printf("Execution time: %.2f ms\n", elapsed_ms);
        printf("SM efficiency: %.2f%%\n", metrics.sm_efficiency * 100);
        printf("Achieved occupancy: %.2f%%\n", metrics.achieved_occupancy * 100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return metrics;
    }
};
```

---

##  **Advanced Techniques**

###  **Multi-Kernel Optimization**

#### **Concurrent Kernel Execution:**
```cpp
// Demonstrate concurrent kernel execution for better SM utilization
__global__ void compute_intensive_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float result = data[tid];

        // CPU-intensive computation
        for (int i = 0; i < 1000; i++) {
            result = sin(result) * cos(result) + sqrt(fabs(result));
        }

        data[tid] = result;
    }
}

__global__ void memory_intensive_kernel(float* src, float* dst, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Memory-intensive pattern
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        dst[i] = src[i] * 2.0f;
    }
}

void demonstrate_concurrent_execution() {
    int N = 1024 * 1024;
    float *d_data1, *d_data2, *d_data3;

    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));
    cudaMalloc(&d_data3, N * sizeof(float));

    // Create streams for concurrent execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int blocks = (N + 255) / 256;

    // Launch kernels concurrently
    compute_intensive_kernel<<<blocks/2, 256, 0, stream1>>>(d_data1, N/2);
    memory_intensive_kernel<<<blocks/2, 256, 0, stream2>>>(d_data2, d_data3, N/2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("Concurrent kernel execution completed\n");

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);
}
```

###  **Dynamic Load Balancing**

#### **Adaptive Work Distribution:**
```cpp
// Dynamic load balancing across SMs
__global__ void dynamic_load_balance_kernel(float* data, int* work_queue,
                                           int* queue_size, int total_work) {
    __shared__ int local_work_items[256];
    __shared__ int local_queue_size;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Initialize shared memory
    if (tid == 0) {
        local_queue_size = 0;
    }
    __syncthreads();

    // Each block tries to get work items
    while (true) {
        int work_item = -1;

        // Thread 0 gets work from global queue
        if (tid == 0) {
            int current_size = atomicAdd(queue_size, -32);  // Try to get 32 items
            if (current_size > 0) {
                int start_idx = max(0, current_size - 32);
                int end_idx = current_size;

                // Copy work items to local queue
                for (int i = start_idx; i < end_idx && i < total_work; i++) {
                    if (local_queue_size < 256) {
                        local_work_items[local_queue_size++] = work_queue[i];
                    }
                }
            }
        }

        __syncthreads();

        // No more work available
        if (local_queue_size == 0) break;

        // Distribute local work among threads
        if (tid < local_queue_size) {
            work_item = local_work_items[tid];
        }

        // Process work item
        if (work_item >= 0) {
            // Variable amount of work per item
            int work_amount = work_item % 100 + 1;

            float result = 0.0f;
            for (int i = 0; i < work_amount; i++) {
                result += sin(work_item + i * 0.1f);
            }

            data[work_item] = result;
        }

        // Clear local queue for next iteration
        if (tid == 0) {
            local_queue_size = 0;
        }
        __syncthreads();
    }
}
```

---

##  **Key Takeaways**

1. ** Architecture Awareness**: Different GPU architectures have different SM configurations - optimize accordingly
2. ** Resource Management**: Balance registers, shared memory, and occupancy for optimal performance
3. ** Warp Scheduling**: High occupancy enables better latency hiding through efficient warp scheduling
4. ** Memory Hierarchy**: Understand cache behavior and memory access patterns for maximum bandwidth
5. ** Pipeline Optimization**: Minimize instruction dependencies and maximize instruction-level parallelism
6. ** Profile-Driven Optimization**: Use profiling tools to identify bottlenecks and optimize accordingly

##  **Related Guides**

- **Next Step**: [Synchronization Guide](../03_synchronization/1_synchronization_basics.md) - Master thread cooperation and barriers
- **Foundation**: [ Warp Execution Guide](3_warp_execution.md) - Warp-level programming
- **Basics**: [ Thread Hierarchy Guide](2_thread_hierarchy.md) - Thread organization fundamentals
- **Overview**: [ Execution Model Overview](1_cuda_execution_model.md) - Quick reference and navigation

---

** Pro Tip**: Think of SMs as individual processors - optimize for their specific capabilities and resource constraints to maximize GPU utilization!
