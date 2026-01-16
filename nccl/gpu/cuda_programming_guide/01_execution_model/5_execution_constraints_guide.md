#  Execution Constraints Guide

Understanding CUDA's execution constraints is crucial for writing efficient and correct GPU applications. These constraints arise from hardware limitations, resource management, and architectural design choices.

**[Back to Overview](1_cuda_execution_model.md)** | **Previous: [Synchronization Guide](../03_synchronization/1_synchronization_basics.md)** | **Next: [Profiling Overview](../05_performance_profiling/1_profiling_overview.md)**

---

##  **Table of Contents**

1. [ Hardware Resource Constraints](#-hardware-resource-constraints)
2. [ Memory Limitations](#-memory-limitations)
3. [ Thread and Block Constraints](#-thread-and-block-constraints)
4. [⏱ Execution Time Limits](#-execution-time-limits)
5. [ Divergence Penalties](#-divergence-penalties)
6. [ Constraint Workarounds](#-constraint-workarounds)
7. [ Resource Optimization](#-resource-optimization)
8. [ Constraint Analysis Tools](#-constraint-analysis-tools)

---

##  **Hardware Resource Constraints**

GPU hardware imposes fundamental limits on execution resources that directly impact kernel design and performance.

###  **Compute Capability Limits**

#### **Per-Architecture Resource Limits:**
```cpp
// Query and display device constraints
void query_device_constraints() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

        // Critical resource constraints
        printf("\n Hardware Constraints:\n");
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions: [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        printf("\n Memory Constraints:\n");
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);

        printf("\n SM Constraints:\n");
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size: %d\n", prop.warpSize);

        printf("\n Performance Constraints:\n");
        printf("  Memory Clock Rate: %.1f MHz\n", prop.memoryClockRate / 1000.0f);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

        printf("\n");
    }
}
```

#### **Architecture-Specific Constraints:**
```cpp
// Architecture-specific constraint handling
__device__ void architecture_specific_constraints() {
    // Kepler (3.x): Basic constraints
    #if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 500
        // Max 1024 threads per block
        // 48KB shared memory per block
        // Limited dynamic parallelism
    #endif

    // Maxwell (5.x): Enhanced features
    #if __CUDA_ARCH__ >= 500 && __CUDA_ARCH__ < 600
        // Improved shared memory banking
        // Better warp divergence handling
        // Enhanced atomic operations
    #endif

    // Pascal (6.x): Unified memory improvements
    #if __CUDA_ARCH__ >= 600 && __CUDA_ARCH__ < 700
        // 64KB shared memory per block (6.0+)
        // Improved memory coalescing
        // Enhanced compute preemption
    #endif

    // Volta/Turing (7.x): Independent thread scheduling
    #if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 800
        // Independent thread scheduling changes synchronization
        // Tensor Core support (7.0+)
        // Enhanced L1 cache
        #warning "Volta+ requires explicit warp synchronization!"
    #endif

    // Ampere (8.x): Multi-Instance GPU
    #if __CUDA_ARCH__ >= 800
        // MIG support
        // Enhanced memory hierarchy
        // Improved async copy operations
    #endif
}

// Compile-time constraint checking
template<int BLOCK_SIZE, int SHARED_MEM_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, 2) // Suggest 2 blocks per SM
constraint_aware_kernel(float* data, int N) {
    // Static assertions for compile-time checks
    static_assert(BLOCK_SIZE <= 1024, "Block size exceeds maximum");
    static_assert(BLOCK_SIZE % 32 == 0, "Block size must be multiple of warp size");
    static_assert(SHARED_MEM_SIZE <= 48 * 1024, "Shared memory exceeds limit");

    __shared__ float shared_data[SHARED_MEM_SIZE / sizeof(float)];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Kernel implementation...
        shared_data[threadIdx.x] = data[tid];
        __syncthreads();

        data[tid] = shared_data[threadIdx.x] * 2.0f;
    }
}
```

###  **Register Pressure Constraints**

#### **Register Usage Analysis:**
```cpp
// Analyze register usage impact on occupancy
__global__ void high_register_usage_kernel(float* data, int N) {
    // High register usage - reduces occupancy
    float reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    float reg9, reg10, reg11, reg12, reg13, reg14, reg15, reg16;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Use many registers
        reg1 = data[tid] + 1.0f;
        reg2 = reg1 * 2.0f;
        reg3 = sin(reg2);
        reg4 = cos(reg2);
        reg5 = reg3 + reg4;
        reg6 = sqrt(reg5);
        reg7 = log(reg6 + 1.0f);
        reg8 = exp(reg7);

        reg9 = reg1 + reg8;
        reg10 = reg2 * reg9;
        reg11 = reg3 - reg10;
        reg12 = reg4 + reg11;
        reg13 = reg5 * reg12;
        reg14 = reg6 / (reg13 + 1.0f);
        reg15 = reg7 + reg14;
        reg16 = reg8 - reg15;

        data[tid] = reg16;
    }
}

// Optimized version with reduced register pressure
__global__ void __launch_bounds__(256, 4) // More blocks per SM
low_register_usage_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Reuse registers to reduce pressure
        float temp = data[tid] + 1.0f;
        temp *= 2.0f;

        float trig_result = sin(temp) + cos(temp);
        temp = sqrt(trig_result);
        temp = log(temp + 1.0f);
        temp = exp(temp);

        data[tid] = temp;
    }
}

// Runtime register usage query
void check_register_usage() {
    // Use CUDA profiler or nvprof to check register usage:
    // nvprof --metrics achieved_occupancy,registers_per_thread ./program

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, high_register_usage_kernel);
    printf("High register kernel uses %d registers\n", attr.numRegs);

    cudaFuncGetAttributes(&attr, low_register_usage_kernel);
    printf("Low register kernel uses %d registers\n", attr.numRegs);
}
```

---

##  **Memory Limitations**

Memory constraints significantly impact kernel design, from shared memory allocation to global memory access patterns.

###  **Shared Memory Constraints**

#### **Shared Memory Bank Conflicts:**
```cpp
// Demonstrate and resolve shared memory bank conflicts
__global__ void bank_conflict_demo(float* input, float* output, int N) {
    __shared__ float shared_data[32][33];  // 33 to avoid bank conflicts

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // BAD: Bank conflicts when accessing same bank
    if (bid == 0) {
        // All threads access same bank (stride of 32)
        for (int i = 0; i < 32; i++) {
            shared_data[i][0] = input[bid * blockDim.x + tid + i * blockDim.x];
        }
    }

    // GOOD: Avoid bank conflicts with proper stride
    if (bid == 1) {
        // Each thread accesses different bank
        for (int i = 0; i < 32; i++) {
            shared_data[tid][i] = input[bid * blockDim.x + tid + i * blockDim.x];
        }
    }

    __syncthreads();

    // Process data...
    if (tid < N) {
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += shared_data[i][tid % 33];
        }
        output[bid * blockDim.x + tid] = sum;
    }
}

// Dynamic shared memory with constraints
template<int THREADS_PER_BLOCK>
__global__ void dynamic_shared_memory_kernel(float* data, int N, int shared_size) {
    extern __shared__ float dynamic_shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Check if we have enough shared memory
    if (shared_size > 48 * 1024) {  // 48KB limit on many architectures
        if (tid == 0) {
            printf("Warning: Requested shared memory (%d bytes) may exceed limit\n",
                   shared_size);
        }
        return;
    }

    // Use dynamic shared memory
    int elements_per_thread = shared_size / (THREADS_PER_BLOCK * sizeof(float));

    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid * elements_per_thread + i;
        if (idx * sizeof(float) < shared_size) {
            dynamic_shared[idx] = data[bid * THREADS_PER_BLOCK + tid];
        }
    }

    __syncthreads();

    // Process shared data...
    if (bid * THREADS_PER_BLOCK + tid < N) {
        data[bid * THREADS_PER_BLOCK + tid] = dynamic_shared[tid];
    }
}
```

#### **Shared Memory Configuration:**
```cpp
// Configure shared memory vs L1 cache trade-off
void configure_shared_memory() {
    // Query current configuration
    cudaSharedMemConfig config;
    cudaDeviceGetSharedMemConfig(&config);

    printf("Current shared memory config: ");
    switch (config) {
        case cudaSharedMemBankSizeDefault:
            printf("Default (4-byte banks)\n");
            break;
        case cudaSharedMemBankSizeFourByte:
            printf("4-byte banks\n");
            break;
        case cudaSharedMemBankSizeEightByte:
            printf("8-byte banks\n");
            break;
    }

    // Set configuration for different use cases

    // For kernels with many 4-byte accesses
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

    // For kernels with many 8-byte accesses (double precision)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Per-function shared memory configuration
    cudaFuncSetSharedMemConfig(dynamic_shared_memory_kernel<256>,
                              cudaSharedMemBankSizeFourByte);
}
```

###  **Global Memory Constraints**

#### **Memory Coalescing Requirements:**
```cpp
// Demonstrate memory coalescing constraints and solutions
__global__ void memory_coalescing_examples(float* input, float* output,
                                          int width, int height) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    // BAD: Non-coalesced access (strided)
    for (int i = tid; i < width * height; i += total_threads) {
        int row = i / width;
        int col = i % width;

        // Strided access - poor coalescing
        output[col * height + row] = input[row * width + col];
    }
}

__global__ void coalesced_transpose(float* input, float* output,
                                   int width, int height) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from input
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Coalesced write to output
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Memory alignment constraints
__global__ void alignment_sensitive_kernel(float4* aligned_data,
                                          float* unaligned_data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N / 4) {
        // GOOD: 16-byte aligned access (float4)
        float4 data = aligned_data[tid];
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        aligned_data[tid] = data;
    }

    // BAD: Potentially unaligned access
    if (tid < N) {
        unaligned_data[tid] = unaligned_data[tid] * 2.0f;
    }
}
```

---

##  **Thread and Block Constraints**

Understanding thread hierarchy constraints is essential for correct kernel design and optimal performance.

###  **Dimensional Constraints**

#### **Block Dimension Limits:**
```cpp
// Explore and handle block dimension constraints
__global__ void dimension_constraint_demo() {
    // Query current block dimensions
    printf("Block dimensions: [%d, %d, %d]\n",
           blockDim.x, blockDim.y, blockDim.z);
    printf("Grid dimensions: [%d, %d, %d]\n",
           gridDim.x, gridDim.y, gridDim.z);

    // Calculate total threads
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int total_blocks = gridDim.x * gridDim.y * gridDim.z;

    printf("Threads per block: %d\n", threads_per_block);
    printf("Total blocks: %d\n", total_blocks);

    // Verify constraints
    if (threads_per_block > 1024) {
        printf("ERROR: Threads per block exceeds maximum!\n");
    }

    if (blockDim.x > 1024 || blockDim.y > 1024 || blockDim.z > 64) {
        printf("ERROR: Block dimension exceeds limits!\n");
    }
}

// Safe kernel launch with constraint checking
template<typename T>
cudaError_t safe_kernel_launch(void (*kernel)(T*, int), T* data, int N,
                              dim3 requested_block_size) {
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Check block size constraints
    int total_threads = requested_block_size.x * requested_block_size.y * requested_block_size.z;
    if (total_threads > prop.maxThreadsPerBlock) {
        printf("ERROR: Requested block size (%d) exceeds maximum (%d)\n",
               total_threads, prop.maxThreadsPerBlock);
        return cudaErrorInvalidConfiguration;
    }

    if (requested_block_size.x > prop.maxThreadsDim[0] ||
        requested_block_size.y > prop.maxThreadsDim[1] ||
        requested_block_size.z > prop.maxThreadsDim[2]) {
        printf("ERROR: Block dimension exceeds limits [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        return cudaErrorInvalidConfiguration;
    }

    // Calculate grid size
    dim3 grid_size((N + requested_block_size.x - 1) / requested_block_size.x);

    // Check grid size constraints
    if (grid_size.x > prop.maxGridSize[0]) {
        printf("ERROR: Grid size (%d) exceeds maximum (%d)\n",
               grid_size.x, prop.maxGridSize[0]);
        return cudaErrorInvalidConfiguration;
    }

    // Launch kernel
    kernel<<<grid_size, requested_block_size>>>(data, N);
    return cudaGetLastError();
}
```

#### **Multi-Dimensional Indexing:**
```cpp
// Handle multi-dimensional data with proper constraint checking
__global__ void multi_dimensional_kernel(float* data, int width, int height, int depth) {
    // 3D thread indexing
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // Bounds checking - crucial for multi-dimensional kernels
    if (x >= width || y >= height || z >= depth) {
        return;  // Thread out of bounds
    }

    // Linear index calculation
    int idx = z * (width * height) + y * width + x;

    // Process element
    data[idx] = data[idx] * (x + y + z + 1);

    // Debug output for first few threads
    if (x < 2 && y < 2 && z < 2) {
        printf("Thread (%d,%d,%d) -> linear index %d\n", x, y, z, idx);
    }
}

// Host function with proper dimension setup
void launch_3d_kernel(float* h_data, int width, int height, int depth) {
    float* d_data;
    size_t size = width * height * depth * sizeof(float);

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Configure 3D thread blocks
    dim3 block_size(8, 8, 8);  // 512 threads per block
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y,
                   (depth + block_size.z - 1) / block_size.z);

    // Verify grid constraints
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (grid_size.x > prop.maxGridSize[0] ||
        grid_size.y > prop.maxGridSize[1] ||
        grid_size.z > prop.maxGridSize[2]) {

        printf("Grid size [%d, %d, %d] exceeds limits [%d, %d, %d]\n",
               grid_size.x, grid_size.y, grid_size.z,
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // Fallback to 1D indexing
        int total_elements = width * height * depth;
        dim3 fallback_block(256);
        dim3 fallback_grid((total_elements + 255) / 256);

        // Need different kernel for 1D indexing...
        printf("Using 1D fallback indexing\n");
    } else {
        multi_dimensional_kernel<<<grid_size, block_size>>>(d_data, width, height, depth);
    }

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
```

---

## **Execution Time Limits**

GPU kernels face various time constraints that can cause execution failures or poor performance.

### **Watchdog Timer Constraints**

#### **Handling Watchdog Timeouts:**
```cpp
// Long-running kernel that may hit watchdog timeout
__global__ void long_running_kernel(float* data, int N, int iterations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Long computation that might timeout
        for (int i = 0; i < iterations; i++) {
            value = sin(value) + cos(value);
            value = sqrt(fabs(value) + 1.0f);
            value = log(value + 1.0f);
            value = exp(value * 0.1f);
        }

        data[tid] = value;
    }
}

// Chunked execution to avoid timeouts
void execute_with_timeout_avoidance(float* h_data, int N, int total_iterations) {
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    const int max_iterations_per_chunk = 1000;  // Avoid timeout
    const int chunks = (total_iterations + max_iterations_per_chunk - 1) / max_iterations_per_chunk;

    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    printf("Executing %d iterations in %d chunks\n", total_iterations, chunks);

    for (int chunk = 0; chunk < chunks; chunk++) {
        int chunk_iterations = min(max_iterations_per_chunk,
                                  total_iterations - chunk * max_iterations_per_chunk);

        printf("Chunk %d: %d iterations\n", chunk, chunk_iterations);

        // Launch kernel for this chunk
        long_running_kernel<<<grid_size, block_size>>>(d_data, N, chunk_iterations);

        // Synchronize to check for errors
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            printf("Error in chunk %d: %s\n", chunk, cudaGetErrorString(error));
            break;
        }

        // Optional: brief pause to let system breathe
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
```

#### **Cooperative Kernel Execution:**
```cpp
// Use cooperative kernels for controlled execution
__global__ void cooperative_long_kernel(float* data, int N, int* progress_flag) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        for (int phase = 0; phase < 10; phase++) {
            // Do work for this phase
            for (int i = 0; i < 100; i++) {
                data[tid] = sin(data[tid]) + phase;
            }

            // Cooperative synchronization point
            grid.sync();

            // Check if we should continue (host can signal stop)
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                if (*progress_flag == 0) {
                    return;  // Host requested stop
                }
                printf("Completed phase %d\n", phase);
            }

            grid.sync();
        }
    }
}

// Host code for cooperative execution
void launch_cooperative_kernel(float* h_data, int N) {
    // Check cooperative launch support
    int device = 0;
    int cooperative_launch = 0;
    cudaDeviceGetAttribute(&cooperative_launch,
                          cudaDevAttrCooperativeLaunch, device);

    if (!cooperative_launch) {
        printf("Cooperative launch not supported\n");
        return;
    }

    float* d_data;
    int* d_progress_flag;
    int h_progress_flag = 1;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_progress_flag, sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_progress_flag, &h_progress_flag, sizeof(int), cudaMemcpyHostToDevice);

    // Setup cooperative launch parameters
    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    void* args[] = {&d_data, &N, &d_progress_flag};

    // Launch cooperative kernel
    cudaLaunchCooperativeKernel((void*)cooperative_long_kernel,
                               grid_size, block_size, args, 0, 0);

    // Monitor progress and can signal stop if needed
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("Cooperative kernel error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_progress_flag);
}
```

---

##  **Divergence Penalties**

Thread divergence within warps can severely impact performance, especially in control-flow intensive code.

###  **Warp Divergence Analysis**

#### **Measuring Divergence Impact:**
```cpp
// Demonstrate different levels of warp divergence
__global__ void divergence_examples(float* data, int N, int pattern) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % 32;

    if (tid < N) {
        float value = data[tid];

        switch (pattern) {
            case 0: // No divergence - all threads take same path
                value = sin(value) + cos(value);
                break;

            case 1: // Half-warp divergence
                if (lane_id < 16) {
                    value = sin(value);
                } else {
                    value = cos(value);
                }
                break;

            case 2: // Maximum divergence - each thread different
                if (lane_id % 4 == 0) {
                    value = sin(value);
                } else if (lane_id % 4 == 1) {
                    value = cos(value);
                } else if (lane_id % 4 == 2) {
                    value = tan(value);
                } else {
                    value = sqrt(fabs(value));
                }
                break;

            case 3: // Data-dependent divergence (worst case)
                if (value > 0.5f) {
                    for (int i = 0; i < 100; i++) {
                        value = sin(value);
                    }
                } else if (value > 0.0f) {
                    for (int i = 0; i < 50; i++) {
                        value = cos(value);
                    }
                } else {
                    value = -value;
                }
                break;
        }

        data[tid] = value;
    }
}

// Analyze divergence performance
void analyze_divergence_performance() {
    const int N = 1024 * 1024;
    float* h_data = new float[N];
    float* d_data;

    // Initialize with random data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&d_data, N * sizeof(float));

    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    // Test different divergence patterns
    for (int pattern = 0; pattern < 4; pattern++) {
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

        // Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        divergence_examples<<<grid_size, block_size>>>(d_data, N, pattern);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Pattern %d execution time: %.3f ms\n", pattern, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    delete[] h_data;
    cudaFree(d_data);
}
```

#### **Divergence Reduction Techniques:**
```cpp
// Reduce divergence through algorithm restructuring
__global__ void divergent_reduction_bad(float* data, float* result, int N) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load data
    shared_data[tid] = (bid * blockDim.x + tid < N) ? data[bid * blockDim.x + tid] : 0.0f;
    __syncthreads();

    // BAD: Highly divergent reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // Causes divergence!
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[bid] = shared_data[0];
    }
}

__global__ void divergent_reduction_good(float* data, float* result, int N) {
    __shared__ float shared_data[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load data
    shared_data[tid] = (bid * blockDim.x + tid < N) ? data[bid * blockDim.x + tid] : 0.0f;
    __syncthreads();

    // GOOD: Convergent reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {  // Contiguous threads - no divergence!
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[bid] = shared_data[0];
    }
}

// Predicated execution to minimize divergence
__global__ void predicated_execution(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Use predicated execution instead of branching
        bool condition1 = (value > 0.5f);
        bool condition2 = (value > 0.0f) && !condition1;
        bool condition3 = !condition1 && !condition2;

        // All threads execute all paths, but results are selectively used
        float result1 = sin(value);           // For condition1
        float result2 = cos(value);           // For condition2
        float result3 = -value;               // For condition3

        // Select result based on condition
        float final_result = condition1 ? result1 :
                           (condition2 ? result2 : result3);

        data[tid] = final_result;
    }
}
```

---

##  **Constraint Workarounds**

When facing hard constraints, several techniques can help work around limitations.

###  **Memory Constraint Workarounds**

#### **Shared Memory Overflow Handling:**
```cpp
// Handle cases where required shared memory exceeds limits
template<int MAX_THREADS>
__global__ void adaptive_shared_memory_kernel(float* data, int N, int requested_shared_size) {
    extern __shared__ float dynamic_shared[];

    // Query available shared memory at runtime
    int available_shared = 48 * 1024;  // Conservative estimate

    if (requested_shared_size <= available_shared) {
        // Use shared memory approach
        int tid = threadIdx.x;
        int elements_per_thread = requested_shared_size / (MAX_THREADS * sizeof(float));

        // Load data into shared memory
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = tid * elements_per_thread + i;
            int global_idx = blockIdx.x * MAX_THREADS * elements_per_thread +
                           tid * elements_per_thread + i;

            if (global_idx < N && idx < requested_shared_size / sizeof(float)) {
                dynamic_shared[idx] = data[global_idx];
            }
        }

        __syncthreads();

        // Process in shared memory
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = tid * elements_per_thread + i;
            if (idx < requested_shared_size / sizeof(float)) {
                dynamic_shared[idx] *= 2.0f;
            }
        }

        __syncthreads();

        // Write back
        for (int i = 0; i < elements_per_thread; i++) {
            int idx = tid * elements_per_thread + i;
            int global_idx = blockIdx.x * MAX_THREADS * elements_per_thread +
                           tid * elements_per_thread + i;

            if (global_idx < N && idx < requested_shared_size / sizeof(float)) {
                data[global_idx] = dynamic_shared[idx];
            }
        }
    } else {
        // Fallback to global memory approach
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < N) {
            // Process directly in global memory
            data[tid] *= 2.0f;
        }
    }
}

// Multi-pass processing for large shared memory requirements
__global__ void multi_pass_shared_processing(float* data, int N, int pass_id, int total_passes) {
    __shared__ float shared_chunk[12 * 1024 / sizeof(float)];  // 12KB chunks

    int tid = threadIdx.x;
    int chunk_size = 12 * 1024 / sizeof(float);
    int elements_this_pass = chunk_size / blockDim.x;

    // Calculate which data to process in this pass
    int base_idx = blockIdx.x * blockDim.x * total_passes * elements_this_pass +
                   pass_id * blockDim.x * elements_this_pass;

    // Load chunk for this pass
    for (int i = 0; i < elements_this_pass; i++) {
        int global_idx = base_idx + tid * elements_this_pass + i;
        int shared_idx = tid * elements_this_pass + i;

        if (global_idx < N && shared_idx < chunk_size) {
            shared_chunk[shared_idx] = data[global_idx];
        }
    }

    __syncthreads();

    // Process chunk
    for (int i = 0; i < elements_this_pass; i++) {
        int shared_idx = tid * elements_this_pass + i;
        if (shared_idx < chunk_size) {
            shared_chunk[shared_idx] = sin(shared_chunk[shared_idx]);
        }
    }

    __syncthreads();

    // Write back
    for (int i = 0; i < elements_this_pass; i++) {
        int global_idx = base_idx + tid * elements_this_pass + i;
        int shared_idx = tid * elements_this_pass + i;

        if (global_idx < N && shared_idx < chunk_size) {
            data[global_idx] = shared_chunk[shared_idx];
        }
    }
}
```

#### **Register Spilling Management:**
```cpp
// Manage register spilling through algorithm restructuring
__global__ void register_spill_heavy_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // This will likely cause register spilling
        float a = data[tid];
        float b = sin(a);
        float c = cos(a);
        float d = tan(a);
        float e = sqrt(fabs(a));
        float f = log(a + 1.0f);
        float g = exp(a * 0.1f);
        float h = pow(a, 2.0f);
        float i = sinh(a);
        float j = cosh(a);
        float k = tanh(a);
        float l = asin(fabs(a));
        float m = acos(fabs(a));
        float n = atan(a);

        // Complex computation using all registers
        float result = (a + b + c + d + e + f + g) * (h + i + j + k + l + m + n);
        data[tid] = result;
    }
}

// Restructured to reduce register pressure
__global__ void register_optimized_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Compute in stages, reusing registers
        float stage1 = sin(value) + cos(value) + tan(value);
        float stage2 = sqrt(fabs(value)) + log(value + 1.0f) + exp(value * 0.1f);
        float stage3 = pow(value, 2.0f) + sinh(value) + cosh(value);
        float stage4 = tanh(value) + asin(fabs(value)) + acos(fabs(value)) + atan(value);

        // Combine stages
        float result = (value + stage1 + stage2) * (stage3 + stage4);
        data[tid] = result;
    }
}

// Use function calls to manage register usage
__device__ float complex_computation_part1(float x) {
    return sin(x) + cos(x) + tan(x) + sqrt(fabs(x));
}

__device__ float complex_computation_part2(float x) {
    return log(x + 1.0f) + exp(x * 0.1f) + pow(x, 2.0f);
}

__device__ float complex_computation_part3(float x) {
    return sinh(x) + cosh(x) + tanh(x) + asin(fabs(x));
}

__global__ void function_based_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Function calls help manage register pressure
        float part1 = complex_computation_part1(value);
        float part2 = complex_computation_part2(value);
        float part3 = complex_computation_part3(value);

        data[tid] = (value + part1) * (part2 + part3);
    }
}
```

---

##  **Resource Optimization**

Optimizing resource usage within constraints is key to maximizing GPU performance.

###  **Occupancy Optimization**

#### **Occupancy Calculator Integration:**
```cpp
// Calculate theoretical occupancy
void calculate_occupancy() {
    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== Occupancy Analysis ===\n");
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

    // Analyze different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    for (int i = 0; i < num_block_sizes; i++) {
        int block_size = block_sizes[i];

        // Calculate occupancy
        int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
        max_blocks_per_sm = min(max_blocks_per_sm, prop.maxBlocksPerMultiProcessor);

        int active_threads_per_sm = max_blocks_per_sm * block_size;
        float occupancy = (float)active_threads_per_sm / prop.maxThreadsPerMultiProcessor;

        printf("Block size %4d: %d blocks/SM, %4d threads/SM, %.1f%% occupancy\n",
               block_size, max_blocks_per_sm, active_threads_per_sm, occupancy * 100);
    }
}

// Runtime occupancy measurement
__global__ void occupancy_test_kernel(float* data, int N, int work_amount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float value = data[tid];

        // Variable work amount to test occupancy impact
        for (int i = 0; i < work_amount; i++) {
            value = sin(value) + cos(value);
        }

        data[tid] = value;
    }
}

void measure_runtime_occupancy() {
    //nvprof --metrics achieved_occupancy program
    printf("Use nvprof --metrics achieved_occupancy to measure runtime occupancy\n");
    printf("Compare theoretical vs achieved occupancy\n");

    const int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Test different configurations
    int block_sizes[] = {128, 256, 512};
    int work_amounts[] = {10, 100, 1000};

    for (int bs = 0; bs < 3; bs++) {
        for (int wa = 0; wa < 3; wa++) {
            dim3 block_size(block_sizes[bs]);
            dim3 grid_size((N + block_size.x - 1) / block_size.x);

            printf("Testing block_size=%d, work_amount=%d\n",
                   block_sizes[bs], work_amounts[wa]);

            occupancy_test_kernel<<<grid_size, block_size>>>(d_data, N, work_amounts[wa]);
            cudaDeviceSynchronize();
        }
    }

    cudaFree(d_data);
}
```

#### **Dynamic Resource Allocation:**
```cpp
// Adaptively allocate resources based on availability
__global__ void adaptive_resource_kernel(float* data, int N,
                                        int available_shared_mem,
                                        int available_registers) {
    extern __shared__ float dynamic_shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Adapt algorithm based on available resources
    if (available_shared_mem >= 16 * 1024) {
        // High shared memory approach
        int elements_per_thread = (available_shared_mem / sizeof(float)) / blockDim.x;

        for (int i = 0; i < elements_per_thread; i++) {
            int global_idx = bid * blockDim.x * elements_per_thread +
                           tid * elements_per_thread + i;
            int shared_idx = tid * elements_per_thread + i;

            if (global_idx < N) {
                dynamic_shared[shared_idx] = data[global_idx];
            }
        }

        __syncthreads();

        // Process in shared memory
        for (int i = 0; i < elements_per_thread; i++) {
            int shared_idx = tid * elements_per_thread + i;
            dynamic_shared[shared_idx] = sin(dynamic_shared[shared_idx]);
        }

        __syncthreads();

        // Write back
        for (int i = 0; i < elements_per_thread; i++) {
            int global_idx = bid * blockDim.x * elements_per_thread +
                           tid * elements_per_thread + i;
            int shared_idx = tid * elements_per_thread + i;

            if (global_idx < N) {
                data[global_idx] = dynamic_shared[shared_idx];
            }
        }
    } else {
        // Low resource approach - direct global memory
        int global_idx = bid * blockDim.x + tid;

        if (global_idx < N) {
            data[global_idx] = sin(data[global_idx]);
        }
    }
}

// Host-side resource management
void launch_with_resource_adaptation(float* data, int N) {
    // Query available resources
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int available_shared = prop.sharedMemPerBlock;
    int available_regs = prop.regsPerBlock;

    printf("Available resources: %d bytes shared, %d registers\n",
           available_shared, available_regs);

    // Choose optimal configuration based on resources
    dim3 block_size;
    int shared_mem_size;

    if (available_shared >= 48 * 1024) {
        block_size = dim3(256);
        shared_mem_size = 32 * 1024;  // Use 32KB
    } else if (available_shared >= 16 * 1024) {
        block_size = dim3(128);
        shared_mem_size = 16 * 1024;  // Use 16KB
    } else {
        block_size = dim3(64);
        shared_mem_size = 0;  // No shared memory
    }

    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

    adaptive_resource_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_data, N, available_shared, available_regs);

    cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
```

---

##  **Constraint Analysis Tools**

###  **Profiling and Analysis**

#### **Comprehensive Constraint Profiler:**
```cpp
class ConstraintProfiler {
private:
    cudaDeviceProp device_props;

public:
    ConstraintProfiler() {
        cudaGetDeviceProperties(&device_props, 0);
    }

    void profile_kernel_constraints(const char* kernel_name,
                                   dim3 grid_size, dim3 block_size,
                                   size_t shared_mem_bytes = 0) {
        printf("\n=== Constraint Analysis: %s ===\n", kernel_name);

        // Block size analysis
        int threads_per_block = block_size.x * block_size.y * block_size.z;
        printf(" Block Configuration:\n");
        printf("  Dimensions: [%d, %d, %d]\n", block_size.x, block_size.y, block_size.z);
        printf("  Threads per block: %d (max: %d)\n",
               threads_per_block, device_props.maxThreadsPerBlock);

        if (threads_per_block > device_props.maxThreadsPerBlock) {
            printf("   ERROR: Exceeds max threads per block!\n");
        }

        // Grid size analysis
        int total_blocks = grid_size.x * grid_size.y * grid_size.z;
        printf("\n Grid Configuration:\n");
        printf("  Dimensions: [%d, %d, %d]\n", grid_size.x, grid_size.y, grid_size.z);
        printf("  Total blocks: %d\n", total_blocks);

        if (grid_size.x > device_props.maxGridSize[0] ||
            grid_size.y > device_props.maxGridSize[1] ||
            grid_size.z > device_props.maxGridSize[2]) {
            printf("   ERROR: Exceeds max grid dimensions!\n");
        }

        // Occupancy analysis
        printf("\n Occupancy Analysis:\n");
        int max_blocks_per_sm = device_props.maxThreadsPerMultiProcessor / threads_per_block;
        max_blocks_per_sm = min(max_blocks_per_sm, device_props.maxBlocksPerMultiProcessor);

        // Account for shared memory constraints
        if (shared_mem_bytes > 0) {
            int sm_shared_mem = device_props.sharedMemPerMultiprocessor;
            int max_blocks_shared = sm_shared_mem / shared_mem_bytes;
            max_blocks_per_sm = min(max_blocks_per_sm, max_blocks_shared);

            printf("  Shared memory per block: %zu bytes\n", shared_mem_bytes);
            printf("  Shared memory limit: %d blocks/SM\n", max_blocks_shared);
        }

        int active_threads_per_sm = max_blocks_per_sm * threads_per_block;
        float theoretical_occupancy = (float)active_threads_per_sm / device_props.maxThreadsPerMultiProcessor;

        printf("  Max blocks per SM: %d\n", max_blocks_per_sm);
        printf("  Active threads per SM: %d\n", active_threads_per_sm);
        printf("  Theoretical occupancy: %.1f%%\n", theoretical_occupancy * 100);

        // Resource efficiency
        printf("\n Resource Efficiency:\n");
        int warps_per_block = (threads_per_block + 31) / 32;
        printf("  Warps per block: %d\n", warps_per_block);
        printf("  Warp efficiency: %.1f%%\n",
               (float)threads_per_block / (warps_per_block * 32) * 100);

        // Memory bandwidth utilization
        printf("\n Memory Analysis:\n");
        float peak_bandwidth = 2.0f * device_props.memoryClockRate * (device_props.memoryBusWidth / 8) / 1.0e6f;
        printf("  Peak memory bandwidth: %.1f GB/s\n", peak_bandwidth);
        printf("  Memory transaction size: Analyze with profiler\n");

        printf("==========================================\n");
    }

    void suggest_optimizations(dim3 current_block_size, size_t shared_mem_bytes) {
        printf("\n Optimization Suggestions:\n");

        int current_threads = current_block_size.x * current_block_size.y * current_block_size.z;

        // Block size suggestions
        if (current_threads % 32 != 0) {
            printf("  • Round block size to multiple of 32 for warp efficiency\n");
        }

        if (current_threads < 128) {
            printf("  • Consider larger block size (≥128) for better occupancy\n");
        }

        if (current_threads > 512 && shared_mem_bytes > 16 * 1024) {
            printf("  • Consider smaller block size to reduce shared memory pressure\n");
        }

        // Shared memory suggestions
        if (shared_mem_bytes > device_props.sharedMemPerBlock) {
            printf("  • Reduce shared memory usage or use multi-pass approach\n");
        }

        // Architecture-specific suggestions
        if (device_props.major >= 7) {
            printf("  • Use __syncwarp() for explicit warp synchronization (Volta+)\n");
        }

        if (device_props.major >= 8) {
            printf("  • Consider using async copy operations (Ampere+)\n");
        }
    }
};

// Usage example
void analyze_kernel_constraints() {
    ConstraintProfiler profiler;

    // Analyze different kernel configurations
    profiler.profile_kernel_constraints("Matrix Multiply",
                                       dim3(128, 128), dim3(16, 16),
                                       16 * 16 * sizeof(float) * 2);

    profiler.profile_kernel_constraints("Vector Add",
                                       dim3(1024), dim3(256), 0);

    profiler.profile_kernel_constraints("Reduction",
                                       dim3(64), dim3(512),
                                       512 * sizeof(float));

    // Get optimization suggestions
    profiler.suggest_optimizations(dim3(256), 8 * 1024);
}
```

---

##  **Key Takeaways**

1. **Know Your Limits**: Always query device properties and respect hardware constraints
2. **Memory Efficiency**: Balance shared memory usage with occupancy requirements
3. **Thread Organization**: Use warp-aligned block sizes and avoid unnecessary divergence
4. **Time Management**: Break long kernels into chunks to avoid watchdog timeouts
5. **Minimize Divergence**: Structure algorithms to reduce warp divergence penalties
6. **Plan Workarounds**: Have fallback strategies for resource-constrained scenarios
7. **Profile and Optimize**: Use profiling tools to identify and resolve constraint bottlenecks
8. **Continuous Monitoring**: Regularly analyze constraint adherence and performance impact

##  **Related Guides**

- **Next Step**: [Optimization Strategies](../05_performance_profiling/5_optimization_strategies.md) - Real-world optimization case studies
- **Foundation**: [Synchronization Guide](../03_synchronization/1_synchronization_basics.md) - Thread coordination patterns
- **Architecture**: [ Streaming Multiprocessors Guide](4_streaming_multiprocessors_deep.md) - SM resource management
- **Overview**: [ Execution Model Overview](1_cuda_execution_model.md) - Quick reference and navigation

---

** Pro Tip**: Constraints aren't roadblocks—they're design parameters! Understanding them deeply enables you to write GPU code that's both correct and optimal.
