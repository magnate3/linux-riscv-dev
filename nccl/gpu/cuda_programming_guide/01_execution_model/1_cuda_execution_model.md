#  CUDA Execution Model - Overview & Quick Reference

Understanding the CUDA execution model is **essential** for writing performant GPU code, diagnosing bottlenecks, and architecting large-scale systems. This overview provides the key concepts and quick references you need.

##  Navigation Guide

###  **Detailed Section Files**
- **[ Thread Hierarchy Complete Guide](2_thread_hierarchy.md)** - Thread/Block/Grid organization, indexing patterns, dimensionality choices
- **[ Warp Execution Advanced Guide](3_warp_execution.md)** - SIMT model, divergence optimization, warp intrinsics, debugging
- **[ Streaming Multiprocessors Deep Dive](4_streaming_multiprocessors_deep.md)** - SM architecture, scheduling, occupancy tuning, resource management
- **[Synchronization Fundamentals](../03_synchronization/1_synchronization_basics.md)** - Thread barriers, cooperative groups, race condition prevention
- **[ Execution Constraints Guide](5_execution_constraints_guide.md)** - Hardware limits, resource tradeoffs, launch configuration validation

---

##  **Execution Hierarchy Quick Reference**

| Level | Size | Scope | Hardware Mapping | Best For | Detailed Guide |
|-------|------|-------|------------------|----------|----------------|
| **Thread** | 1 unit | Private registers | CUDA core | Individual computations | [ Thread Guide](2_thread_hierarchy.md) |
| **Warp** | 32 threads | SIMT execution | Warp scheduler | Lockstep operations | [ Warp Guide](3_warp_execution.md) |
| **Block** | 1-1024 threads | Shared memory + sync | Part of SM | Cooperative algorithms | [ Thread Guide](2_thread_hierarchy.md#blocks) |
| **Grid** | Many blocks | Global scope | Entire GPU | Massive parallelism | [ Thread Guide](2_thread_hierarchy.md#grids) |

##  **Launch Configuration Quick Patterns**

###  **1D Problems (Arrays, Vectors)**
```cpp
int N = 1000000;  // Array size
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);

// Inside kernel:
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < N) data[idx] = /* computation */;
```

###  **2D Problems (Matrices, Images)**
```cpp
dim3 threadsPerBlock(16, 16);  // 256 threads total
dim3 blocksPerGrid((width + 15)/16, (height + 15)/16);

kernel<<<blocksPerGrid, threadsPerBlock>>>(matrix, width, height);

// Inside kernel:
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
if (col < width && row < height) matrix[row * width + col] = /* computation */;
```

###  **3D Problems (Volumes, Tensors)**
```cpp
dim3 threadsPerBlock(8, 8, 8);  // 512 threads total
dim3 blocksPerGrid((width + 7)/8, (height + 7)/8, (depth + 7)/8);

kernel<<<blocksPerGrid, threadsPerBlock>>>(volume, width, height, depth);

// Inside kernel:
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int z = threadIdx.z + blockIdx.z * blockDim.z;
```

##  **Warp Execution Quick Reference**

###  **Common Divergence Patterns (Avoid These)**
```cpp
//  BAD: Half-warp divergence
if (threadIdx.x % 2 == 0) {
    computeEven();
} else {
    computeOdd();
}

//  BAD: Data-dependent branching
if (data[idx] > threshold) {
    processLarge(data[idx]);
} else {
    processSmall(data[idx]);
}
```

###  **Divergence-Free Alternatives**
```cpp
//  GOOD: Predicated execution
bool is_even = (threadIdx.x % 2 == 0);
float result = is_even ? computeEven() : computeOdd();

//  GOOD: Warp-uniform conditionals
if (blockIdx.x % 2 == 0) {  // All threads in block take same path
    processBlock();
}
```

###  **Essential Warp Intrinsics**
| Function | Purpose | Example Use Case |
|----------|---------|------------------|
| `__shfl_sync(mask, var, srcLane)` | Exchange values within warp | Reduction without shared memory |
| `__ballot_sync(mask, predicate)` | Get bitmask of predicate results | Count active threads |
| `__any_sync(mask, predicate)` | True if any thread matches | Early exit conditions |
| `__all_sync(mask, predicate)` | True if all threads match | Validation checks |

##  **SM Resource Limits (A100 Example)**

| Resource | Per SM Limit | Per Block Limit | Impact of Exceeding |
|----------|--------------|-----------------|---------------------|
| **Threads** | 2048 | 1024 | Launch failure |
| **Warps** | 64 | 32 | Reduced occupancy |
| **Blocks** | 32 | - | Reduced occupancy |
| **Shared Memory** | 163 KB | 163 KB | Fewer concurrent blocks |
| **Registers** | 65536 | Varies | Register spills to local memory |

###  **Occupancy Calculator**
```cpp
// Query optimal launch configuration
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// Check occupancy for specific configuration
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, sharedMemSize);
float occupancy = (numBlocks * blockSize / warpSize) / maxWarpsPerSM;
```

##  **Synchronization Quick Patterns**

###  **Basic Block Synchronization**
```cpp
__global__ void collaborative_kernel(float* data) {
    __shared__ float tile[256];

    // Load phase
    tile[threadIdx.x] = data[blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();  // Wait for all loads

    // Compute phase
    float result = tile[threadIdx.x] * 2.0f;
    __syncthreads();  // Wait for all computations

    // Store phase
    data[blockIdx.x * blockDim.x + threadIdx.x] = result;
}
```

###  **Common Synchronization Mistakes**
```cpp
//  DEADLOCK: Not all threads reach sync
if (threadIdx.x < 128) {
    __syncthreads();  // Only half the threads reach this!
}

//  CORRECT: All threads reach sync
bool active = (threadIdx.x < 128);
if (active) {
    // do work
}
__syncthreads();  // All threads reach this
```

##  **Common Performance Issues & Quick Fixes**

###  **Issue: Low Occupancy**
**Symptom:** Nsight Compute shows <50% occupancy
**Quick Fixes:**
- Reduce shared memory usage per block
- Decrease register usage (fewer local variables)
- Increase block size (try 256, 512 threads)
- Check resource usage: `nvcc --ptxas-options=-v`

###  **Issue: Warp Divergence**
**Symptom:** `branch_efficiency < 80%` in profiler
**Quick Fixes:**
- Group similar data together before processing
- Use warp intrinsics instead of shared memory
- Move conditionals outside inner loops
- Sort/partition data to reduce branching

###  **Issue: Launch Configuration Errors**
**Symptom:** `cudaErrorInvalidConfiguration`
**Quick Fixes:**
```cpp
// Validate before launch
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

if (blockSize > prop.maxThreadsPerBlock) {
    // Reduce block size
}

if (sharedMemSize > prop.sharedMemPerBlock) {
    // Reduce shared memory usage
}
```

##  **Profiling Quick Commands**

###  **Occupancy Analysis**
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed ./app
# Target: >60% SM throughput, >50% active warps
```

###  **Warp Efficiency Check**
```bash
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./app
# Target: >90% (higher is better)
```

###  **Resource Usage Check**
```bash
nvcc --ptxas-options=-v kernel.cu
# Shows registers and shared memory usage per thread/block
```

###  **Launch Configuration Validation**
```bash
ncu --metrics launch__occupancy_limit_warps,launch__occupancy_limit_blocks ./app
# Shows what's limiting occupancy
```

##  **Optimization Decision Tree**

```
GPU Performance Issue?
 Low SM utilization?
    Increase occupancy → [ SM Guide](4_streaming_multiprocessors_deep.md)
    Fix launch configuration → [ Constraints Guide](5_execution_constraints_guide.md)
 Branch divergence issues?
    Optimize warp execution → [ Warp Guide](3_warp_execution.md)
 Thread coordination problems?
    Fix synchronization → [Sync Guide](../03_synchronization/1_synchronization_basics.md)
 Indexing or memory access issues?
    Review thread hierarchy → [ Thread Guide](2_thread_hierarchy.md)
 Need optimization examples?
     Study performance patterns → [Optimization Strategies](../05_performance_profiling/5_optimization_strategies.md)
```

##  **Key Principles**

1. ** Design for Warps**: Think in groups of 32 threads, avoid divergence
2. ** Maximize Occupancy**: Balance resource usage for concurrent execution
3. ** Synchronize Carefully**: Use barriers correctly, avoid deadlocks
4. ** Respect Limits**: Stay within hardware constraints for all resources
5. ** Profile Everything**: Use Nsight Compute/Systems to validate optimizations

** Pro Tip**: Start with correct indexing and basic synchronization, then progressively optimize for warp efficiency and occupancy. Profile at each step to measure improvements!
