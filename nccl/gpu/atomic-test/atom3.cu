#include <cuda/atomic>
#include <iostream>

__global__ void device_kernel(int* shared_var_ptr) {
    // Create a system-scoped atomic reference to the shared variable
    cuda::atomic_ref<int, cuda::thread_scope_system> system_atomic(*shared_var_ptr);

    // Each thread increments the counter atomically at the system scope
    system_atomic.fetch_add(1, cuda::memory_order_release);
}

// Acquire-release synchronization - single kernel with producer/consumer threads
__global__ void acquire_release_kernel(int* result) {
  __shared__ cuda::atomic<int, cuda::thread_scope_block> flag;
  __shared__ int data;

  if (threadIdx.x == 0) {
    flag.store(0, cuda::memory_order_relaxed);
    data = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Producer: write data then release flag
    data = 42;
    flag.store(1, cuda::memory_order_release);
  } else if (threadIdx.x == 1) {
    // Consumer: spin on flag with acquire, then read data
    while (flag.load(cuda::memory_order_acquire) == 0);
    *result = data;  // Guaranteed to see 42 due to acquire-release
  }
}

int main() {
    int* host_ptr;
    const int initial_value = 0;
    const int num_threads = 256;
    const int num_blocks = 1;

    // Allocate pinned (page-locked) host memory for communication
    // Pinned memory is host memory that the device can access directly.
    cudaMallocHost(&host_ptr, sizeof(int));
    *host_ptr = initial_value;

    // Launch kernel
    device_kernel<<<num_blocks, num_threads>>>(host_ptr);

    // Synchronize the device with the host
    // The cudaDeviceSynchronize() function ensures all kernel work is complete
    // and all memory operations are visible to the host.
    cudaDeviceSynchronize();

    // The host can now reliably read the final value modified by all device threads
    int final_value = *host_ptr;

    std::cout << "Expected final value: " << num_threads << std::endl;
    std::cout << "Actual final value: " << final_value << std::endl;
    
    // Verify the result
    if (final_value == num_threads) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Error: Race condition or synchronization failure." << std::endl;
    }

    // Free memory
    cudaFreeHost(host_ptr);

    return 0;
}

