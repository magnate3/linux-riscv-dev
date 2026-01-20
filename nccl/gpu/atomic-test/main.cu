#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/span>

#include <cuda/std/cmath>
//#include <cuda/cmath>
#include <cstddef>
#include <iostream>
#include <limits>

template <int block_size>
__global__ void reduce(cuda::std::span<int const> data, cuda::std::span<int> result) {
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int const index = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = 0;
  if (index < data.size()) {
    sum += data[index];
  }
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(result.front());
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}
template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

int main() {

  // Allocate and initialize input data
  int const N = 1000;
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);
  
  // Allocate output data
  thrust::device_vector<int> kernel_result(1);

  // Compute the sum reduction of `data` using a custom kernel
  constexpr int block_size = 256;
  int const num_blocks = ceil_div(N, block_size);
  //int const num_blocks = cuda::ceil_div(N, block_size);
  reduce<block_size><<<num_blocks, block_size>>>(cuda::std::span<int const>(thrust::raw_pointer_cast(data.data()), data.size()),
                                                 cuda::std::span<int>(thrust::raw_pointer_cast(kernel_result.data()), 1));

  auto const err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  int const custom_result = kernel_result[0];

  // Compute the same sum reduction using Thrust
  int const thrust_result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  // Ensure the two solutions are identical
  std::printf("Custom kernel sum: %d\n", custom_result);
  std::printf("Thrust reduce sum: %d\n", thrust_result);
  assert(kernel_result[0] == thrust_result);
  return 0;
}
