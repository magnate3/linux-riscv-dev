#include <cstdio>
#include <cuda/atomic>

#define EXPECT_BASE(true) \
	do { \
		printf("[%d] Verifying...    Result: ", __LINE__); \
		if(true) { \
			printf("Successful!\n"); \
		} else { \
			printf("Error, Function:%s Line:%d\n", __func__, __LINE__); \
		} \
	} while(0)

#define EXPECT_EQ(x, y)  EXPECT_BASE(x == y)
#define EXPECT_TRUE(expect) EXPECT_EQ(expect, true)
#define CUDA_CHECK(exp)                                                                              \
  do {                                                                                               \
    cudaError_t err = (exp);                                                                         \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s:%d] %s failed: %s\n", __FILE__, __LINE__, #exp, cudaGetErrorString(err)); \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

__global__ void atomic_counter_kernel(cuda::atomic<int, cuda::thread_scope_device>* counter, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    counter->fetch_add(1, cuda::memory_order_relaxed);
  }
}

void test0(){
  const int n = 10000;
  cuda::atomic<int, cuda::thread_scope_device>* d_counter;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(*d_counter)));
  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(*d_counter)));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  atomic_counter_kernel<<<blocks, threads>>>(d_counter, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  int result;
  CUDA_CHECK(cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Atomic counter: %d (expected %d)\n", result, n);
  EXPECT_EQ(result, n);

  CUDA_CHECK(cudaFree(d_counter));
  return ;
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

void test1(){
  int* d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

  // Launch single block with 2 threads - thread 0 produces, thread 1 consumes
  acquire_release_kernel<<<1, 2>>>(d_result);
  CUDA_CHECK(cudaDeviceSynchronize());

  int result;
  CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
  printf("Acquire-release: consumer saw %d (expected 42)\n", result);
  EXPECT_EQ(result, 42);

  CUDA_CHECK(cudaFree(d_result));
  return ;
}

__global__ void atomic_ref_kernel(int* arr, int n, int increments_per_element) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int element = idx % n;

  cuda::atomic_ref<int, cuda::thread_scope_device> ref(arr[element]);
  for (int i = 0; i < increments_per_element; i++) {
    ref.fetch_add(1, cuda::memory_order_relaxed);
  }
}

void test2(){
  const int n = 16;
  const int total_threads = 1024;
  const int increments = 10;

  int* d_arr;
  CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_arr, 0, n * sizeof(int)));

  atomic_ref_kernel<<<total_threads / 256, 256>>>(d_arr, n, increments);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_arr[n];
  CUDA_CHECK(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));

  int expected = (total_threads / n) * increments;
  printf("Atomic ref results: ");
  bool all_correct = true;
  for (int i = 0; i < n; i++) {
    printf("%d ", h_arr[i]);
    if (h_arr[i] != expected) all_correct = false;
  }
  printf("(expected all %d)\n", expected);
  EXPECT_TRUE(all_correct);

  CUDA_CHECK(cudaFree(d_arr));
  return ;
}
int main()
{
    test0();
    test1();
    test2();
    return 0;
}
