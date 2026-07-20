// nvcc test_cuda_api.cu -lcuda -std=c++11 -o api
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>

#define CUDA_CHECK(X)                                                \
  do {                                                               \
    auto result = X;                                                 \
    if (result != cudaSuccess) {                                     \
      const char* p_err_str = cudaGetErrorName(result);              \
      fprintf(stderr, "Failed: Line %d %s.\n", __LINE__, p_err_str); \
      return;                                                        \
    }                                                                \
  } while (0)

#define CU_CHECK(X)                                                            \
  do {                                                                         \
    auto result = X;                                                           \
    if (result != CUDA_SUCCESS) {                                              \
      const char* p_err_str = nullptr;                                         \
      if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {  \
        p_err_str = "Unrecoginzed CU error num";                               \
      }                                                                        \
      fprintf(stderr, "File %s Line %d %s returned %s.\n", __FILE__, __LINE__, \
              #X, p_err_str);                                                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

bool IsCUDAAccessibleMemory(const void* ptr) {
  cudaPointerAttributes attr{};
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  return !(err != cudaSuccess || attr.type == cudaMemoryTypeUnregistered);
}

void show_info(int device) {
  std::cout
      << "***********************Device Info*********************************"
      << std::endl;
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
  std::cout
      << "GPU Device " << device << " name:\t" << deviceProp.name
      << "\nCUDA Capability version:\t" << deviceProp.major << "."
      << deviceProp.minor << "\nMemory:\t"
      << (float)deviceProp.totalGlobalMem / 1024 / 1024 / 1024
      << " GiB\nShares a unified address space with the host:\t"
      << deviceProp.unifiedAddressing << "\nShared memory per block:\t"
      << (float)deviceProp.sharedMemPerBlock / 1024
      << " KiB\nMaximum number of threads per block:\t"
      << deviceProp.maxThreadsPerBlock << "\nWarp size in threads:\t"
      << deviceProp.warpSize << "\nNumber of asynchronous engines:\t"
      << deviceProp.asyncEngineCount
      << "\nCan map host memory with cudaHostAlloc/cudaHostGetDevicePointer:\t"
      << deviceProp.canMapHostMemory
      << "\nCan access host registered memory at the same "
      << "virtual address as the CPU:\t"
      << deviceProp.canUseHostPointerForRegisteredMem
      << "\nCan possibly execute multiple kernels concurrently:\t"
      << deviceProp.concurrentKernels << std::endl;
}

void test_malloc() {
  std::cout
      << "***********************cudaMalloc*********************************"
      << std::endl;
  size_t numElements = 1024;
  size_t bufferSize = numElements * sizeof(float);
  float* deviceBuffer = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&deviceBuffer, bufferSize));
  CUDA_CHECK(cudaFree(deviceBuffer));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Passed!" << std::endl;
}

void test_malloc_host() {
  std::cout
      << "***********************cudaMallocHost*******************************"
      << std::endl;
  const int kBufferSize = 1024 * 1024;
  void* host_buffer;
  CUDA_CHECK(cudaMallocHost(&host_buffer, kBufferSize));
  CUDA_CHECK(cudaFreeHost(host_buffer));
  std::cout << "Passed!" << std::endl;
}

void test_memcpy() {
  std::cout
      << "***********************cudaMemcpy*********************************"
      << std::endl;
  size_t numElements = 1024;
  size_t bufferSize = numElements * sizeof(float);
  float* deviceBuffer = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&deviceBuffer, bufferSize));
  float* hostBuffer = new float[numElements];
  for (size_t i = 0; i < numElements; ++i) {
    hostBuffer[i] = static_cast<float>(i);
  }
  CUDA_CHECK(
      cudaMemcpy(deviceBuffer, hostBuffer, bufferSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t i = 0; i < numElements; ++i) {
    hostBuffer[i] = 0.0;
  }
  CUDA_CHECK(
      cudaMemcpy(hostBuffer, deviceBuffer, bufferSize, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t i = 0; i < numElements; ++i) {
    if (hostBuffer[i] != static_cast<float>(i)) {
      std::cerr << "Data verification failed at index " << i << std::endl;
      return;
    }
  }
  CUDA_CHECK(cudaFree(deviceBuffer));
  delete[] hostBuffer;
  std::cout << "Passed!" << std::endl;
}

void test_memcpy_async() {
  std::cout
      << "***********************cudaMemcpyAsync******************************"
      << std::endl;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  size_t numElements = 1024;
  size_t bufferSize = numElements * sizeof(float);
  float* deviceBuffer = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&deviceBuffer, bufferSize));
  float* hostBuffer = new float[numElements];
  for (size_t i = 0; i < numElements; ++i) {
    hostBuffer[i] = static_cast<float>(i);
  }
  CUDA_CHECK(cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize,
                             cudaMemcpyDefault, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (!IsCUDAAccessibleMemory(deviceBuffer)) {
    std::cerr << "Failed! deviceBuffer" << std::endl;
    return;
  }
  if (IsCUDAAccessibleMemory(hostBuffer)) {
    std::cerr << "Failed! hostBuffer" << std::endl;
    return;
  }
  for (size_t i = 0; i < numElements; ++i) {
    hostBuffer[i] = 0.0;
  }
  delete[] hostBuffer;
  CUDA_CHECK(cudaFree(deviceBuffer));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "Passed!" << std::endl;
}

void test_host_register() {
  std::cout
      << "***********************cudaHostRegister*****************************"
      << std::endl;
  size_t SIZE = 1024;
  size_t bufferSize = SIZE * sizeof(int);
  int* host_ptr = new int[SIZE];
  void* device_ptr = nullptr;
  CUDA_CHECK(cudaHostRegister(host_ptr, bufferSize, cudaHostRegisterDefault));
  CUDA_CHECK(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0));
  if (!IsCUDAAccessibleMemory(device_ptr)) {
    std::cerr << "Failed! device_ptr" << std::endl;
    return;
  }
  if (device_ptr != host_ptr) {
    std::cerr << "Failed! not same" << std::endl;
    return;
  }
  CUDA_CHECK(cudaHostUnregister(host_ptr));
  delete[] host_ptr;
  std::cout << "Passed!" << std::endl;
}

void test_access_peer() {
  std::cout
      << "***********************CanAccessPeer********************************"
      << std::endl;
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::cerr << "Need >=2 device!" << std::endl;
    return;
  }
  int canAccessPeer01 = -1;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1));
  int canAccessPeer10 = -1;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0));
  if (canAccessPeer01 == 1 && canAccessPeer10 == 1) {
    std::cout << "Passed!" << std::endl;
  } else {
    std::cerr << "Failed!" << std::endl;
  }
}

void test_ipc_mem() {
  size_t bufferSize = 1024 * 1024;
  void* device_ptr = nullptr;
  cudaIpcMemHandle_t ipc_handle;
  // use pipe to pass ipc_handle to child process
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    std::cerr << "Failed to create pipe" << std::endl;
    return;
  }
  // Opens an interprocess memory handle exported from another process
  // and returns a device pointer usable in the local process.
  // need another process otherwise cudaErrorDeviceUninitialized
  pid_t pid = fork();
  if (pid == -1) {
    std::cerr << "fork Failed" << std::endl;
    if (device_ptr != nullptr) {
      CUDA_CHECK(cudaFree(device_ptr));
    }
    return;
  } else if (pid == 0) {
    // get ipc handle in child process
    cudaIpcMemHandle_t child_ipc_handle;
    close(pipefd[1]);
    read(pipefd[0], &child_ipc_handle, sizeof(cudaIpcMemHandle_t));
    CUDA_CHECK(cudaSetDevice(0));
    void* return_dev_ptr = nullptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&return_dev_ptr, child_ipc_handle,
                                    cudaIpcMemLazyEnablePeerAccess));
    std::cout << "child process ptr:\t" << return_dev_ptr << std::endl;
    CUDA_CHECK(cudaIpcCloseMemHandle(return_dev_ptr));
    exit(0);
  } else {
    CUDA_CHECK(cudaSetDevice(0));
    show_info(0);
    close(pipefd[0]);
    std::cout
        << "***********************IpcMemHandle*******************************"
        << std::endl;
    CUDA_CHECK(cudaMalloc((void**)&device_ptr, bufferSize));
    CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handle, device_ptr));
    std::cout << "main process ptr:\t" << device_ptr << std::endl;
    write(pipefd[1], &ipc_handle, sizeof(cudaIpcMemHandle_t));
    int status;
    waitpid(pid, &status, 0);
    CUDA_CHECK(cudaFree(device_ptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Passed!" << std::endl;
  }
}

size_t GetGranularity(int dev_id) {
  size_t granularity = 0;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
  prop.location.id = dev_id;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, flags));
  std::cout << "Allocation Granularity:" << granularity << std::endl;
  if (granularity < 512 * 1024 * 1024) granularity = 512 * 1024 * 1024;
  return granularity;
}

#define AlignUp(X, ALIGN_SIZE) \
  (((X) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE))

void test_cumem() {
  // cuMem* api is CUDA Driver API, low level API
  std::cout << "***********************cuMem********************************"
            << std::endl;
  auto granularity = GetGranularity(0);
  size_t SIZE = 512 * 1024 * 1024 + 2048;
  CUdeviceptr reserved;
  auto aligned_size = AlignUp(SIZE, granularity);
  CU_CHECK(cuMemAddressReserve(&reserved, aligned_size, granularity, 0, 0));
  auto local_size = aligned_size / 2;
  CUmemGenericAllocationHandle local_alloc_handle;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  prop.location.id = 0;
  CU_CHECK(cuMemCreate(&local_alloc_handle, local_size, &prop, 0));
  // Given a CUDA memory handle, create a shareable memory allocation
  // handle that can be used to share the memory with other processes.
  // The recipient process can convert the shareable handle back into
  // a CUDA memory handle using cuMemImportFromShareableHandle and
  // map it with cuMemMap.
  int shareble_handle_fd = -1;
  CU_CHECK(cuMemExportToShareableHandle(
      &shareble_handle_fd, local_alloc_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  CUmemGenericAllocationHandle import_alloc_handle;
  CU_CHECK(cuMemImportFromShareableHandle(
      &import_alloc_handle, (void*)(uintptr_t)shareble_handle_fd,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CU_CHECK(cuMemMap(reserved, local_size, 0, import_alloc_handle, 0));
  close(shareble_handle_fd);
  CUmemAccessDesc madesc;
  madesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  madesc.location.id = 0;
  madesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_CHECK(cuMemSetAccess(reserved, local_size, &madesc, 1));
  CU_CHECK(cuMemUnmap(reserved, local_size));
  CU_CHECK(cuMemRelease(local_alloc_handle));
  CU_CHECK(cuMemAddressFree(reserved, aligned_size));
  std::cout << "Passed!" << std::endl;
}

int main() {
  test_ipc_mem();  // must call before other tests
  test_malloc();
  test_malloc_host();
  test_memcpy();
  test_memcpy_async();
  test_host_register();
  test_access_peer();
  test_cumem();
  return 0;
}
