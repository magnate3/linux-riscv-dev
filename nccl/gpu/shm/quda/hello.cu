// https://forums.developer.nvidia.com/t/does-anybody-have-experience-on-cudahostregister-zero-copy-memory/22539/3

#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define SIZE 10

#include <cuda.h>

// Kernel definition, see also section 4.2.3 of Nvidia Cuda Programming Guide

__global__ void vecAdd(float *A, float *B, float *C)
{
  int i = threadIdx.x;

  //	A[i] = 0;
  //	B[i] = i;
  C[i] = A[i] + B[i];
  printf("Kernel: A[%d]=%f, B[%d]=%f, C[%d]=%f\n", i, A[i], i, B[i], i, C[i]);
}

void *map_alloc(size_t size)
{
  return mmap(NULL, size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
}
 void *device_malloc_( size_t size)
  {
    void *ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate device memory )\n");
      exit(0);
    }
#ifdef HOST_DEBUG
    cudaMemset(ptr, 0xff, size);
#endif
    return ptr;
  }
  /**
   * Perform a cuMemAlloc with error-checking.  This function is to
   * guarantee a unique memory allocation on the device, since
   * cudaMalloc can be redirected (as is the case with QDPJIT).  This
   * should only be called via the device_pinned_malloc() macro,
   * defined in malloc_quda.h.
   */
  void *device_pinned_malloc_( size_t size)
  {
    void *ptr;
    CUresult err = cuMemAlloc((CUdeviceptr*)&ptr, size);
    if (err != CUDA_SUCCESS) {
      printf("ERROR: Failed to allocate device memory )\n");
    }
#ifdef HOST_DEBUG
    cudaMemset(ptr, 0xff, size);
#endif
    return ptr;
  }


  /**
   * Perform a standard malloc() with error-checking.  This function
   * should only be called via the safe_malloc() macro, defined in
   * malloc_quda.h
   */
  void *safe_malloc_(size_t size)
  {
    void *ptr = malloc(size);
    if (!ptr) {
      printf("ERROR: Failed to allocate device memory )\n");
    }
#ifdef HOST_DEBUG
    memset(ptr, 0xff, size);
#endif
    return ptr;
  }

  /**
   * Under CUDA 4.0, cudaHostRegister seems to require that both the
   * beginning and end of the buffer be aligned on page boundaries.
   * This local function takes care of the alignment and gets called
   * by pinned_malloc_() and mapped_malloc_()
   */
  static void *aligned_malloc(size_t size)
  {
    void *ptr;
#if (CUDA_VERSION > 4000)
    ptr = malloc(size);
#else
    static int page_size = getpagesize();
    size_t base_size = ((size + page_size - 1) / page_size) * page_size; // round up to the nearest multiple of page_size
    posix_memalign(&ptr, page_size, base_size);
#endif
    if (!ptr) {
      printf("ERROR: Failed to allocate device memory )\n");
    }
    return ptr;
  }
  /**
   * Allocate page-locked ("pinned") host memory.  This function
   * should only be called via the pinned_malloc() macro, defined in
   * malloc_quda.h
   *
   * Note that we do not rely on cudaHostAlloc(), since buffers
   * allocated in this way have been observed to cause problems when
   * shared with MPI via GPU Direct on some systems.
   */
  void *pinned_malloc_(size_t size)
  {
    void *ptr = aligned_malloc(size);


     static int page_size = getpagesize();
    size_t base_size = ((size + page_size - 1) / page_size) * page_size;
    cudaError_t err = cudaHostRegister(ptr, base_size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate device memory )\n");
    }
#ifdef HOST_DEBUG
    memset(ptr, 0xff, base_size);
#endif
    return ptr;
  }


  /**
   * Allocate page-locked ("pinned") host memory, and map it into the
   * GPU address space.  This function should only be called via the
   * mapped_malloc() macro, defined in malloc_quda.h
   */
  void *mapped_malloc_(size_t size)
  {
    void *ptr = aligned_malloc(size);
    static int page_size = getpagesize();
    size_t base_size = ((size + page_size - 1) / page_size) * page_size;
    cudaError_t err = cudaHostRegister(ptr, base_size, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to allocate device memory )\n");
    }
#ifdef HOST_DEBUG
    memset(ptr, 0xff, base_size);
#endif
    return ptr;
  }
  /**
   * Free device memory allocated with device_malloc().  This function
   * should only be called via the device_free() macro, defined in
   * malloc_quda.h
   */
  void device_free_(void *ptr)
  {
    if (!ptr) {
      printf("ERROR: Attempt to free NULL device pointer \n");
    }
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to free device memory \n");
    }
  }


  /**
   * Free device memory allocated with device_pinned malloc().  This
   * function should only be called via the device_pinned_free()
   * macro, defined in malloc_quda.h
   */
  void device_pinned_free_( void *ptr)
  {
    if (!ptr) {
      printf("ERROR: Attempt to free NULL device pointer \n");
    }
    CUresult err = cuMemFree((CUdeviceptr)ptr);
    if (err != CUDA_SUCCESS) {
      printf("ERROR: Attempt to free NULL device pointer \n");
    }
  }


  /**
   * Free host memory allocated with safe_malloc(), pinned_malloc(),
   * or mapped_malloc().  This function should only be called via the
   * host_free() macro, defined in malloc_quda.h
   */
  void host_free_( void *ptr)
  {
    if (!ptr) {
      printf("ERROR: Attempt to free NULL host pointer \n");
    }
    free(ptr);
  }
void vecAddTest(int num,size_t bytes,float *devPtrB, float *devPtrC)
{

      float *d_reduce=0;
      float *h_reduce=0;
      float *hd_reduce=0;
      cudaDeviceProp deviceProp;
      // reduction buffer size

      if (!d_reduce) d_reduce = (float*) device_malloc_(bytes);

      // these arrays are actually oversized currently (only needs to be QudaSumFloat3)

      // if the device supports host-mapped memory then use a host-mapped array for the reduction
      if (!h_reduce) {
	// only use zero copy reductions when using 64-bit
	if(deviceProp.canMapHostMemory) {
	  h_reduce = (float *) mapped_malloc_(bytes);
          fprintf(stderr, "Device %d can map host memory!\n", 0);
	  cudaHostGetDevicePointer(&hd_reduce, h_reduce, 0); // set the matching device pointer
	} else
	  {
	    h_reduce = (float*) pinned_malloc_(bytes);
	    hd_reduce = d_reduce;
	  }
	memset(h_reduce, 0, bytes); // added to ensure that valgrind doesn't report h_reduce is unitialised
      }

      vecAdd<<<1, num>>>(hd_reduce, devPtrB, devPtrC);
      cudaDeviceSynchronize();

      //checkCudaError();
      if (d_reduce) {
	device_free_(d_reduce);
      }
      if (h_reduce) {
	host_free_(h_reduce);
      }
}


int main()
{
  int N = SIZE;

  //	round up the size of the array to be a multiple of the page size

  size_t memsize = ((SIZE * sizeof(float) + 4095) / 4096) * 4096;

  cudaDeviceProp deviceProp;

  // Get properties and verify device 0 supports mapped memory

  cudaGetDeviceProperties(&deviceProp, 0);

  if (!deviceProp.canMapHostMemory)
    {
      fprintf(stderr, "Device %d cannot map host memory!\n", 0);
      exit(EXIT_FAILURE);
    }

  fprintf(stderr, "uni addr: %u\n", deviceProp.unifiedAddressing);
  fprintf(stderr, "can use host pointer: %u\n",
          deviceProp.canUseHostPointerForRegisteredMem);

  // set the device flags for mapping host memory

  cudaSetDeviceFlags(cudaDeviceMapHost);

  float *A, *B, *C;

  float *devPtrA, *devPtrB, *devPtrC;

  //	use valloc instead of malloc
  A = (float *)map_alloc(memsize);
  B = (float *)map_alloc(memsize);
  C = (float *)map_alloc(memsize);

  cudaHostRegister(A, memsize, cudaHostRegisterMapped);
  cudaHostRegister(B, memsize, cudaHostRegisterMapped);
  cudaHostRegister(C, memsize, cudaHostRegisterMapped);

  for (int i = 0; i < SIZE; i++)
    {
      A[i] = B[i] = i;
    }

  cudaHostGetDevicePointer((void **)&devPtrA, (void *)A, 0);
  fprintf(stderr, "%p =? %p\n", devPtrA, A);

  {
    cudaPointerAttributes attr;
    cudaError_t rc = cudaPointerGetAttributes(&attr, (void *)A);
    if (rc != cudaSuccess)
      {
        fprintf(stderr, "fail\n");
      }
    fprintf(stderr, "prop[%p]: dev %u, dptr %p, hptr %p\n", A, attr.device,
            attr.devicePointer, attr.hostPointer);
  }

  {
    cudaPointerAttributes attr;
    cudaError_t rc = cudaPointerGetAttributes(&attr, (void *)devPtrA);
    if (rc != cudaSuccess)
      {
        fprintf(stderr, "fail\n");
      }
    fprintf(stderr, "prop[%p]: dev %u, dptr %p, hptr %p\n", devPtrA,
            attr.device, attr.devicePointer, attr.hostPointer);
  }

  cudaHostGetDevicePointer((void **)&devPtrB, (void *)B, 0);
  cudaHostGetDevicePointer((void **)&devPtrC, (void *)C, 0);

  vecAdd<<<1, N>>>(devPtrA, devPtrB, devPtrC);

  cudaDeviceSynchronize();

  for (int i = 0; i < SIZE; i++) printf("C[%d]=%f\n", i, C[i]);

  vecAddTest(N,memsize,devPtrB, devPtrC);
  for (int i = 0; i < SIZE; i++) printf("C[%d]=%f\n", i, C[i]);
  cudaHostUnregister(A);
  cudaHostUnregister(B);
  cudaHostUnregister(C);

  // free(A);
  munmap(A, memsize);
  munmap(B, memsize);
  munmap(C, memsize);
}
