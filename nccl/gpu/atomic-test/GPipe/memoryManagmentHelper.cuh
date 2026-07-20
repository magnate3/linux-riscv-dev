#ifndef MEMORY_MANAGMENT
#define MEMORY_MANAGMENT

#include "cuda.h"
#include <builtin_types.h>
#include "common.cuh"

#define PRINT_PHISICAL_ALLOCATION 0
typedef int ShareableHandle;

static const char *_cudaGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}

typedef struct
{
	CUdeviceptr ptr;
	size_t alloc_size;
	CUcontext cuda_context;
	CUmemGenericAllocationHandle mem_handle;
} memoryProperties;

__host__ void validateDeviceIsSupported()
{
	int device;
	CUDA_CHECK(cudaGetDevice(&device));

	int deviceSupportsVmm;
	CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device);
	if (deviceSupportsVmm != 0) {
		return;
	}

	dbg_printf("Virtual memory API is unsupported by device");
	exit(-1);
}

__host__ void cleanMemoryMaping(memoryProperties prop)
{
	CUDA_RESULT_CHECK(cuMemUnmap(prop.ptr, prop.alloc_size));
	CUDA_RESULT_CHECK(cuMemAddressFree(prop.ptr, prop.alloc_size));
	if (prop.mem_handle != 0)
		CUDA_RESULT_CHECK(cuMemRelease(prop.mem_handle));

	// CUDA_RESULT_CHECK(cuCtxDestroy(prop.cuda_context));
}

__host__ void closeSharableHandle(ShareableHandle shHandle)
{
	close(shHandle);
}
static memoryProperties importAndMapMemory(ShareableHandle ipc_handle, size_t buffer_size)
{
	CUmemAccessDesc access;
	int	cuda_dindex = 0;
	CUmemGenericAllocationHandle mem_handle;
	memoryProperties memProperties;
	memProperties.alloc_size = buffer_size;
	memProperties.mem_handle = 0;

	CUDA_RESULT_CHECK(cuInit(0));
	// CUdevice cuda_device;
	// CUDA_RESULT_CHECK(cuDeviceGet(&cuda_device, cuda_dindex));
	// CUDA_RESULT_CHECK(cuCtxCreate(&(memProperties.cuda_context), CU_CTX_SCHED_AUTO, cuda_device));

	dbg_printf("Sharable_handle in producer: %d\n", ipc_handle);

	// import shared memory handle
	CUDA_RESULT_CHECK(cuMemImportFromShareableHandle(&mem_handle,
									(void *)(uintptr_t)(ipc_handle),
									CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

	// reserve virtual address space
	CUDA_RESULT_CHECK(cuMemAddressReserve(&(memProperties.ptr), memProperties.alloc_size, 0, 0UL, 0));

	// map device memory
	CUDA_RESULT_CHECK(cuMemMap(memProperties.ptr, memProperties.alloc_size, 0, mem_handle, 0));

	CUDA_RESULT_CHECK(cuMemRelease(mem_handle));

	access.location.id = cuda_dindex;
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	CUDA_RESULT_CHECK(cuMemSetAccess(memProperties.ptr, buffer_size, &access, 1));

	dbg_printf("Successfully import memory\n");
	close(ipc_handle);
	return memProperties;
}

static memoryProperties allocateSharableMemory(size_t memory_size, ShareableHandle *shHandle)
{
	size_t		granularity;
	CUmemAllocationProp prop;
	CUmemAccessDesc access;
	int	cuda_dindex = 0;
	memoryProperties memProperties;

	CUDA_RESULT_CHECK(cuInit(0));
	// CUdevice cuda_device;
	// CUDA_RESULT_CHECK(cuDeviceGet(&cuda_device, cuda_dindex));
	// CUDA_RESULT_CHECK(cuCtxCreate(&(memProperties.cuda_context), CU_CTX_SCHED_AUTO, cuda_device));

	// check allocation granularity
	memset(&prop, 0, sizeof(CUmemAllocationProp));
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = cuda_dindex;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

	CUDA_RESULT_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
									   CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

	// round up buffer_size by the granularity
	memProperties.alloc_size = ROUND_UP(memory_size, granularity);
	CUDA_RESULT_CHECK(cuMemCreate(&(memProperties.mem_handle), memProperties.alloc_size, &prop, 0));

#if PRINT_PHISICAL_ALLOCATION
	// confirm physical memory consumption
	system("nvidia-smi");
#endif

	CUDA_RESULT_CHECK(cuMemAddressReserve(&(memProperties.ptr), memProperties.alloc_size, 0, 0UL, 0));
	CUDA_RESULT_CHECK(cuMemMap(memProperties.ptr, memProperties.alloc_size, 0, memProperties.mem_handle, 0));

	access.location = prop.location;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	CUDA_RESULT_CHECK(cuMemSetAccess(memProperties.ptr, memProperties.alloc_size, &access, 1));

	// export the above allocation to sharable handle
	CUDA_RESULT_CHECK(cuMemExportToShareableHandle(shHandle, memProperties.mem_handle,
									  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
									  0));

	dbg_printf("Sharable_handle '%d' was created\n", *shHandle);
	return memProperties;
}


#endif
