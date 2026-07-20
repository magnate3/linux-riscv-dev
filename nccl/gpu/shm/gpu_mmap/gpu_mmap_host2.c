#include <assert.h>
#include <errno.h>
#include <libgen.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
// C
//#include <cstdlib>
#include <assert.h>

// C++
#include <iostream>
#include <memory>
#include <chrono>

#include "gpu_mmap_kernel.h"

static int		cuda_dindex = 0;
static int		verbose = 0;
static int		num_children = 2;
static size_t	buffer_size = (1UL << 30);	/* 1GB */

#define Elog(fmt, ...)							   \
	do {                                           \
		fprintf(stderr,"%s:%d  " fmt "\n",         \
				__FILE__,__LINE__, ##__VA_ARGS__); \
		exit(1);								   \
	} while(0)


// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)
#if 0
// NCCL
#define NCCLCHECK(cmd) do {                                   \
    ncclResult_t res = cmd;                                   \
if (res != ncclSuccess) {                                     \
    fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n",        \
        __FILE__, __LINE__, ncclGetErrorString(res));         \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// NVML
#define NVMLCHECK(cmd) do {                                   \
    nvmlReturn_t res = cmd;                                   \
    if (res != NVML_SUCCESS) {                                \
        fprintf(stderr, "Failed, NVML error %s:%d '%s'\n",    \
            __FILE__, __LINE__, nvmlErrorString(res));        \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)
#endif

static inline const char *errorName(CUresult rc)
{
	const char *error_name;

	if (cuGetErrorName(rc, &error_name) != CUDA_SUCCESS)
		error_name = "UNKNOWN";
	return error_name;
}

static CUdevice		cuda_device = -1;
static CUcontext	cuda_context = NULL;
static CUmodule		cuda_module = NULL;
static CUdeviceptr	cuda_buffer = 0UL;
static CUdeviceptr	cuda_result = 0UL;

int check_error(cudaError_t status) {
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error: %s\n", s);
        //assert(0);
        //snprintf(buffer, 256, "CUDA Error: %s", s);
        //printf(buffer);
	return -1;
   }
   return 0;
}
#define CUCHECKGOTO(cmd, res, label) do {                     \
    CUresult err = cmd;                                 \
    if( err != CUDA_SUCCESS ) {                               \
      res = -1;                           \
      goto label;                                             \
    }                                                         \
} while(false)
static void host_printf_cuda_buff(char * buff, int size)
{
     int i = 0;
     int max =  size > 32 ?  32 : size;
     cudaError_t status = cudaHostAlloc((void**)&buff, size, cudaHostAllocMapped);
     if(check_error(status))
     {
	 return ;
     }
     printf("%s ",__func__);
     //for(;i< size; ++i)
     for(;i< max; ++i)
	 printf("%d",buff[i]);
     printf("\n");
}
int verify_memory(int dev_idx, CUdeviceptr d_ptr, int size){

    CUcontext ctx2;
    CUdevice device;
    CUstream stream;
    int res = 0;
    uint8_t* host_mem = new uint8_t[size];
    // SPRINTF(devIdx, "%d", selectedDevices[0]);


    CUCHECKGOTO(cuDeviceGet(&device, dev_idx),res,fail);
    CUCHECKGOTO(cuCtxCreate(&ctx2, 0, device),res,fail);
    CUCHECKGOTO(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),res,fail);

    //std::vector<char> verification_buffer(size);
    CUCHECKGOTO(cuMemcpyDtoHAsync(host_mem, d_ptr , size, stream),res,fail);
    CUCHECKGOTO(cuStreamSynchronize(stream),res,fail);
    for (int i = 0; i < size; ++i) {
        //printf("%02x ", host_mem[i]);
    }

    delete[] host_mem;
  return res;
fail:
  delete[] host_mem;
  return res;
#if 0
  // The contents should have the id of the sibling just after me
  char compareId = (char)VAR_ID;
  for (unsigned long long j = 0; j < DATA_BUF_SIZE; j++) {
    if (verification_buffer[j] != compareId) {
      printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
             (int)verification_buffer[j], (int)compareId);
      break;
    }
  }
#endif
}
void print_device_mem(int dev_idx, CUdeviceptr ptr, int size) {
    uint8_t* host_mem = new uint8_t[size];
    CUDACHECK(cudaSetDevice(dev_idx));
    CUDACHECK(cudaMemcpy(host_mem, (void*)ptr, size, cudaMemcpyDeviceToHost));
    printf("Device %d: ", dev_idx);
    for (int i = 0; i < size; ++i) {
        printf("%02x ", host_mem[i]);
    }
    printf("\n");
    delete[] host_mem;
}

void write_random_data(int dev_idx, CUdeviceptr ptr, int size) {
    uint8_t* host_mem = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        host_mem[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(dev_idx));
    CUDACHECK(cudaMemcpy((void*)ptr, host_mem, size, cudaMemcpyHostToDevice));
    delete[] host_mem;
}


static double launch_kernel(CUmemGenericAllocationHandle mem_handle,
							const char *kfunc_name)
{
	CUfunction	cuda_function;
	CUresult	rc;
	size_t		width;
	int			mp_count;
	void	   *kparams[2];
	double		result;

	if (!cuda_module)
	{
		rc = cuModuleLoadData(&cuda_module, gpu_mmap_kernel);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuModuleLoadData: %s", errorName(rc));

		rc = cuModuleGetGlobal(&cuda_result,
							   &width,
							   cuda_module,
							   "gpu_mmap_result");
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuModuleGetGlobal: %s", errorName(rc));
		assert(width == sizeof(double));
	}
	rc = cuModuleGetFunction(&cuda_function, cuda_module, kfunc_name);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetFunction: %s", errorName(rc));

	rc = cuDeviceGetAttribute(&mp_count,
							  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
							  cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGetAttribute: %s", errorName(rc));

	kparams[0] = (void *)&cuda_buffer;
	kparams[1] = (void *)&buffer_size;
	rc = cuLaunchKernel(cuda_function,
						2 * mp_count, 1, 1,
						1024, 1, 1,
						0,
						NULL,
						kparams,
						NULL);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLaunchKernel(kfunc=%s): %s",
			 errorName(rc), kfunc_name );

	rc = cuMemcpyDtoH(&result, cuda_result, sizeof(double));
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemcpyDtoH: %s", errorName(rc));

	return result;
}

typedef struct
{
	int		cuda_dindex;
	size_t	buffer_size;
} ipc_message_t;

static inline void
__server_loop(int sockfd, int ipc_handle)
{
	for (;;)
	{
		int				client;
		struct msghdr	msg;
		struct iovec	iov;
		char			cmsg_buf[CMSG_SPACE(sizeof(int))];
		struct cmsghdr *cmsg;
		ipc_message_t	payload;

		client = accept(sockfd, NULL, NULL);
		if (client < 0)
			Elog("failed on accept: %m");

		/* send a file descriptor using SCM_RIGHTS */
		memset(&msg, 0, sizeof(msg));
		msg.msg_control = cmsg_buf;
		msg.msg_controllen = sizeof(cmsg_buf);

		cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_len = CMSG_LEN(sizeof(int));
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;

		memcpy(CMSG_DATA(cmsg), &ipc_handle, sizeof(ipc_handle));

		memset(&payload, 0, sizeof(payload));
		payload.cuda_dindex = cuda_dindex;
		payload.buffer_size = buffer_size;

		iov.iov_base = &payload;
		iov.iov_len = sizeof(payload);
		msg.msg_iov = &iov;
		msg.msg_iovlen = 1;

		if (sendmsg(client, &msg, 0) < 0)
			Elog("failed on sendmsg(2): %m");

		close(client);
	}
}

typedef struct
{
	int		sockfd;
	int		ipc_handle;
} serv_args_t;

static void *
server_loop(void *pthread_args)
{
	serv_args_t	   *serv_args = (serv_args_t *)pthread_args;

	__server_loop(serv_args->sockfd, serv_args->ipc_handle);

	return NULL;
}

static int
recv_ipc_handle(struct sockaddr_un *addr)
{
	int				client;
	struct msghdr	msg;
    struct iovec	iov;
	struct cmsghdr *cmsg;
	char			cmsg_buf[CMSG_SPACE(sizeof(int))];
	ipc_message_t	payload;
	int				ipc_handle = -1;

	/* connect to the server socket */
	client = socket(AF_UNIX, SOCK_STREAM, 0);
	if (client < 0)
		Elog("failed on socket(2): %m");
	if (connect(client, (struct sockaddr *)addr,
				sizeof(struct sockaddr_un)) != 0)
		Elog("failed on connect(2): %m");

	/* recv a file descriptor using SCM_RIGHTS */
	memset(&msg, 0, sizeof(msg));
	msg.msg_control = cmsg_buf;
	msg.msg_controllen = sizeof(cmsg_buf);

	iov.iov_base = &payload;
	iov.iov_len = sizeof(payload);

	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	if (recvmsg(client, &msg, 0) <= 0)
		Elog("failed on recvmsg(2): %m");

	cmsg = CMSG_FIRSTHDR(&msg);
	if (!cmsg)
		Elog("message has no control header");
	if (cmsg->cmsg_len == CMSG_LEN(sizeof(int)) &&
		cmsg->cmsg_level == SOL_SOCKET &&
		cmsg->cmsg_type == SCM_RIGHTS)
	{
		memcpy(&ipc_handle, CMSG_DATA(cmsg), sizeof(int));
	}
	else
		Elog("unexpected control header");

	cuda_dindex = payload.cuda_dindex;
	buffer_size = payload.buffer_size;

	return ipc_handle;
}

static int child_main(struct sockaddr_un *addr)
{
	int			ipc_handle = recv_ipc_handle(addr);
	CUresult	rc;
	CUmemGenericAllocationHandle mem_handle;
	CUmemAccessDesc access;
	double		sum;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", errorName(rc));

	rc = cuDeviceGet(&cuda_device, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", errorName(rc));

	rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", errorName(rc));

	/* import shared memory handle */
	rc = cuMemImportFromShareableHandle(&mem_handle,
										(void *)(uintptr_t)ipc_handle,
									CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemImportFromShareableHandle: %s", errorName(rc));

	/* reserve virtual address space */
	rc = cuMemAddressReserve(&cuda_buffer,
							 buffer_size,
							 0, 0UL, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemAddressReserve: %s", errorName(rc));

	/* map device memory */
	rc = cuMemMap(cuda_buffer, buffer_size, 0, mem_handle, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemMap: %s", errorName(rc));

	access.location.id = cuda_dindex;
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	rc = cuMemSetAccess(cuda_buffer, buffer_size, &access, 1);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemSetAccess: %s", errorName(rc));

	/* launch GPU kernel for initialization and total sum */
	sum = launch_kernel(mem_handle, "gpu_mmap_kernel");
	printf("total sum [%u]: %f\n", getpid(), sum);
	return 0;
}
int ummap_alloc(void *addr, size_t size)
{
  return munmap(addr, size);
}
void *map_alloc(size_t size)
{
  return mmap(NULL, size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
}

int init_device(int dev_idx)
{ 
    cudaDeviceProp deviceProp;

    // Get properties and verify device 0 supports mapped memory

   cudaGetDeviceProperties(&deviceProp, dev_idx);

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
}
static int export_gpu_memory(void)
{
	CUresult	rc;
	size_t		granularity;
	CUmemAllocationProp prop;
	CUmemGenericAllocationHandle mem_handle;
	CUmemAccessDesc access;
	int			ipc_handle;	/* POSIX file handle */
	double		sum;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", errorName(rc));

	rc = cuDeviceGet(&cuda_device, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", errorName(rc));

	rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", errorName(rc));

	/* check allocation granularity */
	memset(&prop, 0, sizeof(CUmemAllocationProp));
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = cuda_dindex;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

	rc = cuMemGetAllocationGranularity(&granularity, &prop,
									   CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemGetAllocationGranularity: %s", errorName(rc));

	/* round up buffer_size by the granularity */
	buffer_size = (buffer_size + granularity - 1) & ~(granularity - 1);
	rc = cuMemCreate(&mem_handle, buffer_size, &prop, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemCreate: %s", errorName(rc));
#if 0
	/* confirm physical memory consumption */
	system("nvidia-smi");
#endif
	rc = cuMemAddressReserve(&cuda_buffer, buffer_size, 0, 0UL, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemAddressReserve: %s", errorName(rc));

	rc = cuMemMap(cuda_buffer, buffer_size, 0, mem_handle, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemMap: %s", errorName(rc));

	access.location = prop.location;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	rc = cuMemSetAccess(cuda_buffer, buffer_size, &access, 1);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemSetAccess: %s", errorName(rc));

	/* launch GPU kernel for initialization and total sum */
	launch_kernel(mem_handle, "gpu_mmap_init");
	sum = launch_kernel(mem_handle, "gpu_mmap_kernel");
	printf("total sum [master]: %f\n", sum);
#if 1
	//host_printf_cuda_buff((char*)cuda_buffer,buffer_size);
        //print_device_mem(0,mem_handle,buffer_size);
        //write_random_data(0,mem_handle,buffer_size);
#endif
	/* export the above allocation to sharable handle */
	rc = cuMemExportToShareableHandle(&ipc_handle, mem_handle,
									  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
									  0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemExportToShareableHandle: %s", errorName(rc));

	return ipc_handle;
}

static void usage(char *argv0, int exitcode)
{
	fprintf(stderr, "usage: %s [options]\n"
			"  -d <GPU index>         : default 0\n"
			"  -n <num of children>   : default 2\n"
			"  -l <buffer size in MB> : default 1024MB\n"
			"  -v                     : enables verbose output\n"
			"  -h                     : print this message\n",
			basename(argv0));
	exit(1);
}

int main(int argc, char *argv[])
{
	int			sockfd;
	int			i, c;
	struct sockaddr_un addr;
	pthread_t	thread;
	serv_args_t	serv_args;

	while ((c = getopt(argc, argv, "d:n:l:hv")) >= 0)
	{
		switch (c)
		{
			case 'd':
				cuda_dindex = atoi(optarg);
				break;
			case 'n':
				num_children = atoi(optarg);
				if (num_children < 1 || num_children > 100)
					Elog("number of children is out of range: %d\n",
						 num_children);
				break;
			case 'l':
				buffer_size = atol(optarg);
				if (buffer_size < 128 || buffer_size > 65536)
					Elog("buffer size is out of range: %ld[MB]",
						 buffer_size);
				buffer_size <<= 20;
				break;
			case 'h':
				usage(argv[0], 0);
				break;
			case 'v':
				verbose = 1;
				break;
			default:
				usage(argv[0], 1);
				break;
		}
	}

	/*
	 * open unix domain server socket
	 */
	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		Elog("failed on socket(2): %m");

	addr.sun_family = AF_UNIX;
	sprintf(addr.sun_path, "/tmp/gpu_mmap_%u.sock", getpid());
	if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
		Elog("%u failed on bind(2): %m", getpid());

	if (listen(sockfd, num_children) < 0)
		Elog("failed on listen(2): %m");

	/*
	 * spawn child processes
	 */
	for (i=0; i < num_children; i++)
	{
		pid_t	child = fork();

		if (child == 0)
		{
			close(sockfd);
			return child_main(&addr);
		}
		else if (child < 0)
		{
			fprintf(stderr, "failed on fork(2): %m\n");
			killpg(0, SIGTERM);
			return 1;
		}
	}
	serv_args.sockfd = sockfd;
	serv_args.ipc_handle = export_gpu_memory();
	if (pthread_create(&thread, NULL, server_loop, &serv_args) != 0)
	{
		fprintf(stderr, "failed on pthread_create(3): %m\n");
		killpg(0, SIGTERM);
		return 1;
	}

	/* wait for exit of child processes */
	while (1)
	{
		int		status;

		if (wait(&status) != 0)
		{
			if (errno == EEXIST)
				break;
			Elog("failed on wait(2): %m");
		}
	}
	/* cleanup */
	unlink(addr.sun_path);

	return 0;
}
