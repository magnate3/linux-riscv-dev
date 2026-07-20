#ifndef GPIPE
#define GPIPE

#include <cuda/atomic>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include "memoryManagmentHelper.cuh"
#include "connectionHelper.cuh"

#define PIPE_PATH "/tmp/"

typedef struct GPipeData
{
	cuda::atomic<size_t, cuda::thread_scope_system> _tail;
	cuda::atomic<size_t, cuda::thread_scope_system> _head;
	int *_messagesQueue;
} GPipeData_t;

typedef struct
{
	size_t _queue_size;
	size_t _threads_count;
	size_t _alloc_size;
} GpipeInitArguments;


#endif


