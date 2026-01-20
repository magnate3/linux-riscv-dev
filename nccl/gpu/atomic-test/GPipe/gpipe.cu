#include "gpipe.cuh"
#include <math.h>

typedef struct PipeConsts
{
	size_t queueSize;
	size_t blockSize;
	size_t blockSlotCount;
	size_t queueSlotCount;

	size_t userBufferSlotCount;
	size_t LastSlotOccupancy;
	size_t maxBlocksPerThread;

	size_t blockSyncFactor;
} pipeConsts;

template <typename message_t>
struct GPipe
{
	GPipeData_t *_gpipeData;
	pipeConsts _pipeConsts;
	size_t _threadsCount;
	memoryProperties _memProperties;
	const char* _fullPath;
	bool _isConsumer;

	__host__ pipeConsts getPipeSizeData(int threadsCount, int size_multiplier, int sync_rate_target)
	{
		pipeConsts consts;
		size_t userBufferSize = threadsCount * sizeof(message_t);
		size_t maxSlotsPerMessage =  ROUND_UP_DIVISION(sizeof(message_t), sizeof(int));
		consts.blockSyncFactor = getClosestFactor(sync_rate_target, maxSlotsPerMessage);

		consts.blockSlotCount = threadsCount * consts.blockSyncFactor;
		consts.blockSize = consts.blockSlotCount  * sizeof(int);
		consts.queueSize = ROUND_UP(size_multiplier * userBufferSize, consts.blockSize);
		consts.queueSlotCount = consts.queueSize / sizeof(int);

		consts.LastSlotOccupancy = (userBufferSize % sizeof(int));
		consts.LastSlotOccupancy = consts.LastSlotOccupancy == 0 ? 4 : consts.LastSlotOccupancy;
		consts.userBufferSlotCount = ROUND_UP_DIVISION(userBufferSize, sizeof(int));
		consts.maxBlocksPerThread = ROUND_UP_DIVISION(consts.userBufferSlotCount, consts.blockSlotCount);

		// Print all const
		dbg_printf("Block size is %lu\n", consts.blockSize);
		dbg_printf("Block slot count is %lu\n", consts.blockSlotCount);
		dbg_printf("Queue size is %lu\n", consts.queueSize);
		dbg_printf("Queue slot count is %lu\n", consts.queueSlotCount);
		dbg_printf("Last slot Occupancy is %lu\n", consts.LastSlotOccupancy);
		dbg_printf("userBuffer slot count is %lu\n", consts.userBufferSlotCount);
		dbg_printf("Max slots per thread count is %lu\n", consts.maxBlocksPerThread);
		dbg_printf("Sync rate is set to %lu\n", consts.blockSyncFactor);

		return consts;
	}

	__host__ char* getPipeFullPath(const char* pipe_name)
	{
		const char* path = PIPE_PATH;
		char *name = new char[strlen(pipe_name) + strlen(path) + 1];
		sprintf(name, "%s%s", path, pipe_name);
		return name;
	}

	GPipe(const char* pipe_name, bool isConsumer, int size_multiplier, int threadsCount, int sync_rate_target)
	{
		_fullPath = getPipeFullPath(pipe_name);
		_pipeConsts = getPipeSizeData(threadsCount, size_multiplier, sync_rate_target);
		_threadsCount = threadsCount;
		_isConsumer = isConsumer;
		_gpipeData = nullptr;
	}

	__host__ void gclose()
	{
		cleanMemoryMaping(_memProperties);
	}

	__host__ void initGpipeData(GPipeData_t *gpipeData)
	{
		cuda::atomic<size_t, cuda::thread_scope_system> *head;
		cuda::atomic<size_t, cuda::thread_scope_system> *tail;
		head = new cuda::atomic<size_t, cuda::thread_scope_system>(0);
		tail = new cuda::atomic<size_t, cuda::thread_scope_system>(0);

		int* data = new int[_pipeConsts.queueSlotCount];
		int *ptr = (int*)((char*)(_gpipeData) + sizeof(GPipeData_t));

		CUDA_CHECK(cudaMemcpy(&(gpipeData->_head), head, sizeof(*head), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(&(gpipeData->_tail), tail, sizeof(*tail), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(&(gpipeData->_messagesQueue), &ptr, sizeof(int*), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(ptr, data, _pipeConsts.queueSize, cudaMemcpyHostToDevice));
		delete(data);
	}

	__host__ void initConsumer()
	{
		dbg_printf("Initialize consumer with socket name %s\n", _fullPath);

		// Create sharable memory
		ShareableHandle ipc_handle;
		size_t alloc_size = sizeof(GPipeData_t) + _pipeConsts.queueSize;
		_memProperties = allocateSharableMemory(alloc_size, &ipc_handle);
		dbg_printf("Consumer initialize: finish allocation\n");
		_gpipeData = (GPipeData_t *)(_memProperties.ptr);
		initGpipeData(_gpipeData);

		// Create arguments
		GpipeInitArguments initArguments;
		initArguments._queue_size = _pipeConsts.queueSize;
		initArguments._threads_count = _threadsCount;
		initArguments._alloc_size = _memProperties.alloc_size;

		// Send arguments to producer
		dbg_printf("Consumer queue size %lu\n", initArguments._queue_size);
		dbg_printf("Consumer readers count %lu\n", initArguments._threads_count);
		dbg_printf("Consumer allocation size %lu\n", initArguments._alloc_size);

		dbg_printf("Sending arguments to producer...\n");
		int socket_fd = create_socket(_fullPath);
		send_args(socket_fd, ipc_handle, &initArguments, sizeof(GpipeInitArguments));
		ipcCloseSocket(socket_fd, _fullPath);

		close(ipc_handle);
		dbg_printf("Consumer: successfully sent all arguments to producer\n");
	}

	__host__ void initProducer()
	{
		dbg_printf("Initialize producer with socket name %s\n", _fullPath);
		dbg_printf("Producer queue size %d\n", (int)_pipeConsts.queueSize);
		dbg_printf("Producer readers count %d\n", (int)_threadsCount);

		GpipeInitArguments initArguments;
		ShareableHandle ipc_handle = recv_arguments(_fullPath, &initArguments, sizeof(GpipeInitArguments));

		// Validate initialize arguments
		if (initArguments._queue_size != _pipeConsts.queueSize || initArguments._threads_count != _threadsCount)
		{
			dbg_printf("Error: Mismatching pipe arguments! \n");
			exit(-1);
		}

		_memProperties = importAndMapMemory(ipc_handle, initArguments._alloc_size);
		_gpipeData = (GPipeData_t *)(_memProperties.ptr);

		dbg_printf("Producer successfully initialized\n");
	}

	__host__ void init()
	{
		validateDeviceIsSupported();

		if (_isConsumer)
			initConsumer();
		else
			initProducer();
	}

	__device__ void gread(message_t *messages_buffer)
	{
		const int tid = GetThreadNum();

		// Set pipe wrapper
		GPipeData_t *data = (GPipeData_t *)_gpipeData;
		int *buffer = (int*)messages_buffer;

		int buffer_offset = tid;
		for (int b = 0; b < _pipeConsts.maxBlocksPerThread; b++)
		{
			__shared__ size_t _shared_head;
			if (tid == 0)
			{
				dbg_printf("Read - wait for messages\n");
				_shared_head = data->_head.load(cuda::memory_order::memory_order_relaxed);
				while (data->_tail.load(cuda::memory_order::memory_order_relaxed) - _shared_head < _pipeConsts.blockSlotCount);
				cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
				dbg_printf("Read data from head %lu\n", _shared_head);
			}

			__syncthreads();

			int queue_offset = _shared_head + tid;
			for (int k = 0; k < _pipeConsts.blockSyncFactor; k++)
			{
				if (buffer_offset < _pipeConsts.userBufferSlotCount - 1)
				{
					int queue_offset_modulo = (queue_offset %  _pipeConsts.queueSlotCount);
					// dbg_printf("Read data from index %d in queue size to read offset %d \n", queue_offset, buffer_offset);
					memcpy(buffer + buffer_offset, &(data->_messagesQueue[queue_offset_modulo]), sizeof(int));
				}
				// Is last slot
				else if (buffer_offset == _pipeConsts.userBufferSlotCount - 1)
				{
					dbg_printf("Read remain bytes\n");
					memcpy(buffer + buffer_offset, &(data->_messagesQueue[queue_offset %  _pipeConsts.queueSlotCount]), _pipeConsts.LastSlotOccupancy);
				}
				else
				{
					break;
				}

				queue_offset += _threadsCount;
				buffer_offset += _threadsCount;
			}

			__syncthreads();

			if (tid == 0)
			{
				data->_head.store(_shared_head + _pipeConsts.blockSlotCount, cuda::memory_order::memory_order_release);
			}
		}

		if (tid == 0) dbg_printf("Finish read message\n");
	}

	__device__ void gwrite(message_t *messages_buffer)
	{
		const int tid = GetThreadNum();

		// Set pipe wrapper
		GPipeData_t *data = (GPipeData_t*)_gpipeData;
		int *buffer = (int*)messages_buffer;

		int buffer_offset = tid;
		for (int b = 0; b < _pipeConsts.maxBlocksPerThread; b++)
		{
			__shared__ size_t _shared_tail;
			if (tid == 0)
			{
				dbg_printf("Write - wait for slots\n");
				_shared_tail = data->_tail.load(cuda::memory_order::memory_order_relaxed);
				while (_shared_tail - data->_head.load(cuda::memory_order::memory_order_relaxed) > _pipeConsts.queueSlotCount - _pipeConsts.blockSlotCount);
				cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
				dbg_printf("Write tail number %lu\n", _shared_tail);
				dbg_printf("Write head number %lu\n", data->_head.load(cuda::memory_order::memory_order_relaxed));
			}

			__syncthreads();

			int queue_offset = _shared_tail + tid;
			for (int k = 0; k < _pipeConsts.blockSyncFactor; k++)
			{
				if (buffer_offset < _pipeConsts.userBufferSlotCount - 1)
				{
					int queue_offset_modulo = (queue_offset %  _pipeConsts.queueSlotCount);
					// dbg_printf("Write data from index %d in queue size to buffer in %d \n", queue_offset, buffer_offset);
					memcpy(&(data->_messagesQueue[queue_offset_modulo]), buffer + buffer_offset, sizeof(int));
				}
				else if (buffer_offset == _pipeConsts.userBufferSlotCount - 1)
				{
					dbg_printf("Write remain %llu bytes\n", _pipeConsts.LastSlotOccupancy);
					memcpy(&(data->_messagesQueue[queue_offset % _pipeConsts.queueSlotCount]), buffer + buffer_offset, _pipeConsts.LastSlotOccupancy);
				}
				else
				{
					break;
				}

				queue_offset += _threadsCount;
				buffer_offset += _threadsCount;
			}

			__syncthreads();

			if (tid == 0)
			{
				data->_tail.store(_shared_tail + _pipeConsts.blockSlotCount, cuda::memory_order::memory_order_release);
			}
		}
	}

	__device__ void write_many(message_t** messages, int number_of_messages)
	{
		dbg_printf("write many start\n");

		for (int i = 0; i < number_of_messages; i++)
		{
			gwrite(messages[i]);
		}
	}

	__device__ void clean_queue(message_t* emptyMessage)
	{
		for(int i = 0; i < this->_queueSlotCount; i++)
			_gpipeData->_messagesQueue[i] = *emptyMessage;
	}
};
