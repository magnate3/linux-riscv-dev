
#include <cuda/atomic>
#include <stdio.h>
#include "common.cuh"

template <typename message_t>
struct NaiveGPipe
{
	message_t *_messagesQueue;
	size_t _queue_size;
	size_t _threadsCount;
	cuda::atomic<size_t, cuda::thread_scope_device> _head;
	cuda::atomic<size_t, cuda::thread_scope_device> _tail;

	NaiveGPipe(size_t queue_size, size_t threadsCount)
	{
		_threadsCount = threadsCount;
		_queue_size = queue_size;
		_messagesQueue = nullptr;
	}

	__host__ ~NaiveGPipe()
	{
		CUDA_CHECK(cudaFree((void*)_messagesQueue));
	}

	__host__ void init()
	{
		CUDA_CHECK(cudaMalloc((void**)&_messagesQueue, _queue_size * sizeof(message_t)));
		_head.store(0, cuda::memory_order_seq_cst);
		_tail.store(0, cuda::memory_order_seq_cst);
	}

	__device__ void gread(message_t* message)
	{
		__shared__ int _shared_head;
		const int tid = GetThreadNum();

		if (tid==0)
		{
			_shared_head = _head.load(cuda::memory_order::memory_order_relaxed);
			while (_tail.load(cuda::memory_order::memory_order_relaxed) - _shared_head < _threadsCount);
			cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
		}

		__syncthreads();
		*(message + tid) = _messagesQueue[(_shared_head + tid) %  _queue_size];
		__syncthreads();

		if (tid == 0)
		{
			int new_head = _shared_head + _threadsCount;
			dbg_printf("tid %d - gread: update head position to %d\n",tid, new_head);
			_head.store(new_head, cuda::memory_order::memory_order_release);
		}

	}

	__device__ void gwrite(message_t* message)
	{
		__shared__ int _shared_tail;
		const int tid = GetThreadNum();

		if (tid==0)
		{
			_shared_tail = _tail.load(cuda::memory_order::memory_order_relaxed);
			while (_shared_tail - _head.load(cuda::memory_order::memory_order_relaxed) > _queue_size - _threadsCount);
			cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
		}

		__syncthreads();
		dbg_printf("gwrite Writing message: \"%s\"\n", (message + tid)->content);
		_messagesQueue[(_shared_tail + tid) % _queue_size] = *(message + tid);
		__syncthreads();

		if (tid == 0)
		{
			int new_tail = _shared_tail + _threadsCount;
			dbg_printf("gwrite: update tail position to %d\n", new_tail);
			_tail.store(new_tail, cuda::memory_order::memory_order_release);
		}
	}

	__device__ void clean_queue(message_t* emptyMessage)
	{
		for(int i = 0; i < this->_queue_size; i++)
			_messagesQueue[i] = *emptyMessage;
	}

	__device__ void write_many(message_t** messages, int number_of_messages)
	{
		dbg_printf("write many start\n");

		for (int i = 0; i < number_of_messages; i++)
		{
			write(messages[i]);
		}
	}
};
