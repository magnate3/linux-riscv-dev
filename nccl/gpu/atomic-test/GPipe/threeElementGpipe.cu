
#include <cuda/atomic>
#include <stdio.h>
#include "common.cuh"
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <errno.h>

#define DONE_MSG "done"
#define DONE_MSG_LEN 5

using std::string;
using std::thread;

template <typename message_t>
struct ThreeElementGPipe;

template <typename message_t>
__host__ void producer_mediator(ThreeElementGPipe<message_t>* p)
{
	dbg_printf("Producer mediator has started\n");
	while (true)
	{
		size_t head = p->_head->load(cuda::memory_order::memory_order_relaxed);

		while (p->_tail->load(cuda::memory_order::memory_order_relaxed) - head == 0)
		{
			if (p->terminateFlag)
			{
				if (p->_tail->load(cuda::memory_order::memory_order_acquire) - head == 0)
				{
					cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
					dbg_printf("producer_mediator terminated\n");
					return;
				}
			}
		}

		cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);

		size_t writeSize = write(p->_fd, &(p->_messagesQueue[head % p->_queue_size]), sizeof(message_t));
		if (writeSize != sizeof(message_t))
		{
			perror("producer_mediator: ");
			dbg_printf("Error: producer_mediator: Wrong write size: %lu\n", writeSize);
		}
		else
		{
			p->_head->store(head + 1, cuda::memory_order::memory_order_release);
		}
	}
}

template <typename message_t>
__host__ void consumer_mediator(ThreeElementGPipe<message_t>* p)
{
	dbg_printf("Consumer mediator: start\n");
	while (true)
	{
		const size_t tail = p->_tail->load(cuda::memory_order::memory_order_relaxed);
		while (tail - p->_head->load(cuda::memory_order::memory_order_relaxed) == p->_queue_size) { }

		cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);

		switch(read(p->_fd, &(p->_messagesQueue[tail % p->_queue_size]), sizeof(message_t)))
		{
			case 0:
				break;

			case sizeof(message_t):
				dbg_printf("Consumer mediator transfer message: %s\n", p->_messagesQueue[tail % p->_queue_size].content);
				p->_tail->store(tail + 1, cuda::memory_order::memory_order_release);
				break;

			case -1:
				perror("Error: consumer_mediator: named pipe read failed\n");
				exit(-1);

			default:
				char* receivedMessage = (char*)(p->_messagesQueue + tail % p->_queue_size);
				if (strcmp(receivedMessage, "done") == 0)
				{
					dbg_printf("consumer_mediator terminated\n");
					return;
				}

				dbg_printf("Error: consumer_mediator: Read partial message %s\n", receivedMessage);
				break;
		}
	}
}

template <typename message_t>
struct ThreeElementGPipe
{
	message_t *_messagesQueue;
	size_t _queue_size;
	size_t _threadsCount;
	cuda::atomic<size_t, cuda::thread_scope_system>* _head;
	cuda::atomic<size_t, cuda::thread_scope_system>* _tail;
	const char* _fullPath;
	bool _isConsumer;
	int _fd;
	thread *mediator_thread;
	bool terminateFlag;

	ThreeElementGPipe(const char* pipe_name, size_t queue_size, size_t threadsCount, bool isConsumer)
	{
		_fullPath = pipe_name;
		_threadsCount = threadsCount;
		_queue_size = queue_size;
		_isConsumer = isConsumer;
		_messagesQueue = nullptr;
		mediator_thread = nullptr;
		_head = nullptr;
		_tail = nullptr;
		_fd = -1;
		terminateFlag = false;
	}

	__host__ void gclose()
	{
		terminateFlag = true;
		mediator_thread->join();
		if (!_isConsumer)
			write(_fd, "done", 5*sizeof(char));
		delete(mediator_thread);
		CUDA_CHECK(cudaFreeHost((void*)_messagesQueue));
		close(_fd);
	}

	__host__ void initConsumer()
	{
		dbg_printf("Init consumer with pipe %s\n", _fullPath);
		mkfifo(_fullPath, 0666); // S_IRUSR | S_IWOTH
		_fd = open(_fullPath, O_RDONLY);
		dbg_printf("Consumer pipe is connected\n");

		// Validate pipe parameters
		size_t producer_queue_size, producer_readers_count;
		if (read(_fd, &producer_queue_size, sizeof(size_t)) != sizeof(size_t))
		{
			perror("Error: Consumer queue size");
			dbg_printf("Error: Consumer queue size: Read smaller than expected");
		}
		dbg_printf("Consumer: Read queue size %d\n", (int)producer_queue_size);
		if (read(_fd, &producer_readers_count, sizeof(size_t)) != sizeof(size_t))
		{
			perror("Error: Consumer readers count");
			dbg_printf("Error: Consumer readers count: Read smaller than expected");
		}
		dbg_printf("Consumer: Read readers count %d\n", (int)producer_readers_count);
		if (producer_queue_size != _queue_size || producer_readers_count != _threadsCount)
		{
			perror("Error: Consumer compare args");
			dbg_printf("Error: Mismatching pipe arguments!");
		}

		CUDA_CHECK(cudaMallocHost((void**)&_messagesQueue, _queue_size * sizeof(message_t)));
		mediator_thread = new thread(consumer_mediator<message_t>, this);
	}

	__host__ void initProducer()
	{
		dbg_printf("Init producer with pipe %s\n", _fullPath);
		mkfifo(_fullPath, 0666); // S_IWUSR | S_IROTH
		_fd = open(_fullPath, O_WRONLY);
		dbg_printf("Producer pipe is connected\n");

		// Validate pipe parameters
		dbg_printf("Producer queue size %d\n", (int)_queue_size);
		int write_size = write(_fd, (void*)(&_queue_size), sizeof(size_t));
		if (write_size != sizeof(size_t))
		{
			perror("Error: Producer write size");
			dbg_printf("Error: Write smaller than expected: %d\n", write_size);
		}

		dbg_printf("Producer readers count %d\n", (int)_threadsCount);
		if (write(_fd, (void*)(&_threadsCount), sizeof(size_t)) != sizeof(size_t))
		{
			perror("Error: Producer readers count");
			dbg_printf("Error: Write smaller than expected\n");
		}

		dbg_printf("Producer pipe passed all arguments\n");
		CUDA_CHECK(cudaMallocHost((void**)&_messagesQueue, _queue_size * sizeof(message_t)));
		mediator_thread = new thread(producer_mediator<message_t>, this);
		dbg_printf("Producer pipe finished\n");
	}

	__host__ void init()
	{
		CUDA_CHECK(cudaMallocHost((void **)(&_head), sizeof(cuda::atomic<size_t>)));
		new(_head) cuda::atomic<size_t, cuda::thread_scope_system>(0);
		CUDA_CHECK(cudaMallocHost((void **)(&_tail), sizeof(cuda::atomic<size_t>)));
		new(_tail) cuda::atomic<size_t, cuda::thread_scope_system>(0);

		if (_isConsumer)
			initConsumer();
		else
			initProducer();
	}

	__device__ void gread(message_t* message)
	{
		__shared__ int _shared_head;
		const int tid = GetThreadNum();

		if (tid == 0)
		{
			_shared_head = _head->load(cuda::memory_order::memory_order_relaxed);
			while (_tail->load(cuda::memory_order::memory_order_relaxed) - _shared_head < _threadsCount);
			cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
		}

		__syncthreads();
		memcpy(message + tid, &(_messagesQueue[(_shared_head + tid) %  _queue_size]), sizeof(message_t));
		__syncthreads();

		if (tid == 0)
		{
			_head->store(_shared_head + _threadsCount, cuda::memory_order::memory_order_release);
		}
	}

	__device__ void gwrite(message_t* message)
	{
		__shared__ int _shared_tail;
		const int tid = GetThreadNum();

		if (tid == 0)
		{
			_shared_tail = _tail->load(cuda::memory_order::memory_order_relaxed);
			while (_shared_tail - _head->load(cuda::memory_order::memory_order_relaxed) > _queue_size - _threadsCount);
			cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
		}

		__syncthreads();
		memcpy(&(_messagesQueue[(_shared_tail + tid) % _queue_size]), message + tid, sizeof(message_t));
		__syncthreads();

		if (tid == 0)
		{
			_tail->store(_shared_tail + _threadsCount, cuda::memory_order::memory_order_release);
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
		for(int i = 0; i < this->_queue_size; i++)
			_messagesQueue[i] = *emptyMessage;
	}
};
