#define Pipe NaiveGPipe<message>

#include "naiveGpipe.cu"
#include "testCommon.cu"

void RunPipeExperiment(GpuTimer h_consumerTimers[NUMBER_OF_REQUESTS], GpuTimer h_producerTimers[NUMBER_OF_REQUESTS], int msg_per_req)
{
	// Create reduction array
	int* d_reductionResults;
	CUDA_CHECK(cudaMalloc((void**)&d_reductionResults, sizeof(message)*NUMBER_OF_THREADS));

	// Create CUDA stream
	cudaStream_t consumer_stream;
	cudaStream_t producer_stream;
	CUDA_CHECK(cudaStreamCreate(&consumer_stream));
	CUDA_CHECK(cudaStreamCreate(&producer_stream));

	cudaEvent_t copy_done_event;
	cudaEventCreate(&copy_done_event);

	// Create Pipe
	Pipe h_pipe(PIPE_QUEUE_SIZE_IN_MESSAGET, NUMBER_OF_THREADS);
	h_pipe.init();
	Pipe* d_pipe;
	CUDA_CHECK(cudaMalloc((void**)&d_pipe, sizeof(Pipe)));
	CUDA_CHECK(cudaMemcpyAsync(d_pipe, &h_pipe, sizeof(Pipe), cudaMemcpyHostToDevice, consumer_stream));

	// Create messages array
	message *d_messages;
	message *h_messages = create_messages(msg_per_req*NUMBER_OF_THREADS);
	CUDA_CHECK(cudaMalloc((void**)&d_messages, sizeof(message)*msg_per_req*NUMBER_OF_THREADS));
	CUDA_CHECK(cudaMemcpyAsync(d_messages, h_messages, sizeof(message)*msg_per_req*NUMBER_OF_THREADS, cudaMemcpyHostToDevice, consumer_stream));

	cudaEventRecord(copy_done_event, consumer_stream);

	// Copy timers to device
	GpuTimer *d_consumerTimers, *d_producerTimers;
	CUDA_CHECK(cudaMalloc((void**)&d_consumerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS));
	CUDA_CHECK(cudaMalloc((void**)&d_producerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS));
	CUDA_CHECK(cudaMemcpyAsync(d_consumerTimers, h_consumerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyHostToDevice, consumer_stream));
	CUDA_CHECK(cudaMemcpyAsync(d_producerTimers, h_producerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyHostToDevice, producer_stream));

	cudaStreamWaitEvent(producer_stream, copy_done_event, 0);

	message * d_recived_messages_arr;
	CUDA_CHECK(cudaMalloc((void**)&d_recived_messages_arr, sizeof(message)*msg_per_req*NUMBER_OF_THREADS));

	// Run kernels
	CUDA_CHECK(cudaDeviceSynchronize());
	consumer_kernel <<<1, NUMBER_OF_THREADS, 0, consumer_stream>>> (d_pipe, d_messages, msg_per_req, d_consumerTimers, d_recived_messages_arr, d_reductionResults);
	producer_kernel <<<1, NUMBER_OF_THREADS, 0, producer_stream>>> (d_pipe, d_messages, msg_per_req, d_producerTimers);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Copy results to host
	CUDA_CHECK(cudaMemcpy(h_consumerTimers, d_consumerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_producerTimers, d_producerTimers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_pipe));
	CUDA_CHECK(cudaFree(d_consumerTimers));
	CUDA_CHECK(cudaFree(d_producerTimers));
	CUDA_CHECK(cudaFree(d_messages));
	CUDA_CHECK(cudaFree(d_recived_messages_arr));
	CUDA_CHECK(cudaFree(d_reductionResults));
	delete(h_messages);
}

void RunTest()
{
	int msg_per_req_arr[NUMBER_OF_RUNS] = MASSAGES_PER_REQUEST_ARR;
	GpuTimer h_consumerTimers[NUMBER_OF_RUNS][NUMBER_OF_REQUESTS];
	GpuTimer h_producerTimers[NUMBER_OF_RUNS][NUMBER_OF_REQUESTS];

	for (int run_num = 0; run_num < NUMBER_OF_RUNS; run_num++)
	{
		printf("Run number %d\n", run_num + 1);
		RunPipeExperiment(h_consumerTimers[run_num], h_producerTimers[run_num], msg_per_req_arr[run_num]);
	}

	const char* producer_title = "naive_pipe_producer";
	const char* consumer_title = "naive_pipe_consumer";
	save_results(h_producerTimers, msg_per_req_arr, producer_title);
	save_results(h_consumerTimers, msg_per_req_arr, consumer_title);
}

int main(int argc, char *argv[])
{
    SetTestArguments(argc, argv);
	printf("Start naive GPU pipe test\n");
	RunTest();
	printf("done");

	return 0;
}
