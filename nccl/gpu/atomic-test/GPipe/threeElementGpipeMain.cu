#define Pipe ThreeElementGPipe<message>

#include "threeElementGpipe.cu"
#include "testCommon.cu"
#define PIPE_NAME "/tmp/myfifo2"

void runPipeExperiment(GpuTimer h_timers[NUMBER_OF_REQUESTS], int msg_per_req, bool isConsumer)
{
	// Create reduction array
	int* d_reductionResults;
	CUDA_CHECK(cudaMalloc((void**)&d_reductionResults, sizeof(message)*NUMBER_OF_THREADS));

	// Create messages array
	message *d_messages;
	message *h_messages = create_messages(msg_per_req*NUMBER_OF_THREADS);
	CUDA_CHECK(cudaMalloc((void**)&d_messages, sizeof(message)*msg_per_req*NUMBER_OF_THREADS));
	CUDA_CHECK(cudaMemcpy(d_messages, h_messages, sizeof(message)*msg_per_req*NUMBER_OF_THREADS, cudaMemcpyHostToDevice));

	// Create Pipe
	Pipe h_pipe(PIPE_NAME, PIPE_QUEUE_SIZE_IN_MESSAGET, NUMBER_OF_THREADS, isConsumer);
	h_pipe.init();
	Pipe* d_pipe;
	CUDA_CHECK(cudaMalloc((void**)&d_pipe, sizeof(Pipe)));
	CUDA_CHECK(cudaMemcpy(d_pipe, &h_pipe, sizeof(Pipe), cudaMemcpyHostToDevice));

	// Copy timers to device
	GpuTimer *d_timers;
	CUDA_CHECK(cudaMalloc((void**)&d_timers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS));
	CUDA_CHECK(cudaMemcpy(d_timers, h_timers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyHostToDevice));

	message * d_recived_messages_arr;
	CUDA_CHECK(cudaMalloc((void**)&d_recived_messages_arr, sizeof(message)*msg_per_req*NUMBER_OF_THREADS));

	CUDA_CHECK(cudaDeviceSynchronize());
	// Run kernels
	if (isConsumer)
		consumer_kernel <<<1, NUMBER_OF_THREADS, 0>>> (d_pipe, d_messages, msg_per_req, d_timers, d_recived_messages_arr, d_reductionResults);
	else
		producer_kernel <<<1, NUMBER_OF_THREADS, 0>>> (d_pipe, d_messages, msg_per_req, d_timers);

	CUDA_CHECK(cudaDeviceSynchronize());
	dbg_printf("%s: Finish kernel sync\n", isConsumer ? "Consumer" : "Producer");
	h_pipe.gclose();

	// Copy results to host
	CUDA_CHECK(cudaMemcpy(h_timers, d_timers, sizeof(GpuTimer)*NUMBER_OF_REQUESTS, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_pipe));
	CUDA_CHECK(cudaFree(d_timers));
	CUDA_CHECK(cudaFree(d_messages));
	CUDA_CHECK(cudaFree(d_recived_messages_arr));
	CUDA_CHECK(cudaFree(d_reductionResults));
	delete(h_messages);
}

void RunTest()
{
	int number_of_messages[NUMBER_OF_RUNS] = MASSAGES_PER_REQUEST_ARR;

	const char * testPipe = "testPipe";
	mkfifo(testPipe, 0666);

	int pid = fork();
	bool isConsumer = !(pid == 0);
	const char* process_name = (isConsumer) ? "consumer" : "producer";
	const char* doneMessage = DONE_MSG;
	char receivedMessage[DONE_MSG_LEN];
	int _fd = open(testPipe, (isConsumer) ? O_WRONLY : O_RDONLY );

	GpuTimer h_Timers[NUMBER_OF_RUNS][NUMBER_OF_REQUESTS];
	for (int run_num = 0; run_num < NUMBER_OF_RUNS; run_num++)
	{
		printf("%s: Run number %d\n", process_name, run_num + 1);
		runPipeExperiment(h_Timers[run_num], number_of_messages[run_num], isConsumer);
		dbg_printf("%s finished run number %d\n", process_name, run_num);

		// Sync
		if (isConsumer)
		{
			if (write(_fd, (void*)(doneMessage), sizeof(char) * 5) == -1)
			{
				perror("Consumer sync write");
				exit(-1);
			}
		}
		else
		{
			if (read(_fd, (void*)(receivedMessage), sizeof(char) * 5) == -1)
			{
				perror("Producer sync read");
				exit(-1);
			}

			if (strcmp(receivedMessage, doneMessage) != 0)
				dbg_printf("Error: Something went wrong, message is not done\n");
		}
	}

	char title[100];
	sprintf(title, "Three_Element_Gpipe_%s", process_name);
	save_results(h_Timers, number_of_messages, title);
}

int main(int argc, char *argv[])
{
    SetTestArguments(argc, argv);
	printf("Start three element GPU pipe test\n");
	RunTest();
	printf("done\n");

	return 0;
}

