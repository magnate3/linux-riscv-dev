#include "timer.h"
#include "common.cuh"
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define NUMBER_OF_RUNS 1 // 22 //8
#define MASSAGES_PER_REQUEST_ARR { 200 }
// {1 25, 50, 100, 200, 300, 400, 500, 600, 700, 800 }
// { 1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000};
#define NUMBER_OF_REQUESTS 1000 // 100
#define TIMERS_OUTPUT_FILE "Results/timers"
#define MESSAGE_SIZE 400 // 80 // 5000
#define NUMBER_OF_THREADS 256 // 128
#define PIPE_QUEUE_SIZE_IN_MESSAGET SIZE_MULTIPLIER*NUMBER_OF_THREADS
#define CHECK_RECEIVED_MESSAGES false

using std::string;
using std::endl;
using std::ostream;

int SIZE_MULTIPLIER = 1;
int SYNC_RATE_TARGET = 5;

void SetTestArguments(int argc, char *argv[])
{
    int opt; 
      
    // put ':' in the starting of the 
    // string so that program can  
    //distinguish between '?' and ':'  
    while((opt = getopt(argc, argv, "m:r:")) != -1)  
    {  
        switch(opt)  
        {  
            case 'm': 
		printf("Update size multiplier to %s\n", optarg);
                SIZE_MULTIPLIER = atoi(optarg);
		break;

            case 'r': 
		printf("Update sync rate target to %s\n", optarg);
                SYNC_RATE_TARGET = atoi(optarg);
		break;
            
            case ':':  
                printf("option needs a value\n"); 
	       	exit(-1);	
                break;  

            case '?':  
                printf("unknown option: %c\n", optopt); 
		exit(-1);
                break;  
        }  
    }  
}

typedef struct {
	char content[MESSAGE_SIZE];
} message;

__host__ message create_message(int index) {
	message custome_message;
	int number_of_letters = ('z' - 'a' + 1);
	custome_message.content[0] = '0' + (index % 10);
	for (int i = 1; i < MESSAGE_SIZE - 1; i++) {
		//custome_message.content[i] = 'a' + (i-1)%number_of_letters;
		custome_message.content[i] = 'a' + (rand()) % number_of_letters;
	}

	custome_message.content[MESSAGE_SIZE - 1] = '\0';
	return custome_message;
}

__host__ message* create_messages(int number_of_messages) {
	message* messages_array = new message[number_of_messages];
	for (int i = 0; i < number_of_messages; i++) {
		messages_array[i] = create_message(i);
	}

	return messages_array;
}

__device__ inline bool compareMessages(message* expected, message* actual) {
	for (int i = 0; i < MESSAGE_SIZE; i++) {
		if (expected->content[i] != actual->content[i]) {
			printf("Error: expected \"%s\", but actual is \"%s\"\n",
					expected->content, actual->content);
			return false;
		}
	}

	return true;
}

__device__ inline bool compareMessagesArray(message* expected, message* actual,	int messeges_count)
{
	bool res = true;
	for (int i = 0; i < messeges_count; i++)
	{
		if (!compareMessages(&(expected[i]), &(actual[i]))) {
			res = false;
			printf("comparing massages: Failed at message #%d\n", i);
		}
	}

	return res;
}

int getSyncRate()
{
	size_t userBufferSize = NUMBER_OF_THREADS * sizeof(message);
	size_t maxSlotsPerMessage =  ROUND_UP_DIVISION(sizeof(message), sizeof(int));
	return getClosestFactor(SYNC_RATE_TARGET, maxSlotsPerMessage);
}

void save_results(GpuTimer results[NUMBER_OF_RUNS][NUMBER_OF_REQUESTS], int number_of_messages[], const char* title) {
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	char filePath[1000];
	sprintf(filePath, "%s_%s_QS%d_Req%d_Run%d_Thread%d_MS%d_CR%d_SR%d.csv",
		TIMERS_OUTPUT_FILE,
		title,
		PIPE_QUEUE_SIZE_IN_MESSAGET,
		NUMBER_OF_REQUESTS,
		NUMBER_OF_RUNS,
		NUMBER_OF_THREADS,
		MESSAGE_SIZE,
		prop.clockRate,
		getSyncRate());

	std::ofstream output(filePath);

	// Print titles
	for (int i = 0; i < NUMBER_OF_RUNS; i++) {
		output << "Run " << i + 1 << " Latency M[" << number_of_messages[i] * NUMBER_OF_THREADS	<< "],";
	}
	output << endl;

	// Print data
	for (int request = 0; request < NUMBER_OF_REQUESTS; request++) {
		for (int run = 0; run < NUMBER_OF_RUNS; run++) {
			output << results[run][request].Elapsed() << ",";
		}

		output << endl;
	}

	output.close();
}

__global__ void producer_kernel(Pipe *pipe, message* messages,
		int number_of_messages, GpuTimer timers[]) {
	const int tid = threadIdx.x;
	if (tid == 0)
		dbg_printf("Producer kernel start\n");

	for (int i = 0; i < NUMBER_OF_REQUESTS; i++) {
		__syncthreads();
		if (tid == 0)
			timers[i].Start();
		for (int j = 0; j < number_of_messages * NUMBER_OF_THREADS; j += NUMBER_OF_THREADS)
		{
			dbg_printf("tid %d - Producer kernel Write (R:%d, M:%d)\n", tid, i, j+tid);
			message* message_buffer = &(messages[j]);
			pipe->gwrite(message_buffer);
		}
		if (tid == 0)
			timers[i].Stop();
	}

	if (tid == 0)
		dbg_printf("Producer kernel finished\n");
}

__device__ void reduce(int* in, int* out, int length)
{
	int tid = threadIdx.x;
	int half_length = length / 2;

	// First iteration
	for (int i = tid; i < half_length; i += blockDim.x)
	{
		out[i] = in[i] + in[i + half_length];
	}
	half_length /= 2;
	__syncthreads();

	// Next iterations
	while (half_length > 0)
	{
		for (int i = tid; i < half_length; i += blockDim.x)
		{
			out[i] = out[i] + out[i + half_length];
		}

		__syncthreads();
		half_length /= 2;
	}
}

__device__ void consume(int* in, int* out, int reductionLength, int totalLength)
{
	reduce(in, out, reductionLength);

	if (threadIdx.x == 0)
	{
		for (int *i = in + reductionLength; i < in + totalLength; i++)
		{
			out[0] += *i;
		}
	}
}

__global__ void consumer_kernel(
		Pipe *pipe,
		message* messages,
		int number_of_messages,
		GpuTimer timers[],
		message* recived_messages,
		int *reductionArray)
{
	int reductionSize = (NUMBER_OF_THREADS * sizeof(message)) / sizeof(int);
	int length = pow(2, floor(log2f(reductionSize)));

	const int tid = threadIdx.x;
	if (tid == 0)
		dbg_printf("Consumer kernel start\n");

	for (int i = 0; i < NUMBER_OF_REQUESTS; i++) {
		__syncthreads();

		if (tid == 0)
			timers[i].Start();
		for (int j = 0; j < number_of_messages * NUMBER_OF_THREADS; j += NUMBER_OF_THREADS)
		{
			dbg_printf("tid %d - Consumer kernel Read (R:%d, M:%d)\n", tid, i, j+tid);
			message* message_buffer = &(recived_messages[j]);
			pipe->gread(message_buffer);
			consume((int*)message_buffer, reductionArray, reductionSize, length);
			// dbg_printf("tid %d - Consumer kernel: read message: %s\n", GetThreadNum(), recived_messages[j+tid].content);
		}

		if (tid == 0)
			timers[i].Stop();

		__syncthreads();

		if (tid == 0) dbg_printf("Finish internal test loop\n");
		if (tid == 0 && CHECK_RECEIVED_MESSAGES) {
			compareMessagesArray(messages, recived_messages, number_of_messages * NUMBER_OF_THREADS);
		}
	}

	if (tid == 0)
		dbg_printf("Consumer kernel finished\n");
}
