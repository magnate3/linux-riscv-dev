// much cleaner removed the non in unified memory checks.
#include <cuda/atomic>
#include <cstdio>
#include <string.h>
#include <unistd.h>

using namespace cuda;

#define SAFE(x) if (0 != x) { abort(); }

__global__ void consumer_gpu(atomic<int>* flag, int* data, int* result) {
    while (flag->load(memory_order_acquire) == 0) {}
    *result = *data;
}

__global__ void producer_gpu(atomic<int> *flag, int* data, int * result){
    // Producer sequences
    *data = 42;
    flag->store(1, memory_order_release);
}

__global__ void init_using_gpu_kernel(atomic<int> *flag, int *data){
    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
}

// I needed this weird way of passing pointers to pointers 
// Need to find a better way, but this is similar to the base code without using 
// function call, as this maps in the same way as of the base code.
void allocate_unified_mem(atomic<int> **flag_t, int **data_t, int **result_t){
    // Flag in unified memory
    SAFE(cudaMallocManaged(flag_t, sizeof(atomic<int>)));
    SAFE(cudaMallocManaged(data_t, sizeof(int)));
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(result_t, sizeof(int)));
}

void init(atomic<int> *flag, int* data, const char *initis){
    if (strcmp(initis, "cpu") == 0){
        // Initial values: data = <unknown>, flag = 0
        flag->store(0, memory_order_relaxed);
    }
    if (strcmp(initis, "gpu") == 0){
        // currently 1 block, 1 thread.
        init_using_gpu_kernel<<<1,1>>>(flag, data);
    }
}

void producer(atomic<int> *flag, int* data, int *result, const char *produceris){
    if (strcmp(produceris, "cpu") == 0){
        // Producer sequences
        *data = 42;
        flag->store(1, memory_order_release);
    }
    if (strcmp(produceris, "gpu") == 0){
        producer_gpu<<<1, 1>>>(flag, data, result);
    }
}

void consumer(atomic<int> *flag, int* data, int *result, const char *consumeris){
    if (strcmp(consumeris, "cpu") == 0){
        // Consumer sequences
        while (flag->load(memory_order_acquire) == 0) {}
        *result = *data;
    }
    if (strcmp(consumeris, "gpu") == 0){
        consumer_gpu<<<1, 1>>>(flag, data, result);
    }
}

int main(int argc, char* argv[]) {
    // define what all variables you need.
    atomic<int>* flag;
    int* data;
    int* result;

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));
    SAFE(cudaMallocManaged(&data, sizeof(int)));
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result, sizeof(int)));

    
    // allocate the unified memory 
    // Error: variable "result" is used before its value is se
    // using pointer-to-pointer solve above error.
    // allocate_unified_mem(&flag, &data, &result);

    // choose locations of threads
    const char *initis = "cpu";
    const char *produceris = "cpu";
    const char *consumeris = "gpu";

    // IPC
    init(flag, data, initis);
    consumer(flag, data, result, consumeris);
    sleep(10);
    producer(flag, data, result, produceris);
    
    
    // only synchronize if gpu is involved.
    if (!(strcmp(initis, "gpu") != 0 && strcmp(produceris, "gpu") != 0 && strcmp(consumeris, "gpu") != 0)) {
        // // Wait for consumer to finish
        SAFE(cudaDeviceSynchronize());
    }
    // // Print the result
    printf("%d (expected 42)\n", *result);

    return 0;
}
