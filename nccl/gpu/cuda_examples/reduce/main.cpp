#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "kernel.h"

// a CPU reduction function for testing
// only a real reduction, just get an output of size = NUM_THREAD_BLOCKS
// since each thread block will output a value in its data range [block_id*NUM_THREADS_PER_BLOCK, (block_id+1)*NUM_THREADS_PER_BLOCK)
// so here we align the output of CPU with GPU. 
void cpu_reduction(float* input, float* output, unsigned int n, const unsigned int NUM_THREAD_BLOCKS, const unsigned int num_per_thread = 1) {
    for (size_t i = 0; i < NUM_THREAD_BLOCKS; i++) {
        float cur = 0.0;
        for (size_t j = 0; j < NUM_THREADS_PER_BLOCK * num_per_thread; j++) {
            cur += input[i * NUM_THREADS_PER_BLOCK * num_per_thread + j];
        }
        output[i] = cur;
    }
}

// fill the input data with random values
void fill_data(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // data[i] = static_cast<float>(rand()) / RAND_MAX;
        data[i] = 1.0f; // for simplicity, fill with 1.0
    }
}

// check result
void check(float* ref, float* output, const unsigned int n, int version) {
    for (unsigned int i = 0; i < n; ++i) {
        if (fabs(ref[i] - output[i]) > 1e-5) {
            std::cerr << "Error: Mismatch at index " << i << " for version " << version 
                      << ". Expected: " << ref[i] << ", Got: " << output[i] << std::endl;
            return;
        }
    }
    std::cout << "Kernel version " << version << " passed the check." << std::endl;
}


launch_kernel_t get_launch_kernel(int version) {
    switch (version) {
        case 0:
            return launch_reduce_0;
        case 1:
            return launch_reduce_1;
        case 2:
            return launch_reduce_2;
        default:
            std::cerr << "Error: Unsupported kernel version." << std::endl;
            return nullptr;
    }
}

void test_kernel(int version, float* input, float* ref, unsigned int number, unsigned int num_thread_blocks) {
    float* h_output = new float[num_thread_blocks]();
    // allocate memory on the GPU
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, number * sizeof(float));
    cudaMalloc((void**)&d_output, num_thread_blocks * sizeof(float));
    cudaMemcpy(d_input, input, number * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Data copied to GPU." << std::endl;

    launch_kernel_t kernel = get_launch_kernel(version);
    if (kernel) {
        kernel(d_input, d_output, number);
    }

    // error check
    cudaMemcpy(h_output, d_output, num_thread_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    check(ref, h_output, num_thread_blocks, version);

    // check performance 
    int warmup = 50;
    int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < warmup; i++) {
        kernel(d_input, d_output, number);
    }
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel(d_input, d_output, number);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel version " << version << " executed in " << milliseconds / iterations << " ms per iteration." << std::endl;
    std::cout << "Throughput: " << (number * sizeof(float)) / (milliseconds * 1e6) * iterations << " GB/s" << std::endl;

    // free memory
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}


int main(int argc, char* argv[]) {
    const unsigned int N = 2u * 1024u * 1024u * 1024u; // default size of input data
    // const unsigned int N = 1024u; // default size of input data

    // check cuda device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found." << std::endl;
        return 1;
    }
    cudaSetDevice(0); // Set to the first CUDA device
    std::cout << "Using CUDA device 0 for reduction." << std::endl;

    unsigned int num_thread_blocks = N / NUM_THREADS_PER_BLOCK;

    if (N % NUM_THREADS_PER_BLOCK != 0) {
        std::cerr << "Error: Number must be a multiple of " << NUM_THREADS_PER_BLOCK << "." << std::endl;
        return 1;
    }

    // allocate memory for input data
    float* input = new float[N];
    float* ref_0 = new float[num_thread_blocks]();
    float* ref_1 = new float[num_thread_blocks / NUM_PER_THREAD]();

    fill_data(input, N);
    std::cout << "Input data filled with random values." << std::endl;

    // perform CPU reduction for reference
    cpu_reduction(input, ref_0, N, num_thread_blocks);
    std::cout << "CPU reduction completed." << std::endl;

    // perform CPU reduction for reference with half the thread blocks
    cpu_reduction(input, ref_1, N, num_thread_blocks / NUM_PER_THREAD, NUM_PER_THREAD);
    std::cout << "CPU reduction for half thread blocks completed." << std::endl;

    // test the kernel
    // kernel version 0 is the original kernel, each thread load 1 value
    std::cout << "Testing kernel version 0..." << std::endl;
    test_kernel(0, input, ref_0, N, num_thread_blocks);
    std::cout << "Kernel version 0 test completed." << std::endl;

    // kernel version 1 is the optimized kernel, each thread load NUM_PER_THREAD values
    std::cout << "Testing kernel version 1..." << std::endl;
    test_kernel(1, input, ref_1, N, num_thread_blocks / NUM_PER_THREAD);
    std::cout << "Kernel version 1 test completed." << std::endl;

    // kernel version 2 is the optimized kernel, each thread load NUM_PER_THREAD values and use warp shuffle to reduce block-wise synchronization
    std::cout << "Testing kernel version 2..." << std::endl;
    test_kernel(2, input, ref_1, N, num_thread_blocks / NUM_PER_THREAD);
    std::cout << "Kernel version 2 test completed." << std::endl;

    // free memory
    delete[] input;
    delete[] ref_0;
    delete[] ref_1;
    std::cout << "Test completed successfully." << std::endl;
    return 0;
}