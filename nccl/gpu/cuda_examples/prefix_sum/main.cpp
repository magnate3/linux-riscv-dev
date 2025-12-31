#include <iostream>
#include <cmath>
#include "kernel.h"

void fill_data(int* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1; // assign int value 1
    }
}

// prefix sum, exclusive scan
void cpu_prefix_sum(int* input, int* output, const unsigned int n) {
    output[0] = 0;
    for (unsigned int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

void check(int* ref, int* output, const unsigned int n, int version) {
    std::cout << "output[100] = " << output[100] << std::endl;
    std::cout << "ref[100] = " << ref[100] << std::endl;
    for (unsigned int i = 0; i < n; ++i) {
        if (ref[i] != output[i]) {
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
            return launch_prefix_sum_v0;
        default:
            std::cerr << "Error: Unsupported kernel version." << std::endl;
            return nullptr;
    }
}

void test_kernel(int version, int* input, int* ref, unsigned int number) {
    int* h_output = new int[number]();
    // allocate memory on the GPU
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, number * sizeof(int));
    cudaMalloc((void**)&d_output, number * sizeof(int));
    cudaMemcpy(d_input, input, number * sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Data copied to GPU." << std::endl;

    launch_kernel_t kernel = get_launch_kernel(version);
    if (kernel) {
        kernel(d_input, d_output, number);
    }

    // error check
    cudaMemcpy(h_output, d_output, number * sizeof(int), cudaMemcpyDeviceToHost);
    check(ref, h_output, number, version);

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
    std::cout << "Throughput: " << (number * sizeof(int)) / (milliseconds * 1e6) * iterations << " GB/s" << std::endl;

    // free memory
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const unsigned int N = 1024u * 1024u * 1024u; // default size of input data
    // const unsigned int N = 2u * 1024u * 1024u; // alternative size
    int* input = new int[N];
    int* ref = new int[N]();

    // initialize input data
    fill_data(input, N);

    cpu_prefix_sum(input, ref, N);
    std::cout << "CPU prefix sum computed." << std::endl;

    int version = 0; // kernel version to test
    test_kernel(version, input, ref, N);
}