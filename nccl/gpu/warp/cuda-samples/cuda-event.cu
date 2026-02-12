#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 0.13f;
}

int main() {
    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    float *h_data = (float*)malloc(1024 * sizeof(float));
    float *d_data;
    cudaMalloc((void**)&d_data, 1024 * sizeof(float));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpyAsync(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(start, stream1);

    cudaStreamWaitEvent(stream2, start, 0);
    kernel<<<2, 512, 0, stream2>>>(d_data);
    cudaEventRecord(stop, stream2);

    cudaStreamWaitEvent(stream1, stop, 0);
    cudaMemcpyAsync(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);

    // Verify results
    printf("Kernel executed successfully and data copied to host.\n");
    printf("Data[1022]: %f\n", h_data[1022]); // Print first
    printf("Data[1023]: %f\n", h_data[1023]); // Print second element to verify

    // Free resources
    cudaFree(d_data);
    free(h_data);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}