#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 0.13f;

}

int main() {
    float *d_data;
    float *h_data = (float*)malloc(1024 * sizeof(float));

    cudaMalloc((void**)&d_data, 1024 * sizeof(float));
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // kernel<<<grid dim, block dim, shared memory, stream>>>
    kernel<<<2, 512, 0, stream1>>>(d_data);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaStreamSynchronize(stream2);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        cudaFree(d_data);
        free(h_data);
        return -1;
    }

    // Verify results
    printf("Kernel executed successfully and data copied to host.\n");
    printf("Data[1022]: %f\n", h_data[1022]); // Print first
    printf("Data[1023]: %f\n", h_data[1023]); // Print second element to verify

    // Free resources
    cudaFree(d_data);
    free(h_data);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}