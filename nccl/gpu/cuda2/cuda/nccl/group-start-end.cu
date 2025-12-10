// management of multiple gpus (with and withoug groupstart/groupend)
 
#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define NUM_DEVICES 2
#define COUNT 1024

// Error-checking macros.
#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                  \
    if( e != cudaSuccess ) {                              \
        printf("Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
} while(0)

#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t r = cmd;                                  \
    if (r != ncclSuccess) {                                \
        printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
} while(0)

int main() {
    int nDevices = NUM_DEVICES;
    int devices[NUM_DEVICES] = {0, 1};
    ncclComm_t comms[NUM_DEVICES];
    cudaStream_t streams[NUM_DEVICES];
    float* d_buffers[NUM_DEVICES];

    // --------------------------
    // NCCL Communicator Setup
    // --------------------------
    // Here we use ncclCommInitAll to initialize communicators for all local GPUs.
    NCCLCHECK(ncclCommInitAll(comms, nDevices, devices));

    // --------------------------
    // Allocate device memory and create streams.
    // --------------------------
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMalloc(&d_buffers[i], COUNT * sizeof(float)));
        // Initialize the device memory to zeros.
        CUDACHECK(cudaMemset(d_buffers[i], 0, COUNT * sizeof(float)));
    }

    // --------------------------
    // Create CUDA events for timing.
    // --------------------------
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    float timeWithoutGroup = 0.0f;
    float timeWithGroup = 0.0f;

    // ****************** Without Grouping ****************** //
    // Each NCCL call is issued one-by-one.
    CUDACHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        NCCLCHECK(ncclAllReduce(
            (const void*)d_buffers[i],
            (void*)d_buffers[i],
            COUNT,
            ncclFloat,
            ncclSum,
            comms[i],
            streams[i]
        ));
    }
    // Ensure that all operations have completed.
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaEventElapsedTime(&timeWithoutGroup, start, stop));
    printf("Time without group: %f ms\n", timeWithoutGroup);

    // ****************** With Grouping ****************** //
    // The NCCL group call allows concurrent enqueuing across devices.
    CUDACHECK(cudaEventRecord(start, 0));
    ncclGroupStart();
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        NCCLCHECK(ncclAllReduce(
            (const void*)d_buffers[i],
            (void*)d_buffers[i],
            COUNT,
            ncclFloat,
            ncclSum,
            comms[i],
            streams[i]
        ));
    }
    ncclGroupEnd();
    // Synchronize all streams.
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    CUDACHECK(cudaEventRecord(stop, 0));
    CUDACHECK(cudaEventSynchronize(stop));
    CUDACHECK(cudaEventElapsedTime(&timeWithGroup, start, stop));
    printf("Time with group: %f ms\n", timeWithGroup);

    // --------------------------
    // Cleanup: Free device memory, destroy streams and communicators.
    // --------------------------
    for (int i = 0; i < nDevices; i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaFree(d_buffers[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    return 0;
}
