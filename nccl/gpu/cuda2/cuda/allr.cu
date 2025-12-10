#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <nccl.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_NCCL(call) \
    do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", \
                __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));

    if (numDevices < 8) {
        printf("This example requires at least two GPUs.\n");
        return 0;
    }

    const int numRanks = 8;
    int devices[numRanks] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> sizes = {4, 1<<20}; // Up to 16M floats

    std::ofstream csv("nccl_benchmark.csv");
    csv << "N,Device,Time_ms,Bandwidth_MBps,FreeMem_MB,TotalMem_MB\n";

    for (int N : sizes) {
        printf("\n--- Testing N = %d elements (%.2f MB per GPU) ---\n", N, N * sizeof(float) / (1024.0 * 1024));

        ncclComm_t comms[numRanks];
        float* sendbuffs[numRanks];
        float* recvbuffs[numRanks];
        cudaStream_t streams[numRanks];
        cudaEvent_t start[numRanks], stop[numRanks];

        // Setup
        for (int i = 0; i < numRanks; ++i) {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_CUDA(cudaMalloc(&sendbuffs[i], N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&recvbuffs[i], N * sizeof(float)));
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
            CHECK_CUDA(cudaEventCreate(&start[i]));
            CHECK_CUDA(cudaEventCreate(&stop[i]));

            std::vector<float> data(N, float(i + 1));
            CHECK_CUDA(cudaMemcpy(sendbuffs[i], data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        }

        CHECK_NCCL(ncclCommInitAll(comms, numRanks, devices));

        // Record start
        for (int i = 0; i < numRanks; ++i) {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_CUDA(cudaEventRecord(start[i], streams[i]));
        }

        // AllReduce
        for (int i = 0; i < numRanks; ++i) {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_NCCL(ncclAllReduce((const void*)sendbuffs[i],
                                     (void*)recvbuffs[i],
                                     N,
                                     ncclFloat,
                                     ncclSum,
                                     comms[i],
                                     streams[i]));
        }

        // Record stop
        for (int i = 0; i < numRanks; ++i) {
            CHECK_CUDA(cudaSetDevice(devices[i]));
            CHECK_CUDA(cudaEventRecord(stop[i], streams[i]));
            CHECK_CUDA(cudaEventSynchronize(stop[i]));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start[i], stop[i]));

            float bytes = 2.0f * N * sizeof(float); // read + write
            float mbps = bytes / (ms * 1000.0f); // MB/s

            size_t freeMem = 0, totalMem = 0;
            CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));

            printf("Device %d Time: %.3f ms | Bandwidth: %.3f MB/s | Free: %.1f MB | Total: %.1f MB\n",
                   devices[i], ms, mbps, freeMem / (1024.0 * 1024), totalMem / (1024.0 * 1024));

            csv << N << "," << devices[i] << "," << ms << "," << mbps << ","
                << (freeMem / (1024.0 * 1024)) << "," << (totalMem / (1024.0 * 1024)) << "\n";
        }

        // Cleanup
        for (int i = 0; i < numRanks; ++i) {
            ncclCommDestroy(comms[i]);
            cudaFree(sendbuffs[i]);
            cudaFree(recvbuffs[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(start[i]);
            cudaEventDestroy(stop[i]);
        }
    }

    csv.close();
    return 0;
}
