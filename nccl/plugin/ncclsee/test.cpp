#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <nccl.h>
#ifdef __GNUC__
// For __rdtsc intrinsic on GCC/Clang (x86)
#include <x86intrin.h>
#endif

// ------------------------------------------------------------------
// From the prompt: calibrate() and gettime() for measuring time
// ------------------------------------------------------------------
static double freq = -1;

__attribute__((visibility("hidden"))) void calibrate() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t timeCycles = __rdtsc();
    double time = - tv.tv_sec * 1e6 - tv.tv_usec;
    uint64_t total = 0ULL;

    // Dummy loop to let some time pass
    for (int i = 0; i < 10000; i++) {
        total += __rdtsc();
    }
    (void)total; // avoid unused variable warning

    gettimeofday(&tv, NULL);
    timeCycles = __rdtsc() - timeCycles;  // Compute elapsed CPU cycles
    time += tv.tv_sec * 1e6 + tv.tv_usec; // Compute elapsed real-world time (microseconds)
    freq = timeCycles / time;             // cycles / microseconds => cycles per microsecond
}

__attribute__((visibility("hidden"))) double gettime() {
    // Return current timestamp in microseconds based on calibrated freq
    return __rdtsc() / freq;
}

// ------------------------------------------------------------------
// Helper macro for checking NCCL errors
// ------------------------------------------------------------------
#define NCCL_CALL(cmd) do {                            \
    ncclResult_t r = cmd;                               \
    if (r != ncclSuccess) {                             \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",      \
                __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

// ------------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------------
#define CUDA_CALL(cmd) do {                            \
    cudaError_t e = cmd;                               \
    if (e != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",     \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                            \
    }                                                  \
} while(0)

// ------------------------------------------------------------------
// Example main
// ------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // We will assume at least 2 GPUs are available
    // For demonstration, we'll use exactly 2.
    // (You can modify 'numGPUs' and 'devs' accordingly for more GPUs.)
    int numGPUs = 2;
    int devs[2] = {0, 1};  // Use GPU 0 and GPU 1

    // Amount of data per GPU (e.g., 256 floats)
    size_t N = 256;

    // Initialize device buffers and streams
    float* d_sendBuff[2];
    float* d_recvBuff[2];
    cudaStream_t streams[2];
    for (int i = 0; i < numGPUs; ++i) {
        CUDA_CALL(cudaSetDevice(devs[i]));
        CUDA_CALL(cudaMalloc(&d_sendBuff[i], N * sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_recvBuff[i], N * sizeof(float)));
        CUDA_CALL(cudaStreamCreate(&streams[i]));

        // (Optional) Initialize send buffers to some values
        // For demonstration, we won't do a separate host/device copy.
        // If needed, you can create a host buffer and cudaMemcpyAsync here.
        CUDA_CALL(cudaMemset(d_sendBuff[i], i + 1, N * sizeof(float)));
        CUDA_CALL(cudaMemset(d_recvBuff[i], 0,   N * sizeof(float)));
    }

    // Create NCCL communicators
    ncclComm_t comms[2];
    NCCL_CALL(ncclCommInitAll(comms, numGPUs, devs));

    // Calibrate TSC frequency before timing
    calibrate();

    // Measure start time
    double start = gettime();

    // Do a small for-loop with multiple AllReduce calls
    int numIterations = 10;
    for (int iter = 0; iter < numIterations; ++iter) {
        // Launch an AllReduce on each device
        for (int i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(devs[i]));
            // Each GPU calls ncclAllReduce
            NCCL_CALL(ncclAllReduce(
                (const void*)d_sendBuff[i],
                (void*)d_recvBuff[i],
                N,
                ncclFloat,
                ncclSum,
                comms[i],
                streams[i]));
        }
        // Synchronize all streams to ensure the operation completes
        // before next iteration (or measure outside the loop if you prefer).
        for (int i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(devs[i]));
            CUDA_CALL(cudaStreamSynchronize(streams[i]));
        }
    }

    // Measure end time
    double end = gettime();

    double elapsed = end - start; // in microseconds

    // Print timing result
    printf("Total time for %d iterations of ncclAllReduce: %.3f microseconds\n",
           numIterations, elapsed);
    printf("Average per iteration: %.3f microseconds\n",
           elapsed / numIterations);

    // Cleanup
    for (int i = 0; i < numGPUs; ++i) {
        NCCL_CALL(ncclCommDestroy(comms[i]));
        CUDA_CALL(cudaFree(d_sendBuff[i]));
        CUDA_CALL(cudaFree(d_recvBuff[i]));
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
