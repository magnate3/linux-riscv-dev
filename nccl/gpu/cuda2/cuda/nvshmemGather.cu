/*
nvcc nvshmem_allgather_example.cu -o nvsh     -I/usr/include/nvshmem_12     -L/usr/lib/sbsa-linux-gnu/nvshmem/12     -lnvshmem_host -lnvshmem_device -lcudart     -arch=sm_100 -rdc=true -Wno-deprecated-gpu-targets
nvshmrun -np 4 ./nvsh
*/

// printf, snptinf, setenv, malloc, free
#include <cstdio>
#include <cstdlib>
// apis for one-sided, symmetric-memory communication across GPUs
#include <nvshmem.h>
#include <nvshmemx.h>
// dvice management, streams, event, memory copies
#include <cuda_runtime.h>

#define NGPUS            4 // number of GPUs
#define SHARD_ELEMS     30   // per GPU
constexpr int ROWS_PER_SHARD = 3;
constexpr int COLS            = SHARD_ELEMS / ROWS_PER_SHARD;
static_assert(ROWS_PER_SHARD * COLS == SHARD_ELEMS, "ROWS_PER_SHARD * COLS must equal SHARD_ELEMS"); // printing in a 2D grid
using TYPE = float; // easy switch 

// initializing each shard
__global__ void init_shard(TYPE* shard, int rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SHARD_ELEMS) {
        shard[idx] = rank * 1.0f + idx * 1e-3f; // flat index + offset
    }
}

int main(int argc, char** argv) {
    // heap configuration
    constexpr int OVERSUB_FACTOR = 4;
    size_t base     = (size_t)SHARD_ELEMS * NGPUS * sizeof(TYPE); // total data
    size_t sym_size = base * OVERSUB_FACTOR; // over-allocate 4x to avoid out-of-memory
    char buf[64];
    snprintf(buf, sizeof(buf), "%zu", sym_size);
    setenv("NVSHMEM_SYMMETRIC_SIZE", buf, /*overwrite=*/1);

    // NVSHMEM initialization
    nvshmem_init(); // bootsraps lib
    int rank   = nvshmem_my_pe(); // rach id
    int nprocs = nvshmem_n_pes(); // total count

    cudaSetDevice(rank); // pin each process to a unique GPU
    // non-blocking stream for communication work (overlap with default-stream compute)
    cudaStream_t comm_stream;
    cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);

    // CUDA events for benchmark
    cudaEvent_t start_ev, end_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&end_ev);

    // returns memory from symmetric heap visible at same address in all PEs
    TYPE* shard_buf  = (TYPE*)nvshmem_malloc(SHARD_ELEMS * sizeof(TYPE)); // space for this rank's local data
    TYPE* gather_buf = (TYPE*)nvshmem_malloc(SHARD_ELEMS * nprocs * sizeof(TYPE)); // space for all rank's data after all-gather

    // launch unit_shard on comm_stream
    int threads = 256;
    int blocks  = (SHARD_ELEMS + threads - 1) / threads;
    init_shard<<<blocks, threads, 0, comm_stream>>>(shard_buf, rank);  // shard
    nvshmemx_barrier_all_on_stream(comm_stream); // all ranks finish
    cudaStreamSynchronize(comm_stream); // waits kernel and barrier

    // copy shard to host, to print it
    {
        TYPE host_shard[SHARD_ELEMS];
        cudaMemcpy(host_shard, shard_buf,
                   SHARD_ELEMS * sizeof(TYPE),
                   cudaMemcpyDeviceToHost);
        printf("[Rank %d] local shard (%dx%d):\n", rank, ROWS_PER_SHARD, COLS);
        for (int r = 0; r < ROWS_PER_SHARD; ++r) {
            printf("  ");
            for (int c = 0; c < COLS; ++c) {
                printf("%0.3f ", host_shard[r * COLS + c]);
            }
            printf("\n");
        }
    }

    // sync print
    nvshmem_barrier_all();

    // benchmark all-gather
    nvshmemx_barrier_all_on_stream(comm_stream);
    cudaEventRecord(start_ev, comm_stream);

    size_t bytes = SHARD_ELEMS * sizeof(TYPE);
    for (int p = 0; p < nprocs; ++p) { // loop over rank
        TYPE* dst = gather_buf + p * SHARD_ELEMS;
        if (p == rank) {
            cudaMemcpyAsync(dst, shard_buf, bytes,
                            cudaMemcpyDeviceToDevice, comm_stream); // local device-to-device
        } else {
            nvshmem_getmem_nbi(dst, shard_buf, bytes, p); // non-blocking one-sided "get" from PE
        }
    }
    nvshmem_quiet(); // block until outstanding non-blocking complete
    nvshmemx_barrier_all_on_stream(comm_stream);
    cudaEventRecord(end_ev, comm_stream);
    cudaStreamSynchronize(comm_stream);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, end_ev);
    printf("[Rank %d] all-gather time: %.3f ms\n", rank, elapsed_ms); // all gather time

    // sync before print
    nvshmem_barrier_all();

    // rank 0 print gather buff 2D
    if (rank == 0) {
        int total_rows = ROWS_PER_SHARD * nprocs;
        TYPE* host_all = (TYPE*)malloc(total_rows * COLS * sizeof(TYPE));
        cudaMemcpy(host_all, gather_buf,
                   total_rows * COLS * sizeof(TYPE),
                   cudaMemcpyDeviceToHost);

        printf("[Rank %d] gather_buf (%dx%d):\n", rank, total_rows, COLS);
        for (int r = 0; r < total_rows; ++r) {
            printf("  ");
            for (int c = 0; c < COLS; ++c) {
                printf("%0.3f ", host_all[r * COLS + c]);
            }
            printf("\n");
        }
        free(host_all);
    }

    // cleanup
    cudaEventDestroy(start_ev);
    cudaEventDestroy(end_ev);
    cudaStreamDestroy(comm_stream);
    nvshmem_free(shard_buf);
    nvshmem_free(gather_buf);
    nvshmem_finalize();
    return 0;
}
