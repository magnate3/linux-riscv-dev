#include "multimem.cuh"

constexpr int NUM_DEVICES = 8;
constexpr size_t SIZE = 1024 * 1024 * 1024;

constexpr int NUM_ITERS = 3;
constexpr int NUM_WARMUP_ITERS = 1;

int main() {

    assert(NUM_DEVICES > 1);
    assert(SIZE >= 1024 * 1024 && SIZE % (1024 * 1024) == 0);
    
    // Initialize the mc struct
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;
    mc_object<float> *mc;
    mc_malloc<float>(&mc, device_ids, NUM_DEVICES, SIZE);

    // Setup the data
    assert(SIZE % sizeof(float) == 0);

    int nelem = SIZE / sizeof(float);
    float **host_mats = (float**)malloc(NUM_DEVICES * sizeof(float*));
    srand(static_cast<unsigned int>(time(nullptr))); // random seed
    
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] = (float*)malloc(SIZE);
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            host_mats[dev_idx][i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (i < 10)
                printf("%f ", host_mats[dev_idx][i]);
        }
        printf("... (%d elements)\n", nelem);
    }

    float *expected = (float*)malloc(SIZE);
    printf("Expected: ");
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            expected[i] += host_mats[dev_idx][i];
        }
        for (int iter = 1; iter < NUM_ITERS + NUM_WARMUP_ITERS; ++iter) {
            expected[i] *= NUM_DEVICES; 
        }
        if (i < 10)
            printf("%f ", expected[i]);
    }
    printf("... (%d elements)\n", nelem);

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy((void*)mc[dev_idx].mem_va_, host_mats[dev_idx], SIZE, cudaMemcpyHostToDevice));
    }

    // Perform the reduction
    KittenGang gang(device_ids, NUM_DEVICES); // threadpool
    for (int i = 0; i < NUM_WARMUP_ITERS; ++i) { // warmup
        mc_all_reduce_sum_f32(gang, mc, NUM_DEVICES);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERS; ++i) {
        mc_all_reduce_sum_f32(gang, mc, NUM_DEVICES);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_ITERS;
    printf("Time for %lu MiB allreduce using multimem: %lf ms\n", SIZE / (1024 * 1024), avg_time * 1e3);
    printf("NVLink BW: %lf GB/s\n", 4 * (SIZE / (1024. * 1024. * 1024.)) / avg_time);

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy(host_mats[dev_idx], (void*)mc[dev_idx].mem_va_, SIZE, cudaMemcpyDeviceToHost));
    }

    // Verify the results
    float TOL = 1e-2;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            if (i < 10)
                printf("%f ", host_mats[dev_idx][i]);
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                fprintf(stderr, "Mismatch at device %d, index %d: expected %f, got %f\n", dev_idx, i, expected[i], host_mats[dev_idx][i]);
                exit(1);
            }
        }
        printf("... (%d elements)\n", nelem);
    }

    /*
        Cleanup and exit
    */
   mc_free(mc, NUM_DEVICES);

    // Free resources
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        free(host_mats[dev_idx]);
    }
    free(host_mats);
    free(expected);

    printf("Done\n");
    return 0;
}
