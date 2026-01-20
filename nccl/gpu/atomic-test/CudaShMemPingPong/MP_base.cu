#ifndef GHCONSISTENCYTEST_
#define GHCONSISTENCYTEST_

#include <iostream>
#include <cstdint>
#include <thread>
#include <getopt.h>

#include "structs.cuh"
// #include "cpu_data_functions.hpp"
// #include "gpu_data_functions.cuh"
#include "cpu_pingpong.hpp"



/**
 * Parameters:
 *  1. Whether flag and data are in the same cacheline or not
 *      a. Cacheline is 64 bytes  => 32 bytes for flag and 32 bytes for data
 *      b. Cacheline is 128 bytes => 64 bytes for flag and 64 bytes for data
 *  2. Different combinations of flag scopes (cta, block, gpu, sys)
 * */


void run_ping_pong_functions(Allocator allocator) {
    // std::cout << get_cpu_freq() << std::endl;
    // std::cout << get_gpu_freq() << std::endl;

    if (allocator != CUDA_MALLOC) {
        host_device_fetch_add(allocator);
    } else {
        device_device_fetch_add(allocator);
    }

    if (allocator != CUDA_MALLOC) {
        host_ping_device_pong_base(allocator);
        device_ping_host_pong_base(allocator);
        host_ping_device_pong_decoupled(allocator);
        device_ping_host_pong_decoupled(allocator);
    } 

    device_ping_device_pong_base(allocator);
    device_ping_device_pong_decoupled(allocator);
    // host_ping_host_pong_decoupled();

    if (allocator != CUDA_MALLOC) {
        host_ping_device_pong_assymetric(allocator);
        device_ping_host_pong_assymetric(allocator);
    }
}

int main(int argc, char** argv) {

    // do a getopt for a -m flag, and the inputs are either MALLOC or HOST

    Allocator allocator;

    int opt;
    while ((opt = getopt(argc, argv, "m:")) != -1) {
        switch (opt) {
            case 'm':
                if (strcmp(optarg, "MALLOC") == 0) {
                    std::cout << "Using Malloc" << std::endl;
                    allocator = MALLOC;
                } else if (strcmp(optarg, "HOST") == 0) {
                    std::cout << "Using Host" << std::endl;
                    allocator = CUDA_MALLOC_HOST;
                } else if (strcmp(optarg, "UM") == 0) {
                    std::cout << "Using UM" << std::endl;
                    allocator = UM;
                } else if (strcmp(optarg, "CUDA_MALLOC") == 0) {
                    std::cout << "Using CUDA_MALLOC" << std::endl;
                    allocator = CUDA_MALLOC;
                } else {
                    std::cout << "Invalid argument" << std::endl;
                    return 1;
                }
                break;
            default:
                std::cout << "Invalid argument" << std::endl;
                return 1;
        }
    }

    run_ping_pong_functions(allocator);

    return 0;
}




#endif 
