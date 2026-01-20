#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// GPU kernel - consumer
__global__ void consumer_kernel(int *flags, int** address_write, int K, int array_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < K) {
        printf("GPU thread %d started\n", tid);
        
        // Request data from CPU by setting flag to 1
        flags[tid] = 1;
        printf("GPU thread %d requesting data (flag set to 1)\n", tid);
        __threadfence_system();   // ensure CPU sees the update
        
        // Wait until CPU marks data as ready (flag = 2)
        while (flags[tid] != 2) {
            __threadfence_system();   // check for updates from CPU
            //printf("GPU thread %d waiting for data (current flag = %d)\n", tid, flags[tid]);
        }
        
        printf("GPU thread %d received data notification (flag = 2)\n", tid);
        __threadfence_system();   // ensure we see the latest data
        
        // Process the data at the provided address
        if (address_write[tid] != nullptr) {
            int *data_ptr = address_write[tid];
            printf("GPU thread %d processing array at address %p:\n", tid, data_ptr);
            
            // Increment each element and print
            for (int i = 0; i < array_size; i++) {
                int original = data_ptr[i];
                data_ptr[i] += 1;  // Increment the value
                printf("  GPU thread %d: array[%d] = %d -> %d\n", 
                       tid, i, original, data_ptr[i]);
            }
            
            // Calculate and print sum after increment
            int sum = 0;
            for (int i = 0; i < array_size; i++) {
                sum += data_ptr[i];
            }
            printf("GPU thread %d: sum after increment = %d\n", tid, sum);
        } else {
            printf("GPU thread %d: ERROR - null data address!\n", tid);
        }
        
        // Mark as done (flag = 3)
        atomicExch(&flags[tid], 3);
        __threadfence_system();
        printf("GPU thread %d completed (flag set to 3)\n", tid);
    }
}

int main() {


std::cout << "Enter number of Threads: ";
    int K;
    std::cin >> K;

    

    
    const int array_size = 10;  // Each array has 10 elements

    int host_array[10]= {0,1,2,3,4,5,6,7,8,9};
  /*  
int **host_arrays = new int*[K];
std::srand(std::time(nullptr));  // Seed RNG

for (int i = 0; i < K; i++) {
    host_arrays[i] = new int[array_size];  // allocate each array
    for (int j = 0; j < array_size; j++) {
        host_arrays[i][j] = std::rand() % 100;  // random numbers 0–99
    }
}*/
    
 
    
    // Declare unified memory for flags and address_write
    int *flags,*d_sample1;
    int **address_write;


     cudaMalloc(&d_sample1, array_size * sizeof(int));
    
    cudaError_t err;

    cudaStream_t copy_stream;
cudaStreamCreate(&copy_stream);
    


    // Allocate unified memory for flags
    err = cudaMallocManaged(&flags, K * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating unified memory for flags: " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "\nUnified memory for flags allocated" << std::endl;

    
    // Allocate unified memory for address_write
    err = cudaMallocManaged(&address_write, K * sizeof(int*));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating unified memory for address_write: " 
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Unified memory for address_write allocated" << std::endl;
    
    // Initialize flags with 0 and address_write with nullptr
    for (int i = 0; i < K; i++) {
        flags[i] = 0;
        address_write[i] = nullptr;
    }
    std::cout << "\nFlags and address_write initialized" << std::endl;
    
    // Launch GPU kernel
    int threads_per_block = 1024;
    int blocks = (K + threads_per_block - 1) / threads_per_block;
    

    std::cout << "\nLaunching kernel with " << blocks << " blocks and " 
              << threads_per_block << " threads per block" << std::endl;


              //KERNEL CALL
    consumer_kernel<<<blocks, threads_per_block>>>(flags, address_write, K, array_size);
    

    // CPU producer loop - serve data when requested
    std::cout << "\nCPU starting to monitor requests..." << std::endl;

    

    bool all_done = false;
    while (!all_done) {
        all_done = true;
        
        for (int i = 0; i < K; i++) {
            // Check if thread i is requesting data
            if (flags[i] == 1) {
                std::cout << "CPU: Thread " << i << " requested data (flag = 1)" << std::endl;

                
               err = cudaMemcpyAsync(d_sample1,host_array, array_size*sizeof(int), cudaMemcpyHostToDevice,copy_stream);


                                if (err != cudaSuccess) {
                        std::cerr << "Error while copying data to GPU chunk " << i << ": "
                                << cudaGetErrorString(err) << std::endl;
                        return -1;
                    }



                
                // Update the address in unified memory
                address_write[i] = d_sample1;

                cudaStreamSynchronize(copy_stream);
                
                // Set flag to 2 to indicate data is ready
                std::cout << "DONE COPYING" << std::endl;
                flags[i] = 2;
                
            }
            
            // Check if any thread is still working
            if (flags[i] != 3) {
                all_done = false;
            }
        }
    }
    
    std::cout << "\nAll GPU threads completed!" << std::endl;
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();


   // cudaMemcpy(host_array,d_sample1, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    
   
    // Free unified memory
    cudaFree(flags);
    cudaFree(address_write);
    
    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}
