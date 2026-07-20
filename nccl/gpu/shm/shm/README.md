
```
shm# ./test_cuda_api 
***********************Device Info*********************************
GPU Device 0 name:      NVIDIA GeForce RTX 3090
CUDA Capability version:        8.6
Memory: 23.5605 GiB
Shares a unified address space with the host:   1
Shared memory per block:        48 KiB
Maximum number of threads per block:    1024
Warp size in threads:   32
Number of asynchronous engines: 2
Can map host memory with cudaHostAlloc/cudaHostGetDevicePointer:        1
Can access host registered memory at the same virtual address as the CPU:       1
Can possibly execute multiple kernels concurrently:     1
***********************IpcMemHandle*******************************
main process ptr:       0x752648200000
child process ptr:      0x752648200000
Passed!
***********************cudaMalloc*********************************
Passed!
***********************cudaMallocHost*******************************
Passed!
***********************cudaMemcpy*********************************
Passed!
***********************cudaMemcpyAsync******************************
Passed!
***********************cudaHostRegister*****************************
Passed!
***********************CanAccessPeer********************************
Need >=2 device!
***********************cuMem********************************
Allocation Granularity:2097152
Passed!
```
