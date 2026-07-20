#include "cuda_runtime.h"
#include <iostream>
using namespace std;
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount) ;
    cout << "deviceCount: " << deviceCount << endl;
    if (deviceCount == 0){
    cout << "error: no devices supporting CUDA.\n";
    exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);
    //1.设备。
    cout << devProps.name << endl; // GeForce 610M
    cout << devProps.major << "." << devProps.minor << endl; // 2.1
    cout << devProps.totalConstMem << endl; // 65536 = 2^16 = 64K
    cout << devProps.totalGlobalMem << endl; // 1073741824 = 2^30 = 1G
    cout << devProps.unifiedAddressing << endl; // 1
    cout <<"warpSize: "<< devProps.warpSize << endl; // 32
    //2.多处理器。
    cout << devProps.multiProcessorCount << endl; // 1
    cout << "maxThreadsPerMultiProcessor: "<< devProps.maxThreadsPerMultiProcessor << endl; // 1536
    //for (auto x : devProps.maxGridsize) cout << x << " "; cout << endl; // 65535 65535 65535
    cout << devProps.regsPerMultiprocessor << endl; // 32768 = 2^15 = 32K
    cout << devProps.sharedMemPerMultiprocessor << endl; // 49152
    // 3.Block。
    cout <<"maxThreadsPerBlock: " << devProps.maxThreadsPerBlock << endl; // 1024
    for (auto x : devProps.maxThreadsDim) cout << x << " "; cout << endl; // 1024 1024 64
    cout << devProps.regsPerBlock << endl; // 32768 = 2^15 = 32K
    cout << devProps.sharedMemPerBlock << endl; // 49152
}
