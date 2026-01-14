
# cuda

```
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
```

```
deviceCount: 1
NVIDIA GeForce RTX 3090
8.6
65536
25297879040
1
warpSize: 32
82
maxThreadsPerMultiProcessor: 1536
65536
102400
maxThreadsPerBlock: 1024
1024 1024 64 
65536
49152
```

maxThreadsPerBlock：1个block内最多容纳的线程数量，一般为1024。           
warpSize：线程束的大小，一般为32。1个block最多有1024个线程，但只有32个线程可以同时执行，它们的线程编号连续，执行同一条指令处理不同的数据。每个block内的线程数最好设置成32的整数倍，如果设置32n+x(0<x<32）个线程，最后一次仍有32个线程运行，但只有前×个线程的工作有效。    
maxThreadsPerMultiProcessor：一个多处理器最多同时调度的线程数，是warpSize的整数倍。       







# NCCL C++ Examples

| **Cases**                  | **Node require** | **Description**                                           |  
|----------------------------|------------------|-----------------------------------------------------------|
| one_device_per_thread      | 1                | One Device(1 GPU) per Process or Thread                   |
| multi_devices_per_thread   | 1                | Multiple Devices(more than one GPU) per Process or Thread |
| nonblocking_double_streams | 1                | One rank has two communicators.                           |
| nccl_with_mpi              | 1                | Run with Open MPI                                         |
| node_server/node_client    | 2                | Using socket for init                                     |


## Compile

Clone this git lib to your local env, such as /home/xky/

Requirements:
* CUDA
* NVIDIA NCCL (optimized for NVLink)
* Open-MPI (option)

Recommend using docker images：

```shell
docker pull nvcr.io/nvidia/pytorch:24.07-py3
```

If there is docker-ce, run docker:
```shell
sudo docker run  --net=host --gpus=all -it -e UID=root --ipc host --shm-size="32g" \
-v /home/xky/:/home/xky \
-u 0 \
--name=nccl2 nvcr.io/nvidia/pytorch:24.07-py3 bash
```
Others:
```shell
docker run \
  --runtime=nvidia \
  --privileged \
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidia1:/dev/nvidia1 \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidia3:/dev/nvidia3 \
  --device /dev/nvidia4:/dev/nvidia4 \
  --device /dev/nvidia5:/dev/nvidia5 \
  --device /dev/nvidia6:/dev/nvidia6 \
  --device /dev/nvidia7:/dev/nvidia7 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device /dev/infiniband:/dev/infiniband \
  -v /usr/local/bin/:/usr/local/bin/ \
  -v /opt/cloud/cce/nvidia/:/usr/local/nvidia/ \
  -v /home/xky/:/home/xky \
  --ipc host \
  --net host \
  -it \
  -u root \
  --name nccl_env \
nvcr.io/nvidia/pytorch:24.07-py3 bash
```


Enter the git directory and run makefile
```shell
cd /home/xky/BasicCUDA/nccl/
make
```
If there is MPI lib in env, could compile MPI case:
```shell
make mpi
```

## Run 

### Single node

```shell
./multi_devices_per_thread
./one_devices_per_thread
./nonblocking_double_streams
```

Set DEBUG=1 would print some debug information.  
Could change ranks number by set '--nranks'. e.g:

```shell
DEBUG=1 ./nonblocking_double_streams --nranks 8
```

MPI case run:
```shell
mpirun -n 6 --allow-run-as-root ./nccl_with_mpi
```

### Multi nodes

Two nodes case: using socket connection for nccl init.

Server run in one:
```shell
./node_server
```

Client run in another one, e.g. Server IP: 10.10.1.1
```shell
./node_client --hostname 10.10.1.1
```

Add some envs:
```shell
# server:
NCCL_DEBUG=INFO NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 ./node_server --port 8066 --nranks 8
# client:
NCCL_DEBUG=INFO  NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 ./node_client --hostname 10.10.1.1 --port 8066  --nranks 8
```

