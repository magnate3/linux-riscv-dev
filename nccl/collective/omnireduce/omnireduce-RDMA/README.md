# OmniReduce-RDMA

## Getting Started
The simplest way to start is to use our [docker image](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/docker). We provide a [tutorial](https://github.com/sands-lab/omnireduce/blob/master/omnireduce-RDMA/docs/tutorial.md) to help you run RDMA-based OmniReduce with docker image quickly.
Below, we introduce how to build and use OmniReduce.

### Building
OmniReduce is built to run on Linux and the dependencies include CUDA, ibverbs and Boost C++ library.
To build OmniReduce, run:

    git clone https://github.com/sands-lab/omnireduce
    cd omnireduce-RDMA
    make USE_CUDA=ON
	
```
apt-get -y install libboost-program-options-dev
apt install libboost-thread-dev -y
```

```
root@ubuntux86:/workspace/omnireduce/omnireduce-RDMA# python3
Python 3.8.10 (default, Mar 18 2025, 20:04:55) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> nccl_version = torch.cuda.nccl.version()
>>> print(f"NCCL version: {nccl_version}")
NCCL version: (2, 14, 3)
>>> 
```

```
root@ubuntux86:/workspace/omnireduce/omnireduce-RDMA# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:18:20_PST_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0
root@ubuntux86:/workspace/omnireduce/omnireduce-RDMA# 
```
docker images

```
nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 
 docker run --name torch -itd  --rm -v /work/fedAgg/:/workspace    --network=host  --shm-size=4g --ulimit memlock=-1 --cap-add=NET_ADMIN --privileged=true torch-x86:v1
 docker exec -it  torch bash
```

### Examples
Basic examples are provided under the [example](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/example) folder. 
To reproduce the evaluation in our SIGCOMM'21 paper, find the code at this [repo](https://github.com/sands-lab/omnireduce-experiments).

## Frameworks Integration
OmniReduce is only integrated with PyTorch currently. The integration method is under the [frameworks_integration](https://github.com/sands-lab/omnireduce/tree/master/omnireduce-RDMA/frameworks_integration/pytorch_patch) folder.

## Limitations

- Only support AllReduce operation
- Only support int32 and float data type
