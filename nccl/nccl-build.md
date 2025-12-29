
# nvcc


```
nvcc my_file.cu -Xcompiler -fvisibility=hidden
__attribute__((visibility(...))) or compiler flags like -fvisibility=hidden) t
```

```
NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist)
```

```
#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI
```
printf
```
printf "$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)"
```


-fvisibility=hidden    

```
 g++ -I. -I/workspace/nccl-dev/nccl/build/include -DCUDA_MAJOR=11 -DCUDA_MINOR=6 -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -std=c++11 -Wvla -I /usr/local/cuda/include    -O3 -g -DPROFAPI -Iinclude -c transport/net_socket.cc -o /workspace/nccl-dev/nccl/build/obj/transport/net_socket.o 
```

```
root@d37d7af2450e:/workspace/nccl-dev/nccl# g++ -I. -I/workspace/nccl-dev/nccl/build/include -I src/include/ -DCUDA_MAJOR=11 -DCUDA_MINOR=6 -fPIC -fvisibility=default -Wall -Wno-unused-function -Wno-sign-compare -std=c++11 -Wvla -I /usr/local/cuda/include    -O3 -g -DPROFAPI -Iinclude -c src/transport/net_socket.cc -o /workspace/nccl-dev/nccl/build/obj/transport/net_socket.o
root@d37d7af2450e:/workspace/nccl-dev/nccl# 
```


```
CXXFLAGS = -O2 -Wall -g
MODIFIED_FLAGS := $(shell echo $(CXXFLAGS) | sed 's/-O2/-O3/')

all:
	@echo "Modified CXXFLAGS: $(MODIFIED_FLAGS)"

```

# nvcc 链接cuda

```
NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt
```

# docker 

编译的docker镜像   
```
root@ubuntux86:# docker images | grep 54478aaec63b
nvidia/cuda                                                          11.6.1-cudnn8-devel-ubuntu20.04         54478aaec63b   2 years ago     8.53GB
root@ubuntux86:#
```

```
 docker run --name nccl2 -itd  --rm -v /work/nccl:/workspace    --network=my_net  --shm-size=4g --ulimit memlock=-1 --cap-add=NET_ADMIN --privileged=true  --ip=172.20.0.80  54478aaec63b
```


```
root@ubuntux86:# git branch
* (HEAD detached at origin/v2.17-racecheck)
  master
root@ubuntux86:# 
```


```
whereis nvcc
nvcc: /usr/local/cuda-11.6/bin/nvcc /usr/local/cuda-11.6/bin/nvcc.profile
```



```

$ make -j16 src.build

……

Archiving  objects                             > /workspace/nccl-dev/nccl/build/obj/collectives/device/colldevice.a
make[2]: Leaving directory '/workspace/nccl-dev/nccl/src/collectives/device'
Linking    libnccl.so.2.17.1                   > /workspace/nccl-dev/nccl/build/lib/libnccl.so.2.17.1
Archiving  libnccl_static.a                    > /workspace/nccl-dev/nccl/build/lib/libnccl_static.a
make[1]: Leaving directory '/workspace/nccl-dev/nccl/src'
```


# nccl 加载cuda(NCCL_CUDA_PATH)


```
static void initOnceFunc() {
  do {
    char* val = getenv("CUDA_LAUNCH_BLOCKING");
    ncclCudaLaunchBlocking = val!=nullptr && val[0]!=0 && !(val[0]=='0' && val[1]==0);
  } while (0);

  CUresult res;
  /*
   * Load CUDA driver library
   */
  char path[1024];
  char *ncclCudaPath = getenv("NCCL_CUDA_PATH");
  if (ncclCudaPath == NULL)
    snprintf(path, 1024, "%s", "libcuda.so");
  else
    snprintf(path, 1024, "%s%s", ncclCudaPath, "libcuda.so");
```


# nccl v2.27.7-1 （bug 和nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04不匹配）


```
nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
```
nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04中安装的nccl    

```
Python 3.8.10 (default, Mar 18 2025, 20:04:55) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> nccl_version = torch.cuda.nccl.version()
>>> 
>>> print(f"NCCL version: {nccl_version}")
NCCL version: (2, 14, 3)
>>> 
```

编译如下版本出错   
```
root@ubuntux86:# git branch
* (HEAD detached at v2.27.7-1)
  master
```


```
root@ubuntux86:/# python3 --version
Python 3.8.10
root@ubuntux86:/# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:18:20_PST_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0
root@ubuntux86:/# 
```

##  build  v2.27_sym_memory with cuda:12.8.1-devel-ubuntu22.04


```
docker run --name nccl -itd  --rm -v /work/nccl/:/workspace    --network=host  --shm-size=4g --ulimit memlock=-1 --cap-add=NET_ADMIN --privileged=true nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04
```

###  Build NCCL
```
git clone https://github.com/NVIDIA/nccl/
cd nccl
git checkout v2.27_sym_memory # on commit dec8621
make -j src.build NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

### Build NCCL-test
```
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-test # on commit 59072b7
git checkout v2.27_sym_memory
make NCCL_HOME=/path/to/nccl-test/build
```