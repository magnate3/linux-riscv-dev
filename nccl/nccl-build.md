
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

编译如下版本出错(编译symmtric函数出错)   
```
root@ubuntux86:# git branch
  master
* v2.27_sym_memory
root@ubuntux86:# 
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
+ 编译

+  bug: Undefined reference to 'ncclDevFuncTable'     
```
dos2unix src/device/generate.py
```

+  bug: Is a directory: '/workspace/nccl-dev/nccl-latest/build/obj/device/gensrc/symmetric'   


```
root@ubuntux86:# rm  -rf build/obj/device/gensrc/symmetric    
```



应用程序   
```
root@82adaca144df:/workspace/nccl-latest-dev/nccl_graphs# g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

##  build  v2.27_sym_memory with cuda:12.8.1-devel-ubuntu22.04(编译成功)


cuda:12.8.1-devel-ubuntu22.04 也能编译 origin/v2.17-racecheck    

```
docker run --name nccl -itd  --rm -v /work/nccl/:/workspace    --network=host  --shm-size=4g --ulimit memlock=-1 --cap-add=NET_ADMIN --privileged=true nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04
```

```
root@ubuntux86:/# g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntux86:/# gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntux86:/# 
```

###  Build NCCL
```
git clone https://github.com/NVIDIA/nccl/
cd nccl
git checkout v2.27_sym_memory # on commit dec8621
make -j src.build NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

+  bug: Undefined reference to 'ncclDevFuncTable'       
```
dos2unix src/device/generate.py
```

+  bug: Is a directory: '/workspace/nccl-dev/nccl-latest/build/obj/device/gensrc/symmetric'      


```
root@ubuntux86:# rm  -rf build/obj/device/gensrc/symmetric    
```

+ make log     

```
Compiling       build/obj/device/gensrc/reduce_scatter_sumpostdiv_u8.cu
Compiling       build/obj/device/gensrc/reduce_sum_bf16.cu
Compiling       build/obj/device/gensrc/reduce_sum_f16.cu
Compiling       build/obj/device/gensrc/reduce_sum_f32.cu
Compiling       build/obj/device/gensrc/reduce_sum_f64.cu
Compiling       build/obj/device/gensrc/reduce_sum_f8e4m3.cu
Compiling       build/obj/device/gensrc/reduce_sum_f8e5m2.cu
Compiling       build/obj/device/gensrc/reduce_sum_u32.cu
Compiling       build/obj/device/gensrc/reduce_sum_u64.cu
Compiling       build/obj/device/gensrc/reduce_sum_u8.cu
Compiling       build/obj/device/gensrc/reduce_sumpostdiv_u32.cu
Compiling       build/obj/device/gensrc/reduce_sumpostdiv_u64.cu
Compiling       build/obj/device/gensrc/reduce_sumpostdiv_u8.cu
Compiling       build/obj/device/gensrc/device_table.cu
Compiling       build/obj/device/gensrc/symmetric/all_gather.cu
Compiling       build/obj/device/gensrc/symmetric/all_reduce_sum_bf16.cu
Compiling       build/obj/device/gensrc/symmetric/all_reduce_sum_f16.cu
Compiling       build/obj/device/gensrc/symmetric/all_reduce_sum_f32.cu
Compiling       build/obj/device/gensrc/symmetric/all_reduce_sum_f8e4m3.cu
Compiling       build/obj/device/gensrc/symmetric/all_reduce_sum_f8e5m2.cu
Compiling       build/obj/device/gensrc/symmetric/reduce_scatter_sum_bf16.cu
Compiling       build/obj/device/gensrc/symmetric/reduce_scatter_sum_f16.cu
Compiling       build/obj/device/gensrc/symmetric/reduce_scatter_sum_f32.cu
Compiling       build/obj/device/gensrc/symmetric/reduce_scatter_sum_f8e4m3.cu
Compiling       build/obj/device/gensrc/symmetric/reduce_scatter_sum_f8e5m2.cu
Compiling       src/device/onerank.cu
```

```
make -j src.build NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

```
root@ubuntux86:/workspace/nccl-dev/nccl-latest# export NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
root@ubuntux86:/workspace/nccl-dev/nccl-latest# make -j src.build                                                     
make -C src build BUILDDIR=/workspace/nccl-dev/nccl-latest/build
make[1]: Entering directory '/workspace/nccl-dev/nccl-latest/src'
NVCC_GENCODE is -gencode=arch=compute_86,code=sm_86
make[2]: Entering directory '/workspace/nccl-dev/nccl-latest/src/device'
NVCC_GENCODE is -gencode=arch=compute_86,code=sm_86
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/common.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/onerank.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_gather.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_reduce_sum_bf16.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_reduce_sum_f16.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_reduce_sum_f32.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_reduce_sum_f8e4m3.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/all_reduce_sum_f8e5m2.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/reduce_scatter_sum_bf16.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/reduce_scatter_sum_f16.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/reduce_scatter_sum_f32.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/reduce_scatter_sum_f8e4m3.cu.o'
nvlink warning : SM Arch ('sm_86') not found in '/workspace/nccl-dev/nccl-latest/build/obj/device/genobj/symmetric/reduce_scatter_sum_f8e5m2.cu.o'
make[2]: Leaving directory '/workspace/nccl-dev/nccl-latest/src/device'
Linking    libnccl.so.2.27.0                   > /workspace/nccl-dev/nccl-latest/build/lib/libnccl.so.2.27.0
Archiving  libnccl_static.a                    > /workspace/nccl-dev/nccl-latest/build/lib/libnccl_static.a
make[1]: Leaving directory '/workspace/nccl-dev/nccl-latest/src
```

### Build NCCL-test
```
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-test # on commit 59072b7
git checkout v2.27_sym_memory
make NCCL_HOME=/path/to/nccl-test/build
```
##   host_table.cc

```
Dependencies    build/obj/device/gensrc/device_table.cu
Compiling       build/obj/device/gensrc/host_table.cc
```

## 应用

 Upgrade your operating system (e.g., to Ubuntu 22.04 or later, which provides GLIBC 2.34)
 
```
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `cudaGetDriverEntryPoint'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `dlclose@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `__cudaRegisterFunction'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `dlerror@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_mutexattr_init@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_detach@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_condattr_setpshared@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_join@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `__cudaRegisterFatBinaryEnd'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_setname_np@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `pthread_mutex_trylock@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `dlsym@GLIBC_2.34'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `ncclSymImplemented(ncclFunc_t, int, ncclDataType_t)'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `__cudaUnregisterFatBinary'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `stat@GLIBC_2.33'
/usr/bin/ld: /workspace/nccl-dev/nccl-latest/build/lib//libnccl.so: undefined reference to `dlopen@GLIBC_2.34'
```