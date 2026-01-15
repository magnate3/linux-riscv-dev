# cuda-gemm

doc: [一个性能达到cuBLAS 97%的Hopper DenseGEMM实现](https://zhuanlan.zhihu.com/p/1890055094562701466)

## build & run

```
docker pull nvcr.io/nvidia/tritonserver:24.05-py3
```

+ nvcc版本   
```
 nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```


> ##  gemm
+ -std=c++20编译cuda-gemm
```bash
/cuda-gemm/gemm# make
nvcc -std=c++20 -O3 -DNDEBUG -w --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a -lcudart -lcublas -lcublasLt -lcuda  -lcupti  gemm.cu -o gemm
```

```
cuda-gemm/gemm# ./gemm 
Impl Average time (ns): 0
Impl Performance: inf TFLOPS
cuBLAS Average time (ns): 2.00226e+06
cuBLAS Performance: 68.6418 TFLOPS
Incorrect!
```

> ## simplegemm

```
cuda-gemm/simplegemm# make gemm
nvcc -std=c++17 -O3 -DNDEBUG -w --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a -lcublas -lcuda  main.cu -o gemm
```

```
simplegemm# ./gemm 
about to run gemm
CUDA error at pingpong.cu:578: no kernel image is available for execution on the device
```