

```
TORCH_CUDA_LIB=`python3 -c 'from torch.utils import cpp_extension; print(f"{cpp_extension.TORCH_LIB_PATH}/libtorch_cuda.so")' 2>/dev/null`
```

```
whereis libnccl
libnccl: /usr/lib/x86_64-linux-gnu/libnccl.so
```

```
nm -DC /usr/lib/x86_64-linux-gnu/libnccl.so |  grep nccl
                 w TLS init function for ncclDebugNoWarn
0000000000093910 T ncclAllGather
00000000000935d0 T ncclAllReduce
```


# test1 (cuLaunchKernel kernel.ptx)
```
./kernel_test 
PTX kernel execution time: 0.025568 ms
Runtime kernel execution time: 0.02016 ms
Both kernels produced correct results.
```

# test2
```
root@ubuntu:/pytorch/kernel/test2# ./kernel_test 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0
```


#  test3

+ 从libvec.a中cudaGetFuncBySymbol    


```
cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)VecAdd2);
```

```
test3# ./kernel_test 
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```


+ 从main2.cu中直接cudaGetFuncBySymbol     


```
 cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)VecAdd3);
```


```
test3# ./kernel_test2
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```

+ 从libvec.a中cudaGetFuncBySymbol   

```
 cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)invoke_VecAdd);
```

```
test3# ./kernel_test3 
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```

# make bug


+  ptxas error   : Undefined reference to 'VecAdd2' in '<input>'    
```    
```
typedef void (*fp)(double *, double *, double *, int);
extern "C" __global__ void VecAdd2(double *a, double *b, double *c, int n);
__device__ fp kernelPtr = VecAdd2;
//__device__ fp kernelPtr = VecAdd2;
```


```
ptxas error   : Undefined reference to 'VecAdd2' in '<input>'
```




+ undefined reference to `invoke_VecAdd(double*, double*, double*, int)'   


invoke_VecAdd前面加上extern "C" __global__ 

