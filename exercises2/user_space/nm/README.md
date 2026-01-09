

只查看导出的动态符号 (推荐):   
```
root@ubuntux86:# nm -D  lib64/libcudart.so 
                 w __cxa_finalize
                 w __gmon_start__
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
root@ubuntux86:# 
```


+ g++编译

```
 c++filt  ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t
ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t
```

```
nm -D /workspace/nccl-latest-dev/libcudart-with-kernel/lib64/libcudart.so  | grep ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t
000000000001beec T _Z42ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_tP11ncclDevCommmP12ncclWorkList
```

+ nvcc编译

```
nm -D /workspace/nccl-dev/nccl-latest/libcudart-with-kernel/lib64/libcudartKernel.so   | grep ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t
00000000000056e0 T _Z42ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_tP11ncclDevCommmP12ncclWorkList
```