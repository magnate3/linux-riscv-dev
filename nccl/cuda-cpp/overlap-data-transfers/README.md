
[如何在 CUDA C/C++ 中实现数据传输的重叠](https://developer.nvidia.cn/blog/how-overlap-data-transfers-cuda-cc/)   



```
root@ubuntu:/pytorch/cuda-cpp/overlap-data-transfers# nvcc  --std=c++11 async.cu -o async_test
root@ubuntu:/pytorch/cuda-cpp/overlap-data-transfers# ./async_test 
Device : NVIDIA GeForce RTX 3090
Time for sequential transfer and execute (ms): 1.519616
  max error: 1.192093e-07
Time for asynchronous V1 transfer and execute (ms): 0.984352
  max error: 1.192093e-07
Time for asynchronous V2 transfer and execute (ms): 0.970208
  max error: 1.192093e-07
```
```
root@ubuntu:/pytorch/cuda-cpp/overlap-data-transfers# ./overlap_async 
Device : NVIDIA GeForce RTX 3090
Time for sequential transfer and execute (ms): 1.516928
  max error: 1.192093e-07
Time for asynchronous V1 transfer and execute (ms): 0.988960
  max error: 1.192093e-07
Time for asynchronous V2 transfer and execute (ms): 0.974272
  max error: 1.192093e-07
root@ubuntu:/pytorch/cuda-cpp/overlap-data-transfers# ./overlap_data_transfers 
Device : NVIDIA GeForce RTX 3090
Time for sequential transfer and execute (ms): 1.528448
  max error: 1.192093e-07
Time for asynchronous V1 transfer and execute (ms): 0.997440
  max error: 1.192093e-07
Time for asynchronous V2 transfer and execute (ms): 0.981280
  max error: 1.192093e-07
root@ubuntu:/pytorch/cuda-cpp/overlap-data-transfers#
```

 

# Models

```
The simplest CUDA program consists of three steps, including copying the memory from host to device, kernel execution, and copy the memory from device to host. In our particular example, we have the following facts or assumptions:

The memory copy (host to device, device to host) time is linearly dependent on the size of the memory for copy.
The GPU would never be fully utilized.
The kernel execution could be divided into 
 smaller kernel executions and each smaller kernel execution would only take 1/N
 of the time the original kernel execution takes.
The memory copy time from host to device, kernel execution, and memory copy time from device to host are the same.
Each CUDA engine executes commands or kernels in order.
We could come up with two models, including a serial model and a concurrent model, to implement the program.
```

![images](cuda-stream.png)


```
Serial Model
In the serial model, we first copy the input memory from host to device first, then execute the kernel to compute the output, and finally copy the output memory from device back to host.

Concurrent Model
In the concurrent model, we make memory copy from host to device, kernel executions, and memory copy from device to host, asynchronous. We divided the memory into 
 trunks. In our particular example above, we set 
. After finishing copying the first trunk from host to device, we launch the smaller kernel execution to process the first trunk. In the meantime, the host to device (H2D) engine becomes available and it proceeds to copy the second trunk from host to device. Once the first trunk has been processed by the kernel, the output memory would be copied from device to host using the device to host engine (D2H) engine. In the meantime, the host to device (H2D) engine and the kernel engine becomes available and they proceed to copy the third trunk from host to device and process for the second trunk respectively.

From the figure above, we could see that the concurrent model would only take half of the time the serial model would take.

The question then becomes how do we write a CUDA program such that the commands for each of the trunks are executed in order, and different trunks could be executed concurrently. The answer is to use CUDA stream.
```
 

```
To implement the concurrent model, instead of calling cudaMemcpy, we call cudaMemcpyAsync and launch kernel with the stream specified so that they will return to the host thread immediately after call.
```