

```
 python3
Python 3.8.10 (default, Mar 18 2025, 20:04:55) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from torch.utils import cpp_extension;
No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
>>> from torch.utils import cpp_extension
>>> print(f"{cpp_extension.TORCH_LIB_PATH}")
/usr/local/lib/python3.8/dist-packages/torch/lib
>>> 
```


# make

```
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

```
make
nvcc prune_gate.cu ../stream_manager.cpp -lcublas -L /usr/local/lib/python3.8/dist-packages/torch/lib -lc10_cuda -lcublas -lcublasLt -ltorch_cuda_cu  -ltorch_python -I /usr/local/lib/python3.8/dist-packages/torch/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/TH -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/THC -o test_prune_gate  
prune_gate.cu: In function 'int main(int, char**)':
prune_gate.cu:44:8: warning: format '%d' expects argument of type 'int', but argument 4 has type 'long int' [-Wformat=]
   44 |         printf("%ld %ld (%d)\n", gate_idx[i], n_gate_idx[i], lec[gate_idx[i]]);
      |        ^~~~~~~~~~~~~~~~                              ~~~~~~~~~~~~~~~~
      |                                                                     |
      |                                                                     long int
nvcc limit.cu ../stream_manager.cpp -lcublas -L /usr/local/lib/python3.8/dist-packages/torch/lib -lc10_cuda -lcublas -lcublasLt -ltorch_cuda_cu  -ltorch_python -I /usr/local/lib/python3.8/dist-packages/torch/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/TH -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/THC -o test_limit  
nvcc assign.cu ../stream_manager.cpp -lcublas -L /usr/local/lib/python3.8/dist-packages/torch/lib -lc10_cuda -lcublas -lcublasLt -ltorch_cuda_cu  -ltorch_python -I /usr/local/lib/python3.8/dist-packages/torch/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/TH -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/THC -o test_assign  
assign.cu: In function 'int main(int, char**)':
assign.cu:34:8: warning: format '%d' expects argument of type 'int', but argument 2 has type 'long int' [-Wformat=]
   34 |         printf("%d ", gate_idx[i]);
      |        ^~~~~  ~~~~~~~~~~~
      |                         |
      |                         long int
assign.cu:59:8: warning: format '%d' expects argument of type 'int', but argument 2 has type 'long int' [-Wformat=]
   59 |         printf("%d ", pos[i]);
      |        ^~~~~  ~~~~~~
      |                    |
      |                    long int
nvcc counting.cu ../stream_manager.cpp -lcublas -L /usr/local/lib/python3.8/dist-packages/torch/lib -lc10_cuda -lcublas -lcublasLt -ltorch_cuda_cu  -ltorch_python -I /usr/local/lib/python3.8/dist-packages/torch/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/TH -I /usr/local/lib/python3.8/dist-packages/torch/lib/include/THC -o test_counting  
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_assign 
Segmentation fault (core dumped)
```

docker加上--privileged    
```
sudo docker run  --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" --privileged  -v /pytorch:/pytorch -u 0  --entrypoint bash  --name=fedml fedml-x86:v2 
```


```
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_counting  1 2
ref lec
1 0 
CUDA error at ../stream_manager.cpp:52 code=100(cudaErrorNoDevice) "cudaSetDevice(device)" 
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_counting  1 1
ref lec
1 
CUDA error at ../stream_manager.cpp:52 code=100(cudaErrorNoDevice) "cudaSetDevice(device)"
```


```
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_counting  1 1
ref lec
1 
lec
1 
root@ubuntu:/pytorch/fastmoe/cuda/tests# nvidia-smi 
Wed Dec 31 06:54:43 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:01:00.0 Off |                  N/A |
| 31%   33C    P8             25W /  350W |       2MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_counting  1 2
ref lec
1 0 
lec
1 0 
root@ubuntu:/pytorch/fastmoe/cuda/tests#
```