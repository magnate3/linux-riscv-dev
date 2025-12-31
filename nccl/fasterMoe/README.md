


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

#  fastmoe

```
sudo git remote -v
origin  https://github.com/laekov/fastmoe.git (fetch)
origin  https://github.com/laekov/fastmoe.git (push)
```


```
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

```
root@ubuntu:/pytorch/fastmoe/cuda/tests# ./test_counting  1 2
ref lec
1 0 
lec
1 0 
root@ubuntu:/pytorch/fastmoe/cuda/tests# 
```