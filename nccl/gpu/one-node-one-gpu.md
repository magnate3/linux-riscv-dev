
```
 export CUDA_VISIBLE_DEVICES=0
```

```
/pytorch/cuda# python3 single-node-single-cpu.py 
Process Process-1:
Process Process-8:
Process Process-7:
Process Process-4:
Process Process-5:
Process Process-6:
Process Process-3:
Process Process-2:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 5 and rank 0 both on CUDA device 1000
Traceback (most recent call last):
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 6 and rank 0 both on CUDA device 1000
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 3 and rank 0 both on CUDA device 1000
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 4 and rank 0 both on CUDA device 1000
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 1000
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
Traceback (most recent call last):
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 7 and rank 0 both on CUDA device 1000
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/pytorch/cuda/single-node-single-cpu.py", line 13, in f
    model = DDP(model, device_ids=[device])
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/parallel/distributed.py", line 655, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/utils.py", line 112, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 2 and rank 0 both on CUDA device 1000
```


# single-gpu-mutil-procs
```
/pytorch/cuda# mpirun -np 2 ./single-gpu-mutil-procs
myrank 0, localRank 0 cuda set device 0 
myrank 1, localRank 1 cuda set device 1 
Failed: Cuda error single-gpu-mutil-procs.cu:141 'invalid device ordinal'
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------

rank = 0, gpuid = 0, sendbuff =
0.03 0.42 0.08 0.77 0.14 0.92 0.25 0.83 0.73 0.86 0.50 0.55 0.02 0.04 0.76 0.32 

--------------------------------------------------------------------------
mpirun.real detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[36637,1],1]
  Exit code:    1
--------------------------------------------------------------------------
```