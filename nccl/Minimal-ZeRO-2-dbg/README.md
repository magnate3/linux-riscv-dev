
```
make
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -I./ -I/usr/local/mpi/include/ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o zero2_demo main.cu utils.cu model.cu zero_optimizer.cu  -lnccl -lmpi -L/usr/local/mpi/lib/
```

```
mpirun -np 1 ./zero2_demo


--- Parameters after Step 10000 (Rank 0 view) ---
[Rank 0] W (10 elements): [-99.9008, -99.9704, -99.9043, -100.0062, -99.9659, -99.8945, -99.9459, -99.9589, -99.9515, -99.8524]
[Rank 0] b (10 elements): [-9.9914, -9.9914, -9.9914, -9.9914, -9.9914, -9.9914, -9.9914, -9.9914, -9.9914, -9.9914]
------------------------------------

--- Training Loop Finished ---
*** The MPI_Comm_rank() function was called after MPI_FINALIZE was invoked.
*** This is disallowed by the MPI standard.
*** Your MPI job will now abort.
[ubuntu:19880] Local abort after MPI_FINALIZE started completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun.real detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[52749,1],0]
  Exit code:    1
--------------------------------------------------------------------------
```