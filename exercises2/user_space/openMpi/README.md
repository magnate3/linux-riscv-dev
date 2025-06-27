
# os


```
uname -r
5.15.0-138-generic
```

# make    

```
cd ~
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.bz2
tar xf openmpi-4.1.4.tar.bz2
cd openmpi-4.1.4
./configure --prefix=/usr/local/openmpi 2>&1 | tee config.out
make -j 80 all 2>&1 | tee make.out
make install 2>&1 | tee install.out
sed -i '1i\export PATH=/usr/local/openmpi/bin:$PATH' ~/.bashrc
source ~/.bashrc
```
执行以下命令，检查OpenMPI是否正常安装。    

```
mpiexec --version
mpiexec (OpenRTE) 4.1.4

Report bugs to http://www.open-mpi.org/community/help/
```

## mpi-benchmarks


```
git clone --branch IMB-v2021.3 https://github.com/intel/mpi-benchmarks.git
```

```
# cd mpi-benchmarks/src_c
# make all
```

### 单节点测试
```
root@ubuntu:~/openMpi/mpi-benchmarks/src_c# export OMPI_ALLOW_RUN_AS_ROOT=1
root@ubuntu:~/openMpi/mpi-benchmarks/src_c# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
root@ubuntu:~/openMpi/mpi-benchmarks/src_c# mpirun -np 2 ./IMB-MPI1 pingpong
--------------------------------------------------------------------------
WARNING: No preset parameters were found for the device that Open MPI
detected:

  Local host:            ubuntu
  Device name:           mlx5_0
  Device vendor ID:      0x02c9
  Device vendor part ID: 4125

Default device parameters will be used, which may result in lower
performance.  You can edit any of the files specified by the
btl_openib_device_param_files MCA parameter to set values for your
device.

NOTE: You can turn off this warning by setting the MCA parameter
      btl_openib_warn_no_device_params_found to 0.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
WARNING: There is at least non-excluded one OpenFabrics device found,
but there are no active ports detected (or Open MPI was unable to use
them).  This is most certainly not what you wanted.  Check your
cables, subnet manager configuration, etc.  The openib BTL will be
ignored for this job.

  Local host: ubuntu
--------------------------------------------------------------------------
#----------------------------------------------------------------
#    Intel(R) MPI Benchmarks 2018, MPI-1 part
#----------------------------------------------------------------
# Date                  : Fri Jun 27 02:00:00 2025
# Machine               : x86_64
# System                : Linux
# Release               : 5.15.0-138-generic
# Version               : #148-Ubuntu SMP Fri Mar 14 19:05:48 UTC 2025
# MPI Version           : 3.1
# MPI Thread Environment: 


# Calling sequence was: 

# ./IMB-MPI1 pingpong

# Minimum message length in bytes:   0
# Maximum message length in bytes:   4194304
#
# MPI_Datatype                   :   MPI_BYTE 
# MPI_Datatype for reductions    :   MPI_FLOAT
# MPI_Op                         :   MPI_SUM  
#
#

# List of Benchmarks to run:

# PingPong

#---------------------------------------------------
# Benchmarking PingPong 
# #processes = 2 
#---------------------------------------------------
       #bytes #repetitions      t[usec]   Mbytes/sec
            0         1000         0.26         0.00
            1         1000         0.33         2.99
            2         1000         0.34         5.93
            4         1000         0.34        11.88
            8         1000         0.32        24.72
           16         1000         0.43        37.19
           32         1000         0.42        76.06
           64         1000         0.51       126.09
          128         1000         0.46       277.53
          256         1000         0.52       496.50
          512         1000         0.65       781.83
         1024         1000         0.77      1330.71
         2048         1000         1.04      1970.07
         4096         1000         1.64      2504.66
         8192         1000         1.86      4410.07
        16384         1000         2.33      7031.50
        32768         1000         3.30      9919.61
        65536          640         5.27     12440.57
       131072          320         8.04     16297.31
       262144          160        14.30     18336.70
       524288           80        25.58     20493.27
      1048576           40        70.45     14884.37
      2097152           20       203.09     10326.33
      4194304           10       403.60     10392.17


# All processes entering MPI_Finalize

[ubuntu:598920] 3 more processes have sent help message help-mpi-btl-openib.txt / no device params found
[ubuntu:598920] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[ubuntu:598920] 1 more process has sent help message help-mpi-btl-openib.txt / no active ports found
root@ubuntu:~/openMpi/mpi-benchmarks/src_c# 
```

#  多接点



[集群机器搭建多节点MPI运行环境](https://cloud.tencent.com/developer/article/2156525)   