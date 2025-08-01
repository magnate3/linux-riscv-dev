
# mpitutorial --  Wes Kendall   

[MPI Reduce and Allreduce](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/zh_cn/)   


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

## OpenMPI中使用UCX¶
UCX在OpenMPI是默认的pml，在OpenSHMEM中是默认的spml，一般安装好设置号后无需用户自己设置就可使用，用户也可利用下面方式显式指定：   

在Open MPI中显式指定采用UCX：   
mpirun --mca pml ucx --mca osc ucx ...   
在OpenSHMEM显示指定采用UCX：    
oshrun --mca spml ucx ...    

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
[mpirun多机多进程-多节点运行nccl-tests](https://zhuanlan.zhihu.com/p/682530828)    
```
在10.22.116.221机器上

1)运行：ssh-keygen -t rsa

2)然后拍两下回车（均选择默认）

ls /root/.ssh/id_rsa.pub
/root/.ssh/id_rsa.pub

3)运行： 

ssh-copy-id -i /root/.ssh/id_rsa.pub root@10.22.116.220

或普通用户:

ssh-copy-id NAME@IP

4)再输入220机器上的root密码

此时，再ssh root@10.22.116.220 or ssh  10.22.116.220，则不需要密码了。相互之间scp，也不需要密码
```


示例一：测试N个节点间allreduce通信模式效率，每个节点开启2个进程，获取不同消息粒度下的通信时间。     

```
/opt/intel/impi/2018.3.222/bin64/mpirun -genv I_MPI_DEBUG 5 -np <N*2> -ppn 2 -host <node0>,...,<nodeN> /opt/intel-mpi-benchmarks/2019/IMB-MPI1 -npmin 2 -msglog 19:21 allreduce    
```     
示例二：测试N个节点间alltoall通信模式效率，每个节点开启1个进程，获取不同消息粒度下的通信时间。    
```
/opt/intel/impi/2018.3.222/bin64/mpirun -genv I_MPI_DEBUG 5 -np <N> -ppn 1 -host <node0>,...,<nodeN> /opt/intel-mpi-benchmarks/2019/IMB-MPI1 -npmin 1 -msglog 15:17 alltoall
```

```
mpirun  -np 2  -host 10.22.116.220,10.22.116.221 --mca btl_openib_if_include   mlx5_1  ./IMB-MPI1 -npmin 1 -msglog 15:17 alltoall
```

基于ucx   
```
mpirun  -np 2  -host 10.22.116.220,10.22.116.221  -x UCX_NET_DEVICES=mlx5_1:1  ./IMB-MPI1 -npmin 1 -msglog 15:17 alltoall
```

```
--------------------------------------------------------------------------
WARNING: No preset parameters were found for the device that Open MPI
detected:

  Local host:            node2
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
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           node2
  Local device:         mlx5_1
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
[1753684564.462789] [node2:2290969:0]     ucp_context.c:1774 UCX  WARN  UCP version is incompatible, required: 1.14, actual: 1.12 (release 1)
#----------------------------------------------------------------
#    Intel(R) MPI Benchmarks 2018, MPI-1 part
#----------------------------------------------------------------
# Date                  : Mon Jul 28 06:36:04 2025
# Machine               : x86_64
# System                : Linux
# Release               : 5.15.0-138-generic
# Version               : #148-Ubuntu SMP Fri Mar 14 19:05:48 UTC 2025
# MPI Version           : 3.1
# MPI Thread Environment: 


# Calling sequence was: 

# ./IMB-MPI1 -npmin 1 -msglog 15:17 alltoall

# Minimum message length in bytes:   0
# Maximum message length in bytes:   131072
#
# MPI_Datatype                   :   MPI_BYTE 
# MPI_Datatype for reductions    :   MPI_FLOAT
# MPI_Op                         :   MPI_SUM  
#
#

# List of Benchmarks to run:

# Alltoall

#----------------------------------------------------------------
# Benchmarking Alltoall 
# #processes = 1 
# ( 1 additional process waiting in MPI_Barrier)
#----------------------------------------------------------------
       #bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]
            0         1000         0.03         0.03         0.03
        32768         1000         0.94         0.94         0.94
        65536          640         1.73         1.73         1.73
       131072          320         3.69         3.69         3.69

#----------------------------------------------------------------
# Benchmarking Alltoall 
# #processes = 2 
#----------------------------------------------------------------
       #bytes #repetitions  t_min[usec]  t_max[usec]  t_avg[usec]
            0         1000         0.03         0.03         0.03
        32768         1000        10.01        10.02        10.02
        65536          640        18.44        18.63        18.53
       131072          320        25.41        25.54        25.47


# All processes entering MPI_Finalize

[node2:2290964] 3 more processes have sent help message help-mpi-btl-openib.txt / no device params found
[node2:2290964] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
[node2:2290964] 1 more process has sent help message help-mpi-btl-openib-cpc-base.txt / no cpcs for port
```

## test2

```
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print a message from each process
    printf("Hello, World! I am process %d of %d.\n", rank, size);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
```

```
mpicc hello.c -o hello

```

```
root@node:~/openMpi/mpi-benchmarks/demo# mpirun  --mca btl_openib_warn_no_device_params_found 0  -n 2 ./hello
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           node
  Local device:         mlx5_1
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
Hello, World! I am process 0 of 2.
Hello, World! I am process 1 of 2.
[node:3622506] 1 more process has sent help message help-mpi-btl-openib-cpc-base.txt / no cpcs for port
[node:3622506] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
root@node:~/openMpi/mpi-benchmarks/demo# mpirun  --mca btl_openib_warn_no_device_params_found 0  -n 3 ./hello
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           node
  Local device:         mlx5_1
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
Hello, World! I am process 0 of 3.
Hello, World! I am process 1 of 3.
Hello, World! I am process 2 of 3.
[node:3622520] 2 more processes have sent help message help-mpi-btl-openib-cpc-base.txt / no cpcs for port
[node:3622520] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
root@node:~/openMpi/mpi-benchmarks/demo# 
```


```
root@node:~/openMpi/mpi-benchmarks/demo#
mpirun  -np 2  -host 10.22.116.220,10.22.116.221  --mca btl_openib_warn_no_device_params_found 0  -x UCX_NET_DEVICES=mlx5_1:1  ./hello -npmin 1 -msglog 15:17 alltoall
--------------------------------------------------------------------------
No OpenFabrics connection schemes reported that they were able to be
used on a specific port.  As such, the openib BTL (OpenFabrics
support) will be disabled for this port.

  Local host:           node2
  Local device:         mlx5_1
  Local port:           1
  CPCs attempted:       rdmacm, udcm
--------------------------------------------------------------------------
[1753757184.080841] [node2:2292458:0]     ucp_context.c:1774 UCX  WARN  UCP version is incompatible, required: 1.14, actual: 1.12 (release 1)
Hello, World! I am process 0 of 2.
Hello, World! I am process 1 of 2.
[node:3622441] 1 more process has sent help message help-mpi-btl-openib-cpc-base.txt / no cpcs for port
[node:3622441] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
```
# UNREACHABLE in file server/pmix_server.c
```
mpirun -np 16 hello_mpifh
```

```
[scu-ce.gpu-0-0:13935] PMIX ERROR: UNREACHABLE in file server/pmix_server.c at line 2198
[scu-ce.gpu-0-0:13935] PMIX ERROR: UNREACHABLE in file server/pmix_server.c at line 2198
[scu-ce.gpu-0-0:13935] PMIX ERROR: UNREACHABLE in file server/pmix_server.c at line 2198
[scu-ce.gpu-0-0:13935] PMIX ERROR: UNREACHABLE in file server/pmix_server.c at line 2198
[scu-ce.gpu-0-0:13935] PMIX ERROR: UNREACHABLE in file server/pmix_server.c at line 2198
......
```
我们添加运行参数   
```
mpirun --mca btl vader,self -np 16 hello_mpifh
```
成功运行。    

我们发现，OpenMPI-4.1.1版本运行不再是老版本那样直接mpirun就可以运行了，需添加参数。但有时候我们对参数不清楚，或者有些计算软件是封包的，使用参数会报错，因此，我们用alias方法，把mpirun+参数简化为mpirun，方法如下：     
在用户环境变量文件.bashrc，或者OpenMPI环境文件中添加如下一行    
alias mpirun='mpirun --mca btl vader,self'   
加载生效后，我们运行命令   
mpirun -np 16 hello_mpifh   
运行输出结果正确      


#  Caught signal 11
```
Caught signal 11 (Segmentation fault: address not mapped to object at address 0x2410)
==== backtrace (tid:2333777) ====
 0  /lib/x86_64-linux-gnu/libucs.so.0(ucs_handle_error+0x2e4) [0x7fb4416c6fc4]
 1  /lib/x86_64-linux-gnu/libucs.so.0(+0x24fec) [0x7fb4416cafec]
 2  /lib/x86_64-linux-gnu/libucs.so.0(+0x251aa) [0x7fb4416cb1aa]
````


+ 查看ucx_info 命令
```
ucx_info  
ucx_info: symbol lookup error: ucx_info: undefined symbol: uct_md_query_v2
```

```
ucx_info -d -t rc
ucx_info: symbol lookup error: ucx_info: undefined symbol: uct_md_query_v2
```



+ ucx 编译


```
root@node:~/rdma-bench/ucx# ls ./build/
bin  etc  include  lib  share
root@node:~/rdma-bench/ucx# find ./ -name '*ibucs.so*'
./build/lib/libucs.so.0.0.0
./build/lib/libucs.so
./build/lib/libucs.so.0
./src/ucs/.libs/libucs.so.0.0.0
./src/ucs/.libs/libucs.so
./src/ucs/.libs/libucs.so.0.0.0T
./src/ucs/.libs/libucs.so.0
```

```
~/openMpi/openmpi-4.1.4# ./configure --prefix=/usr/local/openmpi  --with-ucx=/root/rdma-bench/ucx/build> config.out 2>&1       
```


```
Rank 0 uses GPU 0
[node2:2600718:0:2600718] Caught signal 11 (Segmentation fault: address not mapped to object at address 0x2410)
==== backtrace (tid:2600718) ====
 0  /root/rdma-bench/ucx/build/lib/libucs.so.0(ucs_handle_error+0x2e4) [0x7fc7401dba94]
 1  /root/rdma-bench/ucx/build/lib/libucs.so.0(+0x36c8d) [0x7fc7401dbc8d]
 2  /root/rdma-bench/ucx/build/lib/libucs.so.0(+0x36f76) [0x7fc7401dbf76]
 3  /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(_Z24ncclTopoGetSystemFromXmlP7ncclXmlPP14ncclTopoSystem+0x1ec) [0x7fc74319e80b]
 4  /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(_Z17ncclTopoGetSystemP8ncclCommPP14ncclTopoSystem+0x212) [0x7fc74319f07a]
 5  /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(+0x7949f) [0x7fc74311049f]
 6  /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(ncclCommInitRank+0x297) [0x7fc7431108df]
 7  ./mpi_nccl_bcast(+0x3688) [0x55e25ec74688]
 8  /lib/x86_64-linux-gnu/libc.so.6(+0x29d90) [0x7fc742d44d90]
 9  /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80) [0x7fc742d44e40]
10  ./mpi_nccl_bcast(+0x2345) [0x55e25ec73345]
=================================
[node2:2600718] *** Process received signal ***
[node2:2600718] Signal: Segmentation fault (11)
[node2:2600718] Signal code:  (-6)
[node2:2600718] Failing at address: 0x27af0e
[node2:2600718] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7fc742d5d520]
[node2:2600718] [ 1] /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(_Z24ncclTopoGetSystemFromXmlP7ncclXmlPP14ncclTopoSystem+0x1ec)[0x7fc74319e80b]
[node2:2600718] [ 2] /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(_Z17ncclTopoGetSystemP8ncclCommPP14ncclTopoSystem+0x212)[0x7fc74319f07a]
[node2:2600718] [ 3] /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(+0x7949f)[0x7fc74311049f]
[node2:2600718] [ 4] /root/rdma-bench/NCCL_GP/build/lib/libnccl.so.2(ncclCommInitRank+0x297)[0x7fc7431108df]
[node2:2600718] [ 5] ./mpi_nccl_bcast(+0x3688)[0x55e25ec74688]
[node2:2600718] [ 6] /lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x7fc742d44d90]
[node2:2600718] [ 7] /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80)[0x7fc742d44e40]
[node2:2600718] [ 8] ./mpi_nccl_bcast(+0x2345)[0x55e25ec73345]
[node2:2600718] *** End of error message ***
[node2:2600719] 1 more process has sent help message help-mpi-btl-openib.txt / no device params found
[node2:2600719] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
Segmentation fault (core dumped)
```
+  gdb ./mpi_nccl_bcast   
```
Thread 1 "mpi_nccl_bcast" received signal SIGSEGV, Segmentation fault.
0x00007ffff7f2480b in ncclTopoGetSystemFromXml (xml=0x7fffdc0f4010, topoSystem=0x555555b05328) at graph/topo.cc:547
547       for (int s=0; s<topNode->nSubs; s++) {
(gdb) n
ucs_error_signal_handler (signo=11, info=0x7ffff4f298f0, context=0x7ffff4f297c0) at debug/debug.c:1048
1048    {
(gdb) bt
#0  ucs_error_signal_handler (signo=11, info=0x7ffff4f298f0, context=0x7ffff4f297c0) at debug/debug.c:1048
#1  <signal handler called>
#2  0x00007ffff7f2480b in ncclTopoGetSystemFromXml (xml=0x7fffdc0f4010, topoSystem=0x555555b05328) at graph/topo.cc:547
#3  0x00007ffff7f2507a in ncclTopoGetSystem (comm=0x555555b02f90, system=0x555555b05328) at graph/topo.cc:679
#4  0x00007ffff7e9649f in ncclCommInitRankDev (newcomm=0x7fffffffe1d0, nranks=1, commId=..., myrank=0, cudaDev=0, config=0x7fffffffe050) at init.cc:1603
#5  0x00007ffff7e968df in ncclCommInitRank (newcomm=0x7fffffffe1d0, nranks=1, commId=..., myrank=0) at init.cc:1644
#6  0x0000555555557688 in main (argc=1, argv=0x7fffffffe3b8) at mpi_nccl_bcast.c:67
(gdb) 
```


更换export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/test2/nccl_topo.xml和 export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/test2/nccl_graph.xml     
```
Rank 0 uses GPU 0
node2:2601564:2601564 [0] NCCL INFO Bootstrap : Using eno8303:172.22.116.221<0>
node2:2601564:2601564 [0] NCCL INFO NET/Plugin : Plugin load (libnccl-net.so) returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory
node2:2601564:2601564 [0] NCCL INFO NET/Plugin : No plugin found, using internal implementation
node2:2601564:2601564 [0] NCCL INFO cudaDriverVersion 12020
NCCL version 2.18.3+cuda.
node2:2601564:2601564 [0] NCCL INFO NCCL_TOPO_FILE set by environment to /root/rdma-bench/NCCL_GP/test2/nccl_topo.xml
node2:2601564:2601564 [0] NCCL INFO Loading topology file /root/rdma-bench/NCCL_GP/test2/nccl_topo.xml
node2:2601564:2601564 [0] NCCL INFO Loading unnamed topology

node2:2601564:2601564 [0] graph/xml.cc:155 NCCL WARN XML Parse : expected >, got '
'
node2:2601564:2601564 [0] NCCL INFO graph/xml.cc:177 -> 3
node2:2601564:2601564 [0] NCCL INFO graph/xml.cc:299 -> 3
node2:2601564:2601564 [0] NCCL INFO graph/xml.cc:201 -> 3
node2:2601564:2601564 [0] NCCL INFO graph/xml.cc:314 -> 3
node2:2601564:2601564 [0] NCCL INFO graph/topo.cc:602 -> 3
node2:2601564:2601564 [0] NCCL INFO init.cc:1603 -> 3
node2:2601564:2601564 [0] NCCL INFO init.cc:1644 -> 3
NCCL error at mpi_nccl_bcast.c:67 'internal error - please report this issue to the NCCL developers'
[node2:2601565] 1 more process has sent help message help-mpi-btl-openib.txt / no device params found
[node2:2601565] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error message
```

#  ORTE_ERROR_LOG: Data unpack had inadequate space in file base/plm_base_launch_support.c at line 1213


```
hwloc-info --version
hwloc-info 2.7.0
```
 
 