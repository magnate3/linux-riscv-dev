

```
git checkout v1.19.0-rc1
./autogen.sh
./contrib/configure-release --prefix=/root/rdma-benckmark/ucx/build
make -j8
make install
```

```
root@ubuntu:~/rdma-benckmark/ucx/build# ./bin/ucx_info  -d | grep -i rdma
# Connection manager: rdmacm
root@ubuntu:~/rdma-benckmark/ucx/build# ./bin/ucx_info  -b | grep -i rdma
#define HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ 0
#define HAVE_NETLINK_RDMA         1
#define RDMA_DEVICE_TYPE_SMI      1
#define RDMA_NLDEV_ATTR_DEV_TYPE  99
#define UCX_CONFIGURE_FLAGS       "--disable-logging --disable-debug --disable-assertions --disable-params-check --prefix=/root/rdma-benckmark/ucx/build"
#define uct_MODULES               ":ib:rdmacm:cma"
root@ubuntu:~/rdma-benckmark/ucx/build# 
```



```
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy ./bin/ucx_perftest -t tag_bw -m host -s 10000000 -n 10 -p 9999 &
```
```
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy ./bin/ucx_perftest 10.22.116.220 -t tag_bw -m host -s 100000000 -n 10 -p 9999
[1751618583.034650] [ubuntu2:744810:0]        perftest.c:800  UCX  WARN  CPU affinity is not set (bound to 48 cpus). Performance may be impacted.
+--------------+--------------+------------------------------+---------------------+-----------------------+
|              |              |       overhead (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+
|    Stage     | # iterations | 50.0%ile | average | overall |  average |  overall |  average  |  overall  |
+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+
[1751618583.086184] [ubuntu2:744810:0]     ucp_context.c:1321 UCX  WARN  transport 'cuda_copy' is not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
Final:                    10      0.191  8162.713  8162.713    11683.30   11683.30         123         123
```


```
root@ubuntu:~/rdma-bench/ucx# export CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy 
root@ubuntu:~/rdma-bench/ucx#  ./bin/ucx_perftest 10.22.116.220 -t tag_bw -m host -s 100000000 -n 10 -p 9999
```

+ export


```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/rdma-benckmark/ucx/build/lib
export PATH=/root/rdma-benckmark/ucx/build/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy
ucx_perftest -t tag_bw -m host -s 10000000 -n 10 -p 9999 
```

+ gdb


```
ucx_perf_run  -> …… ->  uct_ib_reg_mr
ucx_perf_run  -> …… ->  uct_rc_verbs_iface_poll_rx_common
ucx_perf_run  -> …… -> uct_ud_verbs_post_send
```







+ shell

```
 rm libucm.so libucm.so.0
 ln -sf libucm.so  libucm.so.0.0.0
 ln -sf libucm.so.0  libucm.so.0.0.0
 rm libucp.so   libucp.so.0
 ln -sf libucp.so.0.0.0  libucp.so
 ln -sf libucp.so.0.0.0  libucp.so.0
 rm   libucs_signal.so.0  libucs_signal.so
 ln -sf libucs_signal.so.0.0.0   libucs_signal.so  
 ln -sf libucs_signal.so.0.0.0   libucs_signal.so.0
 rm libucs.so.0 libucs.so
 ln -sf libucs.so.0.0.0  libucs.so
 ln -sf libucs.so.0.0.0  libucs.so.0
 rm libuct.so  libuct.so.0
 ln -sf libuct.so.0.0.0  libuct.so
 ln -sf libuct.so.0.0.0  libuct.so.0
```

#    ucp_client_server  example


```
root@ubuntu:~/rdma-bench/ucx# cd build/share/ucx/examples/
root@ubuntu:~/rdma-bench/ucx/build/share/ucx/examples# pwd
/root/rdma-bench/ucx/build/share/ucx/examples
root@ubuntu:~/rdma-bench/ucx/build/share/ucx/examples# ls
hello_world_util.h  ucp_client_server.c  ucp_hello_world.c  ucp_util.h  uct_hello_world.c
root@ubuntu:~/rdma-bench/ucx/build/share/ucx/examples# pwd
/root/rdma-bench/ucx/build/share/ucx/examples
root@ubuntu:~/rdma-bench/ucx/build/share/ucx/examples# gcc  ucp_client_server.c -lucp -lucs -o ucp_client_server -I/root/rdma-bench/ucx/build/include -L/root/rdma-bench/ucx/build/lib
root@ubuntu:~/rdma-bench/ucx/build/share/ucx/examples# 
```

```
export LD_LIBRARY_PATH=$PWD/install-debug/lib
./ucp_client_server &
./ucp_client_server -a 10.22.116.220 -p 13337 -c tag

```

```