

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



```
(gdb) bt
#0  uct_ud_mlx5_post_send (max_log_sge=2147483647, neth=0x7fffe7bfe850, wqe_size=<optimized out>, ctrl=0x555555867140, ce_se=0 '\000', ep=0x555555660b90, iface=0x5555556df2c0)
    at ud/ud_mlx5.c:103
#1  uct_ud_mlx5_ep_am_bcopy (tl_ep=0x555555660b90, id=<optimized out>, pack_cb=<optimized out>, arg=<optimized out>, flags=<optimized out>) at ud/ud_mlx5.c:438
#2  0x00007ffff7f84a34 in uct_ep_am_bcopy (flags=1, arg=0x7ffff7503930, pack_cb=0x7ffff7f83c80 <ucp_wireup_msg_pack>, id=1 '\001', ep=<optimized out>)
    at /root/rdma-bench/ucx/src/uct/api/uct.h:3074
#3  ucp_wireup_msg_progress (self=0x555555672c08) at wireup/wireup.c:176
#4  0x00007ffff7f8231a in ucp_wireup_ep_progress_pending (self=0x55555566b928) at wireup/wireup_ep.c:121
#5  0x00007ffff771f3ba in uct_ud_ep_do_pending (arbiter=arbiter@entry=0x5555556df908, group=group@entry=0x555555660bb8, elem=elem@entry=0x55555566b930, arg=arg@entry=0x1)
    at ud/base/ud_ep.c:1654
#6  0x00007ffff7e3cc16 in ucs_arbiter_dispatch_nonempty (arbiter=0x5555556df908, per_group=per_group@entry=1, cb=0x7ffff771f2f0 <uct_ud_ep_do_pending>, cb_arg=cb_arg@entry=0x1)
    at datastruct/arbiter.c:321
#7  0x00007ffff76107c6 in ucs_arbiter_dispatch (per_group=1, cb=<optimized out>, cb_arg=0x1, arbiter=0x5555556df908) at /root/rdma-bench/ucx/src/ucs/datastruct/arbiter.h:386
#8  uct_ud_iface_progress_pending (is_async=1, iface=0x5555556df2c0) at /root/rdma-bench/ucx/src/uct/ib/ud/base/ud_iface.h:471
#9  uct_ud_mlx5_iface_async_progress (ud_iface=0x5555556df2c0) at ud/ud_mlx5.c:615
#10 0x00007ffff760baef in uct_ud_iface_async_progress (iface=0x5555556df2c0) at /root/rdma-bench/ucx/src/uct/ib/ud/base/ud_inl.h:275
#11 uct_ud_mlx5_iface_async_handler (fd=<optimized out>, events=<optimized out>, arg=0x5555556df2c0) at ud/ud_mlx5.c:712
#12 0x00007ffff7e3099a in ucs_async_handler_invoke (events=1 '\001', handler=0x55555583dfc0) at async/async.c:268
#13 ucs_async_handler_dispatch (handler=handler@entry=0x55555583dfc0, events=events@entry=1 '\001') at async/async.c:290
#14 0x00007ffff7e314fb in ucs_async_dispatch_handlers (handler_ids=handler_ids@entry=0x7ffff7503c04, count=count@entry=1, events=1 '\001') at async/async.c:322
#15 0x00007ffff7e344bb in ucs_async_thread_ev_handler (callback_data=<optimized out>, events=<optimized out>, arg=0x7ffff7503d70) at async/thread.c:88
#16 0x00007ffff7e56f39 in ucs_event_set_wait (event_set=<optimized out>, num_events=num_events@entry=0x7ffff7503d6c, timeout_ms=<optimized out>, 
    event_set_handler=event_set_handler@entry=0x7ffff7e34480 <ucs_async_thread_ev_handler>, arg=arg@entry=0x7ffff7503d70) at sys/event_set.c:215
#17 0x00007ffff7e34bb2 in ucs_async_thread_func (arg=0x5555556152c0) at async/thread.c:131
#18 0x00007ffff7c27ac3 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
#19 0x00007ffff7cb9850 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
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
export UCX_LOG_LEVEL=debug
./uct_hello_world -d mlx5_1:1 -t rc_verbs
./uct_hello_world -d mlx5_1:1 -t rc_verbs -n 10.22.116.220  
```

# folly


```
./build/fbcode_builder/getdeps.py install-system-deps --recursive
python3 ./build/fbcode_builder/getdeps.py --allow-system-packages build
```