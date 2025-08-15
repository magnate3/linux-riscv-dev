
# mookcake

[注释](https://github.com/sunshenao/mc/tree/b67619f63a8bc128069953588a585a04f60c78fa/mooncake-transfer-engine)

[Mooncake Transfer Engine 发布啦！](https://libfeng.com/posts/mooncake-transfer-engine/)         

#  编译 

```
# For debian/ubuntu
apt-get install -y build-essential \
               cmake \
               libibverbs-dev \
               libgoogle-glog-dev \
               libgtest-dev \
               libjsoncpp-dev \
               libnuma-dev \
               libcurl4-openssl-dev \
               libhiredis-dev

# For centos/alibaba linux os
yum install cmake \
            gflags-devel \
            glog-devel \
            libibverbs-devel \
            numactl-devel \
            gtest \
            gtest-devel \
            boost-devel \
            openssl-devel \
            hiredis-devel \
            libcurl-devel \
```
安装 yalantinglibs    

```


git clone https://github.com/alibaba/yalantinglibs.git
cd yalantinglibs
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
make -j$(nproc)
make install
```


## transfer_engine_bench

```
cmake -DUSE_REDIS=ON -DUSE_HTTP=off -S . -B build
make -j $(nproc)
```


# run 
+ server

```
ibdev2netdev 
mlx5_0 port 1 ==> enp61s0f0np0 (Down)
mlx5_1 port 1 ==> enp61s0f1np1 (Up)
```
export MC_GID_INDEX=3    
```
root@ubuntu:~/rdma-benckmark/Mooncake/mooncake-transfer-engine/build# ./example/transfer_engine_bench  --mode=target    --metadata_server=redis://10.10.16.251:6379  --local_server_name=10.22.116.220:12345 --device_name=enp61s0f1np1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0710 09:44:43.595530 3571482 transfer_engine.cpp:387] Metrics reporting is disabled (set MC_TE_METRIC=1 to enable)
I0710 09:44:43.595564 3571482 transfer_engine.cpp:44] Transfer Engine starting. Server: 10.22.116.220:12345, Metadata: redis://10.10.16.251:6379, ip_or_host_name: 10.22.116.220, rpc_port: 12345
I0710 09:44:43.595701 3571482 transfer_metadata_plugin.cpp:1053] Found active interface eno8303 with IP 172.22.116.220
I0710 09:44:43.595710 3571482 transfer_metadata_plugin.cpp:1053] Found active interface enp61s0f1np1 with IP 10.22.116.220
I0710 09:44:43.595724 3571482 transfer_engine.cpp:112] Transfer Engine RPC using new RPC mapping, listening on 172.22.116.220:16279
E0710 09:44:43.599644 3571482 rdma_context.cpp:464] No matched device found: enp61s0f1np1
E0710 09:44:43.599650 3571482 rdma_context.cpp:66] Failed to open device enp61s0f1np1 on port  with GID 3
W0710 09:44:43.599655 3571482 rdma_transport.cpp:448] Disable device enp61s0f1np1
I0710 09:44:43.605105 3571482 transfer_engine_bench.cpp:422] numa node num: 2
^CI0710 09:46:46.844035 3571482 transfer_engine_bench.cpp:370] Received signal 2, stopping target server...
root@ubuntu:~/rdma-benckmark/Mooncake/mooncake-transfer-engine/build# ./example/transfer_engine_bench  --mode=target    --metadata_server=redis://10.10.16.251:6379  --local_server_name=10.22.116.220:12345 --device_name=mlx5_1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0710 09:47:03.725802 3571485 transfer_engine.cpp:387] Metrics reporting is disabled (set MC_TE_METRIC=1 to enable)
I0710 09:47:03.725838 3571485 transfer_engine.cpp:44] Transfer Engine starting. Server: 10.22.116.220:12345, Metadata: redis://10.10.16.251:6379, ip_or_host_name: 10.22.116.220, rpc_port: 12345
I0710 09:47:03.725956 3571485 transfer_metadata_plugin.cpp:1053] Found active interface eno8303 with IP 172.22.116.220
I0710 09:47:03.725961 3571485 transfer_metadata_plugin.cpp:1053] Found active interface enp61s0f1np1 with IP 10.22.116.220
I0710 09:47:03.725973 3571485 transfer_engine.cpp:112] Transfer Engine RPC using new RPC mapping, listening on 172.22.116.220:16253
I0710 09:47:03.754519 3571485 rdma_context.cpp:125] RDMA device: mlx5_1, LID: 0, GID: (GID_Index 3) 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:16:74:dc
I0710 09:47:04.304725 3571485 transfer_engine_bench.cpp:422] numa node num: 2
```

+ listen    

```
(gdb) bt
#0  listen () at ../sysdeps/unix/syscall-template.S:120
#1  0x00005555555aa4dc in mooncake::SocketHandShakePlugin::startDaemon (this=0x555555603410, listen_port=<optimized out>, sockfd=<optimized out>)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transfer_metadata_plugin.cpp:634
#2  0x0000555555598d3a in mooncake::TransferMetadata::startHandshakeDaemon(std::function<int (mooncake::TransferMetadata::HandShakeDesc const&, mooncake::TransferMetadata::HandShakeDesc&)>, unsigned short, int) (this=this@entry=0x555555603140, on_receive_handshake=..., listen_port=listen_port@entry=15808, sockfd=sockfd@entry=5) at /usr/include/c++/11/bits/shared_ptr_base.h:1295
#3  0x00005555555af6b0 in mooncake::RdmaTransport::startHandshakeDaemon (this=0x555555604fa0, local_server_name=...)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:461
#4  0x00005555555b2f7f in mooncake::RdmaTransport::install (this=0x555555604fa0, local_server_name="10.22.116.220:12345", meta=..., topo=...)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:71
#5  0x000055555558b995 in mooncake::MultiTransport::installTransport (this=0x5555555fdf90, proto="rdma", topo=std::shared_ptr<mooncake::Topology> (use count 4, weak count 0) = {...})
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:222
#6  0x0000555555594f32 in mooncake::TransferEngine::installTransport (this=this@entry=0x555555602d00, proto="rdma", args=<optimized out>)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transfer_engine.cpp:192
#7  0x000055555556cd2d in target () at /root/rdma-bench/Mooncake/mooncake-transfer-engine/example/transfer_engine_bench.cpp:391
#8  0x0000555555568619 in main (argc=<optimized out>, argv=<optimized out>) at /root/rdma-bench/Mooncake/mooncake-transfer-engine/example/transfer_engine_bench.cpp:454
(gdb) 
```

```
epoll_wait
ibv_fork_init
```

+ client

```
ibdev2netdev 
mlx5_0 port 1 ==> enp61s0f0np0 (Down)
mlx5_1 port 1 ==> enp61s0f1np1 (Up)
```

export MC_GID_INDEX=3   
```
 root@ubuntu2:~/rdma-bench/Mooncake/mooncake-transfer-engine/build# ./example/transfer_engine_bench    --metadata_server=redis://10.10.16.251:6379   --segment_id=10.22.116.220:12345 --local_server_name=10.22.116.221:12346 --device_name=mlx5_1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0710 09:47:17.601840 795140 transfer_engine.cpp:387] Metrics reporting is disabled (set MC_TE_METRIC=1 to enable)
I0710 09:47:17.601876 795140 transfer_engine.cpp:44] Transfer Engine starting. Server: 10.22.116.221:12346, Metadata: redis://10.10.16.251:6379, ip_or_host_name: 10.22.116.221, rpc_port: 12346
I0710 09:47:17.601989 795140 transfer_metadata_plugin.cpp:1053] Found active interface eno8303 with IP 172.22.116.221
I0710 09:47:17.601994 795140 transfer_metadata_plugin.cpp:1042] Skipping interface docker0 (not UP or not RUNNING)
I0710 09:47:17.601996 795140 transfer_metadata_plugin.cpp:1053] Found active interface enp61s0f1np1 with IP 10.22.116.221
I0710 09:47:17.601999 795140 transfer_metadata_plugin.cpp:1053] Found active interface enp61s0f1np1 with IP 10.22.116.222
I0710 09:47:17.602002 795140 transfer_metadata_plugin.cpp:1053] Found active interface enp61s0f1np1 with IP 10.22.116.223
I0710 09:47:17.602015 795140 transfer_engine.cpp:112] Transfer Engine RPC using new RPC mapping, listening on 172.22.116.221:15222
I0710 09:47:17.627082 795140 rdma_context.cpp:125] RDMA device: mlx5_1, LID: 0, GID: (GID_Index 3) 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:16:74:dd
I0710 09:47:28.185002 795148 transfer_engine_bench.cpp:238] Worker 3 stopped!
I0710 09:47:28.185706 795150 transfer_engine_bench.cpp:238] Worker 5 stopped!
I0710 09:47:28.186408 795151 transfer_engine_bench.cpp:238] Worker 6 stopped!
I0710 09:47:28.187132 795153 transfer_engine_bench.cpp:238] Worker 8 stopped!
I0710 09:47:28.187856 795149 transfer_engine_bench.cpp:238] Worker 4 stopped!
I0710 09:47:28.188613 795152 transfer_engine_bench.cpp:238] Worker 7 stopped!
I0710 09:47:28.189307 795145 transfer_engine_bench.cpp:238] Worker 0 stopped!
I0710 09:47:28.190043 795146 transfer_engine_bench.cpp:238] Worker 1 stopped!
I0710 09:47:28.190783 795156 transfer_engine_bench.cpp:238] Worker 11 stopped!
I0710 09:47:28.191483 795147 transfer_engine_bench.cpp:238] Worker 2 stopped!
I0710 09:47:28.192220 795154 transfer_engine_bench.cpp:238] Worker 9 stopped!
I0710 09:47:28.192926 795155 transfer_engine_bench.cpp:238] Worker 10 stopped!
I0710 09:47:28.192991 795140 transfer_engine_bench.cpp:350] numa node num: 2
I0710 09:47:28.193001 795140 transfer_engine_bench.cpp:352] Test completed: duration 10.01, batch count 12375, throughput 10.37 GB/s
```

```
(gdb) bt
#0  mooncake::RdmaContext::construct (this=this@entry=0x555555605170, num_cq_list=1, num_comp_channels=1, port=1 '\001', gid_index=3, max_cqe=4096, max_endpoints=256)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:63
#1  0x00005555555b0a0e in mooncake::RdmaTransport::initializeRdmaResources (this=0x555555604fd0)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:442
#2  0x00005555555b2f56 in mooncake::RdmaTransport::install (this=0x555555604fd0, local_server_name="10.22.116.221:12346", meta=..., topo=...)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp:58
#3  0x000055555558b995 in mooncake::MultiTransport::installTransport (this=0x5555555fdf90, proto="rdma", topo=std::shared_ptr<mooncake::Topology> (use count 4, weak count 0) = {...})
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/multi_transport.cpp:222
#4  0x0000555555594f32 in mooncake::TransferEngine::installTransport (this=this@entry=0x555555602d50, proto="rdma", args=<optimized out>)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transfer_engine.cpp:192
#5  0x000055555556e001 in initiator () at /root/rdma-bench/Mooncake/mooncake-transfer-engine/example/transfer_engine_bench.cpp:297
#6  0x0000555555568636 in main (argc=<optimized out>, argv=<optimized out>) at /root/rdma-bench/Mooncake/mooncake-transfer-engine/example/transfer_engine_bench.cpp:452
(gdb) c
```

```
(gdb) bt
#0  mooncake::RdmaEndPoint::construct (this=this@entry=0x7fffe8000dc0, cq=0x555555607b90, num_qp_list=num_qp_list@entry=2, max_sge_per_wr=max_sge_per_wr@entry=4, 
    max_wr_depth=max_wr_depth@entry=256, max_inline_bytes=max_inline_bytes@entry=64) at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp:41
#1  0x00005555555c6846 in mooncake::SIEVEEndpointStore::insertEndpoint (this=0x555555603eb0, peer_nic_path="10.22.116.220:12345@mlx5_1", context=0x555555605170)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/endpoint_store.cpp:147
#2  0x00005555555b8718 in mooncake::RdmaContext::endpoint (this=0x555555605170, peer_nic_path="10.22.116.220:12345@mlx5_1")
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp:277
#3  0x00005555555c2522 in mooncake::WorkerPool::performPostSend (this=<optimized out>, thread_id=<optimized out>)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:223
#4  0x00005555555c3642 in mooncake::WorkerPool::transferWorker (this=0x555555609020, thread_id=1)
    at /root/rdma-bench/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:397
#5  0x00007ffff7d66253 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
#6  0x00007ffff7af5ac3 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
#7  0x00007ffff7b87850 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
```