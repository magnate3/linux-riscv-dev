
#  什么是DCT

[什么是DCT](https://zhuanlan.zhihu.com/p/567720023)   


# DC RTT ping test

## Description
dcping is an example code for testing network Round Trip Time.
It shows how to use the Mellanox DC QP to implement RDMA Write operatinos, and the verbs extended CQ to get time stamps for each transation.

The server (passive side) application creates a DCT QP, allowing RDMA WRITE's.
The client (active size) creates a DCI QP. Once addressing is resolved by the client, it issues 2 sequeancial's RDMA WRITE's and request a completion events. Then it will check the time-stamp diff between the 2 hardward timestamps to calculate the full RTT. It does that in a loop to get multiple results and calc min/avg/max RTT's.

this example code demo's the following API's:
1. [mlx5dv_create_qp()](https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/man/mlx5dv_create_qp.3.md) for DCT & DCI  
2. RDMA_CM external QP for DC address resolution:  
2a. mlx5dv_reserved_qpn_alloc()  
2b. ibv_query_ece()/ibv_set_ece()  
2c. rdma_set_local_ece()/rdma_get_remote_ece()  
3. ibv_qp_ex WR's sends on the Mellanox DCI QP ([man ibv_wr_post](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_wr_post.3.md)):  
3a. ibv_wr_start()  
3b. ibv_wr_rdma_write()  
3c. mlx5dv_set__dc_addr()  
3d. ibv_wr_set_sge()  
3e. ibv_wr_complete()  
4. ibv_cq_ex ([man ibv_create_cq_ex](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_wr_post.3.md)):  
4a. ibv_start_poll()  
4b. ibv_wc_read_completion_ts()  
4c. ibv_end_poll() 

ibv_create_srq    

![images](rd.png)

## Build:
```sh
$ make clean
$ make
```

## Run Server:
```sh
$ ./dcping -s -a 192.192.20.13 -d
created cm_id 0x158aef0
rdma_bind_addr successful on address: <192.192.20.13:7174>
rdma_listen
created pd 0x158a5c0
created channel 0x158ab30
created cq 0x158e3b0
created srq 0x158e5d8
created qp 0x1592018 (qpn=4700)
hw_clocks_kHz = 78125
allocated & registered buffers...
server ready, waiting for client connection requests...
waiting for client events ...
got cm event: RDMA_CM_EVENT_CONNECT_REQUEST(4) status=0, cm_id 0x1592560
accepting client connection request from <192.192.20.13:57929> (cm_id 0x1592560)
waiting for client events ...
got cm event: RDMA_CM_EVENT_ESTABLISHED(9) status=0, cm_id 0x1592560
client connection established (cm_id 0x1592560)
waiting for client events ...
got cm event: RDMA_CM_EVENT_DISCONNECTED(10) status=0, cm_id 0x1592560
client connection disconnected (cm_id 0x1592560)
waiting for client events ...
^C
```

## Run Client:
```sh
$ ./dcping -s -a 10.22.116.221 -d
created cm_id 0x5629e7a996c0
rdma_bind_addr successful on address: <10.22.116.221:7174>
rdma_listen
created pd 0x5629e7a9a210
allocated & registered buffers...
server ready, waiting for client connection requests...
waiting for client events ...
got cm event: RDMA_CM_EVENT_CONNECT_REQUEST(4) status=0, cm_id 0x5629e7a9a8b0
accepting client connection request from <10.22.116.220:40524> (cm_id 0x5629e7a9a8b0)
created channel 0x5629e7a9a270
created cq 0x5629e7a9ab10
created srq 0x5629e7a9f0b8
update ECE from CM responce to QP...(0x15b3, 0x30000000)
created qp 0x5629e7a9f228 (qpn=49133)
hw_clocks_kHz = 1000000
update ECE from QP to CM...(0x15b3, 0)
waiting for client events ...
got cm event: RDMA_CM_EVENT_ESTABLISHED(9) status=0, cm_id 0x5629e7a9a8b0
client connection established (cm_id 0x5629e7a9a8b0)
waiting for client events ...
got cm event: RDMA_CM_EVENT_DISCONNECTED(10) status=0, cm_id 0x5629e7a9a8b0
client connection disconnected (cm_id 0x5629e7a9a8b0)
waiting for client events ...
```


