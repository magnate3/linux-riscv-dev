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


```
ibv_create_srq(cb->pd, &srq_attr)
```

## Build:
```sh
$ make clean
$ make
```

## Run Server:
```sh
$  ./dcping -s -a 10.22.116.221 -d
created cm_id 0x556d0a72e6c0
rdma_bind_addr successful on address: <10.22.116.221:7174>
rdma_listen
created pd 0x556d0a72f210
allocated & registered buffers...
server ready, waiting for client connection requests...
waiting for client events ...
got cm event: RDMA_CM_EVENT_CONNECT_REQUEST(4) status=0, cm_id 0x556d0a72f8b0
accepting client connection request from <10.22.116.220:40493> (cm_id 0x556d0a72f8b0)
created channel 0x556d0a72f270
created cq 0x556d0a72fb10
created srq 0x556d0a7340b8
update ECE from CM responce to QP...(0x15b3, 0x30000000)
created qp 0x556d0a734228 (qpn=49216)
hw_clocks_kHz = 1000000
update ECE from QP to CM...(0x15b3, 0)
waiting for client events ...
got cm event: RDMA_CM_EVENT_ESTABLISHED(9) status=0, cm_id 0x556d0a72f8b0
client connection established (cm_id 0x556d0a72f8b0)
waiting for client events ...
got cm event: RDMA_CM_EVENT_DISCONNECTED(10) status=0, cm_id 0x556d0a72f8b0
client connection disconnected (cm_id 0x556d0a72f8b0)
waiting for client events ...

^C
```

## Run Client:
```sh
$ ./dcping -c -a 10.22.116.221 -C 100 -D 100 -d
created cm_id 0x55f8868636c0
got cm event: RDMA_CM_EVENT_ADDR_RESOLVED(0) status=0, cm_id 0x55f8868636c0
got cm event: RDMA_CM_EVENT_ROUTE_RESOLVED(2) status=0, cm_id 0x55f8868636c0
rdma_resolve_addr/rdma_resolve_route successful to server: <10.22.116.221:40493>
created pd 0x55f886864820
created channel 0x55f886860420
created cq 0x55f886864880
created qp 0x55f886864b88 (qpn=36854)
hw_clocks_kHz = 1000000
allocated & registered buffers...
update ECE from QP to CM...(0x15b3, 0x30000000)
rdma_connecting...
got cm event: RDMA_CM_EVENT_CONNECT_RESPONSE(5) status=0, cm_id 0x55f8868636c0
got server param's: dctn=49216, buf=93926815085520, size=64, rkey=1930098
update ECE from CM responce to QP...(0x15b3, 0)
modify QP...
created ah (0x55f886869550)
rdma_connect successful
connected to server, starting DC RTT test
[total = 100] rtt = 0.012 / 0.430 / 0.460 usec <min/avg/max>
done DC RTT test
dcping_free_buffers called on cb 0x55f88685f830
dcping_free_qp/srq/cq/pd called on cb 0x55f88685f830
destroy cm_id 0x55f8868636c0
```
