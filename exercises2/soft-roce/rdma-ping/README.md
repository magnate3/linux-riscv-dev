# test


## ibv_devinfo   

```
root@test:~/rdma# rdma link
link irdma0/1 state DOWN physical_state DISABLED netdev enp23s0f0 
link irdma1/1 state ACTIVE physical_state LINK_UP netdev enp23s0f1 
ibv_devinfo -d irdma1  -v
hca_id: irdma1
        transport:                      InfiniBand (0)
        fw_ver:                         1.54
        node_guid:                      6efe:54ff:fe3d:8a39
        sys_image_guid:                 6cfe:543d:8a39:0000
        vendor_id:                      0x8086
        vendor_part_id:                 5522
        hw_ver:                         0x2
        phys_port_cnt:                  1
        max_mr_size:                    0x200000000000
        page_size_cap:                  0x40201000
        max_qp:                         65533
        max_qp_wr:                      4063
        device_cap_flags:               0x00229000
                                        RC_RNR_NAK_GEN
                                        MEM_WINDOW
                                        MEM_MGT_EXTENSIONS
                                        Unknown flags: 0x8000
        max_sge:                        13
        max_sge_rd:                     13
        max_cq:                         131069
        max_cqe:                        1048574
        max_mr:                         2097150
        max_pd:                         262141
        max_qp_rd_atom:                 127
        max_ee_rd_atom:                 0
        max_res_rd_atom:                0
        max_qp_init_rd_atom:            255
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_NONE (0)
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         2097150
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  65536
        max_mcast_qp_attach:            8
        max_total_mcast_qp_attach:      524288
        max_ah:                         131072
        max_fmr:                        0
        max_srq:                        0
        max_pkeys:                      1
        local_ca_ack_delay:             0
        general_odp_caps:
        rc_odp_caps:
                                        NO SUPPORT
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        NO SUPPORT
        xrc_odp_caps:
                                        NO SUPPORT
        completion timestamp_mask:                      0x000000000001ffff
        core clock not supported
        device_cap_flags_ex:            0x229000
        tso_caps:
                max_tso:                        0
        rss_caps:
                max_rwq_indirection_tables:                     0
                max_rwq_indirection_table_size:                 0
                rx_hash_function:                               0x0
                rx_hash_fields_mask:                            0x0
        max_wq_type_rq:                 0
        packet_pacing_caps:
                qp_rate_limit_min:      0kbps
                qp_rate_limit_max:      0kbps
        tag matching not supported
        num_comp_vectors:               48
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               1
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x7fffffff
                        port_cap_flags:         0x04050000
                        port_cap_flags2:        0x0000
                        max_vl_num:             invalid value (0)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           1
                        gid_tbl_len:            32
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           4X (2)
                        active_speed:           25.0 Gbps (32)
                        phys_state:             LINK_UP (5)
                        GID[  0]:               fe80::6efe:54ff:fe3d:8a39, RoCE v2
                        GID[  1]:               ::ffff:10.22.116.221, RoCE v2
```

## test

```
./rc_pingpong -h
./rc_pingpong: invalid option -- 'h'
Usage:
  ./rc_pingpong            start a server and wait for connection
  ./rc_pingpong <host>     connect to server at <host>

Options:
  -p, --port=<port>      listen on/connect to port <port> (default 18515)
  -d, --ib-dev=<dev>     use IB device <dev> (default first device found)
  -i, --ib-port=<port>   use port <port> of IB device (default 1)
  -s, --size=<size>      size of message to exchange (default 4096)
  -m, --mtu=<size>       path MTU (default 1024)
  -r, --rx-depth=<dep>   number of receives to post at a time (default 500)
  -n, --iters=<iters>    number of exchanges (default 1000)
  -l, --sl=<sl>          service level value
  -e, --events           sleep on CQ events (default poll)
  -g, --gid-idx=<gid index> local port gid index
  -f, --fname=<filename> use a mmapable file for the RDMA MR
```

## server

```
./rc_pingpong -d irdma1  -g  1  -p 7777
  local address:  LID 0x0001, QPN 0x000021, PSN 0x6dc420, GID ::ffff:10.22.116.221
  remote address: LID 0x0001, QPN 0x000027, PSN 0x82ab1e, GID ::ffff:10.22.116.220
8192000 bytes in 0.01 seconds = 4453.38 Mbit/sec
1000 iters in 0.01 seconds = 14.72 usec/iter
```

## client




```
./rc_pingpong 10.22.116.221  -d irdma0  -g  1  -p 7777  
  local address:  LID 0x0001, QPN 0x000027, PSN 0x82ab1e, GID ::ffff:10.22.116.220
  remote address: LID 0x0001, QPN 0x000021, PSN 0x6dc420, GID ::ffff:10.22.116.221
8192000 bytes in 0.01 seconds = 4460.05 Mbit/sec
1000 iters in 0.01 seconds = 14.69 usec/iter
```

## tcp listen

```
netstat -pan | grep 7777
tcp        0      0 0.0.0.0:7777            0.0.0.0:*               LISTEN      45069/./rc_pingpong
```



## error Failed to modify QP to RTR

+ server

```
./rc_pingpong -d irdma1  -g  0  -p 7777
  local address:  LID 0x0001, QPN 0x000022, PSN 0x872bcc, GID fe80::6efe:54ff:fe3d:8a39
Failed to modify QP to RTR
Couldn't connect to remote QP
```
采用-g  1    

+ client

```
./rc_pingpong 10.22.116.221  -d irdma0  -g  0  -p 7777  
  local address:  LID 0x0001, QPN 0x000028, PSN 0x93b207, GID fe80::6efe:54ff:fe3d:ba88
client read: No space left on device
Couldn't read remote address
```
采用-g  1    

## rping


```
./rping -s 10.22.116.221  -v -C 3
server ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
server ping data: rdma-ping-1: BCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrs
server ping data: rdma-ping-2: CDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrst
server DISCONNECT EVENT...
wait for RDMA_READ_ADV state 9
```
```
/rping -c -a 10.22.116.221 -v -C 3
ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
ping data: rdma-ping-1: BCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrs
ping data: rdma-ping-2: CDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrst
```

##  RDMA_RC_example    


```
./RDMA_RC_example -g 1 -d  rocep23s0f1 
./RDMA_RC_example -g 1 -d  rocep23s0f0  10.22.116.221
```


# intel

+ iwarp

```
mkdir /sys/kernel/config/rdma_cm/irdma0
echo 16 > /sys/kernel/config/rdma_cm/irdma0/ports/1/default_roce_tos
```

+ roce

```
modprobe irdma roce_ena=1
```

```
mkdir /sys/kernel/config/rdma_cm/rocep23s0f0
echo 96 > /sys/kernel/config/rdma_cm/rocep23s0f0/ports/1/default_roce_tos
```

```
mkdir /sys/kernel/config/rdma_cm/rocep23s0f1
echo 96 > /sys/kernel/config/rdma_cm/rocep23s0f1/ports/1/default_roce_tos
```
![images](ecn.png)

+ 程序开启

```
 attr.ah_attr.grh.traffic_class = 96|0x2;
```

+ server

```
./RDMA_RC_example -g 1 -d  rocep23s0f1  
 ------------------------------------------------
 Device name : "rocep23s0f1"
 IB port : 1
 TCP port : 19875
 GID index : 1
 UDP source port : 0
 ------------------------------------------------

waiting on port 19875 for TCP connection
TCP connection was established
searching for IB devices in host
found 2 device(s)
dev.max_qp = 65533
fill the buffer with 'SEND operation from server'
going to send the message: 'SEND operation from server'
MR was registered with addr=0x559260033020, lkey=0xb23c2b0a, rkey=0xb23c2b0a, flags=0x7
QP was created, QP number=0x9

Local LID = 0x1
Remote address = 0x558ef74f7990
Remote rkey = 0xf1b87ea1
Remote QP number = 0x8
Remote LID = 0x1
Remote GID = 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:16:74:dc
QP state was change to RTS

Test 1: server use 'RDMA send' to client
Send Request was posted
completion was found in CQ with status 0xc
got bad completion with status: 0xc, vendor syndrome: 0x10008
poll completion failed

test result is 1
```

+ client

```
./RDMA_RC_example -g 1 -d  rocep23s0f0  10.22.116.221
 ------------------------------------------------
 Device name : "rocep23s0f0"
 IB port : 1
 IP : 10.22.116.221
 TCP port : 19875
 GID index : 1
 UDP source port : 0
 ------------------------------------------------

TCP connection was established
searching for IB devices in host
found 2 device(s)
dev.max_qp = 65533
MR was registered with addr=0x558ef74f7990, lkey=0xf1b87ea1, rkey=0xf1b87ea1, flags=0x7
QP was created, QP number=0x8

Local LID = 0x1
Remote address = 0x559260033020
Remote rkey = 0xb23c2b0a
Remote QP number = 0x9
Remote LID = 0x1
Remote GID = 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:16:74:dd
Receive Request was posted
QP state was change to RTS

Test 1: server use 'RDMA send' to client
completion wasn't found in the CQ after timeout
poll completion failed

test result is 1./RDMA_RC_example -g 1 -d  rocep23s0f0  10.22.116.221
 ------------------------------------------------
 Device name : "rocep23s0f0"
 IB port : 1
 IP : 10.22.116.221
 TCP port : 19875
 GID index : 1
 UDP source port : 0
 ------------------------------------------------

TCP connection was established
searching for IB devices in host
found 2 device(s)
dev.max_qp = 65533
MR was registered with addr=0x558ef74f7990, lkey=0xf1b87ea1, rkey=0xf1b87ea1, flags=0x7
QP was created, QP number=0x8

Local LID = 0x1
Remote address = 0x559260033020
Remote rkey = 0xb23c2b0a
Remote QP number = 0x9
Remote LID = 0x1
Remote GID = 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:16:74:dd
Receive Request was posted
QP state was change to RTS

Test 1: server use 'RDMA send' to client
completion wasn't found in the CQ after timeout
poll completion failed

test result is 1
```

+ switch

![images](ecn2.png)