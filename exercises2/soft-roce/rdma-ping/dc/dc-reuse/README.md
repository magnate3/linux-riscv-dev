
#  ibv_gid  

```
char buf[128];
    sprintf(buf, "::ffff:%s", ip);
    // TODO: correct GID
    union ibv_gid dgid;
    if (inet_pton(AF_INET6, buf, &dgid) != 1)
      throw RDMAException("inet_pton");
```

```
	inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof gid);
	printf("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
	       rem_dest->lid, rem_dest->qpn, rem_dest->psn, gid);
```


```
  char buf[128];
			    union ibv_gid dgid = {0};
			    sprintf(buf, "::ffff:%s", "10:22:116:222");
			    //"00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222"
			    if (inet_pton(AF_INET6, buf, &dgid) != 1)
			    {
				fprintf(stderr, "Failed to get dgid \n");
			    }
			    inet_ntop(AF_INET6,  &dgid,buf, sizeof buf);
			    fprintf(stderr,"************ dgid: %s \n ",buf);
```


```
if (qpt == IBV_QPT_DRIVER && connection_type == DC)
                {
                        #ifdef HAVE_DCS
                        mlx5dv_wr_set_dc_addr_stream(
                                ctx->dv_qp[index],
                                ctx->ah[index],
                                ctx->r_dctn[index],
                                DC_KEY,
                                ctx->dci_stream_id[index]);
                        ctx->dci_stream_id[index] = (ctx->dci_stream_id[index] + 1) & (0xffffffff >> (32 - (user_param->log_active_dci_streams)));
                        #else 
                        mlx5dv_wr_set_dc_addr(
                                ctx->dv_qp[index],
                                ctx->ah[index],
                                ctx->r_dctn[index],
                                DC_KEY);
                        #endif
                }
```


```
warning: Unexpected size of section `.reg-xstate/3312739' in core file.
#0  0x00007f9ceb95be73 in ?? () from /lib/x86_64-linux-gnu/libmlx5.so.1
(gdb) bt
#0  0x00007f9ceb95be73 in ?? () from /lib/x86_64-linux-gnu/libmlx5.so.1
#1  0x000056080632d071 in mlx5dv_wr_set_dc_addr_stream (stream_id=<optimized out>, remote_dc_key=4293844428, remote_dctn=<optimized out>, ah=<optimized out>, mqp=<optimized out>)
    at /usr/include/infiniband/mlx5dv.h:516
#2  _new_post_send (enc=0, connection_type=5, op=IBV_WR_RDMA_WRITE, qpt=IBV_QPT_DRIVER, index=0, inl=0, user_param=0x7ffc201b3e20, ctx=0x7ffc201b3c80) at src/perftest_resources.c:518
#3  new_post_write_sge_dc (ctx=0x7ffc201b3c80, index=0, user_param=0x7ffc201b3e20) at src/perftest_resources.c:611
#4  0x0000560806334e79 in post_send_method (user_param=0x7ffc201b3e20, index=0, ctx=0x7ffc201b3c80) at src/perftest_resources.c:829
#5  post_send_method (user_param=0x7ffc201b3e20, index=0, ctx=0x7ffc201b3c80) at src/perftest_resources.c:824
#6  run_iter_bw (ctx=ctx@entry=0x7ffc201b3c80, user_param=user_param@entry=0x7ffc201b3e20) at src/perftest_resources.c:3789
#7  0x000056080631b680 in main (argc=<optimized out>, argv=<optimized out>) at src/write_dc_bw.c:643
```


```
ctx_connect:
                if (((user_param->connection_type == UD || user_param->connection_type == DC || user_param->connection_type == SRD) &&
                                (user_param->tst == LAT || user_param->machine == CLIENT || user_param->duplex)) ||
                                (user_param->connection_type == SRD && user_param->verb == READ)) {

                        ctx->ah[i] = ibv_create_ah(ctx->pd,&(attr.ah_attr));

                        if (!ctx->ah[i]) {
                                fprintf(stderr, "Failed to create AH\n");
                                return FAILURE;
                        }
                        user_param->ah_allocated = 1;
                }
```


```
 ibv_destroy_ah(ctx->ah[i])
```

# test



## server


server开启两个进程


```
./ib_write_bw -d mlx5_1  -x 3 -c DC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887
./ib_write_bw -d mlx5_1  -x 5 -c DC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8888
```
+ 进程1

```
./ib_write_bw -d mlx5_1  -x 3 -c DC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : DC           Using SRQ      : ON
 PCIe relax order: ON
 ibv_wr* API     : ON
 CQ Moderation   : 100
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0xc0cb PSN 0x6d8c61 RKey 0x053823 VAddr 0x007f7329067000 SRQn 0x00c0ca
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 local address: LID 0000 QPN 0xc0cc PSN 0xf75537 RKey 0x053823 VAddr 0x007f7329867000 SRQn 0x00c0ca
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 remote address: LID 0000 QPN 0x902f PSN 0x953f49 RKey 0x0543ba VAddr 0x007fe8ff6f1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9030 PSN 0xf16536 RKey 0x0543ba VAddr 0x007fe8ffef1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    10000            97.56              97.56              0.001454
---------------------------------------------------------------------------------------
```

+ 进程2    
```
./ib_write_bw -d mlx5_1  -x 5 -c DC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8888

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : DC           Using SRQ      : ON
 PCIe relax order: ON
 ibv_wr* API     : ON
 CQ Moderation   : 100
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 5
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0xc0ce PSN 0x51396 RKey 0x053824 VAddr 0x007f5cfb0da000 SRQn 0x00c0cd
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 local address: LID 0000 QPN 0xc0cf PSN 0xd184c8 RKey 0x053824 VAddr 0x007f5cfb8da000 SRQn 0x00c0cd
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x902f PSN 0x953f49 RKey 0x0543ba VAddr 0x007fe8ff6f1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9030 PSN 0xf16536 RKey 0x0543ba VAddr 0x007fe8ffef1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    10000            97.56              97.56              0.001454
---------------------------------------------------------------------------------------
```


## client    


dc_ctx_modify_qp_to_rts not call ibv_modify_qp   only prepare struct ibv_qp_attr attr  and then call  ibv_create_ah    

```
./ib_write_dc_bw -d mlx5_1  -x 3 -c DC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887 10.22.116.221
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : DC           Using SRQ      : ON
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 100
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x902f PSN 0x953f49 RKey 0x0543ba VAddr 0x007fe8ff6f1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 local address: LID 0000 QPN 0x9030 PSN 0xf16536 RKey 0x0543ba VAddr 0x007fe8ffef1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0xc0cb PSN 0x6d8c61 RKey 0x053823 VAddr 0x007f7329067000 SRQn 0x00c0ca
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 remote address: LID 0000 QPN 0xc0cc PSN 0xf75537 RKey 0x053823 VAddr 0x007f7329867000 SRQn 0x00c0ca
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          10000           0.089888            0.089371            5.585684
 4          10000            0.18               0.18               5.699766
 8          10000            0.36               0.36               5.688586
 16         10000            0.73               0.49               3.820121
 32         10000            1.46               1.46               5.687004
 64         10000            2.92               2.91               5.687807
 128        10000            5.80               5.79               5.656333
 256        10000            11.60              11.59              5.657104
 512        10000            23.27              23.25              5.675114
 1024       10000            46.55              46.46              5.671046
 2048       10000            93.09              92.86              5.667733
 4096       10000            96.95              96.91              2.957551
 8192       10000            97.31              97.24              1.483747
 16384      10000            97.42              97.38              0.742981
 32768      10000            97.51              97.49              0.371888
 65536      10000            97.52              97.52              0.186003
 131072     10000            97.55              97.54              0.093024
 262144     10000            97.55              97.55              0.046517
 524288     10000            97.56              97.56              0.023260
 1048576    10000            97.56              97.56              0.011630
 2097152    10000            97.56              97.56              0.005815
 4194304    10000            97.56              97.56              0.002908
 8388608    10000            97.56              97.56              0.001454
---------------------------------------------------------------------------------------
***********again ******************* 
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : DC           Using SRQ      : ON
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 100
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 remote address: LID 0000 QPN 0x902f PSN 0x953f49 RKey 0x0543ba VAddr 0x007fe8ff6f1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9030 PSN 0xf16536 RKey 0x0543ba VAddr 0x007fe8ffef1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 all qps hand shake 
 remote address: LID 0000 QPN 0x902f PSN 0x953f49 RKey 0x0543ba VAddr 0x007fe8ff6f1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9030 PSN 0xf16536 RKey 0x0543ba VAddr 0x007fe8ffef1000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0xc0ce PSN 0x51396 RKey 0x053824 VAddr 0x007f5cfb0da000 SRQn 0x00c0cd
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0xc0cf PSN 0xd184c8 RKey 0x053824 VAddr 0x007f5cfb8da000 SRQn 0x00c0cd
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
********* begin to iter ************* 
 2          10000           0.090652            0.090361            5.647532
 4          10000            0.18               0.18               5.689612
 8          10000            0.36               0.36               5.690150
 16         10000            0.73               0.73               5.678514
 32         10000            1.46               1.46               5.690305
 64         10000            2.91               2.91               5.678472
 128        10000            5.80               5.80               5.660498
 256        10000            11.60              11.56              5.645724
 512        10000            23.21              23.20              5.663351
 1024       10000            46.55              46.47              5.672658
 2048       10000            93.36              93.19              5.688043
 4096       10000            96.95              96.91              2.957452
 8192       10000            97.23              97.23              1.483537
 16384      10000            97.42              97.40              0.743135
 32768      10000            97.49              97.48              0.371848
 65536      10000            97.53              97.53              0.186014
 131072     10000            97.54              97.54              0.093021
 262144     10000            97.55              97.55              0.046517
 524288     10000            97.56              97.56              0.023259
 1048576    10000            97.56              97.56              0.011630
 2097152    10000            97.56              97.56              0.005815
 4194304    10000            97.56              97.56              0.002908
 8388608    10000            97.56              97.56              0.001454
---------------------------------------------------------------------------------------
```


# IBV_QP_OOO_RW_DATA_PLACEMENT(Out-of-Order (OOO) Data Placement)
In certain fabric configurations, InfiniBand packets for a given QP may take up different paths in a network from source to destination. This results into packets being received in an out-of-order manner. These packets can now be handled instead of being dropped, in order to avoid retransmission. Data will be placed into host memory in an out-of-order manner when out of order messages are received.    

+ OOO Experimental Verbs   
  ibv_exp_create_dct   

  IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT
 

+ OOO Device Attributes   
IBV_EXP_DEVICE_ATTR_OOO_CAPS   
```
enum ib_exp_ooo_flags {
	/*
	 * Device should set IB_EXP_DEVICE_OOO_RW_DATA_PLACEMENT
	 * capability, when it supports handling RDMA reads and writes
	 * received out of order.
	 */
	IB_EXP_DEVICE_OOO_RW_DATA_PLACEMENT	= (1 << 0),
};
```
