

## server    

+ server


```
./ib_write_bw -d mlx5_1  -x 3 -c UC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8887
./ib_write_bw -d mlx5_1  -x 5 -c UC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8888
```


```
 ./ib_write_bw -d mlx5_1  -x 5 -c UC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8887

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
 Connection type : UC           Using SRQ      : OFF
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
 local address: LID 0000 QPN 0x1ac66 PSN 0x438e4e RKey 0x0538e4 VAddr 0x007f9848bc6000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x9008 PSN 0xf4ee36 RKey 0x054395 VAddr 0x007f4b6f5d1000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    5000             98.03              98.03              0.001461
---------------------------------------------------------------------------------------
```


## client 

+ client    

```
 ./ib_write_dc_bw -d mlx5_1  -x 3 -c UC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8887 10.22.116.221
```


```

```