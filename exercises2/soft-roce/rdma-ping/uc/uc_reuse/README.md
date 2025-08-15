

## server    

+ server

server通过ib_write_bw开启两个进程，    

```
./ib_write_bw -d mlx5_1  -x 3 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887
./ib_write_bw -d mlx5_1  -x 5 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8888
```



```
 ./ib_write_bw -d mlx5_1  -x 3 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : UC           Using SRQ      : OFF
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
 local address: LID 0000 QPN 0x1ac7d PSN 0x573c62 RKey 0x053825 VAddr 0x007f214c663000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 local address: LID 0000 QPN 0x1ac7e PSN 0x842ba4 RKey 0x053825 VAddr 0x007f214ce63000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 remote address: LID 0000 QPN 0x9031 PSN 0xe3a562 RKey 0x0543bb VAddr 0x007f670ec83000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9032 PSN 0xa2e0a4 RKey 0x0543bb VAddr 0x007f670f483000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    10000            98.03              98.03              0.001461
---------------------------------------------------------------------------------------
```


```
./ib_write_bw -d mlx5_1  -x 5 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8888

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
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
 local address: LID 0000 QPN 0x1ac7f PSN 0x1e2d62 RKey 0x053826 VAddr 0x007f4f6daac000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 local address: LID 0000 QPN 0x1ac80 PSN 0xbec8a4 RKey 0x053826 VAddr 0x007f4f6e2ac000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x9031 PSN 0xe3a562 RKey 0x0543bb VAddr 0x007f670ec83000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9032 PSN 0xa2e0a4 RKey 0x0543bb VAddr 0x007f670f483000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    10000            98.03              98.03              0.001461
---------------------------------------------------------------------------------------
```


## client 

+ client    

```
 ./ib_write_dc_bw -d mlx5_1  -x 3 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887 10.22.116.221
```


```
./ib_write_dc_bw -d mlx5_1  -x 3 -c UC --qp=2 --report_gbits -s 4096 -m 4096     -a  -F -p 8887 10.22.116.221
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : UC           Using SRQ      : OFF
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
 local address: LID 0000 QPN 0x9031 PSN 0xe3a562 RKey 0x0543bb VAddr 0x007f670ec83000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 local address: LID 0000 QPN 0x9032 PSN 0xa2e0a4 RKey 0x0543bb VAddr 0x007f670f483000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x1ac7d PSN 0x573c62 RKey 0x053825 VAddr 0x007f214c663000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
 remote address: LID 0000 QPN 0x1ac7e PSN 0x842ba4 RKey 0x053825 VAddr 0x007f214ce63000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          10000           0.097264            0.096293            6.018339
 4          10000            0.20               0.20               6.128484
 8          10000            0.39               0.39               6.125593
 16         10000            0.79               0.78               6.123939
 32         10000            1.58               1.57               6.137311
 64         10000            3.13               3.13               6.109445
 128        10000            6.26               6.26               6.114013
 256        10000            12.53              12.52              6.113116
 512        10000            25.13              25.03              6.111939
 1024       10000            50.41              50.29              6.139158
 2048       10000            95.53              95.29              5.816074
 4096       10000            97.67              97.65              2.980000
 8192       10000            97.89              97.85              1.493058
 16384      10000            97.96              97.93              0.747124
 32768      10000            98.00              97.99              0.373791
 65536      10000            98.01              98.01              0.186932
 131072     10000            98.02              98.02              0.093479
 262144     10000            98.03              98.02              0.046742
 524288     10000            98.03              98.03              0.023372
 1048576    10000            98.03              98.03              0.011686
 2097152    10000            98.03              98.03              0.005843
 4194304    10000            98.03              98.03              0.002921
 8388608    10000            98.03              98.03              0.001461
---------------------------------------------------------------------------------------
***********again ******************* 
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 2            Transport type : IB
 Connection type : UC           Using SRQ      : OFF
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
 remote address: LID 0000 QPN 0x9031 PSN 0xe3a562 RKey 0x0543bb VAddr 0x007f670ec83000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9032 PSN 0xa2e0a4 RKey 0x0543bb VAddr 0x007f670f483000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 all qps hand shake 
 remote address: LID 0000 QPN 0x9031 PSN 0xe3a562 RKey 0x0543bb VAddr 0x007f670ec83000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x9032 PSN 0xa2e0a4 RKey 0x0543bb VAddr 0x007f670f483000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x1ac7f PSN 0x1e2d62 RKey 0x053826 VAddr 0x007f4f6daac000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x1ac80 PSN 0xbec8a4 RKey 0x053826 VAddr 0x007f4f6e2ac000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
********* begin to iter ************* 
 2          10000           0.098160            0.097932            6.120723
 4          10000            0.20               0.20               6.137598
 8          10000            0.39               0.39               6.125019
 16         10000            0.79               0.78               6.124224
 32         10000            1.58               1.57               6.138690
 64         10000            3.15               3.14               6.140070
 128        10000            6.28               6.26               6.116126
 256        10000            12.53              12.48              6.091455
 512        10000            25.13              25.07              6.120843
 1024       10000            50.57              50.44              6.157154
 2048       10000            95.53              95.29              5.816138
 4096       10000            97.67              97.61              2.978877
 8192       10000            97.81              97.81              1.492478
 16384      10000            97.96              97.93              0.747125
 32768      10000            98.00              97.98              0.373774
 65536      10000            98.02              98.01              0.186938
 131072     10000            98.02              98.02              0.093478
 262144     10000            98.03              98.02              0.046742
 524288     10000            98.03              98.03              0.023371
 1048576    10000            98.03              98.03              0.011686
 2097152    10000            98.03              98.03              0.005843
 4194304    10000            98.03              98.03              0.002921
 8388608    10000            98.03              98.03              0.001461
---------------------------------------------------------------------------------------

```