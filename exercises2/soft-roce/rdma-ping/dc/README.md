


# server



```
./ib_write_bw -d mlx5_1  -x 5 -c DC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8887

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
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
 local address: LID 0000 QPN 0xc07b PSN 0x79b32b RKey 0x0538fa VAddr 0x007f7feca10000 SRQn 0x00c07a
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x9015 PSN 0x2d1ecc RKey 0x0543a2 VAddr 0x007f31016bd000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    5000             97.57              97.57              0.001454
---------------------------------------------------------------------------------------
```


```
./ib_write_bw -d mlx5_1  -x 5 -c DC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8888

************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
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
 local address: LID 0000 QPN 0xc07d PSN 0x23d010 RKey 0x0538fb VAddr 0x007f7e92355000 SRQn 0x00c07c
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
 remote address: LID 0000 QPN 0x9015 PSN 0x2d1ecc RKey 0x0543a2 VAddr 0x007f31016bd000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 8388608    5000             97.57              97.57              0.001454
---------------------------------------------------------------------------------------
```


# client    

```
./ib_write_dc_bw -d mlx5_1  -x 3 -c DC --qp=1 --report_gbits -s 4096 -m 4096     -a  -F -p 8887 10.22.116.221
dc init attr dc type = MLX5DV_DCTYPE_DCI 
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
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
 local address: LID 0000 QPN 0x9015 PSN 0x2d1ecc RKey 0x0543a2 VAddr 0x007f31016bd000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0xc07b PSN 0x79b32b RKey 0x0538fa VAddr 0x007f7feca10000 SRQn 0x00c07a
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          5000           0.093567            0.092692            5.793246
 4          5000             0.19               0.19               5.910941
 8          5000             0.38               0.38               5.896852
 16         5000             0.76               0.76               5.898842
 32         5000             1.51               1.50               5.868248
 64         5000             3.02               3.02               5.894808
 128        5000             6.01               6.00               5.858409
 256        5000             12.01              12.00              5.857977
 512        5000             23.95              23.94              5.844037
 1024       5000             47.49              47.17              5.757748
 2048       5000             82.96              82.79              5.053389
 4096       5000             96.66              96.65              2.949384
 8192       5000             97.16              97.10              1.481669
 16384      5000             97.34              97.34              0.742633
 32768      5000             97.47              97.45              0.371756
 65536      5000             97.51              97.50              0.185971
 131072     5000             97.54              97.53              0.093015
 262144     5000             97.55              97.55              0.046514
 524288     5000             97.56              97.56              0.023259
 1048576    5000             97.56              97.56              0.011630
 2097152    5000             97.56              97.56              0.005815
 4194304    5000             97.56              97.56              0.002908
 8388608    5000             97.57              97.57              0.001454
---------------------------------------------------------------------------------------
***********again ******************* 
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
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
 remote address: LID 0000 QPN 0x9015 PSN 0x2d1ecc RKey 0x0543a2 VAddr 0x007f31016bd000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 all qps hand shake 
Failed to modify QP 36885 dest
 remote address: LID 0000 QPN 0x9015 PSN 0x2d1ecc RKey 0x0543a2 VAddr 0x007f31016bd000 SRQn 00000000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0xc07d PSN 0x23d010 RKey 0x0538fb VAddr 0x007f7e92355000 SRQn 0x00c07c
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:222
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
********* begin to iter ************* 
 2          5000           0.094395            0.094118            5.882360
 4          5000             0.19               0.19               5.901683
 8          5000             0.38               0.38               5.897791
 16         5000             0.76               0.75               5.891745
 32         5000             1.51               1.50               5.849718
 64         5000             3.03               3.02               5.900123
 128        5000             6.01               6.00               5.860868
 256        5000             12.01              11.99              5.852087
 512        5000             23.95              23.89              5.832768
 1024       5000             46.95              46.65              5.694547
 2048       5000             81.92              81.76              4.990234
 4096       5000             96.66              96.64              2.949200
 8192       5000             97.16              97.10              1.481666
 16384      5000             97.34              97.33              0.742541
 32768      5000             97.47              97.45              0.371753
 65536      5000             97.50              97.50              0.185959
 131072     5000             97.53              97.53              0.093012
 262144     5000             97.55              97.55              0.046514
 524288     5000             97.56              97.55              0.023259
 1048576    5000             97.56              97.56              0.011630
 2097152    5000             97.56              97.56              0.005815
 4194304    5000             97.56              97.56              0.002908
 8388608    5000             97.57              97.57              0.001454
---------------------------------------------------------------------------------------
```