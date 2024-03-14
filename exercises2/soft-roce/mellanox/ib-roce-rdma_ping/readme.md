# build

```
gcc -c rcommon.c -w -o rcommon.o -lpthread -lrdmacm -libverbs
gcc -c rserver.c -w -o rserver.o -lpthread -lrdmacm -libverbs
gcc -c rclient.c -w -o rclient.o -lpthread -lrdmacm -libverbs
gcc -c rping.c -w -o rping.o -lpthread -lrdmacm -libverbs
gcc rcommon.o rserver.o rclient.o rping.o -o rping_test -lpthread -lrdmacm -libverbs
```

# How to test roce!

### no rdma_rxe driver 

mellanox网卡进行roce 不需要rdma_rxe 模块
```
[root@796243dc30b6 mofed_installer]# ethtool -i  eth0
driver: mlx5_core
version: 4.9-5.1.0
firmware-version: 16.27.1016 (MT_0000000011)
expansion-rom-version: 
bus-info: 0000:4b:00.1
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
[root@796243dc30b6 mofed_installer]# 
```

```
[root@796243dc30b6 mofed_installer]# lsmod | grep rdma
rdma_ucm               28672  0 
rdma_cm                57344  2 ib_iser,rdma_ucm
iw_cm                  45056  1 rdma_cm
ib_cm                  53248  3 rdma_cm,ib_ipoib,ib_ucm
ib_uverbs             131072  3 rdma_ucm,mlx5_ib,ib_ucm
ib_core               323584  11 rdma_cm,ib_ipoib,mlx4_ib,iw_cm,ib_iser,ib_umad,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm,ib_ucm
mlx_compat             40960  16 rdma_cm,ib_ipoib,mlx4_core,mlx4_ib,iw_cm,mlx5_fpga_tools,ib_iser,ib_umad,mlx4_en,ib_core,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm,mlx5_core,ib_ucm
[root@796243dc30b6 mofed_installer]# lsmod | grep rdma_rxe
[root@796243dc30b6 mofed_installer]# pwd
/mofed_installer
[root@796243dc30b6 mofed_installer]# pwd
/mofed_installer
[root@796243dc30b6 mofed_installer]# ls
rping_test
```
## server rping
```
[root@796243dc30b6 mofed_installer]# ./rping_test -s -a 194.168.1.3 -v -C 10
server ping data: rdma-ping-0: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqr
server ping data: rdma-ping-1: BCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrs
server ping data: rdma-ping-2: CDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrst
server ping data: rdma-ping-3: DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstu
server ping data: rdma-ping-4: EFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuv
server ping data: rdma-ping-5: FGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvw
server ping data: rdma-ping-6: GHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwx
server ping data: rdma-ping-7: HIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxy
server ping data: rdma-ping-8: IJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz
server ping data: rdma-ping-9: JKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyzA
server DISCONNECT EVENT...
wait for RDMA_READ_ADV state 10
[root@796243dc30b6 mofed_installer]# rdma link show
bash: rdma: command not found
```
 
## client rping
```
[root@e620a32e61c7 mofed_installer]# ./rping_test -c -a 194.168.1.3 -d -C 10
count 10
created cm_id 0x56334a155eb0
cma_event type RDMA_CM_EVENT_ADDR_RESOLVED cma_id 0x56334a155eb0 (parent)
cma_event type RDMA_CM_EVENT_ROUTE_RESOLVED cma_id 0x56334a155eb0 (parent)
rdma_resolve_addr - rdma_resolve_route successful
created pd 0x56334a1558f0
created channel 0x56334a1551d0
created cq 0x56334a158370
created qp 0x56334a15a688
rping_setup_buffers called on cb 0x56334a151780
allocated & registered buffers...
cq_thread started.
cma_event type RDMA_CM_EVENT_ESTABLISHED cma_id 0x56334a155eb0 (parent)
ESTABLISHED
rmda_connect successful
RDMA addr 56334a153070 rkey 3809 len 64
send completion
recv completion
RDMA addr 56334a155180 rkey 390a len 64
send completion
recv completion
RDMA addr 56334a153070 rkey 3809 len 64
send completion
recv completion
RDMA addr 56334a155180 rkey 390a len 64
send completion
recv completion
RDMA addr 56334a153070 rkey 3809 len 64
send completion
recv completion
```

## Main Repos
| Repo | link |
| ------ | ------ |
| rdma-core | https://github.com/linux-rdma/rdma-core |
| librdmacm | https://github.com/ofiwg/librdmacm |