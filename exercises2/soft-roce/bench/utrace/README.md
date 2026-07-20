

# client

+ step1
```
ulimit -n 65535
```

+ step2

```
mkdir log
```

+ run

```
root@ubuntu:~/rdma-benckmark/utrace# rm -rf log/*
root@ubuntu:~/rdma-benckmark/utrace# python3 setup.py -f   scripts/rdma_websearch_test.yaml  -c
Thread 2 All flows are generated. Wait for all flows to finish.
Thread 1 All flows are generated. Wait for all flows to finish.
Thread 0 All flows are generated. Wait for all flows to finish.
Thread 3 All flows are generated. Wait for all flows to finish.

```

```
ps -elf | grep rdma
4 S root        1309       1  0  80   0 -  1588 do_pol May06 ?        00:00:00 /usr/sbin/rdma-ndd --systemd
1 I root        1565       2  0  60 -20 -     0 rescue May06 ?        00:00:00 [rdma_cm]
0 S root      175883  159799  3  80   0 - 903222 do_wai 02:51 pts/0   00:00:17 python3 setup.py -f scripts/rdma_websearch_test.yaml -c
5 S root      181894  175883  0  80   0 - 911418 futex_ 02:51 pts/0   00:00:00 python3 setup.py -f scripts/rdma_websearch_test.yaml -c
5 S root      181896  175883  0  80   0 - 911418 futex_ 02:51 pts/0   00:00:00 python3 setup.py -f scripts/rdma_websearch_test.yaml -c
5 S root      181899  175883  0  80   0 - 911418 futex_ 02:51 pts/0   00:00:00 python3 setup.py -f scripts/rdma_websearch_test.yaml -c
0 S root      182020  181982  0  80   0 -  1653 pipe_r 03:00 pts/1    00:00:00 grep --color=auto rdma
```


# server

```
 python3 setup.py -f scripts/rdma_websearch_test.yaml -s
```

```
ps -elf | grep rdma
4 S root        1303       1  0  80   0 -  1588 do_pol May06 ?        00:00:00 /usr/sbin/rdma-ndd --systemd
1 I root        1583       2  0  60 -20 -     0 rescue May06 ?        00:00:00 [rdma_cm]
0 S root      181786  178422  0  80   0 - 1666548 do_sel 02:38 pts/0  00:00:06 python3 setup.py -f scripts/rdma_websearch_test.yaml -s
0 S root      184877  181714  0  80   0 -  1653 pipe_r 03:00 pts/1    00:00:00 grep --color=auto rdma
```
client完成测试，server要退出，否则接着测试，client会卡死    


# result


```
vim log/rdma_output_flow2988.txt
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : mlx5_1
 Number of qps   : 1            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 PCIe relax order: OFF
 ibv_wr* API     : ON
 TX depth        : 5
 CQ Moderation   : 5
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs     : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0xdc55 PSN 0x37d124 RKey 0x1c3332 VAddr 0x0056000d90c40d
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:220
 remote address: LID 0000 QPN 0x1edcb PSN 0x1aa4c4 RKey 0x1c9d9c VAddr 0x00564c5773f40d
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:22:116:221
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 5133       5                2167.02            1375.93            0.281078
---------------------------------------------------------------------------------------
```

# references

[beegfs-rdma](https://github.com/multiib/beegfs-thesis/tree/e221583627028083d4e1dc3bc51a3b56331548da)