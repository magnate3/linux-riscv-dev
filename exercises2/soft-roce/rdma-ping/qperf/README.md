 
 
# qperf   
 
+ server 
```
qperf 
```
+ client
```
qperf -t 60 -cm1 10.22.116.221  rc_rdma_write_bw
rc_rdma_write_bw:
    bw  =  9.5 GB/sec
```

#  MyPerfv8   

```
./MyPerfv8 start a server and wait for connection
./MyPerfv8 <host> connect to server at <host>

Options:
-p, --port <port> listen on/connect to port <port> (default 18515)

-d, --ib-dev <dev> use IB device <dev> (default first device found)

-i, --ib-port <port> use port <port> of IB device (default 1)

-g, --gid_idx <git index> gid index to be used in GRH (default not used)

-t, --test_interval <interval> perf test interval in us

-o, --test_opcode <opcode> 2 for send, 4 for read, 0 for write

-s, --test_times <num> number of test times

-e, --iter_nums <num> number of iterations per test round(less than 1000)

-m, --Message_size <Bytes> Message's size in Bytes, default 65536.
```

+ client  没有设置 -m 4096    
```
 ./MyPerfv8 -d  mlx5_1  -g 3 -o 0 10.22.116.221
```
+ server     
```
./MyPerfv8  -d  mlx5_1 -m 4096 -o 0 -g 3
```