# rdma_fc
[RDMA Flow Control Samples](https://github.com/michaelbe2/rdma_fc/tree/master)



#  dc_read_fc

+ server


```
./dc_read_fc   -G 3 -p 9999 -n 1 
dc_read_fc.c:1341 INFO  Waiting for 1 connections...
dc_read_fc.c:1299 INFO  Total read bandwidth: 7355.32 MB/s
dc_read_fc.c:1082 INFO  Disconnecting 1 connections
```


+ client     


```
./dc_read_fc 10.22.116.220  -G 3 -p 9999 -n 1 
dc_read_fc.c:1116 INFO  Connection[0] to 10.22.116.220...
dc_read_fc.c:1082 INFO  Disconnecting 1 connections
root@ljtest2:~/rdma-bench/rdma_fc/dc_read_fc# ./dc_read_fc 10.22.116.220  -G 3 -p 9999 -n 1 -v
dc_read_fc.c:1116 INFO  Connection[0] to 10.22.116.220...
dc_read_fc.c:954  DEBUG Got rdma_cm event RDMA_CM_EVENT_ADDR_RESOLVED
dc_read_fc.c:954  DEBUG Got rdma_cm event RDMA_CM_EVENT_ROUTE_RESOLVED
dc_read_fc.c:699  DEBUG Created CQ @0x55d23a5ff540
dc_read_fc.c:716  DEBUG Created SRQ @0x55d23a5ff848
dc_read_fc.c:290  DEBUG Registered buffer 0x55d23a619000 length 1024 lkey 0x1bfefd rkey 0x1bfefd
dc_read_fc.c:290  DEBUG Registered buffer 0x7f75ad017000 length 1048576 lkey 0xa4645 rkey 0xa4645
dc_read_fc.c:675  DEBUG Posted 128 receives
dc_read_fc.c:521  DEBUG mlx5dv_create_qp(0x7f75ad118150,0x7fff2fe47570,0x7fff2fe474d0)
```
