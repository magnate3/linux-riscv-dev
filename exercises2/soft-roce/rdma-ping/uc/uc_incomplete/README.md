
# test


+ server   
```
./uc_pingpong  -d mlx5_1 -g 3 -p 8777  -r 8 -s 4096  
  local address:  LID 0x0000, QPN 0x0001d5, PSN 0xcd69de, GID ::ffff:10.22.116.220
  remote address: LID 0x0000, QPN 0x0001d7, PSN 0x979664, GID ::ffff:10.22.116.221
before start exchange data 
 server num_a : 0 
start exchange data and num_a will change ? 
the CQ is empty, completion wasn't found in the CQ after timeout
Failed status success (0) for wr_id 0, vendor syndrome: 0x0
Failed status success (0) for wr_id 0, vendor syndrome: 0x0
 server num_a : 1024 
*************** incompelte packet recv ***************** 
```

**incompelte packet recv**
+ client

```
./uc_pingpong   -d mlx5_1 -g 3 -p 8777  -r 8 -s 4096 10.22.116.220 
```