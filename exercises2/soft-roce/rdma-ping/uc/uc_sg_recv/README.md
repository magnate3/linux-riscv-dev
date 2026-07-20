
+  在发送方，CPU发出一个RDMA send，这个send是由两个entry构成的sg_list，NIC将它们合并成一个要发送出的网络包，只要***总长度不超过MTU即可***。 

2048 + 2048= mtu_4096  
+  在接收方，CPU预先posts 一个recv，同样由一个两个entry构成的sg_list        

```
./uc_pingpong_sg 10.22.116.221  -d mlx5_1 -g 3 -p 8777  -r 8 -s 2048
  local address:  LID 0x0000, QPN 0x0002b9, PSN 0x0d2f4d, GID ::ffff:10.22.116.220
  remote address: LID 0x0000, QPN 0x0002e8, PSN 0xb21145, GID ::ffff:10.22.116.221
4096000 bytes in 0.01 seconds = 4820.24 Mbit/sec
1000 iters in 0.01 seconds = 6.80 usec/iter
```