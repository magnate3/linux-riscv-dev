
+ client

./uc_pingpong_imm   -d mlx5_1 -g 3 -p 8777  -r 8 -s 40960 10.22.116.221    
```
imm data is 0x10203040 
81920000 bytes in 0.02 seconds = 38075.76 Mbit/sec
1000 iters in 0.02 seconds = 17.21 usec/iter
Max receive completion clock cycles = 23860
Min receive completion clock cycles = 18446744073709551615
Average receive completion clock cycles = 17209.029029
```

+ server

```
./uc_pingpong   -d mlx5_1 -g 3 -p 8777  -r 8 -s 40960
```