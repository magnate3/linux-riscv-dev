#  udpgso_bench_tx

+ make    

```
make -C tools/testing/selftests/net/ 
```

+ port    

```
 ./udpgso_bench_tx  -l 1 -4 -p 8888 -D  172.17.242.27  -S 0 
 
```
不能写成,这样写的话port失效   
```
 ./udpgso_bench_tx  -l 1 -4 -D  172.17.242.27  -S 0 -p 8888
```


```
The first test is to get the performance of userspace payload splitting.
bash# udpgso_bench_tx -l 4 -4 -D "$DST"

The second test is to get the performance of IP fragmentation.
bash# udpgso_bench_tx -l 4 -4 -D "$DST" -f

The third test is to get the performance of UDP GSO.
bash# udpgso_bench_tx -l 4 -4 -D "$DST" -S 0
```