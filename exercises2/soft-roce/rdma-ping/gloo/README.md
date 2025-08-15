
[openxla/xla/xla/backends/cpu/collectives/mpi_collectives.cc](https://github.com/openxla/xla)
# test

```
cmake ../ -DBUILD_TEST=1 -DBUILD_EXAMPLES=1
```

+ term1

 
```
root@gloo# export PREFIX=test1
root@gloo# export SIZE=4096
root@gloo# export  RANK=0
```
 
+ term2
```
root@gloo# export PREFIX=test1
root@gloo# export SIZE=4096
root@gloo# export  RANK=1
```