
# docker 

编译的docker镜像   
```
root@ubuntux86:# docker images | grep 54478aaec63b
nvidia/cuda                                                          11.6.1-cudnn8-devel-ubuntu20.04         54478aaec63b   2 years ago     8.53GB
root@ubuntux86:#
```

```
 docker run --name nccl2 -itd  --rm -v /work/nccl:/workspace    --network=my_net  --shm-size=4g --ulimit memlock=-1 --cap-add=NET_ADMIN --privileged=true  --ip=172.20.0.80  54478aaec63b
```


```
root@ubuntux86:# git branch
* (HEAD detached at origin/v2.17-racecheck)
  master
root@ubuntux86:# 
```


```

$ make -j src.build

……

Archiving  objects                             > /workspace/nccl-dev/nccl/build/obj/collectives/device/colldevice.a
make[2]: Leaving directory '/workspace/nccl-dev/nccl/src/collectives/device'
Linking    libnccl.so.2.17.1                   > /workspace/nccl-dev/nccl/build/lib/libnccl.so.2.17.1
Archiving  libnccl_static.a                    > /workspace/nccl-dev/nccl/build/lib/libnccl_static.a
make[1]: Leaving directory '/workspace/nccl-dev/nccl/src'
```