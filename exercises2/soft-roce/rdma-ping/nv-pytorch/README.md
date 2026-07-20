


+ nccl库的docker镜像:  
[nv-PyTorch-docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.08-py3)

+ 镜像拉取：    



```
docker pull nvcr.io/nvidia/pytorch:24.07-py3
```
启动方式参考：   
```
sudo docker run  --net=host  --name megatron_test  --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" \
-v /home/xky/:/home/xky \
-u 0 \
--name=nccl nvcr.io/nvidia/pytorch:24.07-py3 bash
````
+ 编译：    

建议在镜像容器上面进行编译，工程位置： [CalvinXKY/BasicCUDA](https://github.com/CalvinXKY/BasicCUDA/tree/master/nccl)         

```
cd BasicCUDA/nccl/   
make  
```
获得 alltoall_test可执行文件。

+ 运行方式：    
```
./alltoall_test
```


+ 如果要观察profiling，运行方式：   
```
nvprof --csv -o profile_output.csv ./alltoall_test
```