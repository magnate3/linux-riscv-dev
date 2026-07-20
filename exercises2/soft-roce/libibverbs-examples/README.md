
#  /sys/class/

```
[root@centos7 rdma-core]# ls /sys/class/infiniband
rxe0  rxe1
[root@centos7 rdma-core]# ls /sys/class/infiniband_verbs/
abi_version  uverbs0  uverbs1
[root@centos7 rdma-core]# ls /sys/class/infiniband_cm/
abi_version  rxe0  rxe1  ucm0  ucm1
[root@centos7 rdma-core]# 
```


# ./ibv_devices 
```
[root@centos7 libibverbs-examples]# ls /dev/infiniband/
issm0  issm1  rdma_cm  ucm0  ucm1  umad0  umad1  uverbs0  uverbs1
[root@centos7 libibverbs-examples]# ./ibv_devices 
    device                 node GUID
    ------              ----------------
    rxe0                b20875fffe5fb85e
    rxe1                46a191fffea49c0c
[root@centos7 libibverbs-examples]# 
```

#  ibv_context

```
struct ibv_context {
        struct ibv_device      *device;
        struct ibv_context_ops  ops;
        int                     cmd_fd;
        int                     async_fd;
        int                     num_comp_vectors;
        pthread_mutex_t         mutex;
        void                   *abi_compat;
};
```


```
[root@centos7 libibverbs-examples]# ./ibv_context 
2 RDMA device(s) found:

The device 'rxe0' was opened
cmd_fd 3 and async_fd 4 
         max_mr_size   : 18446744073709551615
         max_mr_size   : 18446744073709551615
The device 'rxe1' was opened
cmd_fd 3 and async_fd 4 
         max_mr_size   : 18446744073709551615
         max_mr_size   : 18446744073709551615
```

#  anon_inode


```
./ibv_context 
2 RDMA device(s) found:

The device 'rxe0' was opened
cmd_fd 3 and async_fd 4 
         max_mr_size   : 18446744073709551615
         max_mr_size   : 18446744073709551615
event channel fd 5 and file_path anon_inode:[infinibandevent]
The device 'rxe1' was opened
cmd_fd 3 and async_fd 4 
         max_mr_size   : 18446744073709551615
         max_mr_size   : 18446744073709551615
event channel fd 5 and file_path anon_inode:[infinibandevent]
```


## anon_inode_getfile