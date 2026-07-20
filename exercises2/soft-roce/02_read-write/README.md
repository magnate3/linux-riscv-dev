
# server

```
[root@centos7 02_read-write]# ./rdma-server   read
listening on port 34778.
received connection request.
send completed successfully.
received MSG_MR. reading message from remote memory...
on_completion: status is not IBV_WC_SUCCESS.
peer disconnected.
```

# client

```
./rdma-client  write  10.11.11.251  34778 
address resolved.
route resolved.
send completed successfully.
received MSG_MR. writing message to remote memory...
on_completion: status is not IBV_WC_SUCCESS.
```

# references

[RDMA read and write with IB verbs](https://thegeekinthecorner.wordpress.com/2010/09/28/rdma-read-and-write-with-ib-verbs/)