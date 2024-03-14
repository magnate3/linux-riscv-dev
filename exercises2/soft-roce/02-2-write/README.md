

# server

```
[root@centos7 02-2-write]# ./rdma-server write
listening on port 60704.
received connection request.
**** recv msg mr 
 send mr rkey is 20832 and lkey is 20832 
send completed successfully.
received MSG_MR. writing message to remote memory...
peer mr rkey is 48129 and lkey is 48129 
send packet to client: I'm server, message from passive/server side with pid 67807 
send completed successfully.
send completed successfully.
recv packet :  I'm client . message from active/client side with pid 6909
peer disconnected.
^C
```

# client

```
 ./rdma-client write 10.11.11.251  60704
address resolved and pad local message.
route resolved.
send mr rkey is 48129 and lkey is 48129 
send completed successfully.
**** recv msg mr 
 received MSG_MR. writing message to remote memory...
peer mr rkey is 20832 and lkey is 20832 
send packet to server :  I'm client . message from active/client side with pid 6909 
send completed successfully.
send completed successfully.
recv packet : I'm server, message from passive/server side with
```