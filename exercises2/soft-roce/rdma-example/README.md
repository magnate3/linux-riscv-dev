# RDMA exmaple

A simple RDMA server client example. The code contains a lot of comments. Here is the workflow that happens in the example: 

Client: 
  1. setup RDMA resources   
  2. connect to the server 
  3. receive server side buffer information via send/recv exchange 
  4. do an RDMA write to the server buffer from a (first) local buffer. The content of the buffer is the string passed with the `-s` argument. 
  5. do an RDMA read to read the content of the server buffer into a second local buffer. 
  6. compare the content of the first and second buffers, and match them. 
  7. disconnect 

Server: 
  1. setup RDMA resources 
  2. wait for a client to connect 
  3. allocate and pin a server buffer
  4. accept the incoming client connection 
  5. send information about the local server buffer to the client 
  6. wait for disconnect

###### How to run      
```text
git clone https://github.com/animeshtrivedi/rdma-example.git
cd ./rdma-example
cmake .
make
``` 
 
###### server

```
[root@centos7 bin]# ./rdma_server 
/root/rdma-example/src/rdma_server.c : 134 : ERROR : Creating cm event channel failed with errno : (-19)/root/prdma-example/src/rdma_server.c : 465 : ERROR : RDMA server failed to start cleanly, ret = -19 
[root@centos7 bin]# 
```
执行rxe_cfg start
![images](../pic/rxe1.png)
```text
 ./bin/rdma_server -a 10.11.11.251 -p 8888
Server is listening successfully at: 10.11.11.251 , port: 8888 
A new connection is accepted from 10.11.11.82 
Client side buffer information is received...
---------------------------------------------------------
buffer attr, addr: 0x2833360 , len: 15 , stag : 0x322 
---------------------------------------------------------
The client has requested buffer length of : 15 bytes 
A disconnect event is received from the client...
Server shut-down is complete 
```
###### client


```text
./rdma_client -a 10.11.11.251 -p 8888 -s "hello world 789"
Passed string is : hello world 789 , with count 15 
Trying to connect to server at : 10.11.11.251 port: 8888 
The client is connected successfully 
---------------------------------------------------------
buffer attr, addr: 0x1f6b4ff0 , len: 15 , stag : 0x516 
---------------------------------------------------------
...
SUCCESS, source and destination buffers match 
Client resource clean up is complete 
```

## Does not have an RDMA device?
In case you do not have an RDMA device to test the code, you can setup SofitWARP software RDMA device on your Linux machine. Follow instructions here: [https://github.com/animeshtrivedi/blog/blob/master/post/2019-06-26-siw.md](https://github.com/animeshtrivedi/blog/blob/master/post/2019-06-26-siw.md).
