

#  scopeid
```
enahisic2i3: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.10.103.82  netmask 255.255.255.0  broadcast 0.0.0.0
        inet6 fec0::a:4a57:2ff:fe64:e7ae  prefixlen 64  scopeid 0x40<site>
        inet6 fe80::4a57:2ff:fe64:e7ae  prefixlen 64  scopeid 0x20<link>
        ether 48:57:02:64:e7:ae  txqueuelen 1000  (Ethernet)
        RX packets 132110284  bytes 17876565629 (17.8 GB)
        RX errors 111  dropped 1277716  overruns 0  frame 0
        TX packets 1887048  bytes 193403401 (193.4 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
scopeid 0x40<site>, scopeid 0x20<link>   

## bind(): No such device

```
        client_addr.sin6_family = AF_INET6;
        inet_pton(AF_INET6, "fe80::4a57:2ff:fe64:e7ae", &client_addr.sin6_addr);
        client_addr.sin6_port = htons(CLIENT_PORT);
        //client_addr.sin6_scope_id = if_nametoindex(ifname);
        /* Bind address and socket together */
        ret = bind(sock_fd, (struct sockaddr*)&client_addr, sizeof(client_addr));
        if(ret == -1) {
                perror("bind()");
                goto err1;
        }
```
加上 client_addr.sin6_scope_id = if_nametoindex(ifname);    

##  connect(): Invalid argument

```
      /* Connect to server running on localhost */
        server_addr.sin6_family = AF_INET6;
        //server_addr.sin6_scope_id = SCOPE_LINK;
        inet_pton(AF_INET6, "fe80::4a57:2ff:fe64:e7a7", &server_addr.sin6_addr);
        //inet_pton(AF_INET6, "::1", &server_addr.sin6_addr);
        server_addr.sin6_port = htons(SERVER_PORT);
```
加上        server_addr.sin6_scope_id = if_nametoindex(ifname);    


#  test

```
[root@bogon socket6]# gcc server.c -o server
[root@bogon socket6]# ./server 
New connection from: fe80::4a57:2ff:fe64:e7ae:8788 ...
Connection closed
```

```
root@ubuntu:~/tcpreplay/socket6# ./client 
Received b from server
```