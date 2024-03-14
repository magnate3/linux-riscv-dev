
 

# ./raw_probe enahisic2i0  10.11.11.82

```
#if 0
    // add filter
    if (setsockopt(sd, SOL_SOCKET, SO_ATTACH_FILTER, &fcode, sizeof(fcode)) == -1) {
        perror("setsockopt");
        close(sd);
        exit(5);
    }
#endif
```
这段代码不能加

在10.11.11.82上执行
```
root@ubuntu:~/sifive-u74# gcc raw_probe.c -o  raw_probe
root@ubuntu:~/sifive-u74# ./raw_probe enahisic2i0  10.11.11.82
Fake MAC address is 48:57:02:64:e7:ac
Sent ARP reply: 10.11.11.82 is 48:57:02:64:e7:ac
Received ICMP ECHO from 10.11.11.81 (code: 0  id: 58865  seq: 1)
Received ICMP ECHO from 10.11.11.81 (code: 0  id: 58865  seq: 2)
Sent ARP reply: 10.11.11.82 is 48:57:02:64:e7:ac
Sent ARP reply: 10.11.11.82 is 48:57:02:64:e7:ac
```

在10.11.11.81上执行
```
[root@bogon ~]# ping 10.11.11.82
PING 10.11.11.82 (10.11.11.82) 56(84) bytes of data.
64 bytes from 10.11.11.82: icmp_seq=1 ttl=64 time=0.122 ms
64 bytes from 10.11.11.82: icmp_seq=1 ttl=64 time=0.182 ms (DUP!)
64 bytes from 10.11.11.82: icmp_seq=2 ttl=64 time=0.100 ms
64 bytes from 10.11.11.82: icmp_seq=2 ttl=64 time=0.144 ms (DUP!)
64 bytes from 10.11.11.82: icmp_seq=3 ttl=64 time=0.108 ms
64 bytes from 10.11.11.82: icmp_seq=3 ttl=64 time=0.161 ms (DUP!)
```

## riscv 上执行

```
[root@riscv]:~/test$:./raw_probe eth1  192.168.5.79
Fake MAC address[  902.566835][ T1283] device eth1 entered promiscuous mode
 is 9e:5f:32:9a:3a:59
Received ICMP EC[  904.112341][ T1283] device eth1 left promiscuous mode
HO from 192.168.5.82 (code: 0  id: 16  seq: 72)
sendto: No buffer space available
```

# 2_Pcap

```
./pcap  eth1
```
 


 
