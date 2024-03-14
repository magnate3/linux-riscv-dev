
# skb->tstamp   skb->skb_mstamp_ns  

## 普通ping

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/ping.png)




## telnet

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/telnet.png)


## 普通udp

```
root@SIG-223:~/5.14# iperf3  -s  
iperf3 -u -c 192.168.137.223 -u
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/udp2.png)

## ./udp_tai -c 1 -i enp0s31f6 -P 1000000 -p 99 -d 60000000 -E -u 6868

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/udp.png)

# tcpdump

## client

```
root@ubuntux86:/work/test/tsn/socket# gcc -lpthread -o udp_tai  udp_tai.c 
/usr/bin/ld: /tmp/ccEfnMmQ.o: in function `set_realtime':
udp_tai.c:(.text+0xec7): undefined reference to `pthread_setaffinity_np'
collect2: error: ld returned 1 exit status
root@ubuntux86:/work/test/tsn/socket# gcc  -o udp_tai  udp_tai.c 
/usr/bin/ld: /tmp/ccRQGzNh.o: in function `set_realtime':
udp_tai.c:(.text+0xec7): undefined reference to `pthread_setaffinity_np'
collect2: error: ld returned 1 exit status
root@ubuntux86:/work/test/tsn/socket# gcc  -o udp_tai  udp_tai.c  -lpthread
root@ubuntux86:/work/test/tsn/socket# 
```

```
ip addr add  192.168.137.82/24   broadcast 192.168.137.255 dev enp0s31f6
./udp_tai -c 1 -i enp0s31f6 -P 1000000 -p 99 -d 60000000 -E
./udp_tai -c 1 -i enp0s31f6 -P 1000000 -p 99 -d 60000000 -E -u 6868
```

## 192.168.137.82 to  192.168.137.223

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/client.png)

## 192.168.137.223 to  192.168.137.82

```
tcpdump -i  enp0s31f6 udp and not src host 192.168.137.82 -env
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/server.png)


# SCM_TXTIME

## e1000e driver

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/hw.png)

## client use  use_so_txtime

```
./udp_tai -c 1 -i enp0s31f6 -P 1000000 -p 99 -d 60000000 -E -u 6868
```

```
if (setsockopt(fd, SOL_SOCKET, SO_PRIORITY, &so_priority, sizeof(so_priority))) {
                pr_err("Couldn't set priority");
                goto no_option;
        }
```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/xtime.png)

***注意udp_tai的skb->priority=3***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/xtime2.png)

***注意udp_tai的skb->priority=3***

```
int __sock_cmsg_send(struct sock *sk, struct msghdr *msg, struct cmsghdr *cmsg,
                     struct sockcm_cookie *sockc)
{
        u32 tsflags;

        switch (cmsg->cmsg_type) {
        case SO_MARK:
                if (!ns_capable(sock_net(sk)->user_ns, CAP_NET_ADMIN))
                        return -EPERM;
                if (cmsg->cmsg_len != CMSG_LEN(sizeof(u32)))
                        return -EINVAL;
                sockc->mark = *(u32 *)CMSG_DATA(cmsg);
                break;
        case SO_TIMESTAMPING_OLD:
                if (cmsg->cmsg_len != CMSG_LEN(sizeof(u32)))
                        return -EINVAL;

                tsflags = *(u32 *)CMSG_DATA(cmsg);
                if (tsflags & ~SOF_TIMESTAMPING_TX_RECORD_MASK)
                        return -EINVAL;

                sockc->tsflags &= ~SOF_TIMESTAMPING_TX_RECORD_MASK;
                sockc->tsflags |= tsflags;
                break;
        case SCM_TXTIME:  ////////////////////////
                if (!sock_flag(sk, SOCK_TXTIME))
                        return -EINVAL;
                if (cmsg->cmsg_len != CMSG_LEN(sizeof(u64)))
                        return -EINVAL;
                sockc->transmit_time = get_unaligned((u64 *)CMSG_DATA(cmsg));
                break;
        /* SCM_RIGHTS and SCM_CREDENTIALS are semantically in SOL_UNIX. */
        case SCM_RIGHTS:
        case SCM_CREDENTIALS:
                break;
        default:
                return -EINVAL;
        }
        return 0;
}
```

## client not use  use_so_txtime

```
./udp_tai -c 1 -i enp0s31f6 -P 1000000 -p 99 -d 60000000 -E -u 6868 -s
```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/xtime3.png)

***注意udp_tai的skb->priority=3***


# 网卡多队列



如果网卡及其驱动支持 RSS/多队列，那你可以会调整 RX queue（也叫 RX channel）的数量。这可以用 ethtool 完成

```
root@ubuntux86:/work/test/tsn/socket# ethtool -l  enp0s31f6
Channel parameters for enp0s31f6:
Cannot get device channel parameters
: Operation not supported
root@ubuntux86:/work/test/tsn/socket# 
```

这意味着驱动没有实现 ethtool 的 get_channels 方法。可能的原因包括：该网卡不支持调整 RX queue 数量，不支持 RSS/multiqueue，或者驱动没有更新来支持此功能。


# tc

```
root@ubuntux86:/work/test/tsn/socket#  tc -g qdisc show dev enp0s31f6
qdisc fq_codel 0: root refcnt 2 limit 10240p flows 1024 quantum 1514 target 5.0ms interval 100.0ms memory_limit 32Mb ecn 
root@ubuntux86:/work/test/tsn/socket# tc -g class   show dev enp0s31f6
root@ubuntux86:/work/test/tsn/socket# 
```


```
root@SIG-223:~/5.14# sed -i 's/\r//' taprio.sh 
root@SIG-223:~/5.14# bash taprio.sh eno2
Base time: 1666605603000000000
Configuration saved to: taprio.batch
root@SIG-223:~/5.14# ethtool -l eno2
Channel parameters for eno2:
Cannot get device channel parameters
: Operation not supported
root@SIG-223:~/5.14# ethtool -l eno1
Channel parameters for eno1:
Cannot get device channel parameters
: No such device
root@SIG-223:~/5.14# bash taprio.sh eno1
Cannot find device "eno1"
Command failed taprio.batch:9
Base time: 1666605642000000000
Configuration saved to: taprio.batch
root@SIG-223:~/5.14# 
```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/enet/socket/taprio.png)