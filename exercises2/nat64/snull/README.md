

#  snull

[kernel-in-action](https://github.com/keithnoguchi/kernel-in-action)


```
root@ubuntux86:# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:# 
```

```
root@ubuntux86:# ip l set up dev sn0
root@ubuntux86:# ip l set up dev sn1
root@ubuntux86:# ip a add 1.1.0.1/24 dev sn0
root@ubuntux86:# ip a add 1.1.1.2/24 dev sn1
root@ubuntux86:# ip r | grep 1.1
1.1.0.0/24 dev sn0 proto kernel scope link src 1.1.0.1 
1.1.1.0/24 dev sn1 proto kernel scope link src 1.1.1.2 
10.11.11.0/24 dev enx00e04c3662aa proto kernel scope link src 10.11.11.82 metric 100 
10.11.12.0/24 dev enp0s31f6 proto kernel scope link src 10.11.12.82 metric 101 
192.168.5.0/24 dev enp0s31f6 proto kernel scope link src 192.168.5.82 metric 101 
```

```
root@ubuntux86:# ping 1.1.0.2 -c 1
PING 1.1.0.2 (1.1.0.2) 56(84) bytes of data.
64 bytes from 1.1.0.2: icmp_seq=1 ttl=64 time=0.107 ms

--- 1.1.0.2 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.107/0.107/0.107/0.000 ms
root@ubuntux86:# ip  a sh sn0
11: sn0: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 00:53:4e:55:4c:30 brd ff:ff:ff:ff:ff:ff
    inet 1.1.0.1/24 scope global sn0
       valid_lft forever preferred_lft forever
    inet6 fe80::6ad6:ea43:c599:d220/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
root@ubuntux86:# ip  a sh sn1
12: sn1: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 00:53:4e:55:4c:31 brd ff:ff:ff:ff:ff:ff
    inet6 fe80::34a0:95be:a52b:e959/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
root@ubuntux86:# 
```

#  header_ops->create


```
static inline int dev_hard_header(struct sk_buff *skb, struct net_device *dev,
                                  unsigned short type,
                                  const void *daddr, const void *saddr,
                                  unsigned int len)
{
        if (!dev->header_ops || !dev->header_ops->create)
                return 0;

        return dev->header_ops->create(skb, dev, type, daddr, saddr, len);
}


int neigh_connected_output(struct neighbour *neigh, struct sk_buff *skb)
{
        struct net_device *dev = neigh->dev;
        unsigned int seq;
        int err;

        do {
                __skb_pull(skb, skb_network_offset(skb));
                seq = read_seqbegin(&neigh->ha_lock);
                err = dev_hard_header(skb, dev, ntohs(skb->protocol),
                                      neigh->ha, NULL, skb->len);
        } while (read_seqretry(&neigh->ha_lock, seq));

        if (err >= 0)
                err = dev_queue_xmit(skb);
        else {
                err = -EINVAL;
                kfree_skb(skb);
        }
        return err;
}
```


```
[ 1287.841211]  neigh_connected_output+0xac/0xf0
[ 1287.841221]  ip6_finish_output2+0x1e8/0x6e0
[ 1287.841231]  ? netif_rx_ni+0xa0/0xa0
[ 1287.841241]  __ip6_finish_output+0xe6/0x2a0
[ 1287.841249]  ip6_finish_output+0x2d/0xb0
[ 1287.841257]  ip6_output+0x77/0x130
[ 1287.841265]  ? __ip6_finish_output+0x2a0/0x2a0
[ 1287.841272]  ip6_local_out+0x45/0x70
[ 1287.841281]  ip6_send_skb+0x23/0x70
[ 1287.841289]  udp_v6_send_skb.isra.0+0x2aa/0x4a0
[ 1287.841296]  udpv6_sendmsg+0xb66/0xd90
[ 1287.841303]  ? ip_reply_glue_bits+0x50/0x50
[ 1287.841315]  inet6_sendmsg+0x65/0x70
[ 1287.841321]  ? inet6_sendmsg+0x65/0x70
[ 1287.841327]  sock_sendmsg+0x48/0x70
[ 1287.841333]  ____sys_sendmsg+0x218/0x290
[ 1287.841338]  ? copy_msghdr_from_user+0x5c/0x90
[ 1287.841345]  ___sys_sendmsg+0x81/0xc0
[ 1287.841351]  ? pipe_write+0xfd/0x620
[ 1287.841364]  ? new_sync_write+0x192/0x1b0
[ 1287.841374]  __sys_sendmsg+0x62/0xb0
[ 1287.841381]  __x64_sys_sendmsg+0x1f/0x30
[ 1287.841387]  do_syscall_64+0x61/0xb0
[ 1287.841393]  ? syscall_exit_to_user_mode+0x27/0x50
[ 1287.841400]  ? do_syscall_64+0x6e/0xb0
[ 1287.841404]  ? syscall_exit_to_user_mode+0x27/0x50
[ 1287.841410]  ? __x64_sys_write+0x1a/0x20
[ 1287.841415]  ? do_syscall_64+0x6e/0xb0
[ 1287.841419]  ? asm_sysvec_apic_timer_interrupt+0xa/0x20
[ 1287.841430]  entry_SYSCALL_64_after_hwframe+0x44/0xae
[ 1287.841439] RIP: 0033:0x7f005ddb5157
```