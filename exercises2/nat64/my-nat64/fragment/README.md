
# ipv6 frags   

```
[root@centos7 nat64_test]# insmod  nat64_device.ko 
[root@centos7 nat64_test]# ip a add 2001:db8::a0a:6751/96 dev nat64
[root@centos7 nat64_test]# ip l set nat64 up
[root@centos7 nat64_test]# ping6 2001:db8::a0a:6752 -s 2000 -c 1
PING 2001:db8::a0a:6752(2001:db8::a0a:6752) 2000 data bytes
2008 bytes from 2001:db8::a0a:6752: icmp_seq=1 ttl=64 time=14.5 ms

--- 2001:db8::a0a:6752 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 14.547/14.547/14.547/0.000 ms
[root@centos7 nat64_test]# 
```

抓包查看   
```
[root@centos7 ~]#  tcpdump -i nat64 ip6 and "ip6[6]==0x2c and ip6[40]==0x3a" -eennvv
tcpdump: listening on nat64, link-type RAW (Raw IP), capture size 262144 bytes
05:10:26.191912 ip: (flowlabel 0x3cc37, hlim 64, next-header Fragment (44) payload length: 1456) 2001:db8::a0a:6751 > 2001:db8::a0a:6752: frag (0xc9a76d72:0|1448) ICMP6, echo request, seq 1
05:10:26.197305 ip: (hlim 64, next-header Fragment (44) payload length: 1456) 2001:db8::a0a:6752 > 2001:db8::a0a:6751: frag (0xc9a76d72:0|1448) ICMP6, echo reply, seq 1
05:10:26.200791 ip: (flowlabel 0x3cc37, hlim 64, next-header Fragment (44) payload length: 568) 2001:db8::a0a:6751 > 2001:db8::a0a:6752: frag (0xc9a76d72:1448|560)
05:10:26.206431 ip: (hlim 64, next-header Fragment (44) payload length: 568) 2001:db8::a0a:6752 > 2001:db8::a0a:6751: frag (0xc9a76d72:1448|560)
```

日志查看  

```
[root@centos7 nat64_test]# dmesg | tail -n 10
[ 9456.125504] tx_packets=0 tx_bytes=0
[ 9457.112037] ipv6 frag , more 1, offset: 0, id 996911049 
[ 9457.117329] tx_packets=0 tx_bytes=0
[ 9457.120806] ipv6 frag , more 0, offset: 1448, id 996911049 
[ 9457.126357] tx_packets=0 tx_bytes=0
[ 9473.801874] device nat64 entered promiscuous mode
[ 9478.718944] ipv6 frag , more 1, offset: 0, id 1919789001 
[ 9478.724326] tx_packets=0 tx_bytes=0
[ 9478.727812] ipv6 frag , more 0, offset: 1448, id 1919789001 
[ 9478.733450] tx_packets=0 tx_bytes=0
[root@centos7 nat64_test]# 
```


# csum_replace2 csum_replace4

```
		csum_replace2(&iph->check, iph->tot_len, tot_len);
		iph->tot_len = tot_len;
```

```

			addr = iphdr->daddr;
			iphdr->daddr = addr_new;
			csum_replace4(&iphdr->check, addr, addr_new);
```


```
 					/* source IP/Port manipulation */
					addr = iphdr->saddr;
					iphdr->saddr = target->dst.u3.ip;
					csum_replace4(&iphdr->check, addr, target->dst.u3.ip);

					/* ToDo: port translation */
					newport = target->dst.u.tcp.port;
					portptr = &tcph->source;

					oldport = *portptr;
					*portptr = newport;
					csum_replace2(&tcph->check, oldport, newport);
					csum_replace4(&tcph->check, addr, target->src.u3.ip);
```

## IP层对于TTL的修改

对于TTL的修改通常发生在网络层对一个报文进行转发的情况下，也就是典型的在linux-2.6.21\net\ipv4\ip_forward.

```
 
int ip_forward(struct sk_buff *skb)
{
……
/*
 * According to the RFC, we must first decrease the TTL field. If
 * that reaches zero, we must reply an ICMP control message telling
 * that the packet's lifetime expired.
 */
if (skb->nh.iph->ttl <= 1)
goto too_many_hops;
……
/* Decrease ttl after skb cow done */
ip_decrease_ttl(iph);
……
too_many_hops:
/* Tell the sender its packet died... */
IP_INC_STATS_BH(IPSTATS_MIB_INHDRERRORS);
icmp_send(skb, ICMP_TIME_EXCEEDED, ICMP_EXC_TTL, 0);
drop:
kfree_skb(skb);
return NET_RX_DROP;
}
```

这里对于TTL的递减和校验和的更新看起来非常简单，但是乍一看也有点让人费解，这个操作通过ip_decrease_ttl函数完成：


```
/* The function in 2.2 was invalid, producing wrong result for
 * check=0xFEFF. It was noticed by Arthur Skawina _year_ ago. --ANK(000625) */
static inline
int ip_decrease_ttl(struct iphdr *iph)
{
u32 check = (__force u32)iph->check;
check += (__force u32)htons(0x0100);
iph->check = (__force __sum16)(check + (check>=0xFFFF));
return --iph->ttl;
}
```

这里从整体上来看就比较简单了。对于这里的操作动作其实非常明确，就是在Ipheader中的ttl字段中递减，这个递减是通过在函数的最后--iph->ttl完成。为了保证校验和同时不变，就需要对最终的校验和执行一个和这个操作相反的操作，也就是递增。从之前的Ip header的定义可以看到，作为一个16bits word，它位于|Time to Live |Protocol|这个word中，所以它的递增操作是加上一个0x0100而不是直接加上一个0x0001。      

##  TCP层在NAT中可能对校验和的修改

正如前面所说的，这个场景主要发生在NAT这样的场景中，这个也是我最早感受到校验和的存在，之后在使用traceroute的时候再次遇到校验和。以TCP协议中端口修改的场景为例，在函数linux-2.6.21\net\ipv4\netfilter\ip_nat_proto_tcp.c:


```
static int
tcp_manip_pkt(struct sk_buff **pskb,
      unsigned int iphdroff,
      const struct ip_conntrack_tuple *tuple,
      enum ip_nat_manip_type maniptype)
{
……
nf_proto_csum_replace4(&hdr->check, *pskb, oldip, newip, 1);
nf_proto_csum_replace2(&hdr->check, *pskb, oldport, newport, 0);
return 1;
}
```

以对于4字节字段的修改为例，可以明显的看到，对于这个地方的实现依然间接，但是并没有TTL操作那么飘逸，相对来说比较直观古朴。

```
linux-2.6.21\net\netfilter\core.c
void nf_proto_csum_replace4(__sum16 *sum, struct sk_buff *skb,
    __be32 from, __be32 to, int pseudohdr)
{
__be32 diff[] = { ~from, to };
if (skb->ip_summed != CHECKSUM_PARTIAL) {
*sum = csum_fold(csum_partial((char *)diff, sizeof(diff),
~csum_unfold(*sum)));
if (skb->ip_summed == CHECKSUM_COMPLETE && pseudohdr)
skb->csum = ~csum_partial((char *)diff, sizeof(diff),
~skb->csum);
} else if (pseudohdr)
*sum = ~csum_fold(csum_partial((char *)diff, sizeof(diff),
csum_unfold(*sum)));
}
```

这里加上~from，所以之前的from + ~from = 0xFFFF，由于前面说过，任何一个非零数加上0xFFFF都等于该数本身，所以加上~from相当于把这个值首先从校验和中清除掉，然后加上to就得到了修正后的数据。至于为什么说之前的数据一定非零呢？因为IP header中的version肯定非零(而且TTL正常情况下也应该大于等于1)。   