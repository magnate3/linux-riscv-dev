

# server （dpdk）
mac   
```
44:a1:91:a4:9c:0b
```

```
./build/helloworld  -c0x1
src: 10.10.103.81:49727, dst: 10.10.103.251:2000, client message: hello world 
```

## udp 校验和计算

```
    udphdr->dgram_cksum = 0;
    udphdr->dgram_cksum = rte_ipv4_udptcp_cksum(iphdr, udphdr);
```

## udp 校验和计算二（网卡不支持udp cksum offload）

```
[root@bogon ~]# tcpdump -i enahisic2i3 udp and host 10.10.103.251 -env
tcpdump: listening on enahisic2i3, link-type EN10MB (Ethernet), capture size 262144 bytes
16:15:32.891406 48:57:02:64:ea:1e > 44:a1:91:a4:9c:0b, ethertype IPv4 (0x0800), length 70: (tos 0x0, ttl 64, id 42233, offset 0, flags [DF], proto UDP (17), length 56)
    10.10.103.81.60125 > 10.10.103.251.sieve-filter: UDP, length 28
16:15:32.891460 44:a1:91:a4:9c:0b > 48:57:02:64:ea:1e, ethertype IPv4 (0x0800), length 62: truncated-ip - 10 bytes missing! (tos 0x0, ttl 64, id 0, offset 0, flags [none], proto UDP (17), length 58)
    10.10.103.251.sieve-filter > 10.10.103.81.60125: UDP, length 30

```

这是因为mbuf->l4_len设置错了    

```
   //mbuf->l4_len = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);
        mbuf->l4_len = len + sizeof(struct rte_udp_hdr);
```


网卡offload udp校验和有问题    
```
    iphdr->hdr_checksum = 0;
    udphdr->dgram_cksum = 0;

        /* Must be set to offload checksum. */
        mbuf->l2_len = sizeof(struct rte_ether_hdr);
        mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
        //mbuf->l4_len = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);
        mbuf->l4_len =  sizeof(struct rte_udp_hdr);
        mbuf->pkt_len = mbuf->l2_len +  mbuf->l3_len  + mbuf->l4_len + len ;
        mbuf->data_len = mbuf->pkt_len;

        /* Enable IPV4 CHECKSUM OFFLOAD */
        mbuf->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM);

        /* Enable UDP TX CHECKSUM OFFLOAD */
        mbuf->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_UDP_CKSUM);
```

# client

配置neigh   
```
[root@bogon tcpreplay]# ip n del  10.10.103.251 dev enahisic2i3  lladdr 44:a1:91:a4:9c:0b
[root@bogon tcpreplay]# ip n add  10.10.103.251 dev enahisic2i3  lladdr 44:a1:91:a4:9c:0b
```


```
[root@bogon tcpreplay]# ./udp_cli2
Socket created successfully
Server's response: client message: hello world 
[root@bogon tcpreplay]# 
```

# references

[udp_generator](https://github.com/carvalhof/udp_generator/blob/master/main.c)