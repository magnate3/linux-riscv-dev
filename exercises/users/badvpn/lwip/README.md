
# debug
```
./lwip/src/include/lwip/debug.h
#define LWIP_DEBUG
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/badvpn/lwip/debug.png)

```
[root@centos7 badvpn]# grep  LWIP_DBG_MIN_LEVEL  -rn *
lwip/src/include/lwip/opt.h:2155: * LWIP_DBG_MIN_LEVEL: After masking, the value of the debug is
lwip/src/include/lwip/opt.h:2159:#ifndef LWIP_DBG_MIN_LEVEL
lwip/src/include/lwip/opt.h:2160:#define LWIP_DBG_MIN_LEVEL              LWIP_DBG_LEVEL_ALL
lwip/src/include/lwip/debug.h:87:                                   ((s16_t)((debug) & LWIP_DBG_MASK_LEVEL) >= LWIP_DBG_MIN_LEVEL)) { \
[root@centos7 badvpn]# 
```

## lwip/src/include/lwip/opt.h
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/badvpn/lwip/on.png)

```

/**
 * NETIF_DEBUG: Enable debugging in netif.c.
 */
#ifndef NETIF_DEBUG
#define NETIF_DEBUG                     LWIP_DBG_ON
//#define NETIF_DEBUG                     LWIP_DBG_OFF
#endif
```

# telnet 8.8.8.8 22

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/badvpn/lwip/tcp.png)

```
[root@centos7 tun2socks]# telnet 8.8.8.8 22
Trying 8.8.8.8...
Connected to 8.8.8.8.
Escape character is '^]'.
Connection closed by foreign host.
[root@centos7 tun2socks]# telnet 8.8.8.8 22
Trying 8.8.8.8...
Connected to 8.8.8.8.
Escape character is '^]'.
Connection closed by foreign host.
[root@centos7 tun2socks]# 
```

# ipv4_header

```

            BAddr dns_addr;
            char buf[28];
            BAddr_InitIPv4(&dns_addr, ipv4_header.destination_address, udp_header.dest_port);
            BAddr_Print(&dns_addr, buf);
            printf("@@@@@@@@@@ ipv4 ip and udp port addr %s ", buf);
            BAddr_InitIPv4(&dns_addr, netif_ipaddr.ipv4, hton16(53));
            BAddr_Print(&dns_addr, buf);
            printf(", netif ip and udp port addr %s ", buf);
            printf(", is_dns %d \n", is_dns);
```

# addr

```
    client->local_addr = baddr_from_lwip(PCB_ISIPV6(newpcb), &newpcb->local_ip, newpcb->local_port);
    client->remote_addr = baddr_from_lwip(PCB_ISIPV6(newpcb), &newpcb->remote_ip, newpcb->remote_port);
    char buf[28];
    BAddr_Print(&client->local_addr, buf);
    printf("********* local addr %s ", buf);
    BAddr_Print(&client->remote_addr, buf);
    printf("remote addr %s ********** \n", buf);
	
	 
```
## BAddr addr
```
 // construct addresses
            BAddr_InitIPv4(&local_addr, ipv4_header.source_address, udp_header.source_port);
            BAddr_InitIPv4(&remote_addr, ipv4_header.destination_address, udp_header.dest_port);
static void addr_socket_to_sys (struct sys_addr *out, BAddr addr)
{
    switch (addr.type) {
        case BADDR_TYPE_IPV4: {
            out->len = sizeof(out->addr.ipv4);
            memset(&out->addr.ipv4, 0, sizeof(out->addr.ipv4));
            out->addr.ipv4.sin_family = AF_INET;
            out->addr.ipv4.sin_port = addr.ipv4.port;
            out->addr.ipv4.sin_addr.s_addr = addr.ipv4.ip;
        } break;
```

```
struct in_pktinfo {
  unsigned int   ipi_ifindex;  /* Interface index */
  struct in_addr ipi_addr;     /* Destination (from header) address */
};
```

# BIPAddr

```
typedef struct {
    int type;
    union {
        uint32_t ipv4;
        uint8_t ipv6[16];
    };
} BIPAddr;

void BIPAddr_InitIPv4 (BIPAddr *addr, uint32_t ip)
void BIPAddr_Print (BIPAddr *addr, char *out)
```


# udpgw_addr

```
struct udpgw_addr_ipv4 addr_ipv4;
            addr_ipv4.addr_ip = con->conaddr.remote_addr.ipv4.ip;
            addr_ipv4.addr_port = con->conaddr.remote_addr.ipv4.port;
        BAddr remote_addr2;
        BAddr_InitIPv4(&remote_addr2, addr_ipv4.addr_ip, addr_ipv4.addr_port);
        char buf[28];
        BAddr_Print(&remote_addr2, buf);
        printf("****** %s  send udpgw header , remote addr =  %s: %d\n", __func__, buf, ntohs(addr_ipv4.addr_port));
```

# BConnection

```
BConnection->fd 对应一个socket fd

```

# printf
```
 
struct sockaddr_in *saddr;

saddr->sin_port;
saddr->sin_addr.s_addr;

struct sys_addr sysaddr;
    addr_socket_to_sys(&sysaddr, o->send.remote_addr);
    printf("****** remote addr = %d, %s: %d\n", sysaddr.addr.ipv4.sin_family, inet_ntoa(sysaddr.addr.ipv4.sin_addr), ntohs(sysaddr.addr.ipv4.sin_port));
```

# tcp_debug_print
```
void tcp_debug_print(struct tcp_hdr *tcphdr)
{
  LWIP_DEBUGF(TCP_DEBUG, ("TCP header:\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("|    %5"U16_F"      |    %5"U16_F"      | (src port, dest port)\n",
                          lwip_ntohs(tcphdr->src), lwip_ntohs(tcphdr->dest)));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("|           %010"U32_F"          | (seq no)\n",
                          lwip_ntohl(tcphdr->seqno)));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("|           %010"U32_F"          | (ack no)\n",
                          lwip_ntohl(tcphdr->ackno)));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("| %2"U16_F" |   |%"U16_F"%"U16_F"%"U16_F"%"U16_F"%"U16_F"%"U16_F"|     %5"U16_F"     | (hdrlen, flags (",
                          TCPH_HDRLEN(tcphdr),
                          (u16_t)(TCPH_FLAGS(tcphdr) >> 5 & 1),
                          (u16_t)(TCPH_FLAGS(tcphdr) >> 4 & 1),
                          (u16_t)(TCPH_FLAGS(tcphdr) >> 3 & 1),
                          (u16_t)(TCPH_FLAGS(tcphdr) >> 2 & 1),
                          (u16_t)(TCPH_FLAGS(tcphdr) >> 1 & 1),
                          (u16_t)(TCPH_FLAGS(tcphdr)      & 1),
                          lwip_ntohs(tcphdr->wnd)));
  tcp_debug_print_flags(TCPH_FLAGS(tcphdr));
  LWIP_DEBUGF(TCP_DEBUG, ("), win)\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(TCP_DEBUG, ("|    0x%04"X16_F"     |     %5"U16_F"     | (chksum, urgp)\n",
                          lwip_ntohs(tcphdr->chksum), lwip_ntohs(tcphdr->urgp)));
  LWIP_DEBUGF(TCP_DEBUG, ("+-------------------------------+\n"));
}
```

```
 */
void
ip4_debug_print(struct pbuf *p)
{
  struct ip_hdr *iphdr = (struct ip_hdr *)p->payload;

  LWIP_DEBUGF(IP_DEBUG, ("IP header:\n"));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(IP_DEBUG, ("|%2"S16_F" |%2"S16_F" |  0x%02"X16_F" |     %5"U16_F"     | (v, hl, tos, len)\n",
                         (u16_t)IPH_V(iphdr),
                         (u16_t)IPH_HL(iphdr),
                         (u16_t)IPH_TOS(iphdr),
                         lwip_ntohs(IPH_LEN(iphdr))));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(IP_DEBUG, ("|    %5"U16_F"      |%"U16_F"%"U16_F"%"U16_F"|    %4"U16_F"   | (id, flags, offset)\n",
                         lwip_ntohs(IPH_ID(iphdr)),
                         (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 15 & 1),
                         (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 14 & 1),
                         (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 13 & 1),
                         (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) & IP_OFFMASK)));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(IP_DEBUG, ("|  %3"U16_F"  |  %3"U16_F"  |    0x%04"X16_F"     | (ttl, proto, chksum)\n",
                         (u16_t)IPH_TTL(iphdr),
                         (u16_t)IPH_PROTO(iphdr),
                         lwip_ntohs(IPH_CHKSUM(iphdr))));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(IP_DEBUG, ("|  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  | (src)\n",
                         ip4_addr1_16(&iphdr->src),
                         ip4_addr2_16(&iphdr->src),
                         ip4_addr3_16(&iphdr->src),
                         ip4_addr4_16(&iphdr->src)));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
  LWIP_DEBUGF(IP_DEBUG, ("|  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  | (dest)\n",
                         ip4_addr1_16(&iphdr->dest),
                         ip4_addr2_16(&iphdr->dest),
                         ip4_addr3_16(&iphdr->dest),
                         ip4_addr4_16(&iphdr->dest)));
  LWIP_DEBUGF(IP_DEBUG, ("+-------------------------------+\n"));
}
```

```
case IP_PROTO_TCP:
      snmp_inc_ipindelivers();
      struct tcp_hdr *tcphdr = (struct tcp_hdr *)((u8_t *)iphdr + iphdr_hlen);
      //printf("************** src port %d and dst  port %d ", ntohs(tcphdr->src),  ntohs(tcphdr->dest));
      printf("############## src ip %d.%d.%d.%d,", ip4_addr1_16(&iphdr->src), ip4_addr2_16(&iphdr->src), ip4_addr3_16(&iphdr->src), ip4_addr4_16(&iphdr->src));
      printf(" src port %d",ntohs(tcphdr->src));
      printf(" dest ip %d.%d.%d.%d,", ip4_addr1_16(&iphdr->dest), ip4_addr2_16(&iphdr->dest), ip4_addr3_16(&iphdr->dest), ip4_addr4_16(&iphdr->dest));
      printf(" dest port %d",ntohs(tcphdr->dest));
      printf("\n");
      //printf("|  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  |  %3"U16_F"  | dst\n", ip4_addr1_16(&iphdr->dest), ip4_addr2_16(&iphdr->dest), ip4_addr3_16(&iphdr->dest), ip4_addr4_16(&iphdr->dest));
      //char buf[32]; 
      //memset(buf,0, sizeof(buf));
      //linux_src_ip.sin_addr.s_addr = ip4_addr_get_u32(iphdr->src);    
      //linux_dst_ip.sin_addr.s_addr = ip4_addr_get_u32(iphdr->dest);    
      //printf("************** src %s: %d and dst  %s: %d \n", inet_ntoa(linux_src_ip), ntohs(tcphdr->src), inet_ntoa(linux_dst_ip), ntohs(tcphdr->dest));
      tcp_input(p, inp);
      break;
```


