/*
 * Driver for NAT64 Virtual Network Interface.
 *
 * Author: Jianying Liu <rssnsj@gmail.com>
 * Date: 2014/01/21
 *
 * This source code is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 */

#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/etherdevice.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/netdevice.h>
#include <linux/if.h>
#include <linux/if_ether.h>
#include <linux/if_arp.h>
#include <net/sock.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <linux/icmpv6.h>
#include <linux/netdev_features.h>
#include <linux/udp.h>
#include <net/ip6_checksum.h>

#include "tayga.h"
//#include "lib_checksum.h"

#define  CONFIG_IP_VS_IPV6 1
struct nat64_if_info {
	struct net_device *dev;
};

static int virt_rs_packet(struct sk_buff *skb, struct net_device *dev)
{

	unsigned char *type;
	struct iphdr *ih;
	__be32 *saddr, *daddr, tmp;
	struct sk_buff *rx_skb;
	int ret;
        int len;
#if 0
	unsigned char tmp_dev_addr[ETH_ALEN];
	struct ethhdr *ethhdr;
	len = skb->len;
	if (len < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return -1;
	}
	//对调ethhdr结构体 "源/目的"MAC地址*/
	ethhdr = (struct ethhdr *)skb->data;
        if(0x0800 != ntohs(ethhdr->h_proto)){
		pr_info("not ip hdr \n");
		return -1;
	}
	memcpy(tmp_dev_addr, ethhdr->h_dest, ETH_ALEN);
	memcpy(ethhdr->h_dest, ethhdr->h_source, ETH_ALEN);
	memcpy(ethhdr->h_source, tmp_dev_addr, ETH_ALEN);
	//对调iphdr结构体"源/目的" IP地址
	ih = (struct iphdr *)(skb->data + sizeof(struct ethhdr));
#else
	len = skb->len;
	if (len < sizeof(struct icmphdr) + sizeof(struct iphdr)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return -1;
	}
        if(ETH_P_IP != ntohs(skb->protocol))
        {
		pr_info("not ip hdr \n");
		return -1;
        }
	ih = (struct iphdr *)(skb->data);
#endif
        if(IPPROTO_ICMP != ih->protocol)
        {
		pr_info("not icmp hdr \n");
		return -1;
	}
	saddr = &ih->saddr;
	daddr = &ih->daddr;
	tmp = *saddr;
	*saddr = *daddr;
	*daddr = tmp;
        
        ih->check=0;
        ih->check = ip_fast_csum((unsigned char *)ih,ih->ihl);
	
	//之前是发送ping包0x08,需要改为0x00,表示接收ping包
	type = skb->data + sizeof(struct ethhdr) + sizeof(struct iphdr);
	*type = 0; 
 
	rx_skb = dev_alloc_skb(skb->len + 2);
	skb_reserve(rx_skb, 2);
	
	memcpy(skb_put(rx_skb, skb->len), skb->data, skb->len);
	rx_skb->dev = dev;	
	rx_skb->ip_summed = CHECKSUM_UNNECESSARY;
	//rx_skb->protocol = eth_type_trans(rx_skb, dev);
        rx_skb->protocol = htons(ETH_P_IP);
	ret=netif_rx(rx_skb);
	
	dev->stats.rx_packets++;        
	dev->stats.rx_bytes += skb->len;
	pr_info("rx_packets=%ld rx_bytes=%ld ret=%d\n",dev->stats.rx_packets,dev->stats.rx_bytes,ret);
	return NETDEV_TX_OK;
}
#if 0
static void host_handle_icmp6(struct pkt *p)
{
}
#endif
static inline u16 swap_u16(u16 val)
{
	return (val << 8) | (val >> 8);
}

static inline u32 swap_u32(u32 val)
{
	val = ((val << 8) & 0xff00ff00) | ((val >> 8) & 0xff00ff);
	return (val << 16) | (val >> 16);
}

static u16 ip_checksum(void *d, int c)
{
	u32 sum = 0xffff;
	u16 *p = d;

	while (c > 1) {
		sum += swap_u16(ntohs(*p++));
		c -= 2;
	}

	if (c)
		sum += swap_u16(*((u8 *)p) << 8);

	while (sum > 0xffff)
		sum = (sum & 0xffff) + (sum >> 16);

	return ~sum;
}

static inline u16 ones_add(u16 a, u16 b)
{
	u32 sum = (u16)~a + (u16)~b;

	return ~((sum & 0xffff) + (sum >> 16));
}

static u16 ip6_checksum(struct ip6 *ip6, u32 data_len, u8 proto)
{
	u32 sum = 0;
	u16 *p;
	int i;

	for (i = 0, p = ip6->src.s6_addr16; i < 16; ++i)
		sum += swap_u16(ntohs(*p++));
	sum += swap_u32(data_len) >> 16;
	sum += swap_u32(data_len) & 0xffff;
	sum += swap_u16(proto);

	while (sum > 0xffff)
		sum = (sum & 0xffff) + (sum >> 16);

	return ~sum;
}
static void host_send_frag(u8 tc, struct in6_addr *src,
		struct in6_addr *dest,u8 next_header, 
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
                //struct ip6_frag ip6_frag;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb;

	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(data_len);
	//header.ip6.payload_length = htons(data_len + sizeof(struct ip6_frag));
	header.ip6.next_header = next_header;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;
        //header.ip6_frag = *ip6_frag;
	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
	skb->protocol = htons(ETH_P_IPV6);
	
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
static void host_send_icmp6(u8 tc, struct in6_addr *src,
		struct in6_addr *dest, struct icmp *icmp,
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
		struct icmp icmp;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb;

	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(sizeof(header.icmp) + data_len);
	header.ip6.next_header = 58;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;
	header.icmp = *icmp;
	header.icmp.cksum = 0;
	header.icmp.cksum = htons(swap_u16(ones_add(ip_checksum(data, data_len),
			ip_checksum(&header.icmp, sizeof(header.icmp)))));
	header.icmp.cksum = htons(swap_u16(ones_add(swap_u16(ntohs(header.icmp.cksum)),
			ip6_checksum(&header.ip6, data_len + sizeof(header.icmp), 58))));

	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
	skb->protocol = htons(ETH_P_IPV6);
	
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
#if 0
static inline __sum16 csum16_add(__sum16 csum, __be16 addend)
{
        uint16_t res = (uint16_t)csum;

        res += (__u16)addend;
        return (__sum16)(res + (res < (__u16)addend));
}

static inline __sum16 csum16_sub(__sum16 csum, __be16 addend)
{
        return csum16_add(csum, ~addend);
}

static inline void csum_replace2(__sum16 *sum, __be16 old, __be16 new)
{
        *sum = ~csum16_add(csum16_sub(~(*sum), old), new);
}
#endif
static inline __wsum ip_vs_check_diff16(const __be32 *old, const __be32 *new,
					__wsum oldsum)
{
	__be32 diff[8] = { ~old[3], ~old[2], ~old[1], ~old[0],
			    new[3],  new[2],  new[1],  new[0] };

	return csum_partial(diff, sizeof(diff), oldsum);
}
static inline __wsum ip_vs_check_diff2(__be16 old, __be16 new, __wsum oldsum)
{
	__be16 diff[2] = { ~old, new };

	return csum_partial(diff, sizeof(diff), oldsum);
}
static inline __wsum ip_vs_check_diff4(__be32 old, __be32 new, __wsum oldsum)
{
	__be32 diff[2] = { ~old, new };

	return csum_partial(diff, sizeof(diff), oldsum);
}
#if 0
static inline void
udp_fast_csum_update(int af, struct udphdr *uhdr,
		     const union nf_inet_addr *oldip,
		     const union nf_inet_addr *newip,
		     __be16 oldport, __be16 newport)
{
#ifdef CONFIG_IP_VS_IPV6
	if (af == AF_INET6)
		uhdr->check =
			csum_fold(ip_vs_check_diff16(oldip->ip6, newip->ip6,
					 ip_vs_check_diff2(oldport, newport,
						~csum_unfold(uhdr->check))));
	else
#endif
		uhdr->check =
			csum_fold(ip_vs_check_diff4(oldip->ip, newip->ip,
					 ip_vs_check_diff2(oldport, newport,
						~csum_unfold(uhdr->check))));
	if (!uhdr->check)
		uhdr->check = CSUM_MANGLED_0;
}
#else
static inline void
udp_fast_csum_update(struct udphdr *uhdr,
		     __be32 * oldip,
		     __be32 * newip,
		     __be16 oldport, __be16 newport)
{
	uhdr->check = csum_fold(ip_vs_check_diff16(oldip, newip,
					 ip_vs_check_diff2(oldport, newport,
						~csum_unfold(uhdr->check))));
	if (!uhdr->check)
		uhdr->check = CSUM_MANGLED_0;
}
#endif
static void host_send_icmp6_first_frag(u8 tc, struct in6_addr *src,
		struct in6_addr *dest, u8 next_header,struct icmp *icmp,struct ip6_frag *ip6_frag,
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
                struct ip6_frag ip6_frag;
		struct icmp icmp;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb;

	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(sizeof(header.icmp) + sizeof(header.ip6_frag)  + data_len);
	header.ip6.next_header = next_header;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;
	header.ip6_frag = *ip6_frag;
	header.icmp = *icmp;
#if 0
	header.icmp.cksum = 0;
	header.icmp.cksum = htons(swap_u16(ones_add(ip_checksum(data, data_len),
			ip_checksum(&header.icmp, sizeof(header.icmp)))));
	header.icmp.cksum = htons(swap_u16(ones_add(swap_u16(ntohs(header.icmp.cksum)),
			ip6_checksum(&header.ip6, data_len + sizeof(header.icmp), next_header))));
#else
        
         csum_replace2(&header.icmp.cksum,
                              htons(ICMPV6_ECHO_REQUEST << 8),
                              htons(ICMPV6_ECHO_REPLY << 8));
#endif
	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
	skb->protocol = htons(ETH_P_IPV6);
	
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
static void host_send_udp_over_ip6(u8 tc, struct in6_addr *src,
		struct in6_addr *dest, struct udphdr*udphdr,
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
		struct  udphdr udphdr;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb;
        //struct udphdr *udp_hdr;
        unsigned int udpoff = sizeof(struct ip6);
        //unsigned int udpoff = sizeof(header);
	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(sizeof(header.udphdr) + data_len);
	header.ip6.next_header = IPPROTO_UDP;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;
	header.udphdr = *udphdr;
        header.udphdr.check = 0;
	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
#if 1
        skb_set_transport_header(skb,udpoff);
        skb->csum = skb_checksum(skb, udpoff, skb->len - udpoff, 0);
        udp_hdr(skb)->check = csum_ipv6_magic(src, dest, skb->len - udpoff, IPPROTO_UDP, skb->csum);
        pr_info("udp check sum %x, skb-len %u \n",header.udphdr.check,skb->len);
        if (0 == header.udphdr.check)
			header.udphdr.check = CSUM_MANGLED_0;
#else
         // udphdr.len = udp header + udp data
        udp_len = ntohs(header.udphdr.len);                               
        skb->csum = csum_partial(skb->data + sizeof(struct ip6), udp_len, 0);
        header.udphdr.check= csum_ipv6_magic(src, dest, udp_len, IPPROTO_UDP, skb->csum);
#endif
	skb->protocol = htons(ETH_P_IPV6);
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
static void host_send_udp_first_frag_over_ip6(u8 tc, struct in6_addr *src,
		struct in6_addr *dest, u8 next_header,struct udphdr *udp,struct ip6_frag *ip6_frag,
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
                struct ip6_frag ip6_frag;
		struct  udphdr udphdr;
	} __attribute__ ((__packed__)) header;
	struct sk_buff * skb;
	struct  udphdr * udphdr;
        __be16 port =0;
        unsigned int udpoff = sizeof(struct ip6) + sizeof(struct ip6_frag);

	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(sizeof(header.udphdr) + sizeof(header.ip6_frag)  + data_len);
	header.ip6.next_header = next_header;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;

	header.ip6_frag = *ip6_frag;
	header.udphdr = *udp;
	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
        skb_set_transport_header(skb,udpoff);
#if 0
        skb->csum = skb_checksum(skb, udpoff, skb->len - udpoff, 0);
        udp_hdr(skb)->check = csum_ipv6_magic(src, dest, skb->len - udpoff, IPPROTO_UDP, skb->csum);
#else
        udphdr = udp_hdr(skb);
        //old five tuple <dest,  udphdr->dest, udp  -->  src, udphdr->source>
        // new five tuple  <src ,  udphdr->source, udp  --> dest,  udphdr->dest>
        //snat
        port = udphdr->dest;
        udphdr->dest = udphdr->source; 
        udp_fast_csum_update(udphdr,(__be32 *)dest,(__be32 *)src,udphdr->dest, udphdr->source);
        //dnat
        udphdr->source = port; 
        udp_fast_csum_update(udphdr,(__be32 *)src,(__be32 *)dest, udphdr->source,port);
#endif
	skb->protocol = htons(ETH_P_IPV6);
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
void handle_ip6(struct sk_buff *skb)
{
     struct ip6 *ip6;
     struct icmp *icmp;
     int len;
     u8 *data;
     u8  more;
     u16 offset  ;
     u8 data_proto;
     struct ip6_frag *ip6_frag;
     bool first_frag = false;
     len = skb->len;
     if (len < sizeof(struct icmp) + sizeof(struct ip6)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return ;
      }
      if(ETH_P_IPV6 != ntohs(skb->protocol))
      {
		pr_info("not ip6 hdr \n");
		goto err1;
      }
      ip6 = (struct ip6 *)(skb->data);
      data_proto = ip6->next_header;
      if(IPPROTO_FRAGMENT == data_proto)
      {
          ip6_frag = (struct ip6_frag *)(ip6 +1);
          more = ntohs(ip6_frag->offset_flags) & IP6_F_MF;
          offset = ntohs(ip6_frag->offset_flags) & IP6_F_MASK;
          pr_info("ipv6 frag , more %u, offset: %u, id %u \n",more,offset,ip6_frag->ident);      
          data_proto = ip6_frag->next_header;
          if(IPPROTO_ICMP6 == data_proto)
          {
              if(more && 0 == offset)
              {
                  first_frag = true;
                  icmp = (struct icmp *)(ip6_frag + 1); 
                  data =  skb->data + sizeof(struct ip6) + sizeof(struct ip6_frag) +  sizeof(struct icmp);
                  len  = len - sizeof(struct ip6) - sizeof(struct icmp) - sizeof(struct ip6_frag);
              }
              else
              {
                  icmp = NULL; 
                  data =  skb->data + sizeof(struct ip6);
                  len  = len - sizeof(struct ip6);
                  host_send_frag((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,&ip6->dest, &ip6->src,IPPROTO_FRAGMENT,data,len,skb->dev);
                  goto out; 
              }
          }
          else if(IPPROTO_UDP == data_proto)
          {
              if(more && 0 == offset)
              {
                  struct udphdr *udp_hdr = (struct udphdr*)(ip6_frag + 1); 
                  data =  skb->data + sizeof(struct ip6)  + sizeof(struct ip6_frag) + sizeof(struct udphdr);
                  len  = len - sizeof(struct ip6) - sizeof(struct ip6_frag)- sizeof(struct udphdr);
	          host_send_udp_first_frag_over_ip6((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,
				&ip6->dest, &ip6->src,IPPROTO_FRAGMENT,
				udp_hdr, ip6_frag,data,len , skb->dev);
              }
              else
              {
                  data =  skb->data + sizeof(struct ip6);
                  len  = len - sizeof(struct ip6);
                  host_send_frag((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,&ip6->dest, &ip6->src,IPPROTO_FRAGMENT,data,len,skb->dev);
              }
              goto out;
          }
      }
      else if(IPPROTO_ICMP6 == data_proto)
      {
          icmp = (struct icmp *)(ip6 + 1); 
          data =  skb->data + sizeof(struct ip6) + sizeof(struct icmp);
          len  = len - sizeof(struct ip6) - sizeof(struct icmp);
      }
      else if(IPPROTO_UDP == data_proto)
      {
          struct udphdr *udp_hdr = (struct udphdr*)(ip6 + 1); 
          //unsigned int udphoff;
          __be16 port = udp_hdr->source;
          //udphoff = sizeof(struct ip6) + sizeof(struct ethhdr);
          udp_hdr->source = udp_hdr->dest;
	  udp_hdr->dest = port;
          data =  skb->data + sizeof(struct ip6) + sizeof(struct udphdr);
          len  = len - sizeof(struct ip6) - sizeof(struct udphdr);

          host_send_udp_over_ip6((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,&ip6->dest, &ip6->src,udp_hdr,data,len,skb->dev);
          goto out;
      }
      /*
       *    first fragment which is not icmp  will be free
       *    middle and last fragment not include proto header,if it not icmp will be free
       *    icmp middle and icmp last fragment will be processed by host_send_frag
       */
      if(IPPROTO_ICMP6 != data_proto)
      {
		pr_info("not icmp6 hdr \n");
		goto err1;
      }
      switch (icmp->type) {
      case ICMPV6_ECHO_REQUEST:
		icmp->type = ICMPV6_ECHO_REPLY;
                if(first_frag) 
                {
		    host_send_icmp6_first_frag((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,
				&ip6->dest, &ip6->src,IPPROTO_FRAGMENT,
				icmp, ip6_frag,data,len , skb->dev);
                }
                else
                {
		    host_send_icmp6((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,
				&ip6->dest, &ip6->src,
				icmp, data,len , skb->dev);
                }
		break;
       default:
           dev_kfree_skb(skb);
           return; 
       }
out:
      dev_kfree_skb(skb);
      return ;
err1:
      dev_kfree_skb(skb);
}
static int nat64_start_xmit(struct sk_buff *skb, struct net_device *dev)
{
	netif_stop_queue(dev);
#if 0
       	skb_orphan(skb);
	skb_dst_drop(skb);
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	nf_reset(skb);
#else
	nf_reset_ct(skb);
#endif

	if (skb_linearize(skb) < 0)
		return NETDEV_TX_OK;
#endif
        if(ETH_P_IPV6 == ntohs(skb->protocol))
        {
             handle_ip6(skb);
        }
        else
        {
	     virt_rs_packet(skb,dev);
	     dev_kfree_skb(skb);
	     dev->stats.tx_packets++; 
	     dev->stats.tx_bytes+=skb->len; 
        }
	pr_info("tx_packets=%ld tx_bytes=%ld\n",dev->stats.tx_packets,dev->stats.tx_bytes);
	netif_wake_queue(dev); 
	return NETDEV_TX_OK;
}
static int nat64_start(struct net_device *dev)
{
	netif_tx_start_all_queues(dev);
	return 0;
}

static int nat64_stop(struct net_device *dev)
{
	netif_tx_stop_all_queues(dev);
	return 0;
}

static const struct net_device_ops nat64_netdev_ops = {
	.ndo_open		= nat64_start,
	.ndo_stop		= nat64_stop,
	.ndo_start_xmit	= nat64_start_xmit,
};

static void nat64_setup(struct net_device *dev)
{
	struct nat64_if_info *nif = (struct nat64_if_info *)netdev_priv(dev);

	/* Point-to-Point interface */
	dev->netdev_ops = &nat64_netdev_ops;
	dev->hard_header_len = 0;
	dev->addr_len = 0;
	dev->mtu = 1500;
	//dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4);

	/* Zero header length */
	dev->type = ARPHRD_NONE;
	dev->flags = IFF_POINTOPOINT | IFF_NOARP | IFF_MULTICAST;
	dev->tx_queue_len = 500;  /* We prefer our own queue length */

	/* Setup private data */
	memset(nif, 0x0, sizeof(nif[0]));
	nif->dev = dev;
}

/* Handle of the NAT64 virtual interface */
static struct net_device *nat64_netdev;
struct proc_dir_entry *g_tayga_proc_dir;

int  __init nat64_module_init(void)
{
	struct net_device *dev;
	int err = -1;


	if (!(dev = alloc_netdev_mqs(sizeof(struct nat64_if_info), "nat64",
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3, 17, 0)
		NET_NAME_UNKNOWN,
#endif
		nat64_setup, 8, 8))) {
		printk(KERN_ERR "tayga: alloc_netdev() failed.\n");
		err = -ENOMEM;
		goto err3;
	}

	if ((err = register_netdev(dev)) < 0)
		goto err4;

	nat64_netdev = dev;
	netif_carrier_on(dev);

	return 0;

err4:
	free_netdev(dev);
err3:
	return err;
}

void __exit nat64_module_exit(void)
{
	unregister_netdev(nat64_netdev);
	free_netdev(nat64_netdev);
}

module_init(nat64_module_init);
module_exit(nat64_module_exit);

MODULE_DESCRIPTION("Kernel NAT64 Transition Module, ported from \"tayga\"");
MODULE_AUTHOR("Jianying Liu <rssnsj@gmail.com>");
MODULE_LICENSE("GPL");

