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
#include <linux/netdev_features.h>
#include <linux/ipv6.h>

#include "tayga.h"
#include "csum.h"

#define SRC_MAC "7226fe61ca65"
#define DST_MAC "22554e946f4d"
#define DBG_FRRAG 0
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
#endif

static int parse_ip6(struct pkt *p)
{
	int hdr_len;

	p->ip6 = (struct ip6 *)(p->data);

	if (p->data_len < sizeof(struct ip6) ||
			(ntohl(p->ip6->ver_tc_fl) >> 28) != 6)
		return -1;

	p->data_proto = p->ip6->next_header;
	p->data += sizeof(struct ip6);
	p->data_len -= sizeof(struct ip6);

	if (p->data_len > ntohs(p->ip6->payload_length))
		p->data_len = ntohs(p->ip6->payload_length);

	while (p->data_proto == 0 || p->data_proto == 43 ||
			p->data_proto == 60) {
		if (p->data_len < 2)
			return -1;
		hdr_len = (p->data[1] + 1) * 8;
		if (p->data_len < hdr_len)
			return -1;
		p->data_proto = p->data[0];
		p->data += hdr_len;
		p->data_len -= hdr_len;
		p->header_len += hdr_len;
	}

	if (p->data_proto == 44) {
		if (p->ip6_frag || p->data_len < sizeof(struct ip6_frag))
			return -1;
		p->ip6_frag = (struct ip6_frag *)p->data;
		p->data_proto = p->ip6_frag->next_header;
		p->data += sizeof(struct ip6_frag);
		p->data_len -= sizeof(struct ip6_frag);
		p->header_len += sizeof(struct ip6_frag);

		if ((p->ip6_frag->offset_flags & htons(IP6_F_MF)) &&
				(p->data_len & 0x7))
			return -1;

		if ((u32)(ntohs(p->ip6_frag->offset_flags) & IP6_F_MASK) +
				p->data_len > 65535)
			return -1;
	}

	if (p->data_proto == 58) {
		if (p->ip6_frag && (p->ip6_frag->offset_flags &
					htons(IP6_F_MASK | IP6_F_MF)))
			return -1; /* fragmented ICMP is unsupported */
		if (p->data_len < sizeof(struct icmp))
			return -1;
		p->icmp = (struct icmp *)(p->data);
	}

	return 0;
}
static void host_handle_icmp6(struct pkt *p)
{
	p->data += sizeof(struct icmp);
	p->data_len -= sizeof(struct icmp);

	switch (p->icmp->type) {
	case 128:
		p->icmp->type = 129;
		host_send_icmp6((ntohl(p->ip6->ver_tc_fl) >> 20) & 0xff,
				&p->ip6->dest, &p->ip6->src,
				p->icmp, p->data, p->data_len, p->dev);
		break;
	}
}
static u16 select_ip4_ipid(void)
{
	static DEFINE_SPINLOCK(ipid_lock);
	static u32 offset = 0;
	u32 ipid;

	spin_lock_bh(&ipid_lock);
	//ipid = gcfg.rand[0] + offset++;
        get_random_bytes(&ipid, sizeof(u32));
	ipid +=  offset++;
	spin_unlock_bh(&ipid_lock);
	return htons(ipid & 0xffff);
}
static void xlate_header_6to4(struct pkt *p, struct ip4 *ip4,
		int payload_length)
{
	ip4->ver_ihl = 0x45;
	ip4->tos = (ntohl(p->ip6->ver_tc_fl) >> 20) & 0xff;
	ip4->length = htons(sizeof(struct ip4) + payload_length);
	if (p->ip6_frag) {
		ip4->ident = htons(ntohl(p->ip6_frag->ident) & 0xffff);
		ip4->flags_offset =
			htons(ntohs(p->ip6_frag->offset_flags) >> 3);
		if (p->ip6_frag->offset_flags & htons(IP6_F_MF))
			ip4->flags_offset |= htons(IP4_F_MF);
	} /* else if (dest && (dest->flags & CACHE_F_GEN_IDENT) &&
			p->header_len + payload_length <= 1280) {
		ip4->ident = htons(dest->ip4_ident++);
		ip4->flags_offset = 0;
		if (dest->ip4_ident == 0)
			dest->ip4_ident++;
	} */ else {
		ip4->ident = select_ip4_ipid();
		ip4->flags_offset = htons(IP4_F_DF);
	}
	ip4->ttl = p->ip6->hop_limit;
	ip4->proto = p->data_proto == 58 ? 1 : p->data_proto;
	ip4->cksum = 0;
}
static u16 convert_cksum(struct ip6 *ip6, struct ip4 *ip4)
{
	u32 sum = 0;
	u16 *p;
	int i;

	sum += ~ip4->src.s_addr >> 16;
	sum += ~ip4->src.s_addr & 0xffff;
	sum += ~ip4->dest.s_addr >> 16;
	sum += ~ip4->dest.s_addr & 0xffff;

	for (i = 0, p = ip6->src.s6_addr16; i < 16; ++i)
		sum += *p++;

	while (sum > 0xffff)
		sum = (sum & 0xffff) + (sum >> 16);

	return sum;
}
static int xlate_payload_6to4(struct pkt *p, struct ip4 *ip4)
{
	u16 *tck;
	u16 cksum;

	if (p->ip6_frag && (p->ip6_frag->offset_flags & ntohs(IP6_F_MASK)))
		return 0;

	switch (p->data_proto) {
	case 58:
		cksum = ~ip6_checksum(p->ip6, ntohs(p->ip6->payload_length) -
							p->header_len, 58);
		cksum = ones_add(swap_u16(ntohs(p->icmp->cksum)), cksum);
		if (p->icmp->type == 128) {
			p->icmp->type = 8;
			p->icmp->cksum = htons(swap_u16(ones_add(cksum, 128 - 8)));
		} else {
			p->icmp->type = 0;
			p->icmp->cksum = htons(swap_u16(ones_add(cksum, 129 - 0)));
		}
		return 0;
	case 17:
		if (p->data_len < 8)
			return -1;
		tck = (u16 *)(p->data + 6);
		if (!*tck)
			return -1; /* drop UDP packets with no checksum */
		break;
	case 6:
		if (p->data_len < 20)
			return -1;
		tck = (u16 *)(p->data + 16);
		break;
	default:
		return 0;
	}
	*tck = ones_add(*tck, convert_cksum(p->ip6, ip4));
	return 0;
}
int __map_ip6_to_ip4(struct in_addr *addr4, 	const struct in6_addr *addr6, int dyn_alloc)
{
 	addr4->s_addr = addr6->s6_addr32[3];
        return 0;
}
static void host_handle_frag(struct pkt *p)
{
       //struct ip6_frag *ip6_frag = p->ip6_frag;
        struct {
		struct ip4 ip4;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb = p->skb;
	//struct ethhdr * ethhdr = NULL;
        if (map_ip6_to_ip4(&header.ip4.dest, &p->ip6->dest, 0)) {
		kfree_skb(skb);
		return;
	}

	if (map_ip6_to_ip4(&header.ip4.src, &p->ip6->src, 1)) {
		kfree_skb(skb);
		return;
	}
     	xlate_header_6to4(p, &header.ip4, p->data_len);
	--header.ip4.ttl;
       	if (xlate_payload_6to4(p, &header.ip4) < 0) {
		kfree_skb(skb);
		return;
	}

	header.ip4.cksum = htons(swap_u16(ip_checksum(&header.ip4, sizeof(header.ip4))));

	skb_pull(skb, (unsigned)(p->data - sizeof(header) - skb->data));
	skb->protocol = htons(ETH_P_IP);
	skb_reset_network_header(skb);
	memcpy(ip_hdr(skb), &header, sizeof(header));
	skb->dev = p->dev;
#if 0
        skb_push(skb,ETH_HLEN);
	ethhdr = (struct ethhdr *)skb->data;
        ethhdr->h_proto = htons(ETH_P_IP);
	memcpy(ethhdr->h_dest, DST_MAC, ETH_ALEN);
	memcpy(ethhdr->h_source, SRC_MAC, ETH_ALEN);
        //skb->protocol = eth_type_trans(skb, skb->dev);
        //skb_set_network_header(skb,ETH_HLEN);
        skb_reset_mac_header(skb);
        skb_pull(skb, ETH_HLEN);
#endif
	p->dev->stats.rx_bytes += skb->len;
	p->dev->stats.rx_packets++;
	netif_rx(skb);
}
static int parse_ip4(struct pkt *p)
{
	p->ip4 = (struct ip4 *)(p->data);

	if (p->data_len < sizeof(struct ip4))
		return -1;

	p->header_len = (p->ip4->ver_ihl & 0x0f) * 4;

	if ((p->ip4->ver_ihl >> 4) != 4 ||
			p->header_len < sizeof(struct ip4) ||
			p->data_len < p->header_len ||
			ntohs(p->ip4->length) < p->header_len )
		return -1;

	if (p->data_len > ntohs(p->ip4->length))
		p->data_len = ntohs(p->ip4->length);

	p->data += p->header_len;
	p->data_len -= p->header_len;
	p->data_proto = p->ip4->proto;

	if (p->data_proto == 1) {
		if (p->ip4->flags_offset & htons(IP4_F_MASK | IP4_F_MF))
			return -1; /* fragmented ICMP is unsupported */
		if (p->data_len < sizeof(struct icmp))
			return -1;
		p->icmp = (struct icmp *)(p->data);
	} else {
		if ((p->ip4->flags_offset & htons(IP4_F_MF)) &&
				(p->data_len & 0x7))
			return -1;

		if ((u32)((ntohs(p->ip4->flags_offset) & IP4_F_MASK) * 8) +
				p->data_len > 65535)
			return -1;
	}
#if 0
        if(p->data_proto == 17 &&((ntohs(p->ip4->flags_offset) &IP4_F_MF) && 0 == (ntohs(p->ip4->flags_offset) & IP4_F_MASK)))
        {
            struct udphdr *udphdr = (struct udphdr *)p->data;
            pr_info("udp first frag , src port %u, dst port %u, udp total len %u \n",ntohs(udphdr->source), ntohs(udphdr->dest), udphdr->len);
        }
#endif
	return 0;

}
#if 0
static void host_handle_icmp4(struct pkt *p)
{
	p->data += sizeof(struct icmp);
	p->data_len -= sizeof(struct icmp);

	switch (p->icmp->type) {
	case 8:
		p->icmp->type = 0;
		host_send_icmp4(p->ip4->tos, &p->ip4->dest, &p->ip4->src,
				p->icmp, p->data, p->data_len, p->dev);
		break;
	}
}
#endif
int __map_ip4_to_ip6(struct in6_addr *addr6, const struct in_addr *addr4)
{
   addr6->s6_addr32[0] = htonl(0x20010db8);
   addr6->s6_addr32[1] = htonl(0x0);
   addr6->s6_addr32[2] = htonl(0x0);
   addr6->s6_addr32[3] = addr4->s_addr;
   return 0;
}
static void xlate_header_4to6(struct pkt *p, struct ip6 *ip6,
		int payload_length)
{
	ip6->ver_tc_fl = htonl((0x6 << 28) | (p->ip4->tos << 20));
	ip6->payload_length = htons(payload_length);
	ip6->next_header = p->data_proto == 1 ? 58 : p->data_proto;
	ip6->hop_limit = p->ip4->ttl;
}

static inline void csum_inv_add(__be16 *sum, __be16 *start, __be16 *end)
{
	__be32	new_sum;

	for(new_sum = *sum; start < end; start++)
		new_sum -= *start;

	*sum = (new_sum & 0xffff) + (new_sum >> 16);
}

static inline void csum_inv_substract(__be16 *sum, __be16 *start, __be16 *end)
{
	__be32	new_sum;

	for(new_sum = *sum; start < end; start++)
		new_sum += *start;

	*sum = (new_sum & 0xffff) + (new_sum >> 16);
}
static int xlate_payload_4to6(struct pkt *p, struct ip6 *ip6)
{
	u16 *tck;
	u16 cksum;

	if (p->ip4->flags_offset & htons(IP4_F_MASK))
		return 0;

	switch (p->data_proto) {
	case 1:
		cksum = ip6_checksum(ip6, ntohs(p->ip4->length) - p->header_len, 58);
		cksum = ones_add(swap_u16(ntohs(p->icmp->cksum)), cksum);
		if (p->icmp->type == 8) {
			p->icmp->type = 128;
			p->icmp->cksum = htons(swap_u16(ones_add(cksum, ~(128 - 8))));
		} else {
			p->icmp->type = 129;
			p->icmp->cksum = htons(swap_u16(ones_add(cksum, ~(129 - 0))));
		}
		return 0;
	case 17:
		if (p->data_len < 8)
			return -1;
		tck = (u16 *)(p->data + 6);
		if (!*tck)
			return -1; /* drop UDP packets with no checksum */
		break;
	case 6:
		if (p->data_len < 20)
			return -1;
		tck = (u16 *)(p->data + 16);
		break;
	default:
		return 0;
	}
#if 1
	*tck = ones_add(*tck, ~convert_cksum(ip6, p->ip4));
#else
	csum_inv_substract(tck, (__be16 *)&p->ip4->src, ((__be16 *)&p->ip4->src) + 4);
        csum_inv_add(tck, (__be16 *)&ip6->src, ((__be16 *)&ip6->src) + 16);
#endif
	return 0;
}
#define IPV6_OFFLINK_MTU 1492
#define G_MTU 1492
static void xlate_4to6_data(struct pkt *p)
{
	struct {
		struct ip6 ip6;
		struct ip6_frag ip6_frag;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb = p->skb, *new_skb;
	int no_frag_hdr = 0;
	u16 off = ntohs(p->ip4->flags_offset);
	int frag_size;

	frag_size = IPV6_OFFLINK_MTU;
	//frag_size = gcfg.ipv6_offlink_mtu;
	//if (frag_size > gcfg.mtu)
	if (frag_size > G_MTU)
		frag_size = G_MTU;
	frag_size -= sizeof(struct ip6);

	if (map_ip4_to_ip6(&header.ip6.dest, &p->ip4->dest)) {
		//host_send_icmp4_error(3, 1, 0, p);
		goto drop_skb;
	}

	if (map_ip4_to_ip6(&header.ip6.src, &p->ip4->src)) {
		//host_send_icmp4_error(3, 10, 0, p);
		goto drop_skb;
	}

	/* We do not respect the DF flag for IP4 packets that are already
	   fragmented, because the IP6 fragmentation header takes an extra
	   eight bytes, which we don't have space for because the IP4 source
	   thinks the MTU is only 20 bytes smaller than the actual MTU on
	   the IP6 side.  (E.g. if the IP6 MTU is 1496, the IP4 source thinks
	   the path MTU is 1476, which means it sends fragments with 1456
	   bytes of fragmented payload.  Translating this to IP6 requires
	   40 bytes of IP6 header + 8 bytes of fragmentation header +
	   1456 bytes of payload == 1504 bytes.) */
	if ((off & (IP4_F_MASK | IP4_F_MF)) == 0) {
                
		if (off & IP4_F_DF) {
			if (G_MTU - MTU_ADJ < p->header_len + p->data_len) {
				//host_send_icmp4_error(3, 4, gcfg.mtu - MTU_ADJ, p);
                                pr_info("drop skb,because of mtu \n");
				goto drop_skb;
			}
			no_frag_hdr = 1;
		} else if (p->data_len <= frag_size) {
			no_frag_hdr = 1;
		}
	}

	xlate_header_4to6(p, &header.ip6, p->data_len);
	--header.ip6.hop_limit;

	if (xlate_payload_4to6(p, &header.ip6) < 0)
		goto drop_skb;

	//if (src)
	//	src->flags |= CACHE_F_SEEN_4TO6;
	//if (dest)
	//	dest->flags |= CACHE_F_SEEN_4TO6;

	if (no_frag_hdr) {
		size_t push_len = (skb->data - (p->data - sizeof(header.ip6)));

		if (skb_headroom(skb) < push_len) {
			struct sk_buff *new_skb = skb_realloc_headroom(skb, push_len);
			if (!new_skb) {
				p->dev->stats.rx_dropped++;
				goto drop_skb;
			}
			kfree_skb(skb);
			skb = new_skb;
		}

		skb_push(skb, push_len);
		skb_reset_network_header(skb);
		memcpy(ipv6_hdr(skb), &header.ip6, sizeof(header.ip6));
		skb->protocol = htons(ETH_P_IPV6);
		skb->dev = p->dev;

		p->dev->stats.rx_bytes += skb->len;
		p->dev->stats.rx_packets++;
		netif_rx(skb);
	} else {
		header.ip6_frag.next_header = header.ip6.next_header;
		header.ip6_frag.reserved = 0;
		header.ip6_frag.ident = htonl(ntohs(p->ip4->ident));

		header.ip6.next_header = 44;

		off = (off & IP4_F_MASK) * 8;
		frag_size = (frag_size - sizeof(header.ip6_frag)) & ~7;

		while (p->data_len > 0) {
			if (p->data_len < frag_size)
				frag_size = p->data_len;

			header.ip6.payload_length =
				htons(sizeof(struct ip6_frag) + frag_size);
			header.ip6_frag.offset_flags = htons(off);
                        //pr_info("ip4 frag %u, ipv6 offset: %u, frag payload %u \n",ntohl(p->ip4->ident),off,frag_size);

			//p->data += frag_size;
			p->data_len -= frag_size;
			off += frag_size;

			if (p->data_len || (p->ip4->flags_offset & htons(IP4_F_MF)))
				header.ip6_frag.offset_flags |= htons(IP6_F_MF);

#if 0
                if(p->data_proto == 17 &&((ntohs(header.ip6_frag.offset_flags) &IP6_F_MF) && 0 == (ntohs(header.ip6_frag.offset_flags) & IP6_F_MASK)))
                {
                    struct udphdr *udphdr = (struct udphdr *)p->data;
                    //udphdr->check = htons(0x4421);
                    pr_info("udp first frag , src port %u, dst port %u, udp total len %u, chk sum 0x%x \n",ntohs(udphdr->source),\
                      ntohs(udphdr->dest), ntohs(udphdr->len), ntohs(udphdr->check));
#if 0
                    u16 total_len = htons(p->data_len);
                    //u16 total_len = htons(udphdr->len) + sizeof(struct udphdr);
                      
		    csum_inv_substract(&udphdr->check, (__be16 *)&total_len, (__be16 *)&total_len + 1 );
		    csum_inv_add(&udphdr->check, (__be16 *)&header.ip6.payload_length, ((__be16 *)&header.ip6.payload_length) + 1);
		    //csum_inv_substract(&udphdr->check, (__be16 *)&header.ip6.payload_length, (__be16 *)&header.ip6.payload_length + 1 );
		    //csum_inv_add(&udphdr->check, (__be16 *)&total_len, ((__be16 *)&total_len) + 1);
#endif
                }
#endif
			new_skb = netdev_alloc_skb(p->dev, sizeof(header) + frag_size);
			if (!new_skb) {
				p->dev->stats.rx_dropped++;
				break;
			}
			memcpy(skb_put(new_skb, sizeof(header)), &header, sizeof(header));
			memcpy(skb_put(new_skb, frag_size), p->data, frag_size);
		        p->data += frag_size;
#if 0
                        pr_info("hop %u , ipv6 frag id  %u, frag offset %u, frag  more %u,next header: %u, payload %u \n",\
                             header.ip6.hop_limit, ntohl(header.ip6_frag.ident),\
                             ntohs(header.ip6_frag.offset_flags)&IP6_F_MASK,\
                             ntohs(header.ip6_frag.offset_flags)&IP6_F_MF,header.ip6_frag.next_header, frag_size);
#endif
			new_skb->protocol = htons(ETH_P_IPV6);

			p->dev->stats.rx_bytes += new_skb->len;
			p->dev->stats.rx_packets++;
			netif_rx(new_skb);
#if DBG_FRRAG
                u_int16_t index = 0, count = 0, udp_payload_size = 0;
                char * udp_payload = NULL;
                if(p->data_proto == 17 &&((ntohs(header.ip6_frag.offset_flags) &IP6_F_MF) && 0 == (ntohs(header.ip6_frag.offset_flags) & IP6_F_MASK)))
                {
                    struct udphdr *udphdr = (struct udphdr *)p->data;
                    pr_info("udp first frag , src port %u, dst port %u, udp total len %u \n",ntohs(udphdr->source), ntohs(udphdr->dest), udphdr->len);
                    udp_payload = (char *) (udphdr +1);
                    udp_payload_size = frag_size - sizeof(struct udphdr);
                }
                else if(p->data_proto == 17 &&(0 != (ntohs(header.ip6_frag.offset_flags) & IP6_F_MASK)))
                {
                    udp_payload = (char *) p->data;
                    udp_payload_size = frag_size;
                }
                if (NULL != udp_payload)
                {
                    while(index < udp_payload_size){
                        if('a' == udp_payload[index++])
                        {
                             ++count; 
                        }
                    }
                    pr_info("ip4 frag %u, ipv6 offset: %u, frag payload %u, number of a %u \n",ntohl(p->ip4->ident),off,frag_size, count);
                }
#else
                //if(p->data_proto == 17 &&((ntohs(header.ip6_frag.offset_flags) &IP6_F_MF) && 0 == (ntohs(header.ip6_frag.offset_flags) & IP6_F_MASK)))
                //{
                //    struct udphdr *udphdr = (struct udphdr *)p->data;
                //    pr_info("udp first frag , src port %u, dst port %u, udp total len %u \n",ntohs(udphdr->source), ntohs(udphdr->dest), udphdr->len);
		//    csum_inv_substract(&udphdr->check, (__be16 *)&p->ip4->src, ((__be16 *)&p->ip4->src) + 4);
		//    csum_inv_add(&udphdr->check, (__be16 *)&header.ip6.src, ((__be16 *)&header.ip6.src) + 16);
                //}
#endif
		}

		kfree_skb(skb);
	}

	return;

drop_skb:
	kfree_skb(skb);
}
void handle_ip4(struct pkt *p)
{
	if (parse_ip4(p) < 0 || p->ip4->ttl == 0 ||
			ip_checksum(p->ip4, p->header_len) ||
			p->header_len + p->data_len != ntohs(p->ip4->length)) {
		p->dev->stats.tx_dropped++;
		kfree_skb(p->skb);
		return;
	}

	if (p->icmp && ip_checksum(p->data, p->data_len)) {
		p->dev->stats.tx_dropped++;
		kfree_skb(p->skb);
		return;
	}

	p->dev->stats.tx_bytes += p->skb->len;
	p->dev->stats.tx_packets++;
        // udp or tcp fragment have no l4 header
	if (p->data_proto != 1 || p->icmp->type == 8 ||
		p->icmp->type == 0)
		xlate_4to6_data(p);
        else {
		kfree_skb(p->skb);
        }
#if 0
	if (p->data_proto ==1 ){

		host_handle_icmp4(p);
	} else {
		if (p->data_proto != 1 || p->icmp->type == 8 ||
				p->icmp->type == 0)
			xlate_4to6_data(p);
	}
#endif
}
void handle_ip6(struct pkt *p)
{
     	if (parse_ip6(p) < 0 || p->ip6->hop_limit == 0 ||
		p->header_len + p->data_len != ntohs(p->ip6->payload_length)) {
		p->dev->stats.tx_dropped++;
		kfree_skb(p->skb);
		return;
	}
        if (p->data_proto == 58)
  	    host_handle_icmp6(p);
        //else if(IPPROTO_FRAGMENT == p->data_proto)
        else if(IPPROTO_UDP == p->data_proto)
        {
            host_handle_frag(p);
        }
        else
        {
	    kfree_skb(p->skb);
        }
	return;
}
static int nat64_start_xmit(struct sk_buff *skb, struct net_device *dev)
{
	struct pkt pbuf;

	skb_orphan(skb);
	skb_dst_drop(skb);
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	nf_reset(skb);
#else
	nf_reset_ct(skb);
#endif

	if (skb_linearize(skb) < 0)
		return NETDEV_TX_OK;

	pbuf.dev = dev;
	pbuf.skb = skb;
	pbuf.ip4 = NULL;
	pbuf.ip6 = NULL;
	pbuf.ip6_frag = NULL;
	pbuf.icmp = NULL;
	pbuf.data_proto = 0;
	pbuf.data = skb->data;
	pbuf.data_len = skb->len;
	pbuf.header_len = 0;

	switch(ntohs(skb->protocol)) {
	case ETH_P_IP:
	        //dev_kfree_skb(skb);
		handle_ip4(&pbuf);
		break;
	case ETH_P_IPV6:
		handle_ip6(&pbuf);
		break;
	default:
		printk(KERN_WARNING "tayga: Unknown protocol %u of packet.\n",
			ntohs(skb->protocol));
		dev->stats.tx_dropped++;
	}

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
        //ipv4 mtu - sizeof(struct iphdr) <  ipv6 mtu -sizeof(struct ip6hdr) - sizeof(struct ip6frag)
	dev->mtu = 1460;
	//dev->mtu = 1500;
	dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4);
	//dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4) + sizeof(ETH_HLEN);

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

