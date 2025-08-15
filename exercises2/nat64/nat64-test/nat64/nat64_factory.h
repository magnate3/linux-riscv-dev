#pragma once

//#include <linux/kernel.h>
//#include <linux/init.h>
//#include <linux/ip.h>
//#include <linux/icmpv6.h>
//#include <linux/inetdevice.h>
//#include <linux/types.h>

//#include <net/ip.h>
//#include <net/tcp.h>
//#include <net/udp.h>
//#include <net/icmp.h>
//#include <net/route.h>
//#include <net/ip6_route.h>

//#include <net/ipv6.h>
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 0)
#include<net/ip6_checksum.h>
#endif
#include "nat64.h"

//#define NAT64_NETDEV_NAME "nat64"

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

static inline void factory_translate_ip4(struct sk_buff *src, struct sk_buff *dst, struct in6_addr *saddr, struct in6_addr *daddr, __u8 nexthdr, int len)
{
	struct iphdr	*iph;
	struct ipv6hdr	*ip6h;
	int		payload_len;
	//__be32		new_sum, tmp;

	//int i;
	//__be16		*cp;

	ip6h = (struct ipv6hdr *)skb_push(dst, sizeof(struct ipv6hdr));
	//iph = ip_hdr(src);
	iph = (struct iphdr *)skb_push(src, len);
	payload_len = dst->len - sizeof(struct ipv6hdr);

	/*	Set IPv6 protocol	*/
	dst->protocol = htons(ETH_P_IPV6);
	skb_reset_network_header(dst);

	/*	Fill IP header fields	*/
	ip6h->version	= 6;
	ip6h->priority	= iph->tos >> 4;
	ip6h->flow_lbl[0] = iph->tos << 4;;
	ip6h->flow_lbl[1] = 0;
	ip6h->flow_lbl[2] = 0;
	ip6h->payload_len = htons(payload_len);
	ip6h->nexthdr	= nexthdr;
	ip6h->hop_limit = iph->ttl;
	//assemble_ipv6(&ip6h->saddr, iph->saddr);
	//ipv6_addr_copy(&ip6h->saddr, saddr);
	ip6h->saddr = *saddr;
	//ipv6_addr_copy(&ip6h->daddr, daddr);
	ip6h->daddr = *daddr;

	//if an unexpired source route option is present then the packet
	//MUST instead be discarded, and an ICMPv4 "Destination
	//Unreachable/Source Route Failed" (Type 3/Code 5) error message
	//SHOULD be returned to the sender.

	switch(nexthdr) {
	case IPPROTO_TCP:
		csum_inv_substract(&(tcp_hdr(dst)->check), (__be16 *)&iph->saddr, ((__be16 *)&iph->saddr) + 4);
		csum_inv_add(&(tcp_hdr(dst)->check), (__be16 *)&ip6h->saddr, ((__be16 *)&ip6h->saddr) + 16);
		break;
	case IPPROTO_UDP:
		csum_inv_substract(&(udp_hdr(dst)->check), (__be16 *)&iph->saddr, ((__be16 *)&iph->saddr) + 4);
		csum_inv_add(&(udp_hdr(dst)->check), (__be16 *)&ip6h->saddr, ((__be16 *)&ip6h->saddr) + 16);
		break;
	case IPPROTO_ICMPV6:
		dst->csum = csum_partial(skb_transport_header(dst), dst->len - sizeof(struct ipv6hdr), 0);
		icmp_hdr(dst)->checksum = csum_ipv6_magic(&ip6h->saddr, &ip6h->daddr, payload_len, nexthdr, dst->csum);
		break;
	}

	dst->ip_summed = CHECKSUM_NONE;

}

static inline void factory_translate_ip6(struct sk_buff *src, struct sk_buff *dst, __be32 saddr, int protocol)
{
	struct iphdr	*iph;
	struct ipv6hdr	*ip6h;
	int		payload_len;

	iph = (struct iphdr *)skb_push(dst, sizeof(struct iphdr));
	ip6h = ipv6_hdr(src);
	payload_len = dst->len - sizeof(struct iphdr);

	/*	Set IPv4 protocol	*/
	dst->protocol = htons(ETH_P_IP);
	skb_reset_network_header(dst);

	/*	Fill IP header fields	*/
	iph->version	= 4;
	iph->ihl	= sizeof(struct iphdr) / 4;
	iph->tos	= (ip6h->priority << 4) | (ip6h->flow_lbl[0] >> 4);
	iph->tot_len	= htons(dst->len);
	//iph->id	= payload_len > 1280 ? /*FIXME random*/ : 0;
	iph->id		= 0;
	/* If data > 1280 set DF. The ICMP too big will be returned to sender */
	/* If data <= 1280 clear DF. This allows packet to be fragmented on the IPv4 side */
	iph->frag_off	= payload_len > 1280 ? htons(IP_DF) : 0;
	iph->ttl	= ip6h->hop_limit;
	iph->protocol	= protocol;
	iph->saddr	= saddr;
	iph->daddr	= extract_ipv4(ip6h->daddr, state.prefix_len);

	/*	Calculate IP header checksum	*/
	ip_send_check(iph);

	switch(protocol) {
	case IPPROTO_TCP:
		csum_inv_substract(&(tcp_hdr(dst)->check), (__be16 *)&ip6h->saddr, ((__be16 *)&ip6h->saddr) + 16);
		csum_inv_add(&(tcp_hdr(dst)->check), (__be16 *)&iph->saddr, ((__be16 *)&iph->saddr) + 4);
		break;
	case IPPROTO_UDP:
		csum_inv_substract(&(udp_hdr(dst)->check), (__be16 *)&ip6h->saddr, ((__be16 *)&ip6h->saddr) + 16);
		csum_inv_add(&(udp_hdr(dst)->check), (__be16 *)&iph->saddr, ((__be16 *)&iph->saddr) + 4);
		break;
	case IPPROTO_ICMP:
		dst->csum = csum_partial(skb_transport_header(dst), dst->len - sizeof(struct iphdr), 0);
		icmp_hdr(dst)->checksum = csum_fold(dst->csum);
		break;
	}

	dst->ip_summed = CHECKSUM_NONE;
}

static inline void factory_clone_icmp(struct sk_buff *src, struct sk_buff *dst, __be16 id, __be16 type)
{
	struct icmphdr	*icmph;
	int		len;

	len = sizeof(struct icmphdr);
	icmph = (struct icmphdr *)skb_push(dst, len);

	//memcpy(icmph, skb_transport_header(src), len);
	memcpy(icmph, skb_push(src, len), len);
	icmph->type = type >> 8;
	icmph->code = type & 0xFF;
	icmph->un.echo.id = id;
	icmph->checksum = 0;

	//printk("nat64: [debug] [icmp_clone] type=%hhu, id=%hu, seq=%hu\n", icmph->type, icmph->un.echo.id, icmph->un.echo.sequence);

	skb_reset_transport_header(dst);
}

static inline void factory_clone_tcp(struct sk_buff *src, struct sk_buff *dst, __be16 sport, __be16 dport, int len)
{
	struct tcphdr	*tcph;
	//int		len;

	//len = tcp_hdrlen(src);
	tcph = (struct tcphdr *)skb_push(dst, len);
	//memcpy(tcph, skb_transport_header(src), len);
	memcpy(tcph, skb_push(src, len), len);
	tcph->source = sport;
	tcph->dest = dport;

	csum_inv_substract(&tcph->check, (__be16 *)src->data, (__be16 *)(src->data + 4));
	csum_inv_add(&tcph->check, (__be16 *)dst->data, (__be16 *)(dst->data + 4));
	//tcph->check = 0;

	skb_reset_transport_header(dst);
}

static inline void factory_clone_udp(struct sk_buff *src, struct sk_buff *dst, __be16 sport, __be16 dport)
{
	struct udphdr	*udph;
	int		len;

	len = sizeof(struct udphdr);
	udph = (struct udphdr*)skb_push(dst, len);
	//memcpy(udph, skb_transport_header(src), len);
	memcpy(udph, skb_push(src, len), len);
	//printk("nat64: [factory] [udp] [debug] src_udph = %02x %02x %02x %02x %02x %02x %02x %02x.\n", *(src->data), *(src->data +1), *(src->data +2), *(src->data +3), *(src->data +4), *(src->data +5), *(src->data +6), *(src->data +7));
	udph->source = sport;
	udph->dest = dport;
	csum_inv_substract(&udph->check, (__be16 *)src->data, (__be16 *)(src->data + 4));
	csum_inv_add(&udph->check, (__be16 *)dst->data, (__be16 *)(dst->data + 4));
	//printk("nat64: [factory] [udp] [debug] dst_udph = %02x %02x %02x %02x %02x %02x %02x %02x.\n", *(dst->data), *(dst->data +1), *(dst->data +2), *(dst->data +3), *(dst->data +4), *(dst->data +5), *(dst->data +6), *(dst->data +7));

	skb_reset_transport_header(dst);
}

static inline void factory_clone_data(struct sk_buff *src, struct sk_buff *dst)
{
	int data_len = src->len;

	if(data_len > 0) {
		skb_push(dst, data_len);
		memcpy(dst->data, src->data, data_len);
	}
}


