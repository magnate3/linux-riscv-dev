/* SPDX-License-Identifier: GPL-2.0-only
   Copyright (c) 2022 */
#include <linux/bpf.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
// #include <linux/icmp.h>

// The parsing helper functions from the packet01 lesson have moved here
#include "../common/parsing_helpers.h"
#include "../common/rewrite_helpers.h"

/* Defines xdp_stats_map */
#include "../common/xdp_stats_kern_user.h"
#include "../common/xdp_stats_kern.h"

#include <stdio.h>

#ifndef AF_INET
#define AF_INET 1
#endif

#ifndef AF_INET6
#define AF_INET6 6
#endif


#ifndef memcpy
#define memcpy(dest, src, n) __builtin_memcpy((dest), (src), (n))
#endif
typedef __u32 __bitwise __wsum;
struct bpf_map_def SEC("maps") static_redirect_8b = {
	.type = BPF_MAP_TYPE_HASH,
	.key_size = sizeof(__u8),
	.value_size = sizeof(__u32),
	.max_entries = 256,
};
struct bpf_map_def SEC("maps") tx_port = {
	.type = BPF_MAP_TYPE_DEVMAP,
	.key_size = sizeof(int),
	.value_size = sizeof(int),
	.max_entries = 256,
};
struct icmpv6_pseudo {
	struct in6_addr saddr;
	struct in6_addr daddr;
	__u32 len;
	__u8 padding[3];
	__u8 nh;
} __attribute__((packed));

#define IPV6_FLOWINFO_MASK bpf_htonl(0x0FFFFFFF)
#define IPV4_SRC_ADDRESS bpf_htonl(0x0a000101) // 10.0.1.1 src address from v6 to v4 (at egress of nat)

static __always_inline __u16 csum_fold_helper(__u32 csum)
{
	__u32 sum;
	sum = (csum >> 16) + (csum & 0xffff);
	sum += (sum >> 16);
	return ~sum;
}
static __always_inline __u16 calculate_icmp_checksum(__u16 *icmph,__u16* ph)
{	
	
	__u16 ret = 0;
	__u32 sum = 0;
	for (int i = 0;i<40;i++)
	{
		sum += *ph++;
	}	
	// sum = 0x20C44;
	for (int i = 0;i<4;i++)
	{
		sum += *icmph++;
	}	
	sum =  (sum >> 16) + (sum & 0xffff);
	sum += (sum >> 16);
	ret =  ~sum;
	return (ret); 
}

static __always_inline int write_icmp(struct icmphdr * icmp, struct icmp6hdr* icmp6)
{
	__u32 mtu, ptr;
	// 	/* These translations are defined in RFC6145 section 5.2 */
	// bpf_printk("inside write icmp with type: %d ,request is %d",icmp6->icmp6_type, ICMPV6_ECHO_REQUEST);
		switch (icmp6->icmp6_type) 
		{
		case ICMPV6_ECHO_REQUEST:
			icmp->type = ICMP_ECHO;
			// icmp->type = 0;
			// bpf_printk("changed type from %d to %d",icmp6->icmp6_type,icmp->type);
			break;
		case ICMPV6_ECHO_REPLY:
			icmp->type = ICMP_ECHOREPLY;
			break;
		case ICMPV6_DEST_UNREACH:
			icmp->type = ICMP_DEST_UNREACH;
			switch(icmp6->icmp6_code) {
			case ICMPV6_NOROUTE:
			case ICMPV6_NOT_NEIGHBOUR:
			case ICMPV6_ADDR_UNREACH:
				icmp->code = ICMP_HOST_UNREACH;
				break;
			case ICMPV6_ADM_PROHIBITED:
				icmp->code = ICMP_HOST_ANO;
				break;
			case ICMPV6_PORT_UNREACH:
				icmp->code = ICMP_PORT_UNREACH;
				break;
			default:
				return -1;
			}
			break;
		case ICMPV6_PKT_TOOBIG:
			icmp->type = ICMP_DEST_UNREACH;
			icmp->code = ICMP_FRAG_NEEDED;

					mtu = bpf_htonl(icmp6->icmp6_mtu) - 20;
					if (mtu > 0xffff)
							return -1;
					icmp->un.frag.mtu = bpf_htons(mtu);
			break;
		case ICMPV6_TIME_EXCEED:
			icmp->type = ICMP_TIME_EXCEEDED;
			break;
			case ICMPV6_PARAMPROB:
					switch (icmp6->icmp6_code) {
					case 0:
							icmp->type = ICMP_PARAMETERPROB;
							icmp->code = 0;
							break;
					case 1:
							icmp->type = ICMP_DEST_UNREACH;
							icmp->code = ICMP_PROT_UNREACH;
							ptr = bpf_ntohl(icmp6->icmp6_pointer);
							/* Figure 6 in RFC6145 - using if statements b/c of
							* range at the bottom
							*/
							if (ptr == 0 || ptr == 1)
									icmp->un.reserved[0] = ptr;
							else if (ptr == 4 || ptr == 5)
									icmp->un.reserved[0] = 2;
							else if (ptr == 6)
									icmp->un.reserved[0] = 9;
							else if (ptr == 7)
									icmp->un.reserved[0] = 8;
							else if (ptr >= 8 && ptr <= 23)
									icmp->un.reserved[0] = 12;
							else if (ptr >= 24 && ptr <= 39)
									icmp->un.reserved[0] = 16;
							else
									return -1;
							break;
					default:
							return -1;
					}
					break;
			default:
				return -1;
			}
	icmp->un.echo.id = icmp6->icmp6_dataun.u_echo.identifier;
	icmp->un.echo.sequence = icmp6->icmp6_dataun.u_echo.sequence;
	return 0;
}
static __always_inline int write_icmp6(struct icmphdr * icmp, struct icmp6hdr* icmp6)
{
	__u32 mtu;
	// 	/* These translations are defined in RFC6145 section 5.2 */
	// bpf_printk("inside write icmp with type: %d ,request is %d",icmp6->icmp6_type, ICMPV6_ECHO_REQUEST);
		switch (icmp->type) {
	case ICMP_ECHO:
		icmp6->icmp6_type = ICMPV6_ECHO_REQUEST;
		break;
	case ICMP_ECHOREPLY:
		icmp6->icmp6_type = ICMPV6_ECHO_REPLY;
		break;
        case ICMP_DEST_UNREACH:
		icmp6->icmp6_type = ICMPV6_DEST_UNREACH;
		switch(icmp->code) {
		case ICMP_NET_UNREACH:
		case ICMP_HOST_UNREACH:
                case ICMP_SR_FAILED:
                case ICMP_NET_UNKNOWN:
                case ICMP_HOST_UNKNOWN:
                case ICMP_HOST_ISOLATED:
                case ICMP_NET_UNR_TOS:
                case ICMP_HOST_UNR_TOS:
			icmp6->icmp6_code = ICMPV6_NOROUTE;
			break;
                case ICMP_PROT_UNREACH:
			icmp6->icmp6_type = ICMPV6_PARAMPROB;
			icmp6->icmp6_code = ICMPV6_UNK_NEXTHDR;
                        icmp6->icmp6_pointer = bpf_htonl(offsetof(struct ipv6hdr, nexthdr));
                case ICMP_PORT_UNREACH:
			icmp6->icmp6_code = ICMPV6_PORT_UNREACH;
			break;
                case ICMP_FRAG_NEEDED:
                        icmp6->icmp6_type = ICMPV6_PKT_TOOBIG;
                        icmp6->icmp6_code = 0;
                        mtu = bpf_ntohs(icmp->un.frag.mtu) + 20;
                        /* RFC6145 section 6, "second approach" - should not be
                         * necessary, but might as well do this
                         */
                        if (mtu < 1280)
                                mtu = 1280;
                        icmp6->icmp6_mtu = bpf_htonl(mtu);
                case ICMP_NET_ANO:
                case ICMP_HOST_ANO:
                case ICMP_PKT_FILTERED:
                case ICMP_PREC_CUTOFF:
                        icmp6->icmp6_code = ICMPV6_ADM_PROHIBITED;
		default:
			return -1;
		}
		break;
        case ICMP_PARAMETERPROB:
                if (icmp->code == 1)
                        return -1;
                icmp6->icmp6_type = ICMPV6_PARAMPROB;
                icmp6->icmp6_code = ICMPV6_HDR_FIELD;
                /* The pointer field not defined in the Linux header. This
                 * translation is from Figure 3 of RFC6145.
                 */
                switch (icmp->un.reserved[0]) {
                case 0: /* version/IHL */
                        icmp6->icmp6_pointer = 0;
                        break;
                case 1: /* Type of Service */
                        icmp6->icmp6_pointer = bpf_htonl(1);
                        break;
                case 2: /* Total length */
                case 3:
                        icmp6->icmp6_pointer = bpf_htonl(4);
                        break;
                case 8: /* Time to Live */
                        icmp6->icmp6_pointer = bpf_htonl(7);
                        break;
                case 9: /* Protocol */
                        icmp6->icmp6_pointer = bpf_htonl(6);
                        break;
                case 12: /* Source address */
                case 13:
                case 14:
                case 15:
                        icmp6->icmp6_pointer = bpf_htonl(8);
                        break;
                case 16: /* Destination address */
                case 17:
                case 18:
                case 19:
                        icmp6->icmp6_pointer = bpf_htonl(24);
                        break;
                default:
                        return -1;
                }
	default:
		return -1;
	}
	icmp6->icmp6_dataun.u_echo.identifier = icmp->un.echo.id;
	icmp6->icmp6_dataun.u_echo.sequence = icmp->un.echo.sequence;
	return 0;
}

SEC("v6_side")
int xdp_nat_v6_func(struct xdp_md *ctx)
{
	void *data_end = (void *)(long)ctx->data_end;
	void *data = (void *)(long)ctx->data;
	
	struct ethhdr *eth = data;
	struct iphdr *iph;
	struct ipv6hdr *ip6h;
	__u32 dst_v4;

	struct ethhdr eth_cpy;
	__u16 h_proto;
	__u64 nh_off;
	int rc;
	struct bpf_fib_lookup fib_params = {};
	int action = XDP_PASS;
	struct iphdr dst_hdr = {
		.version = 4,
                .ihl = 5,
                .frag_off = bpf_htons(1<<14),
        };

	nh_off = sizeof(*eth);
	if (data + nh_off > data_end)
	{
		action = XDP_DROP;
		goto out;
	}

	h_proto = eth->h_proto;

	if (h_proto == bpf_htons(ETH_P_IPV6))
	{
		// bpf_printk("IPv6 packet");
		
		__builtin_memcpy(&eth_cpy, eth, sizeof(eth_cpy));
		ip6h = data + nh_off;

		if (ip6h + 1 > data_end) {
			action = XDP_DROP;
			goto out;
		}
		
		if (ip6h->nexthdr == 0x3b)
		{
			bpf_printk("no next header");
		}
		else if (ip6h->nexthdr == 0x3a)
		{
			bpf_printk("icmp6 header");
		}
		else
		{
			goto out;
		}
		dst_v4 = ip6h->daddr.s6_addr32[3];
		dst_hdr.daddr = dst_v4;
        dst_hdr.saddr = IPV4_SRC_ADDRESS; // 10.0.1.1
		// bpf_printk("ipv4 src %pI4",&dst_hdr.saddr);
        dst_hdr.protocol = ip6h->nexthdr;
		dst_hdr.ttl = ip6h->hop_limit;
        dst_hdr.tos = ip6h->priority << 4 | (ip6h->flow_lbl[0] >> 4);
        dst_hdr.tot_len = bpf_htons(bpf_ntohs(ip6h->payload_len) + sizeof(dst_hdr));
		
		if (dst_hdr.protocol == IPPROTO_ICMPV6) 
		{
			struct icmp6hdr *icmp6 = (void *)ip6h + sizeof(*ip6h);
			if (icmp6 + 1 > data_end)
				return -1;
			struct icmphdr icmp;
			struct icmphdr *new_icmp;
			
			if (write_icmp(&icmp, icmp6) == -1)
			{
				bpf_printk("cant write icmp");
				goto out;
			}

			if (bpf_xdp_adjust_head(ctx, (int)sizeof(*icmp6) - (int)sizeof(icmp)))
			return -1;
			data = (void *)(long)ctx->data;
			data_end = (void *)(long)ctx->data_end;
			new_icmp = (void *)(data + sizeof(struct ethhdr) + sizeof(struct ipv6hdr));

			if (new_icmp + 1 > data_end) {
				bpf_printk("new icmp");
				return -1;
			}

			*new_icmp = icmp;

			// new_icmp->checksum = 0x0000;
			// new_icmp->checksum = calculate_icmp_checksum((__u16 *)new_icmp);
			new_icmp->checksum = csum_fold_helper(bpf_csum_diff((__be32 *)new_icmp, 0,
                                                       (__be32 *)new_icmp, sizeof(new_icmp),
                                                       0));
			// bpf_printk("checksum in icmp %x",new_icmp->checksum);
			dst_hdr.protocol = IPPROTO_ICMP;
		}
		dst_hdr.check = csum_fold_helper(bpf_csum_diff((__be32 *)&dst_hdr, 0,
                                                       (__be32 *)&dst_hdr, sizeof(dst_hdr),
                                                       0));
		if (bpf_xdp_adjust_head(ctx, (int)sizeof(struct ipv6hdr) - (int)sizeof(struct iphdr)))
			return -1;
		
		eth = (void *)(long)ctx->data;
		data = (void *)(long)ctx->data;
		data_end = (void *)(long)ctx->data_end;
		if (eth + 1 > data_end)
			return -1;
		__builtin_memcpy(eth, &eth_cpy, sizeof(*eth));
		eth->h_proto = bpf_htons(ETH_P_IP);
		iph = (void *)(data + sizeof(*eth));

		if (iph + 1 > data_end) {
			bpf_printk("iph out of boundary");
			return -1;
		}

		*iph = dst_hdr;
		fib_params.family = AF_INET;
		fib_params.ipv4_dst = dst_v4;
		// bpf_printk("ipv4 destination %pI4",&fib_params.ipv4_dst);
		fib_params.ifindex = ctx->ingress_ifindex;

		rc = bpf_fib_lookup(ctx, &fib_params, sizeof(fib_params), 0);
		switch (rc)
			{
			case BPF_FIB_LKUP_RET_SUCCESS: /* lookup successful */
				bpf_printk("ifindex redirect %d",fib_params.ifindex);
				memcpy(eth->h_dest, fib_params.dmac, ETH_ALEN);
				memcpy(eth->h_source, fib_params.smac, ETH_ALEN);
				// action = bpf_redirect_map(&tx_port, fib_params.ifindex, 0);
				action = bpf_redirect(fib_params.ifindex, 0);
				bpf_printk("action %d",action);
				goto out;
				break;
			case BPF_FIB_LKUP_RET_BLACKHOLE:   /* dest is blackholed; can be dropped */
			case BPF_FIB_LKUP_RET_UNREACHABLE: /* dest is unreachable; can be dropped */
			case BPF_FIB_LKUP_RET_PROHIBIT:	   /* dest not allowed; can be dropped */
				action = XDP_DROP;
				break;
			case BPF_FIB_LKUP_RET_NOT_FWDED:	/* packet is not forwarded */
				bpf_printk ("route not found, check if routing suite is working properly");
			case BPF_FIB_LKUP_RET_FWD_DISABLED: /* fwding is not enabled on ingress */
			case BPF_FIB_LKUP_RET_UNSUPP_LWT:	/* fwd requires encapsulation */
			case BPF_FIB_LKUP_RET_NO_NEIGH:		/* no neighbor entry for nh */
				bpf_printk("neigh entry missing");
			case BPF_FIB_LKUP_RET_FRAG_NEEDED:	/* fragmentation required to fwd */
				/* PASS */
				break;
			}
	}
out:
		return action;
}

SEC("v4_side")
int xdp_nat_v4_func(struct xdp_md *ctx)
{
	void *data_end = (void *)(long)ctx->data_end;
	void *data = (void *)(long)ctx->data;
	
	struct ethhdr *eth = data;
	struct iphdr *iph;
	struct ipv6hdr *ip6h;
	int iphdr_len;
	struct in6_addr v6_prefix;
	v6_prefix.s6_addr[1] = 0x64;
	v6_prefix.s6_addr[2] = 0xff;
	v6_prefix.s6_addr[3] = 0x9b;
	struct ethhdr eth_cpy;
	__u16 h_proto;
	__u64 nh_off;
	int rc;
	struct bpf_fib_lookup fib_params = {};
	struct in6_addr *fib_dst = (struct in6_addr *)fib_params.ipv6_dst;
	int action = XDP_PASS;
	struct ipv6hdr dst_hdr = {
		.version = 6,
		.saddr = v6_prefix,
		.daddr = v6_prefix
	};

	
	nh_off = sizeof(*eth);
	if (data + nh_off > data_end)
	{
		action = XDP_DROP;
		goto out;
	}

	h_proto = eth->h_proto;
	
	if (h_proto == bpf_htons(ETH_P_IP))
	{
		// bpf_printk("IPv4 packet");
		
		__builtin_memcpy(&eth_cpy, eth, sizeof(eth_cpy));
		iph = data + nh_off;

		if (iph + 1 > data_end) {
			action = XDP_DROP;
			goto out;
		}
		if (iph->daddr != IPV4_SRC_ADDRESS)
		{
			goto out;
		}
		bpf_printk("src address of received packet %pI4",&iph->daddr);
		iphdr_len = iph->ihl * 4;
        if (iphdr_len != sizeof(struct iphdr) || (iph->frag_off & ~bpf_htons(1<<14))) 
		{
                bpf_printk("v4: pkt src/dst %pI4/%pI4 has IP options or is fragmented, dropping\n",
                    &iph->daddr, &iph->saddr);
                goto out;
        }
		dst_hdr.saddr.s6_addr32[3] = iph->saddr;
        dst_hdr.daddr.s6_addr[15] = 0x02;
        dst_hdr.nexthdr = iph->protocol;
        dst_hdr.hop_limit = iph->ttl;
        dst_hdr.priority = (iph->tos & 0x70) >> 4;
        dst_hdr.flow_lbl[0] = iph->tos << 4;
        dst_hdr.payload_len = bpf_htons(bpf_ntohs(iph->tot_len) - iphdr_len);
		if (dst_hdr.nexthdr == IPPROTO_ICMP) 
		{
			struct icmphdr *icmp = (void *)iph + sizeof(*iph);
			if (icmp + 1 > data_end)
				return -1;
			struct icmp6hdr icmp6;
			struct icmp6hdr *new_icmp6;
			
			if (write_icmp6(icmp, &icmp6) == -1)
			{
				bpf_printk("cant write icmp");
				goto out;
			}

			if (bpf_xdp_adjust_head(ctx, (int)sizeof(*icmp) - (int)sizeof(icmp6)))
			return -1;
			data = (void *)(long)ctx->data;
			data_end = (void *)(long)ctx->data_end;
			new_icmp6 = (void *)(data + sizeof(struct ethhdr) + sizeof(struct iphdr));

			if (new_icmp6 + 1 > data_end) {
				bpf_printk("new icmp");
				return -1;
			}

			*new_icmp6 = icmp6;

			struct icmpv6_pseudo ph = 
			{
				.nh = IPPROTO_ICMPV6,
				.saddr = dst_hdr.saddr,
				.daddr = dst_hdr.daddr,
				.len = dst_hdr.payload_len
			};
			new_icmp6->icmp6_cksum = calculate_icmp_checksum((__u16 *)new_icmp6, (__u16 *)&ph);
			bpf_printk("checksum in icmp %x",new_icmp6->icmp6_cksum);
			// new_icmp6->icmp6_cksum = 0x0000;
			// new_icmp6->icmp6_cksum = csum_fold_helper(bpf_csum_diff((__be32 *)new_icmp6, 0,
            //                                            (__be32 *)new_icmp6, sizeof(new_icmp6),
            //                                            0));
			// bpf_printk("checksum in icmp %x",new_icmp6->icmp6_cksum);
			dst_hdr.nexthdr = IPPROTO_ICMPV6;
		}
		// dst_hdr.check = csum_fold_helper(bpf_csum_diff((__be32 *)&dst_hdr, 0,
        //                                                (__be32 *)&dst_hdr, sizeof(dst_hdr),
        //                                                0));


		// bpf_printk("ipv6 destination in hdr %d", &dst_hdr.saddr);
		if (bpf_xdp_adjust_head(ctx, (int)sizeof(struct iphdr) - (int)sizeof(struct ipv6hdr)))
			return -1;
		bpf_printk("adjusted head");
		eth = (void *)(long)ctx->data;
		data = (void *)(long)ctx->data;
		data_end = (void *)(long)ctx->data_end;
		if (eth + 1 > data_end)
			return -1;

		__builtin_memcpy(eth, &eth_cpy, sizeof(*eth));
		eth->h_proto = bpf_htons(ETH_P_IPV6);
		ip6h = (void *)(data + sizeof(*eth));

		if (ip6h + 1 > data_end) {
			bpf_printk("ip6h out of boundary");
			return -1;
		}
		ip6h->saddr.s6_addr32[0] = 0;
		ip6h->saddr.s6_addr32[1] = 0;
		ip6h->saddr.s6_addr32[2] = 0;
		ip6h->saddr.s6_addr32[3] = 0;
		ip6h->daddr.s6_addr32[0] = 0;
		ip6h->daddr.s6_addr32[1] = 0;
		ip6h->daddr.s6_addr32[2] = 0;
		ip6h->daddr.s6_addr32[3] = 0;
		
		*ip6h = dst_hdr;
		ip6h->saddr = dst_hdr.saddr;
		
		fib_params.family = AF_INET6;
		*fib_dst = dst_hdr.daddr;
		// bpf_printk("ipv6 destination %pI6",fib_dst);
		// bpf_printk("ipv6 destination in hdr %pI6",&ip6h->saddr);
		fib_params.ifindex = ctx->ingress_ifindex;

		rc = bpf_fib_lookup(ctx, &fib_params, sizeof(fib_params), 0);
		bpf_printk("rc: %d",rc);
		switch (rc)
			{
			case BPF_FIB_LKUP_RET_SUCCESS: /* lookup successful */
				bpf_printk("ifindex redirect %d",fib_params.ifindex);
				memcpy(eth->h_dest, fib_params.dmac, ETH_ALEN);
				memcpy(eth->h_source, fib_params.smac, ETH_ALEN);
				// action = bpf_redirect_map(&tx_port, fib_params.ifindex, 0);
				action = bpf_redirect(fib_params.ifindex, 0);
				bpf_printk("action %d",action);
				goto out;
				break;
			case BPF_FIB_LKUP_RET_BLACKHOLE:   /* dest is blackholed; can be dropped */
			case BPF_FIB_LKUP_RET_UNREACHABLE: /* dest is unreachable; can be dropped */
			case BPF_FIB_LKUP_RET_PROHIBIT:	   /* dest not allowed; can be dropped */
				action = XDP_DROP;
				break;
			case BPF_FIB_LKUP_RET_NOT_FWDED:	/* packet is not forwarded */
				bpf_printk ("route not found, check if routing suite is working properly");
			case BPF_FIB_LKUP_RET_FWD_DISABLED: /* fwding is not enabled on ingress */
			case BPF_FIB_LKUP_RET_UNSUPP_LWT:	/* fwd requires encapsulation */
			case BPF_FIB_LKUP_RET_NO_NEIGH:		/* no neighbor entry for nh */
				bpf_printk("neigh entry missing");
			case BPF_FIB_LKUP_RET_FRAG_NEEDED:	/* fragmentation required to fwd */
				/* PASS */
				break;
			}
	}
out:
		return action;
}

SEC("xdp_pass")
int xdp_pass_func(struct xdp_md *ctx)
{
	void *data_end = (void *)(long)ctx->data_end;
	void *data = (void *)(long)ctx->data;
	struct iphdr *iph;
	struct ethhdr *eth = data;
	int action = XDP_PASS;
	__u16 h_proto;
	__u64 nh_off;
	nh_off = sizeof(*eth);
	if (data + nh_off > data_end)
	{
		action = XDP_DROP;
		goto out;
	}

	h_proto = eth->h_proto;
	// bpf_printk("XDP PASS: h proto %d",h_proto);
	if (h_proto == bpf_htons(ETH_P_IP))
	{
		// bpf_printk("XDP PASS: IPv4 packet received");
		iph = (void *)(data + sizeof(*eth));
		// void * ippointer = (void *)(data + sizeof(*eth) +1);
		// bpf_printk("ippointer - data %d",ippointer - data);
		if (iph + 1 > data_end) {
			bpf_printk("iph out of boundary");
			return -1;
		}
		// bpf_printk("XDP PASS: ipv4 src %pI4",&iph->saddr);
	}
out:
	return action;
}

char _license[] SEC("license") = "GPL";
