/* SPDX-License-Identifier: GPL-2.0-only
   Copyright (c) 2019-2022 */
#include <linux/bpf.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

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

#define IPV6_FLOWINFO_MASK bpf_htonl(0x0FFFFFFF)
#define IPV4_SRC_ADDRESS bpf_htonl(0x0a000201) // 10.0.2.1

// static __always_inline int parse_contract_spec(void *contract, void *data_end, struct meta_info *meta)
// {
// }

/* Pops the outermost VLAN tag off the packet. Returns the popped VLAN ID on
 * success or negative errno on failure.
 */
// static __always_inline int vlan_tag_pop(struct xdp_md *ctx, struct ethhdr *eth)
// {
// 	void *data_end = (void *)(long)ctx->data_end;
// 	struct ethhdr eth_cpy;
// 	struct vlan_hdr *vlh;
// 	__be16 h_proto;
// 	int vlid;

// 	if (!proto_is_vlan(eth->h_proto))
// 		return -1;

// 	/* Careful with the parenthesis here */
// 	vlh = (void *)(eth + 1);

// 	/* Still need to do bounds checking */
// 	if (vlh + 1 > data_end)
// 		return -1;

// 	/* Save vlan ID for returning, h_proto for updating Ethernet header */
// 	vlid = bpf_ntohs(vlh->h_vlan_TCI);
// 	h_proto = vlh->h_vlan_encapsulated_proto;

// 	/* Make a copy of the outer Ethernet header before we cut it off */
// 	__builtin_memcpy(&eth_cpy, eth, sizeof(eth_cpy));

// 	/* Actually adjust the head pointer */
// 	if (bpf_xdp_adjust_head(ctx, (int)sizeof(*vlh)))
// 		return -1;

// 	/* Need to re-evaluate data *and* data_end and do new bounds checking
// 	 * after adjusting head
// 	 */
// 	eth = (void *)(long)ctx->data;
// 	data_end = (void *)(long)ctx->data_end;
// 	if (eth + 1 > data_end)
// 		return -1;

// 	/* Copy back the old Ethernet header and update the proto type */
// 	__builtin_memcpy(eth, &eth_cpy, sizeof(*eth));
// 	eth->h_proto = h_proto;

// 	return vlid;
// }

/* Pushes a new VLAN tag after the Ethernet header. Returns 0 on success,
 * -1 on failure.
 */
// static __always_inline int vlan_tag_push(struct xdp_md *ctx,
// 		struct ethhdr *eth, int vlid)
// {
// 	void *data_end = (void *)(long)ctx->data_end;
// 	struct ethhdr eth_cpy;
// 	struct vlan_hdr *vlh;

// 	/* First copy the original Ethernet header */
// 	__builtin_memcpy(&eth_cpy, eth, sizeof(eth_cpy));

// 	/* Then add space in front of the packet */
// 	if (bpf_xdp_adjust_head(ctx, 0 - (int)sizeof(*vlh)))
// 		return -1;

// 	/* Need to re-evaluate data_end and data after head adjustment, and
// 	 * bounds check, even though we know there is enough space (as we
// 	 * increased it).
// 	 */
// 	data_end = (void *)(long)ctx->data_end;
// 	eth = (void *)(long)ctx->data;

// 	if (eth + 1 > data_end)
// 		return -1;

// 	/* Copy back Ethernet header in the right place, populate VLAN tag with
// 	 * ID and proto, and set outer Ethernet header to VLAN type.
// 	 */
// 	__builtin_memcpy(eth, &eth_cpy, sizeof(*eth));

// 	vlh = (void *)(eth + 1);

// 	if (vlh + 1 > data_end)
// 		return -1;

// 	vlh->h_vlan_TCI = bpf_htons(vlid);
// 	vlh->h_vlan_encapsulated_proto = eth->h_proto;

// 	eth->h_proto = bpf_htons(ETH_P_8021Q);
// 	return 0;
// }

SEC("v6_side")
int xdp_router_func(struct xdp_md *ctx)
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
                .frag_off = bpf_htons(1<<14), /* set Don't Fragment bit */
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
		// might cause with verifier as copying in else if block
		bpf_printk("IPv6 packet");
		
		__builtin_memcpy(&eth_cpy, eth, sizeof(eth_cpy));
		ip6h = data + nh_off;

		if (ip6h + 1 > data_end) {
			action = XDP_DROP;
			goto out;
		}
		dst_v4 = ip6h->daddr.s6_addr32[3];
		dst_hdr.daddr = dst_v4;
        dst_hdr.saddr = IPV4_SRC_ADDRESS; // 10.0.2.1
        dst_hdr.protocol = ip6h->nexthdr;
        dst_hdr.ttl = ip6h->hop_limit;
        dst_hdr.tos = ip6h->priority << 4 | (ip6h->flow_lbl[0] >> 4);
        dst_hdr.tot_len = bpf_htons(bpf_ntohs(ip6h->payload_len) + sizeof(dst_hdr));


		/* Actually adjust the head pointer */
		if (bpf_xdp_adjust_head(ctx, (int)sizeof(*ip6h) - (int)sizeof(&dst_hdr)))
		// if (bpf_xdp_adjust_head(ctx, -(int)sizeof(&dst_hdr)))
			return -1;
		/* Need to re-evaluate data *and* data_end and do new bounds checking
		* after adjusting head
		*/
		eth = (void *)(long)ctx->data;
		data_end = (void *)(long)ctx->data_end;
		if (eth + 1 > data_end)
			return -1;

		/* Copy back the old Ethernet header and update the proto type */
		__builtin_memcpy(eth, &eth_cpy, sizeof(*eth));
		eth->h_proto = bpf_htons(ETH_P_IP);
		iph = (void *)(eth + sizeof(*eth) +1);

		if (iph + 1 > data_end) {
			return -1;
		}
		iph->version = dst_hdr.version;
		iph->ihl = dst_hdr.ihl;
        iph->frag_off = dst_hdr.frag_off;
		iph->daddr = dst_hdr.daddr;
        iph->saddr = dst_hdr.saddr; // 10.0.2.1
        iph->protocol = dst_hdr.protocol;
        iph->ttl = dst_hdr.ttl;
        iph->tos = dst_hdr.tos;
        iph->tot_len = dst_hdr.tot_len;

		// if (iph + sizeof(&dst_hdr) > data_end) {
		// 	__builtin_memcpy(iph, &dst_hdr, sizeof(*iph));
		// 	// action = XDP_DROP;
		// 	// goto out;
		// 	return -1;
		// }
		// __builtin_memcpy(iph, &dst_hdr, sizeof(*iph));
		// if (iph + sizeof(&dst_hdr) < data_end) {
		// 	// __builtin_memcpy(iph, &dst_hdr, sizeof(*iph));
		// 	// action = XDP_DROP;
		// 	// goto out;
		// 	return -1;
		// }
		fib_params.family = AF_INET;
		fib_params.ipv4_dst = dst_v4;
		fib_params.ifindex = ctx->ingress_ifindex;

		rc = bpf_fib_lookup(ctx, &fib_params, sizeof(fib_params), 0);
		if (rc == BPF_FIB_LKUP_RET_SUCCESS)
		{
			memcpy(eth->h_dest, fib_params.dmac, ETH_ALEN);
			memcpy(eth->h_source, fib_params.smac, ETH_ALEN);
			action = bpf_redirect_map(&tx_port, fib_params.ifindex, 0);
		}
	}
out:
		return xdp_stats_record_action(ctx, action);
}

SEC("xdp_pass")
int xdp_pass_func(struct xdp_md *ctx)
{
	return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
