/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/bpf.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <arpa/inet.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <linux/in.h>
#include <stdbool.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#define OVER(x, d) (x + 1 > (typeof(x))d)
#define TEST_1 0
struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 64);
} xsks_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 64);
} xdp_stats_map SEC(".maps");

#if TEST_1
/* to u64 in host order */
static inline __u64 ether_addr_to_u64(const __u8 *addr)
{
	__u64 u = 0;
	int i;

	for (i = ETH_ALEN - 1; i >= 0; i--)
		u = u << 8 | addr[i];
	return u;
}
#endif
SEC("xdp")
int xdp_sock_prog(struct xdp_md *ctx)
{
     
#if (!TEST_1)
    int index = ctx->rx_queue_index;
    void* data_end = (void*)(long)ctx->data_end;
    void* data = (void*)(long)ctx->data;

    struct ethhdr* eth = data;
    struct iphdr* iph = (struct iphdr*)(eth + 1);

    // Sanity checks
    if (OVER(eth, data_end))
        return XDP_DROP;

    if (eth->h_proto != ntohs(ETH_P_IP))
        return XDP_PASS;

    if (OVER(iph, data_end))
        return XDP_DROP;

    // Check if the packet is an ICMP packet
    if (iph->protocol != IPPROTO_ICMP) {
        return XDP_PASS;
    }

    struct icmphdr* icmp = (struct icmphdr*)(iph + 1);
    if (OVER(icmp, data_end))
        return XDP_DROP;

    // Check if the packet is an ICMP echo request
    if (icmp->type != ICMP_ECHO) {
        return XDP_PASS;
    }
    /* A set entry here means that the correspnding queue_id
     * has an active AF_XDP socket bound to it. */
    if (bpf_map_lookup_elem(&xsks_map, &index))
        return bpf_redirect_map(&xsks_map, index, 0);

    return XDP_PASS;
#else
        int index = ctx->rx_queue_index;
	void *data = (void *)(long)ctx->data;
	void *data_end = (void *)(long)ctx->data_end;
	struct ethhdr *eth = data;
	__u64 offset = sizeof(*eth);

	if ((void *)eth + offset > data_end)
		return 0;

        bpf_printk("queue index %d \t",index);
	bpf_printk("src: %llu, dst: %llu, proto: %u\n", ether_addr_to_u64(eth->h_source),\
		       	ether_addr_to_u64(eth->h_dest), bpf_ntohs(eth->h_proto));
        return XDP_DROP;
#endif
}

char _license[] SEC("license") = "GPL";
