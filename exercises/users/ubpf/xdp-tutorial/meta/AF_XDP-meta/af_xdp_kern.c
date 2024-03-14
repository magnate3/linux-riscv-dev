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
#define LATENCY_MS 200
#define __round_mask(x, y) ((__typeof__(x))((y) - 1))
#define round_up(x, y) ((((x) - 1) | __round_mask(x, y)) + 1)
#define ctx_ptr(ctx, mem) (void *)(unsigned long)ctx->mem
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
    __u32 *val;
    const int siz = sizeof(*val);
    if (bpf_xdp_adjust_meta(ctx, -siz) != 0)
	 return XDP_PASS;
    data = ctx_ptr(ctx, data); // required to re-obtain data pointer
    void *data_meta = ctx_ptr(ctx, data_meta);
    val = (typeof(val))data_meta;

    if ((void *)(val + 1) > data)
          return XDP_PASS;

    *val = LATENCY_MS;
    /* A set entry here means that the correspnding queue_id
     * has an active AF_XDP socket bound to it. */
    if (bpf_map_lookup_elem(&xsks_map, &index))
        return bpf_redirect_map(&xsks_map, index, 0);

    return XDP_PASS;
#else
    return XDP_PASS;
#endif
}

char _license[] SEC("license") = "GPL";
