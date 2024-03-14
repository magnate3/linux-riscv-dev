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
#define MAX_PROGS 4
int handle_icmp(struct xdp_md *ctx);
int handle_udp(struct xdp_md *ctx);
int handle_tcp(struct xdp_md *ctx);
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

struct {
	__uint(type, BPF_MAP_TYPE_PROG_ARRAY);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, MAX_PROGS);
	__array(values, int());
} packet_processing_progs SEC(".maps");
SEC("xdp/icmp")
#if TEST_1
int handle_icmp(struct xdp_md *ctx) {
    // ICMP包的处理逻辑
    bpf_printk("new icmp packet captured (XDP)\n");
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
}
#else
int handle_icmp(struct xdp_md *ctx) {
    // TCP包的处理逻辑
    bpf_printk("new icmp packet captured (XDP)\n");
    return XDP_PASS;
}
#endif

SEC("xdp/tcp")
int handle_tcp(struct xdp_md *ctx) {
    // TCP包的处理逻辑
    bpf_printk("new tcp packet captured (XDP)\n");
    return XDP_PASS;
}

SEC("xdp/udp")
int handle_udp(struct xdp_md *ctx) {
    // UDP包的处理逻辑
    bpf_printk("new udp packet captured (XDP)\n");
    return XDP_PASS;
}



SEC("xdp_classifier")
int packet_classifier(struct xdp_md *ctx)  {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;
    struct iphdr *ip;

    // 检查是否有足够的数据空间
    if ((void *)(eth + 1) > data_end) {
        return XDP_ABORTED;
    }

    // 确保这是一个IP包
    if (eth->h_proto != 8) {
        return XDP_PASS;
    }

    ip = (struct iphdr *)(eth + 1);

    // 检查IP头部是否完整
    if ((void *)(ip + 1) > data_end) {
        return XDP_ABORTED;
    }
    bpf_printk("protocol: %d\n", ip->protocol);
    bpf_printk("icmp: %d,tcp:%d,udp:%d\n", IPPROTO_ICMP, IPPROTO_TCP, IPPROTO_UDP);
    switch (ip->protocol) {
        case IPPROTO_ICMP:
            bpf_printk("icmp\n");
            bpf_tail_call(ctx, &packet_processing_progs, 0);
            break;
        case IPPROTO_TCP:
            bpf_printk("tcp\n");
            bpf_tail_call(ctx, &packet_processing_progs, 1);
            break;
        case IPPROTO_UDP:
            bpf_printk("udp\n");
            bpf_tail_call(ctx, &packet_processing_progs, 2);
            break;
        default:
            bpf_printk("unknown protocol\n");
            break;
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
