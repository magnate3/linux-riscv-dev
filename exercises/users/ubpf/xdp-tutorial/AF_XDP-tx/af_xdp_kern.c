// This is a simple ICMP responder that responds to ICMP echo requests

#include <arpa/inet.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/icmp.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/tcp.h>

#define OVER(x, d) (x + 1 > (typeof(x))d)
#define bpf_memcpy __builtin_memcpy
struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__type(key, __u32);
	__type(value, __u32);
	__uint(max_entries, 64);
} xsks_map SEC(".maps");

/**
 * Replaces a 16-bit value in a checksum with a new value.
 *
 * @param sum Pointer to the checksum value to be updated.
 * @param old The old 16-bit value to be replaced.
 * @param new The new 16-bit value to replace the old value.
 */
static inline void csum_replace2(uint16_t* sum, uint16_t old, uint16_t new) {
    uint16_t csum = ~*sum;  // 1's complement of the checksum (flip all the bits)

    csum += ~old;                   // Subtract the old value from the checksum
    csum += csum < (uint16_t)~old;  // If the subtraction overflowed, add 1 to the checksum

    csum += new;                    // Add the new value to the checksum
    csum += csum < (uint16_t) new;  // If the addition overflowed, add 1 to the checksum

    *sum = ~csum;  // 1's complement of the checksum
}

/**
 * Swaps the source and destination MAC addresses in the Ethernet header.
 *
 * @param eth Pointer to the Ethernet header structure.
 */
static inline void swap_src_dst_mac(struct ethhdr* eth) {
    __u8 tmp[ETH_ALEN];
    bpf_memcpy(tmp, eth->h_source, ETH_ALEN);
    bpf_memcpy(eth->h_source, eth->h_dest, ETH_ALEN);
    bpf_memcpy(eth->h_dest, tmp, ETH_ALEN);
}

/**
 * Swaps the source and destination IP addresses in the given IP header.
 *
 * @param ip The IP header to modify.
 */
static inline void swap_src_dst_ip(struct iphdr* ip) {
    __u32 tmp = ip->saddr;
    ip->saddr = ip->daddr;
    ip->daddr = tmp;
}

SEC("prog")
int xdp_responder(struct xdp_md* ctx) {
    // data points to the start of the packet data, and data_end points to the
    // end of the packet data data_end - data is the length of the packet
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

    // Swap the source and destination MAC addresses
    swap_src_dst_mac(eth);

    // Swap the source and destination IP addresses
    swap_src_dst_ip(iph);

    // Set the ICMP type to ICMP_ECHOREPLY
    icmp->type = ICMP_ECHOREPLY;

    // Recalculate the ICMP checksum
    csum_replace2(&icmp->checksum, ICMP_ECHO, ICMP_ECHOREPLY);

    // Return XDP_TX to send the packet out
    return XDP_TX;
}

char _license[] SEC("license") = "GPL";
