/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/bpf.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if_vlan.h>
#include <arpa/inet.h>
#include <linux/ip.h>
//#include <linux/ipv6.h>
#include <linux/icmp.h>
#include <linux/icmpv6.h>
#include <linux/in.h>
#include <stdbool.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include "tayga.h"

#define OVER(x, d) (x + 1 > (typeof(x))d)
#define MAX_PACKET_OFF 0xffff
#define TEST_1 0
#define TEST_IP6_FRAG  1
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
#define NEXTHDR_NONE              59 
#define IPPROTO_ICMP6    58
#if 0
static inline int
ip6t_ext_hdr(__u8 nexthdr)
{       return (nexthdr == IPPROTO_HOPOPTS) ||
               (nexthdr == IPPROTO_ROUTING) ||
                (nexthdr == IPPROTO_FRAGMENT) ||
                (nexthdr == IPPROTO_ESP) ||
                (nexthdr == IPPROTO_AH) ||
                (nexthdr == IPPROTO_NONE) ||
                (nexthdr == IPPROTO_DSTOPTS);
}
#endif
static int handle_ip6(struct xdp_md *ctx)
{
    int index = ctx->rx_queue_index;
    void* data_end = (void*)(long)ctx->data_end;
    void* data = (void*)(long)ctx->data;
    //char *  currenthdr;
    __u8  currenthdr;
    struct ethhdr* eth = data;
    //struct ipv6hdr* ip6h =  (struct ipv6hdr*)(eth + 1);
    struct ip6 *ip6h =  (struct ip6*)(eth + 1);
    struct icmp6hdr *icmp =  (struct icmp6hdr *)(ip6h +1);
#if TEST_IP6_FRAG
    __u8  more;
    __u16 offset  ;
    struct ip6_frag *ip6_frag;
#endif 
    if (OVER(ip6h, data_end))
        return XDP_DROP;
    currenthdr = ip6h->nexthdr;
    if (currenthdr == 0x3b)
    {
        //bpf_printk("no next header");
        return XDP_PASS;
    }

    if (currenthdr == IPPROTO_ICMP6)
    {
        if (OVER(icmp, data_end))
             return XDP_DROP;
        if (icmp->icmp6_type == ICMPV6_ECHO_REQUEST)
        {
            goto user_xdp;	
          
        }
        return XDP_PASS;
    }
/*
*    if (currenthdr == IPPROTO_ICMP6 && (icmp->icmp6_type != ICMPV6_ECHO_REQUEST || icmp->icmp6_type != ICMPV6_ECHO_REPLY))
*    {
*#if 0
*	if(icmp->icmp6_type != ICMPV6_ECHO_REQUEST)
*        {
*            return XDP_PASS;
*	}
*        bpf_printk("icmp6 ping request header");
*        goto user_xdp;	
*#else
*        return XDP_PASS;
*      
*#endif
*    }
**/
/*
*    currenthdr = ip6h->nexthdr;
*    while (currenthdr != NEXTHDR_NONE && ip6t_ext_hdr(currenthdr)) {
*	 switch (currenthdr) {
*		                 case IPPROTO_FRAGMENT: 
*
*		                 default:
*	 }
*    }
*/
    if(IPPROTO_FRAGMENT == currenthdr)
    {
#if TEST_IP6_FRAG
         ip6_frag = (struct ip6_frag *)(ip6h +1);
        if (OVER(ip6_frag, data_end))
             return XDP_DROP;
	 more = ntohs(ip6_frag->offset_flags) & IP6_F_MF;
         offset = ntohs(ip6_frag->offset_flags) & IP6_F_MASK;
         bpf_printk("frag is icmp6 header, more %u, offset: %u \n",more,offset);
         if(IPPROTO_ICMP6 == ip6_frag->next_header)
	 {
#if 0
              icmp =  (struct icmp6hdr *)(ip6_frag +1);
              if (OVER(icmp, data_end))
                  return XDP_DROP;
	      // this will cause a bug, the frags of ping will not be directed to user space ,except for the first
              if (icmp->icmp6_type == ICMPV6_ECHO_REQUEST)
	      {
                 goto user_xdp;	
	      }
#else

               goto user_xdp;	
#endif
	 }
#else
         bpf_printk("ipv6 frag \n ");
         goto user_xdp;	
#endif
    }
    return XDP_PASS;
user_xdp:
    /* A set entry here means that the correspnding queue_id
     * has an active AF_XDP socket bound to it. */
    if (bpf_map_lookup_elem(&xsks_map, &index))
        return bpf_redirect_map(&xsks_map, index, 0);
    return XDP_PASS;
}
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

    if (eth->h_proto == ntohs(ETH_P_IPV6))
    {
	return handle_ip6(ctx);
    }
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
