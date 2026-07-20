// SPDX-License-Identifier: GPL-2.0
#include <linux/bpf.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/types.h>
#include <stddef.h>
#include <memory.h>
#include <sys/types.h>

#define RR_LB 0
#define MAX_SOCKS 4

#define SEC(NAME) __attribute__((section(NAME), used))

#define __uint(name, val) int(*(name))[val]
#define __type(name, val) typeof(val) *(name)
#define __array(name, val) typeof(val) *(name)[]
//https://raw.githubusercontent.com/torvalds/linux/v4.19/tools/testing/selftests/bpf/bpf_helpers.h
static void *(*bpf_map_lookup_elem)(void *map, void *key) =
	(void *) BPF_FUNC_map_lookup_elem;
static int (*bpf_redirect_map)(void *map, int key, int flags) =
	(void *) BPF_FUNC_redirect_map;
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(int));
    __uint(max_entries, 1);
}qidconf_map SEC(".maps");

struct{
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(int));
	__uint(max_entries, MAX_SOCKS);
}xsks_map SEC(".maps");

struct{
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(int));
    __uint(max_entries, 1);
}rr_map SEC(".maps");

SEC("xdp_sock")
int xdp_sock_prog(struct xdp_md *ctx)
{
    int ipsize = 0;
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
#if RR_LB
	int *qidconf, key = 0, idx=0;
	unsigned int *rr;
#else
	int idx=0;
#endif
    struct ethhdr *eth = data;
    ipsize = sizeof(*eth);
    struct iphdr *ip = data + ipsize;
    ipsize += sizeof(struct iphdr);
    if (data + ipsize > data_end) {
    // not an ip packet, too short. Pass it on
        return XDP_PASS;
    }
    
    // technically, we should also check if it is an IP packet by
    // checking the ethernet header proto field ...
    if (ip->protocol == IPPROTO_ICMP) {
        /*
        qidconf = bpf_map_lookup_elem(&qidconf_map, &key);
        if (!qidconf)
            return XDP_ABORTED;
        if (*qidconf != ctx->rx_queue_index)
            return XDP_PASS;
        */
    #if RR_LB /* NB! RR_LB is configured in xdpsock.h */
        rr = bpf_map_lookup_elem(&rr_map, &key);
        if (!rr)
            return XDP_ABORTED;
        *rr = (*rr + 1) & (MAX_SOCKS - 1);
        idx = *rr;
    #endif
        return bpf_redirect_map(&xsks_map, idx, 0);    
    }
    return XDP_PASS;

}
char _license[] SEC("license") = "GPL";
//clang -g -c -O2 -target bpf -c xdpsock_kern.c -o xdpsock_kern.o
