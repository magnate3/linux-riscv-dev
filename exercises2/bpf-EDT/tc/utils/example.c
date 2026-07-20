#include <linux/types.h>

#include <bpf/bpf_helpers.h>
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/pkt_cls.h>
#include <linux/swab.h>
#include <stdint.h>
#include <linux/ip.h>

struct bpf_elf_map {
        __u32 type;
        __u32 size_key;
        __u32 size_value;
        __u32 max_elem;
        __u32 flags;
        __u32 id;
        __u32 pinning;
        __u32 inner_id;
        __u32 inner_idx;
};

#define NS_PER_SEC      1000000000ULL
// #define PIN_GLOBAL_NS   2

#define MY_CUSTOM_NS 5



#ifndef __section
# define __section(NAME)                  \
   __attribute__((section(NAME), used))
#endif

enum bpf_pin_type {
	PIN_NONE = 0,
	PIN_OBJECT_NS,
	PIN_GLOBAL_NS,
	PIN_CUSTOM_NS,
};

struct bpf_elf_map __section("maps") rate_map = {
        .type           = BPF_MAP_TYPE_HASH,
        .size_key       = sizeof(__u32),
        .size_value     = sizeof(__u64),
        .pinning        = PIN_GLOBAL_NS,
        .max_elem       = 16,
};

struct bpf_elf_map __section("maps") tstamp_map = {
        .type           = BPF_MAP_TYPE_HASH,
        .size_key       = sizeof(__u32),
        .size_value     = sizeof(__u64),
        .max_elem       = 16,
};
                        
int classifier(struct __sk_buff *skb)
{
        void *data_end = (void *)(unsigned long long)skb->data_end;
        __u64 *rate, *tstamp, delay_ns, now, init_rate = 12500000;    /* 100 Mbits/sec */
        void *data = (void *)(unsigned long long)skb->data;
        struct iphdr *ip = data + sizeof(struct ethhdr);
        struct ethhdr *eth = data;
        __u64 len = skb->len;
        long ret;

        now = bpf_ktime_get_ns();

        if (data + sizeof(struct ethhdr) > data_end)
                return TC_ACT_OK;
        if (eth->h_proto != ___constant_swab16(ETH_P_IP))
                return TC_ACT_OK;
        if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end)
                return TC_ACT_OK;

        rate = bpf_map_lookup_elem(&rate_map, &ip->daddr);
        if (!rate) {
                bpf_map_update_elem(&rate_map, &ip->daddr, &init_rate, BPF_ANY);
                bpf_map_update_elem(&tstamp_map, &ip->daddr, &now, BPF_ANY);
                return TC_ACT_OK;
        }

        delay_ns = skb->len * NS_PER_SEC / (*rate);

        tstamp = bpf_map_lookup_elem(&tstamp_map, &ip->daddr);
        if (!tstamp)    /* unlikely */
                return TC_ACT_OK;

        if (*tstamp < now) {
                *tstamp = now + delay_ns;
                skb->tstamp = now;
                return TC_ACT_OK;
        }

        skb->tstamp = *tstamp;
        __sync_fetch_and_add(tstamp, delay_ns);

        return TC_ACT_OK;
}

char _license[] SEC("license") = "GPL";