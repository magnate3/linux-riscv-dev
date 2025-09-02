//go:build ignore

#include "bpf_endian.h"
#include "common.h"
// #include <linux/bpf.h>
// #include <linux/pkt_cls.h>

char __license[] SEC("license") = "GPL";

#define NS_PER_SEC 1000000000ULL
#define PIN_GLOBAL_NS 2

#ifndef __section
#define __section(NAME) __attribute__((section(NAME), used))
#endif

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16);
	__type(key, __u32);
	__type(value, __u64);
} rate_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 16);
	__type(key, __u32);
	__type(value, __u64);
} tstamp_map SEC(".maps");

SEC("classifier/cls")
int classifier(struct __sk_buff *skb) {
	void *data_end = (void *)(unsigned long long)skb->data_end;
	void *data     = (void *)(unsigned long long)skb->data;
	__u64 *rate, *tstamp, delay_ns, now, init_rate = 12500000; /* 100 Mbits/sec */
	struct iphdr *ip   = data + sizeof(struct ethhdr);
	struct ethhdr *eth = data;
	// __u64 len          = skb->len;
	// long ret;

	now = bpf_ktime_get_ns();

	if (data + sizeof(struct ethhdr) > data_end)
		return TC_ACT_OK;
	if (eth->h_proto != bpf_htons(ETH_P_IP))
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
	if (!tstamp) /* unlikely */
		return TC_ACT_OK;
	if (*tstamp < now) {
		*tstamp     = now + delay_ns;
		skb->tstamp = now;
		return TC_ACT_OK;
	}

	skb->tstamp = *tstamp;
	__sync_fetch_and_add(tstamp, delay_ns);

	return TC_ACT_OK;
}
