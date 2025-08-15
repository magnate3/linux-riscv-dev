/* 
SPDX-License-Identifier: GPL-2.0 
AF_XDP的内核态程序
仅仅过滤TCP协议并发送到AF_XDP的用户态程序
*/

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/tcp.h>

#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>

struct {
	__uint(type, BPF_MAP_TYPE_XSKMAP);
	__uint(max_entries, 64);
	__type(key, int);
	__type(value, int);
} xsks_map SEC(".maps");

SEC("xdp")
int xdp_prog(struct xdp_md *ctx)
{
	__u32 off;
	//数据包的起始地址和结束地址
	void *data_end = (void *)(long)ctx->data_end;
	void *data = (void *)(long)ctx->data;
	//以太网头部
	struct ethhdr *eth = data;
	//IP头部
	struct iphdr *ip = data + sizeof(*eth);
	//TCP头部
	// struct tcphdr *tcp = data + sizeof(*eth) + sizeof(*ip);
	//偏移量
	off = sizeof(struct ethhdr);
	if (data + off > data_end) // To pass verifier
		return XDP_PASS;
	//判断是否为IPV4协议
	if (bpf_htons(eth->h_proto) == ETH_P_IP) {
        off += sizeof(struct iphdr);
		if (data + off > data_end) // To pass verifier
			return XDP_PASS;
			//判断是否为TCP协议
		if (ip->protocol == IPPROTO_TCP) {
            int idx = ctx->rx_queue_index;
            /* 如果idx对应网卡队列已绑定xsk并更新到了xsks_map中，数据包就会被redirect到该xsk */
			if (bpf_map_lookup_elem(&xsks_map, &idx)) {
				return bpf_redirect_map(&xsks_map, idx, 0);
			}
		}
	}
	return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
// //判断是否为HTTP GET请求
// if (bpf_htons(tcp->dest) == 80 && tcp->syn == 1 && tcp->ack == 0) {
// 	int idx = ctx->rx_queue_index;
// 	/* 如果idx对应网卡队列已绑定xsk并更新到了xsks_map中，数据包就会被redirect到该xsk */
// 	if (bpf_map_lookup_elem(&xsks_map, &idx)) {
// 	return bpf_redirect_map(&xsks_map, idx, 0);
// 	}
// }