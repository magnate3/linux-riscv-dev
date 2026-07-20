#define KBUILD_MODNAME "xdp_demo2"
#include <linux/bpf.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#include "bpf_helpers.h"
#include "bpf_endian.h"
#define SEC(NAME) __attribute__((section(NAME), used))
//定义一个 map,用于统计协议收包统计
struct bpf_map_def SEC("maps") rxcnt = {
 .type = BPF_MAP_TYPE_PERCPU_ARRAY,
 .key_size = sizeof(u32),
 .value_size = sizeof(long),
 .max_entries = 256,
};

SEC("xdp")
int xdp_drop_the_world(struct xdp_md *ctx) {
 void *data = (void *)(long)ctx->data;
 void *data_end = (void *)(long)ctx->data_end;
 int ipsize = 0;
 __u32 idx;
 u16 port;
 long *value;
 struct ethhdr  *eth = data;
 struct iphdr *ip;
 struct tcphdr *tcp;
 ipsize = sizeof(*eth);
 ip = data + ipsize;
 ipsize += sizeof(struct iphdr);
 if(data + ipsize > data_end){
  return XDP_DROP;
 }
 idx = ip->protocol;
    //判断协议字段，若为icmp则drop，若为TCP则屏蔽掉22端口
 switch(idx){
        case IPPROTO_ICMP:
                value = bpf_map_lookup_elem(&rxcnt,&idx);
  if(value)
  (*value) += 1;  //icmp协议丢包记录++
  return XDP_DROP;
#if 0
 case IPPROTO_TCP:
  tcp = data + ipsize;
  if(tcp + 1 > data_end)
   return XDP_DROP;
  port = bpf_ntohs(tcp->dest);
  if(port == 22){
   value = bpf_map_lookup_elem(&rxcnt,&idx);
   if(value)
     (*value) += 1;  //tcp协议22端口丢包记录++
   return XDP_DROP;
  }
#endif

 }
        return XDP_PASS; 
}
char _license[] SEC("license") = "GPL";
