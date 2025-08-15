#include <linux/bpf.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/in.h>
#include <linux/if_ether.h>
#define SEC(NAME) __attribute__((section(NAME), used))

SEC("xdp")
int xdp_drop_the_world(struct xdp_md *ctx) {
    //从xdp程序的上下文参数获取数据包的起始地址和终止地址
 void *data = (void *)(long)ctx->data;
 void *data_end = (void *)(long)ctx->data_end;
 int ipsize = 0;
 __u32 idx;
    //以太网头部
 struct ethhdr  *eth = data;
    //ip头部
 struct iphdr *ip;
 struct tcphdr *tcp;
    //以太网头部偏移量
 ipsize = sizeof(*eth);
 ip = data + ipsize;
 ipsize += sizeof(struct iphdr);
    //异常数据包，丢弃
 if(data + ipsize > data_end){
  return XDP_DROP;
 }
    //从ip头部获取上层协议
 idx = ip->protocol;
 //如果是icmp协议，则drop掉
    if(idx == IPPROTO_ICMP){    
 return XDP_DROP;
    }
    return XDP_PASS; 
         
}

char _license[] SEC("license") = "GPL";