/**
 * @file    lkm.c
 * @author  Dmytro Shytyi
 * @date    14 Octobre 2018
 * @version 0.1
 * @brief  A "Hello World!" loadable kernel module (LKM) that sends Hello World udp packet.
 * @see https://dmytro.shytyi.net/ for a full description and follow-up descriptions.
*/
 
#include <linux/init.h>             // Macros used to mark up functions e.g., __init __exit
#include <linux/module.h>           // Core header for loading LKMs into the kernel
#include <linux/kernel.h>           // Contains types, macros, functions for the kernel

#include <linux/types.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/fcntl.h>
#include <linux/socket.h>
#include <linux/in.h>
#include <linux/inet.h>
#include <linux/skbuff.h>
#include <linux/if_arp.h>
#include <linux/netdevice.h> 
#include <linux/etherdevice.h>
#include <linux/netdevice.h> 
#include <linux/etherdevice.h>
//#include <linux/string.h>
#include <linux/ip.h> 
#include <linux/udp.h>
            ///< The version of the module
 
 
extern struct sk_buff *  build_udp_skb(struct net_device* dev, uint8_t dest_addr[ETH_ALEN], uint16_t proto);

unsigned int inet_addr(char *str)
{
    int a, b, c, d;
    char arr[4];
    sscanf(str, "%d.%d.%d.%d", &a, &b, &c, &d);
    arr[0] = a; arr[1] = b; arr[2] = c; arr[3] = d;
    return *(unsigned int *)arr;
}

extern struct sk_buff *  build_udp_skb(struct net_device* dev, uint8_t dest_addr[ETH_ALEN], uint16_t proto)
{
  unsigned char* data;
  
  char *srcIP = "192.168.1.222";
  char *dstIP = "192.168.1.111";
  char *hello_world = ">>> KERNEL sk_buff Hello World <<< by Dmytro Shytyi";
  char *hello_world2 = "### 123456 SK_BUFF good night ### on sunday goodby";
  int data_len = 51;

  int udp_header_len = 8;
  int udp_payload_len = data_len;
  int udp_total_len = udp_header_len+udp_payload_len;

  int ip_header_len = 20;
  int ip_payload_len = udp_total_len;
  int ip_total_len = ip_header_len + ip_payload_len;

/* skb */
  struct sk_buff* skb = alloc_skb(ETH_HLEN+ip_total_len, GFP_ATOMIC);//allocate a network buffer
  skb->dev = dev;
  skb->pkt_type = PACKET_OUTGOING;
  skb_reserve(skb, ETH_HLEN+ip_header_len+udp_header_len);//adjust headroom
/* allocate space to data and write it */
  data = skb_put(skb,udp_payload_len);
  memcpy(data, hello_world, data_len);
  memcpy(data, hello_world2, data_len);
/* UDP header */
  struct udphdr* uh = (struct udphdr*)skb_push(skb,udp_header_len);
  uh->len = htons(udp_total_len);
  uh->source = htons(15934);
  uh->dest = htons(15904);

/* IP header */
  struct iphdr* iph = (struct iphdr*)skb_push(skb,ip_header_len);
  iph->ihl = ip_header_len/4;//4*5=20 ip_header_len
  iph->version = 4; // IPv4u
  iph->tos = 0; 
  iph->tot_len=htons(ip_total_len); 
  iph->frag_off = 0; 
  iph->ttl = 64; // Set a TTL.
  iph->protocol = IPPROTO_UDP; //  protocol.
  iph->check = 0; 
  iph->saddr = inet_addr(srcIP);
  iph->daddr = inet_addr(dstIP);

  /* caculate checksum */
  uh->check =0; 
  skb->csum = skb_checksum(skb, iph->ihl*4, skb->len - iph->ihl*4, 0);
  iph->check = ip_fast_csum(iph, iph->ihl);
  uh->check = csum_tcpudp_magic(inet_addr(srcIP), inet_addr(dstIP), skb->len - iph->ihl * 4, IPPROTO_UDP, skb->csum);
  skb->ip_summed = CHECKSUM_NONE;


  /*changing Mac address */
  struct ethhdr* eth = (struct ethhdr*)skb_push(skb, sizeof (struct ethhdr));//add data to the start of a buffer
  skb->protocol = eth->h_proto = htons(proto);
  skb->no_fcs = 1;
  memcpy(eth->h_source, dev->dev_addr, ETH_ALEN);
  memcpy(eth->h_dest, dest_addr, ETH_ALEN);
 

  skb->pkt_type = PACKET_OUTGOING;
  return skb;
} 

