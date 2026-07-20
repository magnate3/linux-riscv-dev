/***********************************************************************
* File: sendUDPWithkernelModule.c
* Copy Right: http://blogold.chinaunix.net/u3/117455/showart_2285390.html
* Author: lion3875 <lion3875@gmail.com>
* Create Date: Unknow
* Abstract Description:
*             To send a UDP packet to another in kernel space.
*
*------------------------Revision History------------------------
* No.    Date        Revised By   Description
* 1      2011/7/28   Sam          make the program work on my machines
*                                 IP address: [10.14.1.122] -> [10.14.1.21]
*                                 MAC address:
*                                 00:e0:4d:8b:3c:d7 -> 20:cf:30:57:1a:18     
* 2      2011/8/7    Sam          To correct the checksum.(fail)
* 3      2011/8/9    Sam          To correct the checksum.(success)
***********************************************************************/





#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/workqueue.h>
#include <linux/in.h>
#include <linux/inet.h>
#include <linux/socket.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/sock.h>



#define IF_NAME "enp125s0f0"
#define SIP     "10.10.16.251"
#define DIP     "10.10.16.82"
#define SPORT   31900
#define DPORT   31900


#define SRC_MAC {0xb0, 0x08, 0x75, 0x5f, 0xb8, 0x5b}
// 48:57:02:64:e7:ab
#define DST_MAC {0x48, 0x57, 0x02, 0x64, 0xe7, 0xab}

struct socket *sock;


static void sock_init(void)
{
        /* "struct ifreq" is defined in linux/if.h.
         * Interface request structure used for socket
         * ioctl's.  All interface ioctl's must have parameter
         * definitions which begin with ifr_name.  The
         * remainder may be interface specific. */
        struct ifreq ifr;


        /* sock_create_kern():创建socket结构.
         * SIOCSIFNAME is defined in include/linux/sockios.h.
         * It is used to set interface name. */
        sock_create_kern(&init_net,PF_INET, SOCK_DGRAM, 0, &sock);
        // copy the interface name to the ifrn_name.
        strcpy(ifr.ifr_name, IF_NAME);
        kernel_sock_ioctl(sock, SIOCSIFNAME, (unsigned long) &ifr);
}


static void send_by_skb(void)
{
        struct net_device *netdev = NULL;
        struct net *net = NULL;
        struct sk_buff *skb = NULL;
        struct ethhdr *eth_header = NULL;
        struct iphdr *ip_header = NULL;
        struct udphdr *udp_header = NULL;
        __be32 dip = in_aton(DIP);
        __be32 sip = in_aton(SIP);
        u8 buf[16] = {"hello world"};
        u16 data_len = sizeof(buf);
        //u16 expand_len = 16;    /* for skb align */
        u8 *pdata = NULL;
        u32 skb_len;
        u8 dst_mac[ETH_ALEN] = DST_MAC;    /* dst MAC */
        u8 src_mac[ETH_ALEN] = SRC_MAC;    /* src MAC */


        /* construct skb
         * sock_net() is defined in include/net/sock.h
         * dev_get_by_name()函数用来取得设备指针,使用该函数
         * 后一定要使用dev_put()函数取消设备引用. */
        sock_init();
        net = sock_net((const struct sock *) sock->sk);
        netdev = dev_get_by_name(net, IF_NAME);


        /* LL_RESERVED_SPACE is defined in include/netdevice.h. */
        /*skb_len = LL_RESERVED_SPACE(netdev) + sizeof(struct iphdr)
                  + sizeof(struct udphdr) + data_len + expand_len;*/
        skb_len = data_len 
		+ sizeof(struct iphdr)
                + sizeof(struct udphdr) 
		+ LL_RESERVED_SPACE(netdev);
        printk("iphdr	: %d\n", sizeof(struct iphdr));
        printk("udphdr	: %d\n", sizeof(struct udphdr));
        printk("data_len: %d\n", data_len);
        printk("skb_len	: %d\n", skb_len);


        /* dev_alloc_skb是一个缓冲区分配函数,主要被设备驱动使用.
         * 这是一个alloc_skb的包装函数, 它会在请求分配的大小上增加
         * 16 Bytes的空间以优化缓冲区的读写效率.*/
        skb = dev_alloc_skb(skb_len);
        if (!skb) {
                return;
        }


        /* fill the skb.具体参照struct sk_buff.
         * skb_reserve()用来为协议头预留空间.
         * PACKET_OTHERHOST: packet type is "to someone else".
         * ETH_P_IP: Internet Protocol packet.
         * CHECKSUM_NONE表示完全由软件来执行校验和. */
        skb_reserve(skb, LL_RESERVED_SPACE(netdev));
        skb->dev = netdev;
        skb->pkt_type = PACKET_OTHERHOST;
        skb->protocol = htons(ETH_P_IP);
        skb->ip_summed = CHECKSUM_NONE;
        skb->priority = 0;

        /* 分配内存给ip头 */
        skb_set_network_header(skb, 0);
        skb_put(skb, sizeof(struct iphdr));

        /* 分配内存给udp头 */
        skb_set_transport_header(skb, sizeof(struct iphdr));
        skb_put(skb, sizeof(struct udphdr));
        
    	/* construct udp header in skb */
        udp_header = udp_hdr(skb);
    	udp_header->source = htons(SPORT);
    	udp_header->dest = htons(DPORT);
        udp_header->check = 0;
       
        /* construct ip header in skb */
	ip_header = ip_hdr(skb);
	ip_header->version = 4;
	ip_header->ihl = sizeof(struct iphdr) >> 2;
	ip_header->frag_off = 0;
	ip_header->protocol = IPPROTO_UDP;
	ip_header->tos = 0;
	ip_header->daddr = dip;
	ip_header->saddr = sip;
	ip_header->ttl = 0x40;
	ip_header->tot_len = htons(skb->len);
	ip_header->check = 0;
	
	
	/* caculate checksum */
	skb->csum = skb_checksum(skb, ip_header->ihl*4, skb->len-ip_header->ihl*4, 0);
	ip_header->check = ip_fast_csum(ip_header, ip_header->ihl);
	udp_header->check = csum_tcpudp_magic(sip, dip, skb->len-ip_header->ihl*4, IPPROTO_UDP, skb->csum);

        /* insert data in skb */
        pdata = skb_put(skb, data_len);
        if (pdata) {
                memcpy(pdata, buf, data_len);
        }
        printk("payload:%20s\n", pdata);

        /* construct ethernet header in skb */
        eth_header = (struct ehthdr *)skb_push(skb, ETH_HLEN);
        memcpy(eth_header->h_dest, dst_mac, ETH_ALEN);
        memcpy(eth_header->h_source, src_mac, ETH_ALEN);
        eth_header->h_proto = htons(ETH_P_IP);       

        /* send packet */
        if (dev_queue_xmit(skb) < 0) {
                dev_put(netdev);
                kfree_skb(skb);
                printk("send packet by skb failed.\n");
                return;
        }
        printk("send packet by skb success.\n");
}

static int __init sendUDP_init(void)
{
        printk("testmod kernel module load!\n");
        send_by_skb();

        return 0;
}

static void __exit sendUDP_exit(void)
{
        sock_release(sock);
        printk("testmod kernel module removed!\n");
}

module_init(sendUDP_init);
module_exit(sendUDP_exit);

MODULE_LICENSE("GPL");
