
/*
 *  Copyright (c) 2011 Finnbarr P. Murphy except for the in_cksum
 *  routine which is from BSD sources. All rights reserved.
 * 
 *  Demonstrates how to spoof an IPv4 ARP and ICMP response
 *  using PF_PACKET and the GNU/Linux Packet Filter
 *
 *  Usage: example device address
 *         example eth0  192.168.0.119
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <libgen.h>
#include <unistd.h>
 
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netpacket/packet.h>
#include <linux/ip.h>
#include <linux/filter.h>
#include <linux/icmp.h>
 
#define MAXPACKETSIZE 200
#define ARPOP_REPLY 2
#define ARPHDR_ETHER 1
#define ETH_ALEN 6
#define IP_ALEN 4
#define IP_DOTLEN 15
 
// use our own IPv4 arp header structure
struct arphdr
{
    unsigned short hw_type;               // hardware type
    unsigned short proto_type;            // protocol type
    char ha_len;                          // hardware address length
    char pa_len;                          // protocol address length
    unsigned short opcode;                // arp opcode
    unsigned char src_addr[ETH_ALEN];     // source MAC address
    unsigned char src_ip[IP_ALEN];        // source IP address
    unsigned char dst_add[ETH_ALEN];      // destination MAC address
    unsigned char dst_ip[IP_ALEN];        // destination IP address
};
 
 
//
// filter out all except broadcast and unicast (BPF format)
//
struct sock_filter macfilter[] =
{
    BPF_STMT(BPF_LD + BPF_W + BPF_ABS, 2),                      // A <- P[2:4]
    BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, 0xffffffff, 0, 2),      // if A != 0xffffffff GOTO LABEL
    BPF_STMT(BPF_LD + BPF_H + BPF_ABS, 0),                      // A <- P[0:2]
    BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, 0x0000ffff, 2, 0),      // if A == 0xffff GOTO ACCEPT
    // LABEL
    BPF_STMT(BPF_LD + BPF_B + BPF_ABS, 0),                      // A <- P[0:1]
    BPF_JUMP(BPF_JMP + BPF_JSET + BPF_K, 0x01, 0, 1),           // if !(A & 1) GOTO REJECT
    // ACCEPT
    BPF_STMT(BPF_RET, 1514),                                    // accept packet
    // REJECT
    BPF_STMT(BPF_RET, 0),                                       // reject packet
};
 
struct sock_filter promiscfilter[] = {
    BPF_STMT(BPF_RET, 1514)
};
 
 
char *
ipaddr_string(char *in)
{
    static char buf[IP_DOTLEN + 1];
    unsigned char *p = in;
 
    snprintf(buf, sizeof(buf), "%d.%d.%d.%d", p[0], p[1], p[2], p[3]);
 
    return (buf);
}
 
 
//
//  straight from the BSD source code
//
uint16_t
in_cksum(unsigned char *addr,
         int len)
{
    int nleft = len;
    const uint16_t *w = (const uint16_t *)addr;
    uint32_t sum = 0;
    uint16_t answer = 0;
 
    while (nleft > 1)  {
        sum += *w++;
        nleft -= 2;
    }
 
    // mop up an odd byte, if necessary
    if (nleft == 1) {
        *(unsigned char *)(&answer) = *(const unsigned char *)w ;
        sum += answer;
    }
 
    // add back carry outs from top 16 bits to low 16 bits
    sum = (sum & 0xffff) + (sum >> 16);
    sum += (sum >> 16);
 
    answer = ~sum;              // truncate to 16 bits
    return answer;
}
 
 
void
usage(char *prog)
{
    printf("Usage: %s interfacename ipaddress (e.g. eth0 192.168.0.119)\n", basename(prog));
}
 
 
int
main(int argc,
     char **argv)
{
    unsigned char arppacket[sizeof(struct arphdr) + sizeof(struct ether_header)];
    char packet[MAXPACKETSIZE], smac[ETH_ALEN];
    struct ether_header *eth, *spoof_eth;
    struct arphdr *arp, *spoof_arp;
    struct iphdr *iphdr, *spoof_iph;
    struct icmphdr *icmphdr, *spoof_icmphdr;
    struct sockaddr addr;
    struct sockaddr_ll lladdr;
    struct sock_filter *filter;
    struct sock_fprog  fcode;
    struct packet_mreq mr;
    struct ifreq iface;
    char *interface, *spoof_packet;
    unsigned int temp32;
    unsigned short temp16;
    int packetsize = MAXPACKETSIZE, spoof_packetsize;
    int sd, n;
    int optionLen = 0;
 
 
    if (argc < 3) {
        usage(argv[0]);
        exit(1);
    }
 
    // check if root
    if (geteuid() || getuid()) {
        printf("ERROR: You must be root to use this utility\n");
        exit(1);
    }
    interface = argv[1];
 
   // open PACKET socket
    if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0)
    {
        perror("socket");
        exit(2);
    }
 
    // get device MAC address
    strncpy(iface.ifr_name, interface, IFNAMSIZ);
    if ((ioctl(sd, SIOCGIFHWADDR, &iface)) == -1) {
        perror("ioctl");
        close(sd);
        exit(3);
    }
 
    // fake MAC address is just last 8 bits of real MAC incremented by 1
    iface.ifr_hwaddr.sa_data[5]++;
 
    memcpy(smac, &(iface.ifr_hwaddr.sa_data), ETH_ALEN);                    // Source IP
    printf("Fake MAC address is %02x:%02x:%02x:%02x:%02x:%02x\n",
        (unsigned char)smac[0], (unsigned char)smac[1],
        (unsigned char)smac[2], (unsigned char)smac[3],
        (unsigned char)smac[4], (unsigned char)smac[5]);
 
 
    // get device index
    strncpy(iface.ifr_name, interface, IFNAMSIZ);
    if (ioctl(sd, SIOCGIFINDEX, &iface) == -1) {
        perror("SIOCGIFINDEX");
        close(sd);
        exit(6);
    }
 
    memset(&mr, 0, sizeof(mr));
    mr.mr_ifindex = iface.ifr_ifindex;
    mr.mr_type = PACKET_MR_PROMISC;
 
    // set promiscous mode
    if (setsockopt(sd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &mr, sizeof(mr)) == -1) {
         perror("setsockopt");
         close(sd);
         exit(7);
    }
    // prepare linux packet filter
    if ((filter = (struct sock_filter *)malloc(sizeof(macfilter))) == NULL) {
        perror("malloc");
        close(sd);
        exit(4);
    }
 
#ifdef PROMISCFILTER
    memcpy(filter, &promiscfilter, sizeof(promiscfilter));
 
    fcode.filter = filter;
    fcode.len = sizeof(promiscfilter)/sizeof(struct sock_filter);
#else
    memcpy(filter, &macfilter, sizeof(macfilter));
 
    // adjust for fake MAC address
    filter[1].k =
       (smac[2] & 0xff) << 24 |
       (smac[3] & 0xff) << 16 |
       (smac[4] & 0xff) << 8  |
       (smac[5] & 0xff);
    filter[3].k =
       (smac[0] & 0xff) << 8 |
       (smac[1] & 0xff);
 
    fcode.filter = filter;
    fcode.len = sizeof(macfilter)/sizeof(struct sock_filter);
#endif
 
#if 0
    // add filter
    if (setsockopt(sd, SOL_SOCKET, SO_ATTACH_FILTER, &fcode, sizeof(fcode)) == -1) {
        perror("setsockopt");
        close(sd);
        exit(5);
    }
#endif 
 
    iphdr = (struct iphdr *)(packet + sizeof(struct ether_header));
    eth   = (struct ether_header *) packet;
    arp   = (struct arphdr *)(packet + sizeof(struct ether_header));
 
   // process packets
    while (1) {
        n = recvfrom(sd, packet, packetsize, 0, NULL, 0);
 
        if (n < 42) {
            perror("recvfrom");
            close(sd);
            exit(8);
        }
 
        // got an ARP match - so send the fake reply
        if (ntohs(eth->ether_type) == ETHERTYPE_ARP  && !strncmp(ipaddr_string(arp->dst_ip), argv[2], IP_DOTLEN)) {
 
            spoof_eth = (struct ether_header *)arppacket;
            spoof_arp = (struct arphdr *)(arppacket + sizeof(struct ether_header));
 
            // build ethernet header
            memcpy(spoof_eth->ether_dhost, eth->ether_shost, ETH_ALEN);         // Destination MAC
            memcpy(spoof_eth->ether_shost, smac, ETH_ALEN);                     // Source MAC
            spoof_eth->ether_type = htons(ETHERTYPE_ARP);                       // Packet type
 
            // build arp header
            spoof_arp->hw_type = htons(ARPHDR_ETHER);                           // Hardware address type
            spoof_arp->proto_type = htons(ETH_P_IP);                            // Protocol address type
            spoof_arp->ha_len = ETH_ALEN;                                       // Hardware address length
            spoof_arp->pa_len = IP_ALEN;                                        // Protocol address length
            spoof_arp->opcode = htons(ARPOP_REPLY);                             // ARP operation type
            memcpy(spoof_arp->src_addr, smac, ETH_ALEN);                        // Sender MAC
            memcpy(spoof_arp->src_ip, arp->dst_ip, IP_ALEN);                    // Source IP
            memcpy(spoof_arp->dst_add, arp->src_addr, ETH_ALEN);                // Target MAC
            memcpy(spoof_arp->dst_ip, arp->src_ip, IP_ALEN);                    // Target IP
 
            strncpy(addr.sa_data, interface, sizeof(addr.sa_data));
 
            printf("Sent ARP reply: %s is %02x:%02x:%02x:%02x:%02x:%02x\n",
               inet_ntoa(*(struct in_addr*)&spoof_arp->src_ip),
               (unsigned char)spoof_arp->src_addr[0], (unsigned char)spoof_arp->src_addr[1],
               (unsigned char)spoof_arp->src_addr[2], (unsigned char)spoof_arp->src_addr[3],
               (unsigned char)spoof_arp->src_addr[4], (unsigned char)spoof_arp->src_addr[5]);
 
            // set up link level information
            lladdr.sll_family = htons(PF_PACKET);
            lladdr.sll_protocol = htons(ETH_P_ALL);
            lladdr.sll_pkttype  = PACKET_OTHERHOST;
            lladdr.sll_halen = ETH_ALEN;
            lladdr.sll_ifindex = iface.ifr_ifindex;
            memcpy(&(lladdr.sll_addr), arp->src_addr, ETH_ALEN);
 
            if (sendto(sd, arppacket, packetsize, 0, (struct sockaddr *)&lladdr, sizeof(lladdr)) < 0) {
                perror("sendto");
                close(sd);
                exit(9);
            }
 

       } 
#if 1
	else if ((ntohs(eth->ether_type) == ETHERTYPE_IP)
                 && !strncmp(ipaddr_string((char *)&(iphdr->daddr)), argv[2], IP_DOTLEN)
                 && (iphdr->protocol == IPPROTO_ICMP) ) {
             icmphdr = (struct icmphdr *)(packet + sizeof (struct ether_header) + sizeof (struct iphdr));
 
             if (icmphdr->type == ICMP_ECHO) {
                 printf("Received ICMP ECHO from %s (code: %u  id: %u  seq: %u)\n", inet_ntoa(*(struct in_addr *)&iphdr->saddr),
                        ntohs(icmphdr->code) , ntohs(icmphdr->un.echo.id) , ntohs(icmphdr->un.echo.sequence));
 
                 // copy received packet so that we can swizzle bits and send back
                 spoof_packetsize = ntohs(iphdr->tot_len) + sizeof(struct ether_header);
                 if ((spoof_packet = (char *)malloc(spoof_packetsize)) == NULL) {
                     perror("malloc");
                     close(sd);
                     exit(10);
                 }
                 memcpy(spoof_packet, packet, spoof_packetsize);
 
                 // fix up ICMP header
                 spoof_icmphdr = ((struct icmphdr *)(spoof_packet + sizeof (struct ether_header) + sizeof (struct iphdr)));
                 spoof_icmphdr->type = ICMP_ECHOREPLY;
                 spoof_icmphdr->checksum = 0x0000;                              // has to be zero for checksum calculation
                 spoof_icmphdr->checksum = in_cksum((char *)spoof_icmphdr,
                      (spoof_packetsize - sizeof (struct ether_header) - sizeof (struct iphdr)) );
 
                // fix up IP header
                 spoof_iph = (struct iphdr *)(spoof_packet + sizeof(struct ether_header));
                 memcpy(&(spoof_iph->saddr), &(iphdr->daddr), IP_ALEN);         // source IP
                 memcpy(&(spoof_iph->daddr), &(iphdr->saddr), IP_ALEN);         // target IP
 
                 // fix up ethernet header
                 spoof_eth = (struct ether_header *)spoof_packet;
                 memcpy(spoof_eth->ether_dhost, eth->ether_shost, ETH_ALEN);    // destination MAC
                 memcpy(spoof_eth->ether_shost, smac, ETH_ALEN);                // source MAC
 
                 // set up link level information
                 lladdr.sll_family = htons(PF_PACKET);
                 lladdr.sll_protocol = htons(ETH_P_ALL);
                 lladdr.sll_pkttype  = PACKET_OTHERHOST;
                 lladdr.sll_halen = ETH_ALEN;
                 lladdr.sll_ifindex = iface.ifr_ifindex;
                 memcpy(&(lladdr.sll_addr), arp->src_addr, ETH_ALEN);
 
                 if (sendto(sd, spoof_packet, spoof_packetsize, 0, (struct sockaddr *)&lladdr, sizeof(lladdr)) < 0) {
                     perror("sendto");
                     free(spoof_packet);
                     close(sd);
                     exit(11);
                 }
 
                 free(spoof_packet);
             }
 
       }
 
   }
#else
	else if (ntohs(eth->ether_type) == ETHERTYPE_IP) {
            optionLen = iphdr->ihl * 4 - sizeof(struct iphdr);
	    printf("ip packet recv and option len %d , proto 0x%x, 0x%x\n", optionLen, iphdr->protocol, IPPROTO_ICMP);
            if(iphdr->protocol == IPPROTO_ICMP) {
	    printf("icmp packet recv \n");
	    }
	}
   }
#endif
 
   close(sd);
 
   exit(0);
}
 
