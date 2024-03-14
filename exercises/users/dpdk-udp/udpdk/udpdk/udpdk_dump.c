//
// Created by leoll2 on 11/19/20.
// Copyright (c) 2020 Leonardo Lai. All rights reserved.
//
// The following code derives in part from netmap pkt-gen.c
//

#include <ctype.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <string.h>

#include <rte_ethdev.h>
#include <rte_byteorder.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_icmp.h>
#include "udpdk_dump.h"
#include "icmp.h"

#define IP_BUF_LEN 24

static const char *
ip_proto_name(uint16_t ip_proto)
{
	static const char * ip_proto_names[] = {
		"IP6HOPOPTS", /**< IP6 hop-by-hop options */
		"ICMP",       /**< control message protocol */
		"IGMP",       /**< group mgmt protocol */
		"GGP",        /**< gateway^2 (deprecated) */
		"IPv4",       /**< IPv4 encapsulation */

		"UNASSIGNED",
		"TCP",        /**< transport control protocol */
		"ST",         /**< Stream protocol II */
		"EGP",        /**< exterior gateway protocol */
		"PIGP",       /**< private interior gateway */

		"RCC_MON",    /**< BBN RCC Monitoring */
		"NVPII",      /**< network voice protocol*/
		"PUP",        /**< pup */
		"ARGUS",      /**< Argus */
		"EMCON",      /**< EMCON */

		"XNET",       /**< Cross Net Debugger */
		"CHAOS",      /**< Chaos*/
		"UDP",        /**< user datagram protocol */
		"MUX",        /**< Multiplexing */
		"DCN_MEAS",   /**< DCN Measurement Subsystems */

		"HMP",        /**< Host Monitoring */
		"PRM",        /**< Packet Radio Measurement */
		"XNS_IDP",    /**< xns idp */
		"TRUNK1",     /**< Trunk-1 */
		"TRUNK2",     /**< Trunk-2 */

		"LEAF1",      /**< Leaf-1 */
		"LEAF2",      /**< Leaf-2 */
		"RDP",        /**< Reliable Data */
		"IRTP",       /**< Reliable Transaction */
		"TP4",        /**< tp-4 w/ class negotiation */

		"BLT",        /**< Bulk Data Transfer */
		"NSP",        /**< Network Services */
		"INP",        /**< Merit Internodal */
		"SEP",        /**< Sequential Exchange */
		"3PC",        /**< Third Party Connect */

		"IDPR",       /**< InterDomain Policy Routing */
		"XTP",        /**< XTP */
		"DDP",        /**< Datagram Delivery */
		"CMTP",       /**< Control Message Transport */
		"TPXX",       /**< TP++ Transport */

		"ILTP",       /**< IL transport protocol */
		"IPv6_HDR",   /**< IP6 header */
		"SDRP",       /**< Source Demand Routing */
		"IPv6_RTG",   /**< IP6 routing header */
		"IPv6_FRAG",  /**< IP6 fragmentation header */

		"IDRP",       /**< InterDomain Routing*/
		"RSVP",       /**< resource reservation */
		"GRE",        /**< General Routing Encap. */
		"MHRP",       /**< Mobile Host Routing */
		"BHA",        /**< BHA */

		"ESP",        /**< IP6 Encap Sec. Payload */
		"AH",         /**< IP6 Auth Header */
		"INLSP",      /**< Integ. Net Layer Security */
		"SWIPE",      /**< IP with encryption */
		"NHRP",       /**< Next Hop Resolution */

		"UNASSIGNED",
		"UNASSIGNED",
		"UNASSIGNED",
		"ICMPv6",     /**< ICMP6 */
		"IPv6NONEXT", /**< IP6 no next header */

		"Ipv6DSTOPTS",/**< IP6 destination option */
		"AHIP",       /**< any host internal protocol */
		"CFTP",       /**< CFTP */
		"HELLO",      /**< "hello" routing protocol */
		"SATEXPAK",   /**< SATNET/Backroom EXPAK */

		"KRYPTOLAN",  /**< Kryptolan */
		"RVD",        /**< Remote Virtual Disk */
		"IPPC",       /**< Pluribus Packet Core */
		"ADFS",       /**< Any distributed FS */
		"SATMON",     /**< Satnet Monitoring */

		"VISA",       /**< VISA Protocol */
		"IPCV",       /**< Packet Core Utility */
		"CPNX",       /**< Comp. Prot. Net. Executive */
		"CPHB",       /**< Comp. Prot. HeartBeat */
		"WSN",        /**< Wang Span Network */

		"PVP",        /**< Packet Video Protocol */
		"BRSATMON",   /**< BackRoom SATNET Monitoring */
		"ND",         /**< Sun net disk proto (temp.) */
		"WBMON",      /**< WIDEBAND Monitoring */
		"WBEXPAK",    /**< WIDEBAND EXPAK */

		"EON",        /**< ISO cnlp */
		"VMTP",       /**< VMTP */
		"SVMTP",      /**< Secure VMTP */
		"VINES",      /**< Banyon VINES */
		"TTP",        /**< TTP */

		"IGP",        /**< NSFNET-IGP */
		"DGP",        /**< dissimilar gateway prot. */
		"TCF",        /**< TCF */
		"IGRP",       /**< Cisco/GXS IGRP */
		"OSPFIGP",    /**< OSPFIGP */

		"SRPC",       /**< Strite RPC protocol */
		"LARP",       /**< Locus Address Resolution */
		"MTP",        /**< Multicast Transport */
		"AX25",       /**< AX.25 Frames */
		"4IN4",       /**< IP encapsulated in IP */

		"MICP",       /**< Mobile Int.ing control */
		"SCCSP",      /**< Semaphore Comm. security */
		"ETHERIP",    /**< Ethernet IP encapsulation */
		"ENCAP",      /**< encapsulation header */
		"AES",        /**< any private encr. scheme */

		"GMTP",       /**< GMTP */
		"IPCOMP",     /**< payload compression (IPComp) */
		"UNASSIGNED",
		"UNASSIGNED",
		"PIM",        /**< Protocol Independent Mcast */
	};

	if (ip_proto < sizeof(ip_proto_names) / sizeof(ip_proto_names[0]))
		return ip_proto_names[ip_proto];
	switch (ip_proto) {
#ifdef IPPROTO_PGM
	case IPPROTO_PGM:  /**< PGM */
		return "PGM";
#endif
	case IPPROTO_SCTP:  /**< Stream Control Transport Protocol */
		return "SCTP";
#ifdef IPPROTO_DIVERT
	case IPPROTO_DIVERT: /**< divert pseudo-protocol */
		return "DIVERT";
#endif
	case IPPROTO_RAW: /**< raw IP packet */
		return "RAW";
	default:
		break;
	}
	return "UNASSIGNED";
}

/* Print the content of the packet in hex and ASCII */
void udpdk_dump_payload(const char *payload, int len)
{
	char buf[128];
	int i, j, i0;
	const unsigned char *p = (const unsigned char *)payload;

    printf("Dumping payload [len = %d]:\n", len);

	/* hexdump routine */
	for (i = 0; i < len; ) {
		memset(buf, ' ', sizeof(buf));
		sprintf(buf, "%5d: ", i);
		i0 = i;
		for (j = 0; j < 16 && i < len; i++, j++)
			sprintf(buf + 7 + j*3, "%02x ", (uint8_t)(p[i]));
		i = i0;
		for (j = 0; j < 16 && i < len; i++, j++)
			sprintf(buf + 7 + j + 48, "%c",
				isprint(p[i]) ? p[i] : '.');
		printf("%s\n", buf);
	}
}

void udpdk_dump_mbuf(struct rte_mbuf *m)
{
    udpdk_dump_payload(rte_pktmbuf_mtod(m, char *), rte_pktmbuf_data_len(m));
    
}
static void ip_format_addr(char *buf, uint16_t size,const uint32_t ip_addr)
{
    snprintf(buf, size, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 ,
             (uint8_t)((ip_addr >> 24) & 0xff),
             (uint8_t)((ip_addr >> 16) & 0xff),
             (uint8_t)((ip_addr >> 8) & 0xff),
             (uint8_t)((ip_addr)&0xff));
}
void udpdk_dump_eth(struct rte_mbuf *m)
{
     struct rte_ether_hdr *eth_hdr;
     struct rte_ipv4_hdr *ip_hdr;
     const char * dst_ip = "10.10.103.251";
     uint32_t ip;
     uint16_t eth_type, total_length;
     uint16_t hdr_len; 
     char buf[IP_BUF_LEN] = {0};
     eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
     eth_type = rte_cpu_to_be_16(eth_hdr->ether_type);
     inet_pton(AF_INET, dst_ip, &ip);
     if (eth_type ==  RTE_ETHER_TYPE_IPV4)
     {
          ip_hdr = (struct rte_ipv4_hdr *)((char *)eth_hdr + sizeof(struct rte_ether_hdr));
          total_length = rte_be_to_cpu_16(ip_hdr->total_length);
          if(ip != ip_hdr->dst_addr){
              return;
          }
          memset(buf,IP_BUF_LEN,0);
          //ip_format_addr(buf,IP_BUF_LEN,ip_hdr->src_addr); 
          ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->src_addr)); 
          printf("src ip : %s, ",buf);
          memset(buf,IP_BUF_LEN,0);
          ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->dst_addr)); 
          //ip_format_addr(buf,IP_BUF_LEN,ip_hdr->dst_addr); 
          printf("dst ip : %s",buf);
          hdr_len = ip4_hdrlen(m);
          printf(" payload len %u, ip hdrlen %u, next proto %s \n",total_length , hdr_len,ip_proto_name(ip_hdr->next_proto_id));
          if(IPPROTO_ICMP == ip_hdr->next_proto_id)
          {
          	struct rte_icmp_hdr *icmph;
                icmph = (struct rte_icmp_hdr *)((char*)ip_hdr + hdr_len); 
                printf("icmp type: %u ", icmph->icmp_type); 
                switch (icmph->icmp_type) {
                    case ICMP_DEST_UNREACH:
                            switch (icmph->icmp_code & 15) {
                            case ICMP_NET_UNREACH:
                                    printf("icmp net unreach\n");
                                    break;
                            case ICMP_HOST_UNREACH:
                                    printf("icmp host unreach\n");
                                    break;
                            case ICMP_PROT_UNREACH:
                                    printf("icmp proto unreach\n");
                                    break;
                            case ICMP_PORT_UNREACH:
                                    printf("icmp port unreach\n");
                                    break;
                            case ICMP_FRAG_NEEDED:
                                    printf("icmp frag needded \n");
                            case ICMP_SR_FAILED:
                                    break;
                            default:
                                    break;
                            }
                     default:
                         break;
                }
           }
    }
}
