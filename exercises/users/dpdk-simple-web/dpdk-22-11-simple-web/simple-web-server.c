/*-
 *   BSD LICENSE
 *
 *   Copyright(c) 2010-2015 Intel Corporation. All rights reserved.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* simple-web.c: Simple WEB Server using DPDK. 
*/

#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_gro.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>
#include <linux/if_arp.h>
//#include <linux/ip.h>
#include <linux/icmp.h>
#include <linux/tcp.h>
#include <arpa/inet.h>
#include "printreq.h"

#define RX_RING_SIZE 128
#define TX_RING_SIZE 512

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

#define TCPMSS 1400

static const struct rte_eth_conf port_conf_default = {
	.rxmode = { .mtu = RTE_ETHER_MAX_LEN }
};

typedef unsigned long int uint32;
typedef unsigned short int uint16;

struct __attribute__((packed)) arp_header
{
	unsigned short arp_hd;
	unsigned short arp_pr;
	unsigned char arp_hdl;
	unsigned char arp_prl;
	unsigned short arp_op;
	unsigned char arp_sha[6];
	unsigned char arp_spa[4];
	unsigned char arp_dha[6];
	unsigned char arp_dpa[4];
};
#define IPV4_MASK                       (0xf0)
#define IPV4_VERION                     (0x40)
#define TCP_HDR_LEN(tcp_hdr)            (((tcp_hdr)->data_off & 0xf0) >> 2)
#if 0
#define ETH_ALEN 6
#define ETH_HLEN 14
#define ETH_ZLEN 60     /* Min. octets in frame without FCS */
#define ETH_FCS_LEN 4
#define ETH_MTU 1500

struct eth_header {
    uint8_t  h_dest[ETH_ALEN];   /* destination eth addr */
    uint8_t  h_source[ETH_ALEN]; /* source ether addr    */
    uint16_t h_proto;            /* packet type ID field */
};
struct ether_addr {
	uint8_t addr_bytes[ETHER_ADDR_LEN]; /**< Addr bytes in tx order */
} __attribute__((__packed__));
#endif
struct rte_ether_addr my_eth_addr;	// My ethernet address
uint32 my_ip;  			// My IP Address in network order
uint16 tcp_port; 		// listen tcp port in network order


int user_init_func(int , char *[]);

char * INET_NTOA(uint32 ip);
void swap_bytes(unsigned char *a, unsigned char *b, int len);
void dump_packet(unsigned char *buf, int len);
void dump_arp_packet(struct ethhdr *eh);
int process_arp(struct rte_mbuf *mbuf, struct ethhdr *eh);

char * INET_NTOA(uint32 ip)	// ip is network order
{
	static char buf[100];
	sprintf(buf,"%d.%d.%d.%d",
		(int)(ip&0xff), (int)((ip>>8)&0xff), (int)((ip>>16)&0xff), (int)((ip>>24)&0xff));
	return buf;
}

void swap_bytes(unsigned char *a, unsigned char *b, int len)
{
	unsigned char t;
	int i;
	if (len <= 0)
		return;
	for (i = 0; i < len; i++) {
		t = *(a + i);
		*(a + i) = *(b + i);
		*(b + i) = t;
	}
}

/*
 * Initializes a given port using global settings and with the RX buffers
 * coming from the mbuf_pool passed as a parameter.
 */
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
	struct rte_eth_conf port_conf = port_conf_default;
	const uint16_t rx_rings = 1, tx_rings = 1;
	uint16_t nb_rxd = RX_RING_SIZE;
	uint16_t nb_txd = TX_RING_SIZE;
	int retval;
	uint16_t q;

	if (port >= rte_eth_dev_count_avail())
		return -1;

	/* Configure the Ethernet device. */
	retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
	if (retval != 0)
		return retval;

	retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
	if (retval != 0)
		return retval;

	/* Allocate and set up 1 RX queue per Ethernet port. */
	for (q = 0; q < rx_rings; q++) {
		retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
				rte_eth_dev_socket_id(port), NULL, mbuf_pool);
		if (retval < 0)
			return retval;
	}

	/* Allocate and set up 1 TX queue per Ethernet port. */
	for (q = 0; q < tx_rings; q++) {
		retval = rte_eth_tx_queue_setup(port, q, nb_txd,
				rte_eth_dev_socket_id(port), NULL);
		if (retval < 0)
			return retval;
	}

	/* Start the Ethernet port. */
	retval = rte_eth_dev_start(port);
	if (retval < 0)
		return retval;

	/* Display the port MAC address. */
	struct rte_ether_addr addr;
	rte_eth_macaddr_get(port, &addr);
	printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
			   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
			port,
			addr.addr_bytes[0], addr.addr_bytes[1],
			addr.addr_bytes[2], addr.addr_bytes[3],
			addr.addr_bytes[4], addr.addr_bytes[5]);

	my_eth_addr = addr;
	/* Enable RX in promiscuous mode for the Ethernet device. */
	// rte_eth_promiscuous_enable(port);

	return 0;
}

void dump_packet(unsigned char *buf, int len)
{
	printf("packet buf=%p len=%d\n",buf,len);
	int i,j;
	unsigned char c;
	for(i=0;i<len;i++) {
		printf("%02X",buf[i]);
		if(i%16 == 7) 
			printf("  ");
		if((i%16)==15 || (i==len-1)) {
			if(i%16 < 7) printf("  ");
			for(j=0;j<15-(i%16);j++) printf("  ");
			printf(" | ");
			for(j=(i-(i%16));j<=i;j++) {
				c=buf[j];
				if((c>31)&&(c<127))
					printf("%c",c);
				else
					printf(".");
			}
			printf("\n");
		}
	}
}

void dump_arp_packet(struct ethhdr *eh)
{
	struct arp_header *ah;
	ah = (struct arp_header*) ((unsigned char *)eh + 14);
	printf("+++++++++++++++++++++++++++++++++++++++\n" );
	printf("ARP PACKET: %p \n",eh);
	printf("ETHER DST MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
		eh->h_dest[0], eh->h_dest[1], eh->h_dest[2], eh->h_dest[3],
		eh->h_dest[4], eh->h_dest[5]);
	printf("ETHER SRC MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
		eh->h_source[0], eh->h_source[1], eh->h_source[2], eh->h_source[3], eh->h_source[4],
		eh->h_source[5]);
	printf("H/D TYPE : %x PROTO TYPE : %x \n",ah->arp_hd,ah->arp_pr);
	printf("H/D leng : %x PROTO leng : %x \n",ah->arp_hdl,ah->arp_prl);
	printf("OPERATION : %x \n", ah->arp_op);
	printf("SENDER MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
		ah->arp_sha[0], ah->arp_sha[1], ah->arp_sha[2], ah->arp_sha[3],
		ah->arp_sha[4], ah->arp_sha[5]);
	printf("SENDER IP address: %02d:%02d:%02d:%02d\n",
		ah->arp_spa[0], ah->arp_spa[1], ah->arp_spa[2], ah->arp_spa[3]);
	printf("TARGET MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
		ah->arp_dha[0], ah->arp_dha[1], ah->arp_dha[2], ah->arp_dha[3],
		ah->arp_dha[4], ah->arp_dha[5]);
	printf("TARGET IP address: %02d:%02d:%02d:%02d\n",
		ah->arp_dpa[0], ah->arp_dpa[1], ah->arp_dpa[2], ah->arp_dpa[3]);
}

// #define DEBUGARP

int process_arp(struct rte_mbuf *mbuf, struct ethhdr *eh)
{
	struct arp_header *ah;
	ah = (struct arp_header*) ((unsigned char *)eh + 14);
#ifdef DEBUGARP
	dump_arp_packet(eh);
#endif
	if(htons(ah->arp_op) != 0x0001) { // ARP request
		return 0;
	}
	if(memcmp((unsigned char*)&my_ip, (unsigned char*)ah->arp_dpa, 4)==0) {
#ifdef DEBUGARP
		printf("Asking me....\n");
#endif
		memcpy((unsigned char*)eh->h_dest, (unsigned char*)eh->h_source, 6);
		memcpy((unsigned char*)eh->h_source, (unsigned char*)&my_eth_addr, 6);
		ah->arp_op=htons(0x2);
		memcpy((unsigned char*)ah->arp_dha, (unsigned char*)ah->arp_sha, 6);
		memcpy((unsigned char*)ah->arp_dpa, (unsigned char*)ah->arp_spa, 4);
		memcpy((unsigned char*)ah->arp_sha, (unsigned char*)&my_eth_addr, 6);
		memcpy((unsigned char*)ah->arp_spa, (unsigned char*)&my_ip, 4);
#ifdef DEBUGARP
		printf("I will reply following \n");
		dump_arp_packet(eh);
#endif
		if(likely(1 == rte_eth_tx_burst(0, 0, &mbuf, 1)))
			return 1;
	}
	return 0;
}

unsigned short packet_chksum(unsigned short *addr,int len);
int process_icmp(struct rte_mbuf *mbuf, struct ethhdr *eh, struct iphdr *iph, int iphdrlen, int len);

unsigned short packet_chksum(unsigned short *addr,int len)
{
	int nleft=len;
	int sum=0;
	unsigned short *w=addr;
	unsigned short answer=0;
	while(nleft>1) {
		sum+=*w++;
		nleft-=2;
	}
	if(nleft==1) {
		*(unsigned char *)(&answer)=*(unsigned char *)w;
		sum+=answer;
        }
	sum=(sum>>16)+(sum&0xffff);
	sum+=(sum>>16);
	answer=~sum;
	return answer;
}

// #define DEBUGICMP

int process_icmp(struct rte_mbuf *mbuf, struct ethhdr *eh, struct iphdr *iph, int iphdrlen, int len)
{
	struct icmphdr *icmph = (struct icmphdr *)((unsigned char*)(iph)+iphdrlen);
#ifdef DEBUGICMP
	printf("icmp type=%d, code=%d\n",icmph->type,icmph->code);
#endif
	if((icmph->type==8) && (icmph->code==0)) {  // ICMP echo req
		memcpy((unsigned char*)eh->h_dest, (unsigned char*)eh->h_source, 6);
		memcpy((unsigned char*)eh->h_source, (unsigned char*)&my_eth_addr, 6);
		memcpy((unsigned char*)&iph->daddr, (unsigned char*)&iph->saddr, 4);
		memcpy((unsigned char*)&iph->saddr, (unsigned char*)&my_ip, 4);
		icmph->type=0;
		icmph->checksum=0;
		icmph->checksum = packet_chksum((unsigned short*)icmph, len - 14 - iphdrlen );
#ifdef DEBUGICMP
		printf("I will send reply\n");
		dump_packet(rte_pktmbuf_mtod(mbuf, unsigned char*), len);
#endif
		int ret	= rte_eth_tx_burst(0, 0, &mbuf, 1);
		if(ret==1) return 1;
		printf("send icmp packet ret = %d\n",ret);
	}
	return 0;
}

u_int16_t tcp_sum_calc(u_int16_t len_tcp, u_int16_t src_addr[], u_int16_t dest_addr[], u_int16_t buff[]);

// function from http://www.bloof.de/tcp_checksumming, thanks to crunsh
u_int16_t tcp_sum_calc(u_int16_t len_tcp, u_int16_t src_addr[], u_int16_t dest_addr[], u_int16_t buff[])
{
	u_int16_t prot_tcp = 6;
	u_int32_t sum = 0;
	int nleft = len_tcp;
	u_int16_t *w = buff;

	/* calculate the checksum for the tcp header and payload */
	while (nleft > 1) {
		sum += *w++;
		nleft -= 2;
	}

	/* if nleft is 1 there ist still on byte left. We add a padding byte (0xFF) to build a 16bit word */
	if (nleft > 0)
		sum += *w & ntohs(0xFF00);	/* Thanks to Dalton */

	/* add the pseudo header */
	sum += src_addr[0];
	sum += src_addr[1];
	sum += dest_addr[0];
	sum += dest_addr[1];
	sum += htons(len_tcp);
	sum += htons(prot_tcp);

	// keep only the last 16 bits of the 32 bit calculated sum and add the carries
	sum = (sum >> 16) + (sum & 0xFFFF);
	sum += (sum >> 16);

	// Take the one's complement of sum
	sum = ~sum;

	return ((u_int16_t) sum);
}

static void set_tcp_checksum(struct iphdr *ip);

static void set_tcp_checksum(struct iphdr *ip)
{
	struct tcphdr *tcph = (struct tcphdr *)((u_int8_t *) ip + (ip->ihl << 2));
	tcph->check = 0;	/* Checksum field has to be set to 0 before checksumming */
	tcph->check = (u_int16_t) tcp_sum_calc((u_int16_t) (ntohs(ip->tot_len) - ip->ihl * 4),
		       (u_int16_t *) & ip->saddr, (u_int16_t *) & ip->daddr, (u_int16_t *) tcph);
	ip->check=0;
	ip->check = packet_chksum((unsigned short *) ip, ip->ihl<<2);
}

//int process_http(unsigned char *http_req, int req_len, unsigned char *http_resp, int *resp_len, int *resp_in_req);

// #define DEBUGTCP

int process_tcp(struct rte_mbuf *mbuf, struct ethhdr *eh, struct iphdr *iph, int iphdrlen);

int process_tcp(struct rte_mbuf *mbuf, struct ethhdr *eh, struct iphdr *iph, int iphdrlen)
{
	struct tcphdr *tcph = (struct tcphdr *)((unsigned char*)(iph)+iphdrlen);
	int pkt_len;
#ifdef DEBUGTCP
	printf("TCP packet, dport=%d\n",ntohs(tcph->dest));
	printf("TCP syn=%d ack=%d fin=%d\n",tcph->syn, tcph->ack, tcph->fin);
#endif
	if (tcph->dest != tcp_port) 
		return 0;

	if (tcph->syn && (!tcph->ack)) {	// SYN packet, send SYN+ACK
#ifdef DEBUGTCP
		printf("SYN packet\n");
#endif
		swap_bytes((unsigned char *)&eh->h_source, (unsigned char *)&eh->h_dest, 6);
		swap_bytes((unsigned char *)&iph->saddr, (unsigned char *)&iph->daddr, 4);
		swap_bytes((unsigned char *)&tcph->source, (unsigned char *)&tcph->dest, 2);
		tcph->ack = 1;
		tcph->ack_seq = htonl(ntohl(tcph->seq) + 1);
		tcph->seq = htonl(1);
		tcph->doff = 20 / 4;
		pkt_len = iph->ihl * 4 + tcph->doff * 4;
		iph->tot_len = htons(pkt_len);
		iph->check = 0;
		rte_pktmbuf_data_len(mbuf) = pkt_len + 14;
		set_tcp_checksum(iph);
#ifdef DEBUGTCP
		printf("I will reply following \n");
		dump_packet((unsigned char *)eh, rte_pktmbuf_data_len(mbuf));
#endif
		int ret = rte_eth_tx_burst(0, 0, &mbuf, 1);
		if(ret == 1)
			return 1;
#ifdef DEBUGTCP
		printf("send tcp packet return %d\n", ret);
#endif
		return 0;
	} else if (tcph->fin) {	// FIN packet, send ACK
#ifdef DEBUGTCP
		fprintf(stderr, "FIN packet\n");
#endif
		swap_bytes((unsigned char *)&eh->h_source, (unsigned char *)&eh->h_dest, 6);
		swap_bytes((unsigned char *)&iph->saddr, (unsigned char *)&iph->daddr, 4);
		swap_bytes((unsigned char *)&tcph->source, (unsigned char *)&tcph->dest, 2);
		swap_bytes((unsigned char *)&tcph->seq, (unsigned char *)&tcph->ack_seq, 4);
		tcph->ack = 1;
		tcph->ack_seq = htonl(ntohl(tcph->ack_seq) + 1);
		tcph->doff = 20 / 4;
		pkt_len = iph->ihl * 4 + tcph->doff * 4;
		iph->tot_len = htons(pkt_len);
		iph->check = 0;
		rte_pktmbuf_data_len(mbuf) = pkt_len + 14;
		set_tcp_checksum(iph);
#ifdef DEBUGTCP
		printf("I will reply following \n");
		dump_packet((unsigned char *)eh, rte_pktmbuf_data_len(mbuf));
#endif
		int ret = rte_eth_tx_burst(0, 0, &mbuf, 1);
		if(ret == 1)
			return 1;
#ifdef DEBUGTCP
		fprintf(stderr, "send tcp packet return %d\n", ret);
#endif
		return 0;
	}  else if (tcph->ack && (!tcph->syn)) {	// ACK packet, send DATA
		pkt_len = ntohs(iph->tot_len);
		int tcp_payload_len = pkt_len - iph->ihl * 4 - tcph->doff * 4;
		int ntcp_payload_len = TCPMSS;
		unsigned char *tcp_payload;
		unsigned char buf[TCPMSS]; // http_respone
		int resp_in_req=0;

#ifdef DEBUGTCP
		printf("ACK pkt len=%d(inc ether) ip len=%d\n", rte_pktmbuf_data_len(mbuf), pkt_len);
		printf("tcp payload len=%d\n", tcp_payload_len);
#endif
		if (tcp_payload_len <= 5) {
#ifdef DEBUGTCP
			printf("tcp payload len=%d too small, ignore\n", tcp_payload_len);
#endif
			return 0;
		}
		tcp_payload = (unsigned char*)iph + iph->ihl * 4 + tcph->doff * 4;
		if(process_http(tcp_payload, tcp_payload_len, buf, &ntcp_payload_len, &resp_in_req)==0)
			return 0;
#ifdef DEBUGTCP
		printf("new payload len=%d :%s:\n",ntcp_payload_len, buf);
#endif
#ifdef DEBUGCLIENT
                int i = 0, count = 0;
                for(;i < tcp_payload_len; ++ i)
                {
                      if('a' == tcp_payload[i])
                          ++ count;
                }
		printf("payload len=%d, number of a %d ,%s:\n",tcp_payload_len,count, tcp_payload);
#endif
		uint32 ack_seq = htonl(ntohl(tcph->seq) + tcp_payload_len);
		swap_bytes((unsigned char *)&eh->h_source, (unsigned char *)&eh->h_dest, 6);
		swap_bytes((unsigned char *)&iph->saddr, (unsigned char *)&iph->daddr, 4);
		swap_bytes((unsigned char *)&tcph->source, (unsigned char *)&tcph->dest, 2);
		if(!resp_in_req)
			memcpy(tcp_payload, buf, ntcp_payload_len);
		pkt_len = ntcp_payload_len + iph->ihl * 4 + tcph->doff * 4;
		iph->tot_len = htons(pkt_len);
#ifdef DEBUGTCP
		fprintf(stderr, "new pkt len=%d\n", pkt_len);
#endif
		tcph->ack = 1;
		tcph->psh = 1;
		tcph->fin = 1;
		tcph->seq = tcph->ack_seq;
		tcph->ack_seq = ack_seq;
		iph->check = 0;
		rte_pktmbuf_data_len(mbuf) = pkt_len + 14;
		set_tcp_checksum(iph);
#ifdef DEBUGTCP
		printf("I will reply following \n");
		dump_packet((unsigned char *)eh, rte_pktmbuf_data_len(mbuf));
#endif
		int ret = rte_eth_tx_burst(0, 0, &mbuf, 1);
		if(ret == 1)
			return 1;
#ifdef DEBUGTCP
		fprintf(stderr, "send tcp packet return %d\n", ret);
#endif
		return 0;
	}
	return 0;
}

/*
 * The lcore main. This is the main thread that does the work, reading from
 * an input port and writing to an output port.
 */
static __attribute__((noreturn)) void
lcore_main(void)
{
	const uint16_t nb_ports = rte_eth_dev_count_avail();
	uint16_t port;
        bool gro_enable = true;
        struct rte_gro_param gro_param = {
        .gro_types = RTE_GRO_TCP_IPV4,
        /* 8*16=128(max) */
        .max_flow_num = 8,
        .max_item_per_flow = 16,
        };
         uint16_t nb_rx;
	/*
	 * Check that the port is on the same NUMA node as the polling thread
	 * for best performance.
	 */
	for (port = 0; port < nb_ports; port++)
		if (rte_eth_dev_socket_id(port) > 0 &&
				rte_eth_dev_socket_id(port) !=
						(int)rte_socket_id())
			printf("WARNING, port %u is on remote NUMA node to "
					"polling thread.\n\tPerformance will "
					"not be optimal.\n", port);

	printf("\nCore %u forwarding packets. [Ctrl+C to quit]\n",
			rte_lcore_id());

	/* Run until the application is quit or killed. */
	for (;;) {
		/* Get burst of RX packets, from first port of pair. */
		int port = 0;
		int i;
		struct rte_mbuf *bufs[BURST_SIZE];
		nb_rx = rte_eth_rx_burst(port, 0, bufs, BURST_SIZE);
	        if (unlikely(nb_rx == 0))
		      continue;
                if(gro_enable)
                {
                    uint32_t i = 0;
                    for (i = 0; i < nb_rx; i++) {
                        struct rte_ether_hdr *ethh = rte_pktmbuf_mtod(bufs[i], struct rte_ether_hdr *);
                        uint16_t type  = ethh->ether_type;

                        bufs[i]->l2_len = sizeof(struct rte_ether_hdr);

                        if (RTE_ETHER_TYPE_IPV4 == rte_cpu_to_be_16(type)) {
                             struct rte_ipv4_hdr *iph = rte_pktmbuf_mtod_offset(bufs[i], struct rte_ipv4_hdr *,
                                 sizeof(struct rte_ether_hdr));
                             if (unlikely((iph->version_ihl & IPV4_MASK) != IPV4_VERION)) {
                                 rte_pktmbuf_free(bufs[i]);
                                 continue;
                             }
                             bufs[i]->l3_len = sizeof(struct rte_ipv4_hdr);
                             struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(bufs[i], struct rte_tcp_hdr *,\
                               sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
                             bufs[i]->l4_len = TCP_HDR_LEN(tcp_hdr);
                             bufs[i]->packet_type = RTE_PTYPE_L3_IPV4 | RTE_PTYPE_L4_TCP;
                    }
                    }
                    nb_rx =  rte_gro_reassemble_burst(bufs, nb_rx, &gro_param);
                }
#ifdef DEBUGPACKET
		printf("got %d packets\n",nb_rx);
#endif
		for(i=0;i<nb_rx;i++) {
			int len = rte_pktmbuf_data_len(bufs[i]);
			struct ethhdr *eh = rte_pktmbuf_mtod(bufs[i], struct ethhdr*);
#ifdef DEBUGPACKET
			dump_packet((unsigned char*)eh, len);
			printf("ethernet proto=%4X\n",htons(eh->h_proto));
#endif
			if(eh->h_proto==htons(0x0800)){  // IPv4 protocol
				struct iphdr *iph;
				iph = (struct iphdr*)((unsigned char*)(eh)+14);
				int iphdrlen=iph->ihl<<2;
#ifdef DEBUGPACKET
				printf("ver=%d, frag_off=%d, daddr=%s pro=%d\n",iph->version,ntohs(iph->frag_off)&0x1FFF,
						INET_NTOA(iph->daddr),iph->protocol);
#endif
				if((iph->version==4)&&((iph->frag_off&htons(0x1FFF))==0)&&(iph->daddr==my_ip)) {  // ipv4
#ifdef DEBUGPACKET
					printf("yes ipv4\n");
#endif
					if(iph->protocol==6 ) { // TCP
						if(process_tcp(bufs[i], eh, iph, iphdrlen))
							continue;
					} else if(iph->protocol==1) { // ICMP
						if(process_icmp(bufs[i], eh, iph, iphdrlen, len))
							continue;
					}
				}
			} else if(eh->h_proto == htons(0x0806)){  // ARP protocol
				if(process_arp(bufs[i], eh))
					continue;
			}
			rte_pktmbuf_free(bufs[i]);
		}
	}
}

/*
 * The main function, which does initialization and calls the per-lcore
 * functions.
 */
int
main(int argc, char *argv[])
{
	struct rte_mempool *mbuf_pool;
	unsigned nb_ports;

	/* Initialize the Environment Abstraction Layer (EAL). */
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

	argc -= ret;
	argv += ret;

	if(argc<3)
		rte_exit(EXIT_FAILURE, "You need tell me my IP and port\n");

	my_ip = inet_addr(argv[1]);

	tcp_port = htons(atoi(argv[2]));

	printf("My IP is: %s, port is %d\n", INET_NTOA(my_ip), ntohs(tcp_port));

	argc-=2;
	argv+=2;

	user_init_func(argc, argv);

	/* Check that there is an even number of ports to send/receive on. */
	nb_ports = rte_eth_dev_count_avail();
	if (nb_ports !=  1)
		rte_exit(EXIT_FAILURE, "Error: need 1 ports, but you have %d\n", nb_ports);

	/* Creates a new mempool in memory to hold the mbufs. */
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,
		MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

	if (mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	/* Initialize port. */
	if (port_init(0, mbuf_pool) != 0)
		rte_exit(EXIT_FAILURE, "Cannot init port %"PRIu16 "\n",
					0);
	printf("My ether addr is: %02X:%02X:%02X:%02X:%02X:%02X",
			my_eth_addr.addr_bytes[0], my_eth_addr.addr_bytes[1],
			my_eth_addr.addr_bytes[2], my_eth_addr.addr_bytes[3],
			my_eth_addr.addr_bytes[4], my_eth_addr.addr_bytes[5]);


	if (rte_lcore_count() > 1)
		printf("\nWARNING: Too many lcores enabled. Only 1 used.\n");

	/* Call lcore_main on the master core only. */
	lcore_main();

	return 0;
}
